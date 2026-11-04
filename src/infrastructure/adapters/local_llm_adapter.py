# src/infrastructure/adapters/local_llm_adapter.py
import base64
import io
import json
import logging
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.domain.exceptions.core_errors import DataIncompleteError, LLMInferenceError
from src.domain.models.agent import OrchestratorDecision
from src.domain.models.telemetry import LLMCallMetrics
from src.domain.ports.llm_engine import ILLMEngine
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_REQUEST = 3
MULTIMODAL_REJECTED_STATUSES = frozenset({400, 422})
TEXT_ONLY_RETRY_STATUSES = frozenset({400, 422})
TEXT_ONLY_RETRY_DELAY_SECONDS = 0.75
ROUTER_ALLOWED_TOOLS = frozenset({"buscar_manual", "analizar_imagen", "responder_directo"})
ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})

_ROUTER_RESPONSE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "tool_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "herramienta": {
                    "type": "string",
                    "enum": ["buscar_manual", "analizar_imagen", "responder_directo"],
                }
            },
            "required": ["herramienta"],
            "additionalProperties": False,
        },
    },
}

_IMAGE_MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}

_ROUTER_SYSTEM_PROMPT = (
    "Eres un clasificador de herramientas para un motor RAG de manuales tecnicos. "
    'Responde SOLO con un objeto JSON con la forma {"herramienta": "buscar_manual" | '
    '"analizar_imagen" | "responder_directo"}, sin explicaciones. '
    "Usa buscar_manual para CUALQUIER pregunta tecnica, conceptual o de procedimiento "
    "que pueda responderse con los manuales: definiciones, funcionamiento, pasos, "
    "configuracion o causas. "
    "Usa analizar_imagen solo si la pregunta depende de la imagen adjunta. "
    "Usa responder_directo SOLO para saludos, despedidas, agradecimientos o charla "
    "sin contenido tecnico."
)

SYSTEM_PROMPT = (
    "Eres un asistente técnico experto y analítico para manuales y documentación técnica. "
    "Tu única fuente de verdad es el contexto técnico que se te proporciona. "
    "{image_instruction}\n"
    "Reglas estrictas:\n"
    "- RESPONDE SIEMPRE con base en el contexto provisto. "
    "Si hay información relevante, aunque sea parcial, úsala.\n"
    "- NO digas 'No dispongo de información' si la respuesta se puede inferir "
    "o encontrar parcialmente en los fragmentos.\n"
    "- Explica paso a paso, procedimientos, rutas, comandos, nombres de herramientas "
    "o configuraciones cuando aparezcan en la documentación.\n"
    "- Cita la fuente y la página al final de las afirmaciones clave usando "
    "el formato [Archivo, Pág] si la metadata está disponible.\n"
    "- No inventes pasos, nombres o valores. Solo si tras revisar TODO el contexto "
    "no hay absolutamente nada relacionado, usa exactamente: "
    "'No dispongo de información en los manuales'.\n"
    "- Prioriza la utilidad práctica para el usuario sobre la cautela excesiva."
)

_IMAGE_INSTRUCTION = (
    "Usa también las imágenes/diagramas adjuntos únicamente si están presentes y son relevantes."
)


def _mime_from_magic(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def _image_mime_type(image_path: str, data: bytes | None = None) -> str:
    if data:
        detected = _mime_from_magic(data)
        if detected is not None:
            return detected
    extension = Path(image_path).suffix.lower().lstrip(".")
    return _IMAGE_MIME_TYPES.get(extension, "image/png")


def _vision_prompt(question: str | None) -> str:
    """Build a bounded vision prompt that stays focused on the operator intent."""
    clean = (question or "").strip()
    if not clean:
        return "Describe este diagrama tecnico brevemente."
    return (
        "Analiza la imagen en relacion con esta consulta del operador. "
        "Describe solo los elementos, valores y relaciones visibles que ayuden a "
        f"responderla.\nConsulta: {clean[:500]}"
    )


def _image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        return int(width), int(height)
    except Exception:
        # A corrupt or unsupported asset is still bounded by bytes and count;
        # pixel accounting simply remains unknown.
        return 0, 0


def _build_user_text(context_str: str, user_input: str, include_images: bool) -> str:
    attachment_note = " y las imágenes adjuntas" if include_images else ""
    return (
        f"Contexto técnico:\n{context_str}\n\n"
        f"Pregunta del usuario:\n{user_input}\n\n"
        f"Instrucción: Usa ÚNICAMENTE el contexto anterior{attachment_note}. "
        "Si existe información relacionada, genera una respuesta completa y "
        "accionable con referencias [Archivo, Pág]."
    )


def _with_no_think(prompt: str, model: str) -> str:
    """Append the Qwen3 no-think hint when the server ignores the flag.

    ``chat_template_kwargs`` is not honoured by every LM Studio build, but the
    ``/no_think`` marker in the prompt is. Other model families receive the
    prompt unchanged.
    """
    if "qwen3" in model.lower():
        return f"{prompt}\n/no_think"
    return prompt


def _server_error_code(response: requests.Response | None) -> str:
    """Map a server error body to a fixed, Zero-Log-safe diagnostic code.

    LM Studio reports validation problems as plain error strings. Only known
    markers are translated; the raw body is never logged because a proxy could
    echo prompt content back to the client.
    """
    if response is None:
        return "unknown"
    try:
        body = response.text[:2048].lower()
    except Exception:
        return "unknown"
    markers = (
        ("response_format", "invalid_response_format"),
        ("context length", "context_length_exceeded"),
        ("too long", "context_length_exceeded"),
        ("not found", "model_not_found"),
        ("loading", "model_loading"),
        ("busy", "model_busy"),
        ("out of memory", "out_of_memory"),
        ("overload", "overloaded"),
    )
    for marker, code in markers:
        if marker in body:
            return code
    return "unsupported_request"


def _safe_error_detail(response: requests.Response | None, limit: int = 512) -> str:
    """Return a bounded, redacted excerpt of an infrastructure error payload.

    The server body is useful for diagnosing HTTP 400s, but it must never leak
    document text, query text, Base64 data or absolute paths. Only an error
    field is extracted, then paths, data URIs and long opaque blobs are
    redacted and the result is truncated.
    """
    if response is None:
        return "none"
    try:
        raw = response.text
    except Exception:
        raw = ""
    try:
        parsed: object = response.json()
    except Exception:
        parsed = raw

    detail: object = parsed
    if isinstance(parsed, dict):
        error = parsed.get("error", parsed)
        if isinstance(error, dict):
            parts = [str(error[key]) for key in ("message", "type", "code") if error.get(key)]
            detail = " | ".join(parts) or parsed
        else:
            detail = error

    text = " ".join(str(detail).split())
    text = re.sub(r"data:[^\s\"']+", "<data-uri>", text)
    text = re.sub(r"(?:[A-Za-z]:\\|/)[^\s\"']{3,}", "<path>", text)
    text = re.sub(r"\b[A-Za-z0-9+/]{80,}={0,2}\b", "<blob>", text)
    return text[:limit] or "empty"


class _CircuitBreaker:
    """Minimal consecutive-failure circuit breaker with automatic recovery."""

    def __init__(self, failure_threshold: int, recovery_seconds: float) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._recovery_seconds = max(0.1, recovery_seconds)
        self._consecutive_failures = 0
        self._open_until: float | None = None

    def allow_request(self) -> bool:
        if self._open_until is None:
            return True
        if time.monotonic() >= self._open_until:
            self._open_until = None
            self._consecutive_failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._open_until = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._open_until = time.monotonic() + self._recovery_seconds


@dataclass(frozen=True)
class _PreparedImage:
    mime_type: str
    base64_data: str
    raw_bytes: int
    encoded_bytes: int
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class _ImageBudget:
    prepared: tuple[_PreparedImage, ...] = ()
    skipped: int = 0
    encoded_bytes: int = 0
    pixels: int = 0
    prep_seconds: float = 0.0

    def drop_last(self) -> "_ImageBudget":
        if not self.prepared:
            return self
        kept = self.prepared[:-1]
        return _ImageBudget(
            prepared=kept,
            skipped=self.skipped,
            encoded_bytes=sum(image.encoded_bytes for image in kept),
            pixels=sum(image.pixels for image in kept),
            prep_seconds=self.prep_seconds,
        )


class LocalLLMAdapter(ILLMEngine):
    def __init__(
        self,
        settings: Settings | None = None,
        endpoint: str | None = None,
        model_chat: str | None = None,
        model_vision: str | None = None,
        timeout: int | None = None,
        allowed_images_dir: str | Path | None = None,
    ) -> None:
        if settings is not None:
            endpoint = endpoint or settings.llm_endpoint
            model_chat = model_chat or settings.llm_model_chat
            model_vision = model_vision or settings.llm_model_vision
            timeout = timeout or settings.llm_timeout_seconds
            self._router_timeout = (
                settings.llm_router_connect_timeout_seconds,
                settings.llm_router_read_timeout_seconds,
            )
            self._generation_connect_timeout = settings.llm_generation_connect_timeout_seconds
            self._generation_read_timeout = float(timeout)
            self._vision_timeout = (
                settings.llm_vision_connect_timeout_seconds,
                settings.llm_vision_read_timeout_seconds,
            )
            self._max_concurrency = settings.llm_max_concurrency
            self._queue_timeout = settings.llm_queue_timeout_seconds
            self._health_cache_seconds = settings.llm_health_cache_seconds
            self._router_max_tokens = settings.llm_router_max_tokens
            self._router_breaker = _CircuitBreaker(
                settings.llm_router_failure_threshold,
                settings.llm_router_recovery_seconds,
            )
            self._max_images = settings.vision_max_images_per_turn
            self._max_bytes_per_image = settings.vision_max_bytes_per_image
            self._max_total_payload_bytes = settings.vision_max_total_payload_bytes
            self._images_root: Path | None = (
                Path(allowed_images_dir)
                if allowed_images_dir is not None
                else settings.extracted_images_dir
            )
        else:
            self._router_timeout = (1.5, 10.0)
            self._generation_connect_timeout = 3.0
            self._generation_read_timeout = float(timeout) if timeout is not None else 120.0
            self._vision_timeout = (3.0, 45.0)
            self._max_concurrency = 1
            self._queue_timeout = 30.0
            self._health_cache_seconds = 30.0
            self._router_max_tokens = 48
            self._router_breaker = _CircuitBreaker(3, 60.0)
            self._max_images = MAX_IMAGES_PER_REQUEST
            self._max_bytes_per_image = 1_500_000
            self._max_total_payload_bytes = 4_000_000
            self._images_root = Path(allowed_images_dir) if allowed_images_dir is not None else None

        self._endpoint = (endpoint or "http://127.0.0.1:1234/v1").rstrip("/")
        self._model_chat = model_chat or "qwen3-14b"
        self._model_vision = model_vision or "qwen3-14b"
        self._timeout = int(timeout) if timeout is not None else 60
        self._headers = {"Content-Type": "application/json"}
        self._session = requests.Session()
        retry = Retry(
            total=0,
            connect=0,
            read=0,
            redirect=0,
            status=0,
            backoff_factor=0,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._semaphore = threading.BoundedSemaphore(self._max_concurrency)
        self._local = threading.local()
        self._health_ok = False
        self._health_checked_at: float | None = None
        try:
            from src.infrastructure.hardware.device_factory import DeviceFactory

            provider, dev = DeviceFactory.select_device()
            logger.info(
                "LLM hardware context: provider=%s device=%s (local HTTP inference)", provider, dev
            )
        except Exception as exc:
            logger.warning("LLM hardware detection skipped: error_type=%s", type(exc).__name__)

    # --- Observability -------------------------------------------------
    def _record_metrics(self, metrics: LLMCallMetrics) -> None:
        self._local.metrics = metrics

    def consume_call_metrics(self) -> LLMCallMetrics | None:
        metrics = cast("LLMCallMetrics | None", getattr(self._local, "metrics", None))
        self._local.metrics = None
        return metrics

    def _acquire_slot(self) -> float:
        started = time.perf_counter()
        if not self._semaphore.acquire(timeout=self._queue_timeout):
            raise LLMInferenceError(
                "El motor de inferencia local esta ocupado. Intente nuevamente."
            )
        return time.perf_counter() - started

    def _release_slot(self) -> None:
        self._semaphore.release()

    # --- Health --------------------------------------------------------
    def check_health(self, force: bool = False) -> bool:
        now = time.monotonic()
        if (
            not force
            and self._health_checked_at is not None
            and now - self._health_checked_at < self._health_cache_seconds
        ):
            return self._health_ok

        try:
            response = self._session.get(f"{self._endpoint}/models", timeout=3)
            self._health_ok = int(response.status_code) == 200
        except requests.exceptions.RequestException:
            self._health_ok = False
        except Exception:
            self._health_ok = False
        self._health_checked_at = now
        return self._health_ok

    def health_check(self) -> bool:
        return self.check_health(force=True)

    # --- Router --------------------------------------------------------
    def _router_prompt(self) -> str:
        """Return the classification prompt with an explicit no-think hint."""
        return _with_no_think(_ROUTER_SYSTEM_PROMPT, self._model_chat)

    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        started = time.perf_counter()
        queue_wait = 0.0
        status = "ok"
        output_tokens: int | None = None
        payload_bytes = 0
        acquired = False

        try:
            if not self._router_breaker.allow_request():
                status = "circuit_open"
                raise LLMInferenceError(
                    "El enrutador LLM esta temporalmente deshabilitado por fallos previos."
                )

            queue_wait = self._acquire_slot()
            acquired = True
            payload: dict[str, object] = {
                "model": self._model_chat,
                "messages": [
                    {"role": "system", "content": self._router_prompt()},
                    {
                        "role": "user",
                        "content": (
                            f"Peticion del operador:\n{user_input}\n"
                            f"Imagen adjunta: {'si' if has_image else 'no'}"
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": self._router_max_tokens,
                # LM Studio accepts json_schema; json_object is rejected with
                # HTTP 400 by current releases.
                "response_format": _ROUTER_RESPONSE_FORMAT,
                # Qwen-family servers honour this flag to skip long reasoning.
                "chat_template_kwargs": {"enable_thinking": False},
            }
            payload_bytes = len(json.dumps(payload).encode("utf-8"))
            url = f"{self._endpoint}/chat/completions"

            try:
                response = self._session.post(
                    url,
                    json=payload,
                    headers=self._headers,
                    timeout=self._router_timeout,
                )
                response.raise_for_status()
            except requests.exceptions.Timeout as exc:
                status = "timeout"
                self._router_breaker.record_failure()
                raise LLMInferenceError("Timeout en el enrutador del LLM local.", exc) from exc
            except requests.exceptions.ConnectionError as exc:
                status = "unavailable"
                self._router_breaker.record_failure()
                raise LLMInferenceError(
                    "No se pudo conectar al enrutador del LLM local.", exc
                ) from exc
            except requests.exceptions.HTTPError as exc:
                status = "http_error"
                response_obj = exc.response
                status_code = response_obj.status_code if response_obj is not None else None
                logger.error(
                    "LLM_TRACE REQUEST_ERROR | stage=router | status=%s | reason=http_error "
                    "| server_error=%s | server_detail=%s",
                    status_code,
                    _server_error_code(response_obj),
                    _safe_error_detail(response_obj),
                )
                self._router_breaker.record_failure()
                raise LLMInferenceError(
                    "El enrutador del LLM local rechazo la peticion.", exc
                ) from exc
            except requests.exceptions.RequestException as exc:
                status = "transport_error"
                self._router_breaker.record_failure()
                raise LLMInferenceError(
                    "Error de comunicacion con el enrutador del LLM local.", exc
                ) from exc

            output_tokens = self._output_tokens(response)
            try:
                decision = self._parse_router_response(response)
            except LLMInferenceError:
                status = "protocol_error"
                self._router_breaker.record_failure()
                raise

            self._router_breaker.record_success()
            return decision
        finally:
            if acquired:
                self._release_slot()
            self._record_metrics(
                LLMCallMetrics(
                    operation="route",
                    status=status,
                    total_seconds=time.perf_counter() - started,
                    queue_wait_seconds=queue_wait,
                    payload_bytes=payload_bytes,
                    output_tokens=output_tokens,
                )
            )

    @staticmethod
    def _parse_router_response(response: requests.Response) -> OrchestratorDecision:
        try:
            body = response.json()
            raw_content = str(body["choices"][0]["message"]["content"]).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            # Never surface the model body: it may contain user content.
            raise LLMInferenceError("El enrutador devolvio una respuesta malformada.", exc) from exc

        clean = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match is None:
            raise LLMInferenceError("El enrutador no devolvio un objeto JSON valido.")
        try:
            data = cast("dict[str, object]", json.loads(match.group(0)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMInferenceError("El enrutador no devolvio un objeto JSON valido.", exc) from exc

        tool = data.get("herramienta") or data.get("tool")
        if not isinstance(tool, str) or tool not in ROUTER_ALLOWED_TOOLS:
            raise LLMInferenceError("El enrutador devolvio una herramienta no permitida.")
        return OrchestratorDecision(razonamiento="", herramienta=tool)

    # --- Image payload budget -----------------------------------------
    @staticmethod
    def _collect_image_paths(context: Sequence[str]) -> list[str]:
        """Extract existing image paths from context headers, preserving order.

        Only the header line is parsed, so the fragment body that follows
        ``IMAGE_PATHS:`` is never mistaken for an asset path.
        """
        image_paths: list[str] = []
        for item in context:
            marker = "IMAGE_PATHS:"
            if marker not in item:
                continue
            remainder = item.split(marker, 1)[1].lstrip()
            header = remainder.splitlines()[0] if remainder else ""
            for raw_path in header.split("|"):
                path = raw_path.strip()
                if path and Path(path).is_file():
                    image_paths.append(path)
        return list(dict.fromkeys(image_paths))

    @staticmethod
    def _strip_image_path_headers(context: Sequence[str]) -> list[str]:
        """Remove image path headers before a text-only generation request.

        ``buscar_manual`` must never forward asset paths or Base64 payloads:
        only the retrieved fragment text reaches the local engine.
        """
        cleaned: list[str] = []
        for item in context:
            lines = [line for line in item.splitlines() if "IMAGE_PATHS:" not in line]
            cleaned.append("\n".join(lines))
        return cleaned

    def _safe_image_path(self, raw_path: str, allowed_root: Path | None) -> Path | None:
        if not raw_path or "\x00" in raw_path:
            return None
        try:
            candidate = Path(raw_path)
            if candidate.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                return None
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_file():
            return None
        if allowed_root is not None:
            try:
                root = allowed_root.resolve()
            except OSError:
                return None
            if not resolved.is_relative_to(root):
                return None
        return resolved

    def _encode_image_to_base64(self, image_path: str) -> str | None:
        try:
            path = Path(image_path)
            if not path.is_file():
                logger.warning("LLM_TRACE image skipped | reason=missing_file")
                return None
            return base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as exc:
            logger.warning(
                "LLM_TRACE image skipped | reason=unreadable | error_type=%s", type(exc).__name__
            )
            return None

    def _prepare_images(self, image_paths: Sequence[str], reserved_bytes: int) -> _ImageBudget:
        started = time.perf_counter()
        prepared: list[_PreparedImage] = []
        skipped = 0
        total_encoded = 0
        total_pixels = 0
        available = max(0, self._max_total_payload_bytes - reserved_bytes)

        for raw_path in image_paths:
            if len(prepared) >= self._max_images:
                skipped += 1
                continue

            safe_path = self._safe_image_path(raw_path, self._images_root)
            if safe_path is None:
                skipped += 1
                continue

            try:
                raw_size = safe_path.stat().st_size
            except OSError:
                skipped += 1
                continue
            if raw_size <= 0 or raw_size > self._max_bytes_per_image:
                skipped += 1
                continue

            encoded = self._encode_image_to_base64(str(safe_path))
            if encoded is None:
                skipped += 1
                continue
            encoded_size = len(encoded)
            if encoded_size > self._max_bytes_per_image or total_encoded + encoded_size > available:
                skipped += 1
                continue

            try:
                raw_data = base64.b64decode(encoded, validate=False)
            except (ValueError, TypeError):
                skipped += 1
                continue

            width, height = _image_dimensions(raw_data)
            prepared.append(
                _PreparedImage(
                    mime_type=_image_mime_type(str(safe_path), raw_data),
                    base64_data=encoded,
                    raw_bytes=raw_size,
                    encoded_bytes=encoded_size,
                    width=width,
                    height=height,
                )
            )
            total_encoded += encoded_size
            total_pixels += width * height

        return _ImageBudget(
            prepared=tuple(prepared),
            skipped=skipped,
            encoded_bytes=total_encoded,
            pixels=total_pixels,
            prep_seconds=time.perf_counter() - started,
        )

    # --- Generation ----------------------------------------------------
    def _build_chat_payload(
        self,
        user_content: list[dict[str, object]] | str,
        model: str,
        *,
        include_images: bool = False,
        no_think: bool = False,
    ) -> dict[str, object]:
        image_instruction = _IMAGE_INSTRUCTION if include_images else ""
        system_prompt = SYSTEM_PROMPT.format(image_instruction=image_instruction).strip()
        if no_think:
            system_prompt = _with_no_think(system_prompt, model)
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

    @staticmethod
    def _build_content(
        text_block: str, images: Sequence[_PreparedImage]
    ) -> tuple[list[dict[str, object]] | str, int]:
        if not images:
            return text_block, 0
        content: list[dict[str, object]] = [{"type": "text", "text": text_block}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{image.base64_data}"},
                }
            )
        return content, len(images)

    def generate_response(
        self,
        context: Sequence[str],
        user_input: str,
        *,
        attach_images: bool = True,
    ) -> str:
        started = time.perf_counter()
        queue_wait = 0.0
        status = "ok"
        output_tokens: int | None = None
        payload_bytes = 0
        budget = _ImageBudget()
        attached = 0
        acquired = False

        try:
            if not context:
                return "No hay informacion relevante en los documentos para responder."

            queue_wait = self._acquire_slot()
            acquired = True
            url = f"{self._endpoint}/chat/completions"
            context_str = "\n\n".join(context)
            prompt_context_str = context_str
            image_paths = self._collect_image_paths(context)
            if attach_images:
                # Estimate the reservation with a neutral text block, prepare
                # the assets, then rebuild the text so the "attached images"
                # note only appears when at least one image is really sent.
                text_block = _build_user_text(context_str, user_input, include_images=False)
                budget = self._prepare_images(
                    image_paths, reserved_bytes=len(text_block.encode("utf-8")) + 2048
                )
                if not budget.prepared:
                    # Nothing could be attached: keep the request text-only and
                    # remove asset headers that reference unavailable images.
                    prompt_context_str = "\n\n".join(self._strip_image_path_headers(context))
                text_block = _build_user_text(
                    prompt_context_str, user_input, include_images=bool(budget.prepared)
                )
            else:
                # Text-only modality: report the detected candidates as
                # skipped without reading, encoding or forwarding any asset.
                prompt_context_str = "\n\n".join(self._strip_image_path_headers(context))
                text_block = _build_user_text(prompt_context_str, user_input, include_images=False)
                budget = _ImageBudget(skipped=len(image_paths))
            user_content, attached = self._build_content(text_block, budget.prepared)
            payload = self._build_chat_payload(
                user_content,
                self._model_vision if attached else self._model_chat,
                include_images=(attached > 0),
                no_think=(attached == 0),
            )
            payload_bytes = len(json.dumps(payload).encode("utf-8"))

            while attached and payload_bytes > self._max_total_payload_bytes:
                budget = budget.drop_last()
                text_block = _build_user_text(
                    prompt_context_str, user_input, include_images=bool(budget.prepared)
                )
                user_content, attached = self._build_content(text_block, budget.prepared)
                payload = self._build_chat_payload(
                    user_content,
                    self._model_vision if attached else self._model_chat,
                    include_images=(attached > 0),
                    no_think=(attached == 0),
                )
                payload_bytes = len(json.dumps(payload).encode("utf-8"))

            logger.info(
                "LLM_TRACE generate_request | fragments=%d | context_chars=%d | user_msg_chars=%d "
                "| images_attached=%d | images_skipped=%d | image_bytes=%d | payload_bytes=%d "
                "| timeout_s=%.1f | model=%s",
                len(context),
                len(prompt_context_str),
                len(text_block),
                attached,
                budget.skipped,
                budget.encoded_bytes,
                payload_bytes,
                self._generation_read_timeout,
                self._model_vision if attached else self._model_chat,
            )

            try:
                if attached:
                    response = self._post_completion(url, payload, self._generation_timeout)
                else:
                    response = self._post_text_completion(url, payload, self._generation_timeout)
            except requests.exceptions.HTTPError as exc:
                response_obj = exc.response
                status_code = response_obj.status_code if response_obj is not None else None
                if not attached or status_code not in MULTIMODAL_REJECTED_STATUSES:
                    status = "http_error"
                    logger.error(
                        "LLM_TRACE REQUEST_ERROR | status=%s | reason=http_error "
                        "| server_error=%s | server_detail=%s",
                        status_code,
                        _server_error_code(response_obj),
                        _safe_error_detail(response_obj),
                    )
                    raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc

                logger.warning(
                    "LLM_TRACE multimodal rejected | status=%d | action=retry_text_only",
                    status_code,
                )
                status = "multimodal_rejected"
                budget = _ImageBudget(
                    prepared=(),
                    skipped=budget.skipped + attached,
                    encoded_bytes=0,
                    pixels=0,
                    prep_seconds=budget.prep_seconds,
                )
                attached = 0
                text_block = _build_user_text(prompt_context_str, user_input, include_images=False)
                payload = self._build_chat_payload(
                    text_block, self._model_chat, include_images=False, no_think=True
                )
                payload_bytes = len(json.dumps(payload).encode("utf-8"))
                try:
                    response = self._post_text_completion(url, payload, self._generation_timeout)
                except requests.exceptions.RequestException as fallback_exc:
                    status = "transport_error"
                    raise LLMInferenceError(
                        "Error de comunicacion con el LLM local.", fallback_exc
                    ) from fallback_exc
            except requests.exceptions.Timeout as exc:
                status = "timeout"
                logger.error("LLM_TRACE TIMEOUT | stage=generation")
                raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
            except requests.exceptions.ConnectionError as exc:
                status = "unavailable"
                logger.error("LLM_TRACE CONNECTION_ERROR | stage=generation")
                raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
            except requests.exceptions.RequestException as exc:
                status = "transport_error"
                logger.error("LLM_TRACE REQUEST_ERROR | stage=generation")
                raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc

            output_tokens = self._output_tokens(response)
            result = self._parse_content(response)
            logger.info(
                "LLM_TRACE RESPONSE_OK | latency_s=%.3f | fragments=%d | context_chars=%d "
                "| images_attached=%d | payload_bytes=%d",
                time.perf_counter() - started,
                len(context),
                len(prompt_context_str),
                attached,
                payload_bytes,
            )
            return result
        finally:
            if acquired:
                self._release_slot()
            self._record_metrics(
                LLMCallMetrics(
                    operation="generate",
                    status=status,
                    total_seconds=time.perf_counter() - started,
                    queue_wait_seconds=queue_wait,
                    image_prep_seconds=budget.prep_seconds,
                    images_attached=attached,
                    images_skipped=budget.skipped,
                    image_bytes=budget.encoded_bytes,
                    image_pixels=budget.pixels,
                    payload_bytes=payload_bytes,
                    output_tokens=output_tokens,
                )
            )

    @property
    def _generation_timeout(self) -> tuple[float, float]:
        return (self._generation_connect_timeout, self._generation_read_timeout)

    def _post_completion(
        self, url: str, payload: dict[str, object], timeout: tuple[float, float]
    ) -> requests.Response:
        """POST a chat completion, surfacing transport errors to the caller."""
        response = self._session.post(url, json=payload, headers=self._headers, timeout=timeout)
        response.raise_for_status()
        return response

    def _post_text_completion(
        self, url: str, payload: dict[str, object], timeout: tuple[float, float]
    ) -> requests.Response:
        """POST a text-only completion with one bounded transient-retry.

        After an aborted router request the local server can reject the next
        text-only call with HTTP 400/422 while it is still settling. One retry
        recovers the turn without ever duplicating image preparation work.
        """
        try:
            return self._post_completion(url, payload, timeout)
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else None
            if status_code not in TEXT_ONLY_RETRY_STATUSES:
                raise
            logger.warning(
                "LLM_TRACE text_only_retry | status=%s | attempt=2 | server_error=%s",
                status_code,
                _server_error_code(response),
            )
            time.sleep(TEXT_ONLY_RETRY_DELAY_SECONDS)
            return self._post_completion(url, payload, timeout)

    # --- Vision --------------------------------------------------------
    def analyze_image(self, image_path: str, question: str | None = None) -> str:
        started = time.perf_counter()
        queue_wait = 0.0
        status = "ok"
        output_tokens: int | None = None
        payload_bytes = 0
        image_bytes = 0
        image_pixels = 0
        acquired = False

        try:
            safe_path = self._safe_image_path(image_path, allowed_root=None)
            if safe_path is None:
                raise DataIncompleteError("No se encontro un archivo de imagen valido.")
            try:
                raw_size = safe_path.stat().st_size
            except OSError as exc:
                raise DataIncompleteError("No se pudo leer el archivo de imagen.", exc) from exc
            if raw_size <= 0 or raw_size > self._max_bytes_per_image:
                raise DataIncompleteError("La imagen excede el tamano permitido.")

            queue_wait = self._acquire_slot()
            acquired = True
            data = safe_path.read_bytes()
            image_bytes = len(data)
            width, height = _image_dimensions(data)
            image_pixels = width * height
            base64_image = base64.b64encode(data).decode("ascii")
            mime = _image_mime_type(str(safe_path), data)
            payload: dict[str, object] = {
                "model": self._model_vision,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _vision_prompt(question),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 512,
            }
            payload_bytes = len(json.dumps(payload).encode("utf-8"))
            url = f"{self._endpoint}/chat/completions"
            try:
                response = self._post_completion(url, payload, self._vision_timeout)
            except requests.exceptions.Timeout as exc:
                status = "timeout"
                raise LLMInferenceError("Timeout en vision del LLM local.", exc) from exc
            except requests.exceptions.ConnectionError as exc:
                status = "unavailable"
                raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
            except requests.exceptions.RequestException as exc:
                status = "transport_error"
                raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc

            output_tokens = self._output_tokens(response)
            return self._parse_content(response)
        finally:
            if acquired:
                self._release_slot()
            self._record_metrics(
                LLMCallMetrics(
                    operation="vision",
                    status=status,
                    total_seconds=time.perf_counter() - started,
                    queue_wait_seconds=queue_wait,
                    images_attached=1 if image_bytes else 0,
                    image_bytes=image_bytes,
                    image_pixels=image_pixels,
                    payload_bytes=payload_bytes,
                    output_tokens=output_tokens,
                )
            )

    @staticmethod
    def _parse_content(response: requests.Response) -> str:
        try:
            body = response.json()
            return str(body["choices"][0]["message"]["content"]).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInferenceError("El LLM local devolvio una respuesta malformada.", exc) from exc

    @staticmethod
    def _output_tokens(response: requests.Response) -> int | None:
        try:
            body = response.json()
        except (ValueError, TypeError):
            return None
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None
        value = usage.get("completion_tokens")
        return value if isinstance(value, int) else None
