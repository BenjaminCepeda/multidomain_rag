# src/infrastructure/adapters/local_llm_adapter.py
import base64
import json
import logging
import os
import re
import time
from collections.abc import Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.domain.exceptions.core_errors import DataIncompleteError, LLMInferenceError
from src.domain.models.agent import OrchestratorDecision
from src.domain.ports.llm_engine import ILLMEngine
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_REQUEST = 4
MULTIMODAL_REJECTED_STATUSES = frozenset({400, 422})
_IMAGE_MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}


def _image_mime_type(image_path: str) -> str:
    extension = os.path.splitext(image_path)[1].lower().lstrip(".")
    return _IMAGE_MIME_TYPES.get(extension, "image/png")


def _build_user_text(context_str: str, user_input: str, include_images: bool) -> str:
    attachment_note = " y las imagenes adjuntas" if include_images else ""
    return (
        f"Contexto tecnico:\n{context_str}\n\n"
        f"Pregunta del usuario:\n{user_input}\n\n"
        f"Instruccion: Usa UNICAMENTE el contexto anterior{attachment_note}. "
        "Si existe informacion relacionada, genera una respuesta completa y "
        "accionable con referencias [Archivo, Pag]."
    )


class LocalLLMAdapter(ILLMEngine):
    def __init__(
        self,
        settings: Settings | None = None,
        endpoint: str | None = None,
        model_chat: str | None = None,
        model_vision: str | None = None,
        timeout: int | None = None,
    ) -> None:
        if settings is not None:
            endpoint = endpoint or settings.llm_endpoint
            model_chat = model_chat or settings.llm_model_chat
            model_vision = model_vision or settings.llm_model_vision
            timeout = timeout or settings.llm_timeout_seconds

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
        try:
            from src.infrastructure.hardware.device_factory import DeviceFactory

            provider, dev = DeviceFactory.select_device()
            logger.info(
                "LLM hardware context: provider=%s device=%s (local HTTP inference)", provider, dev
            )
        except Exception as exc:
            logger.warning("LLM hardware detection skipped: %s", exc)

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self._endpoint}/models", timeout=3)
            return int(response.status_code) == 200
        except Exception:
            return False

    def health_check(self) -> bool:
        return self.check_health()

    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no esta disponible.")

        url = f"{self._endpoint}/chat/completions"
        prompt_sistema = (
            "Eres un cerebro orquestador. Responde SOLO con un objeto JSON. "
            '{"razonamiento": "motivo", "herramienta": "nombre", "parametro": "texto_o_null"}'
        )
        payload = {
            "model": self._model_chat,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Operario: {user_input} | Imagen: {has_image}"},
            ],
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc
        raw_content = str(response.json()["choices"][0]["message"]["content"]).strip()

        clean_content = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.MULTILINE)
        start = clean_content.find("{")
        end = -1
        count = 0
        for i, char in enumerate(clean_content[start:]):
            if char == "{":
                count += 1
            elif char == "}":
                count -= 1
            if count == 0:
                end = start + i + 1
                break

        if start == -1 or end == -1:
            raise LLMInferenceError(
                f"El motor no devolvio un JSON valido. Contenido: {raw_content}"
            )

        data = json.loads(clean_content[start:end])
        return OrchestratorDecision(
            razonamiento=data.get("razonamiento", "Sin descripcion"),
            herramienta=data.get("herramienta", "responder_directo"),
            parametro=data.get("parametro"),
        )

    def _encode_image_to_base64(self, image_path: str) -> str | None:
        try:
            if not os.path.exists(image_path):
                logger.warning(
                    "LLM_TRACE image skipped | reason=missing_file | path=%s", image_path
                )
                return None
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8")
        except OSError as exc:
            logger.warning("LLM_TRACE image skipped | reason=unreadable | error=%s", exc)
            return None

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
                if path and os.path.exists(path):
                    image_paths.append(path)
        return list(dict.fromkeys(image_paths))

    def generate_response(self, context: Sequence[str], user_input: str) -> str:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no esta disponible.")

        if not context:
            return "No hay informacion relevante en los documentos para responder."

        url = f"{self._endpoint}/chat/completions"

        context_str = "\n\n".join(context)

        system_prompt = (
            "Eres un asistente tecnico experto para manuales de impresoras Epson. "
            "Tu unica fuente de verdad es el Contexto tecnico que se te proporciona. "
            "Usa tambien las imagenes/diagramas adjuntos cuando sean relevantes.\n"
            "Reglas estrictas:\n"
            "- RESPONDE SIEMPRE con base en el contexto y en las imagenes adjuntas si existen. "
            "Si hay informacion relevante, aunque sea parcial, usala.\n"
            "- NO digas 'No dispongo de informacion' si la respuesta se puede inferir "
            "o encontrar parcialmente en los fragmentos o imagenes.\n"
            "- Explica paso a paso, procedimientos, rutas de menu, nombres de programas "
            "(Epson Scan 2, Epson Email Print) o configuraciones cuando aparezcan.\n"
            "- Cita la fuente y la pagina al final de las afirmaciones clave usando "
            "el formato [Archivo, Pag].\n"
            "- No inventes pasos, nombres o valores. Solo si tras revisar TODO el contexto "
            "no hay absolutamente nada relacionado, usa exactamente: "
            "'No dispongo de informacion en los manuales'.\n"
            "- Prioriza utilidad practica para el usuario sobre la cautela excesiva."
        )

        image_paths = self._collect_image_paths(context)
        text_block = _build_user_text(context_str, user_input, include_images=bool(image_paths))

        attached_images = 0
        user_content: list[dict[str, object]] | str
        if image_paths:
            content_list: list[dict[str, object]] = [{"type": "text", "text": text_block}]
            for img_path in image_paths[:MAX_IMAGES_PER_REQUEST]:
                b64 = self._encode_image_to_base64(img_path)
                if not b64:
                    continue
                mime = _image_mime_type(img_path)
                content_list.append(
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                )
                attached_images += 1
            user_content = content_list if attached_images else text_block
        else:
            user_content = text_block

        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        payload: dict[str, object] = {
            "model": self._model_chat,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        logger.info(
            "LLM_TRACE generate_response | fragments=%d | context_chars=%d | user_msg_chars=%d "
            "| images_attached=%d | timeout_s=%d | endpoint=%s | model=%s",
            len(context),
            len(context_str),
            len(text_block),
            attached_images,
            self._timeout,
            self._endpoint,
            self._model_chat,
        )
        logger.debug(
            "LLM_TRACE request | system_prompt_chars=%d | user_msg_chars=%d | multimodal=%s",
            len(system_prompt),
            len(text_block),
            bool(attached_images),
        )

        start_ts = time.perf_counter()
        try:
            response = self._post_completion(url, payload)
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if not attached_images or status not in MULTIMODAL_REJECTED_STATUSES:
                elapsed = time.perf_counter() - start_ts
                logger.error("LLM_TRACE REQUEST_ERROR after %.3fs | status=%s", elapsed, status)
                raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc

            logger.warning(
                "LLM_TRACE multimodal rejected | status=%d | action=retry_text_only",
                status,
            )
            attached_images = 0
            text_block = _build_user_text(context_str, user_input, include_images=False)
            messages[1]["content"] = text_block
            start_ts = time.perf_counter()
            try:
                response = self._post_completion(url, payload)
            except requests.exceptions.RequestException as fallback_exc:
                raise LLMInferenceError(
                    "Error de comunicacion con el LLM local.", fallback_exc
                ) from fallback_exc
        except requests.exceptions.Timeout as exc:
            elapsed = time.perf_counter() - start_ts
            logger.error(
                "LLM_TRACE TIMEOUT after %.3fs (%.0fms) | fragments=%d | context_chars=%d",
                elapsed,
                elapsed * 1000,
                len(context),
                len(context_str),
            )
            raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
        except requests.exceptions.ConnectionError as exc:
            elapsed = time.perf_counter() - start_ts
            logger.error(
                "LLM_TRACE CONNECTION_ERROR after %.3fs | endpoint=%s",
                elapsed,
                self._endpoint,
            )
            raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
        except requests.exceptions.RequestException as exc:
            elapsed = time.perf_counter() - start_ts
            logger.error(
                "LLM_TRACE REQUEST_ERROR after %.3fs | %s",
                elapsed,
                exc,
            )
            raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc
        elapsed = time.perf_counter() - start_ts
        logger.info(
            "LLM_TRACE RESPONSE_OK | latency_s=%.3f | latency_ms=%.0f | fragments=%d "
            "| context_chars=%d | images_attached=%d",
            elapsed,
            elapsed * 1000,
            len(context),
            len(context_str),
            attached_images,
        )
        try:
            return str(response.json()["choices"][0]["message"]["content"]).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInferenceError("El LLM local devolvio una respuesta malformada.", exc) from exc

    def _post_completion(self, url: str, payload: dict[str, object]) -> requests.Response:
        """POST a chat completion, surfacing transport errors to the caller."""
        response = self._session.post(
            url, json=payload, headers=self._headers, timeout=self._timeout
        )
        response.raise_for_status()
        return response

    def analyze_image(self, image_path: str) -> str:
        if not self.check_health():
            raise LLMInferenceError("El motor LLM no esta disponible.")

        if not os.path.exists(image_path):
            raise DataIncompleteError(f"Archivo no encontrado: {image_path}")
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        url = f"{self._endpoint}/chat/completions"
        payload = {
            "model": self._model_vision,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe este diagrama tecnico brevemente."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
        }
        try:
            response = self._session.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMInferenceError("Timeout en inferencia del LLM local.", exc) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMInferenceError("No se pudo conectar al LLM local.", exc) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMInferenceError("Error de comunicacion con el LLM local.", exc) from exc
        return str(response.json()["choices"][0]["message"]["content"]).strip()
