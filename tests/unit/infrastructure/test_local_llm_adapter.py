import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import requests

from src.domain.exceptions.core_errors import LLMInferenceError
from src.infrastructure.adapters import local_llm_adapter
from src.infrastructure.adapters.local_llm_adapter import (
    MAX_IMAGES_PER_REQUEST,
    LocalLLMAdapter,
    _CircuitBreaker,
    _safe_error_detail,
)
from src.infrastructure.config.settings import Settings


class _FakeResponse:
    def __init__(
        self,
        content: str = "respuesta",
        status_code: int = 200,
        body: str | None = None,
    ) -> None:
        self._content = content
        self.status_code = status_code
        self._body = body

    @property
    def text(self) -> str:
        return self._body if self._body is not None else json.dumps(self.json())

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)  # type: ignore[arg-type]

    def json(self) -> dict[str, Any]:
        if self._body is not None:
            parsed = json.loads(self._body)
            if isinstance(parsed, dict):
                return parsed
        return {"choices": [{"message": {"content": self._content}}]}


def _adapter() -> LocalLLMAdapter:
    return LocalLLMAdapter(
        endpoint="http://llm.test/v1",
        model_chat="test-model",
        timeout=5,
    )


def _capture_payload(monkeypatch: pytest.MonkeyPatch, adapter: LocalLLMAdapter) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["payload"] = json.loads(json.dumps(kwargs["json"]))
        return _FakeResponse()

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    return captured


def _context_with_images(paths: Sequence[Path]) -> list[str]:
    header = "|".join(str(path) for path in paths)
    return [
        "=== FRAGMENTO 1 ===\n"
        "FUENTE: manual.pdf\n"
        "PAGINA: 7\n"
        "IMAGE_PATHS: "
        f"{header}\n"
        "\n"
        "Cuerpo del fragmento que no debe interpretarse como ruta."
    ]


def test_generate_response_with_images_does_not_reference_missing_user_msg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    result = adapter.generate_response(_context_with_images([image]), "muestrame el grafico")

    assert result == "respuesta"
    system_prompt = captured["payload"]["messages"][0]["content"]
    assert "imágenes/diagramas adjuntos" in system_prompt
    content = captured["payload"]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Cuerpo del fragmento" in content[0]["text"]
    assert "imágenes adjuntas" in content[0]["text"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_generate_response_without_images_keeps_plain_text_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(["=== FRAGMENTO 1 ===\nTexto tecnico"], "pregunta")

    content = captured["payload"]["messages"][1]["content"]
    assert isinstance(content, str)
    assert "Texto tecnico" in content


def test_text_only_generation_skips_images_and_strips_asset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    def _forbidden_prepare(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("text-only modality must not prepare images")

    monkeypatch.setattr(adapter, "_prepare_images", _forbidden_prepare)

    adapter.generate_response(_context_with_images([image]), "pregunta", attach_images=False)

    content = captured["payload"]["messages"][1]["content"]
    system_prompt = captured["payload"]["messages"][0]["content"]
    assert isinstance(content, str)
    assert "Cuerpo del fragmento" in content
    assert "IMAGE_PATHS" not in content
    assert "data:image" not in content
    assert str(image) not in content
    assert "imágenes adjuntas" not in content
    assert "imágenes/diagramas adjuntos" not in system_prompt
    assert "imágenes" not in system_prompt

    metrics = adapter.consume_call_metrics()
    assert metrics is not None
    assert metrics.operation == "generate"
    assert metrics.images_attached == 0
    assert metrics.images_skipped == 1
    assert metrics.image_bytes == 0
    assert metrics.image_pixels == 0
    assert metrics.image_prep_seconds == 0.0
    assert captured["payload"]["model"] == "test-model"


def test_system_prompt_is_domain_agnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(["texto tecnico"], "pregunta", attach_images=False)

    system_prompt = captured["payload"]["messages"][0]["content"]
    assert "asistente técnico" in system_prompt
    assert "documentación técnica" in system_prompt
    for forbidden in ("Epson", "impresora", "manuales de impresoras", "Epson Scan"):
        assert forbidden not in system_prompt


def test_text_only_generation_adds_no_think_hint_for_qwen3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LocalLLMAdapter(
        endpoint="http://llm.test/v1",
        model_chat="qwen3-14b",
        timeout=5,
    )
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(["texto tecnico"], "pregunta", attach_images=False)

    system_prompt = captured["payload"]["messages"][0]["content"]
    assert "/no_think" in system_prompt
    content = captured["payload"]["messages"][1]["content"]
    assert isinstance(content, str)


def test_text_only_400_is_retried_once_before_succeeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    payloads: list[dict[str, Any]] = []
    responses = [_FakeResponse(status_code=400), _FakeResponse(content="respuesta texto")]

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        payloads.append(json.loads(json.dumps(kwargs["json"])))
        return responses[len(payloads) - 1]

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    monkeypatch.setattr(local_llm_adapter, "TEXT_ONLY_RETRY_DELAY_SECONDS", 0.0)

    result = adapter.generate_response(["texto tecnico"], "pregunta", attach_images=False)

    assert result == "respuesta texto"
    assert len(payloads) == 2
    for payload in payloads:
        assert isinstance(payload["messages"][1]["content"], str)
        assert "data:image" not in payload["messages"][1]["content"]
    metrics = adapter.consume_call_metrics()
    assert metrics is not None
    assert metrics.status == "ok"
    assert metrics.images_attached == 0
    assert metrics.image_bytes == 0


def test_text_only_400_retry_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    calls: list[int] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append(1)
        return _FakeResponse(status_code=400)

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    monkeypatch.setattr(local_llm_adapter, "TEXT_ONLY_RETRY_DELAY_SECONDS", 0.0)

    with pytest.raises(LLMInferenceError):
        adapter.generate_response(["texto tecnico"], "pregunta", attach_images=False)

    assert len(calls) == 2
    metrics = adapter.consume_call_metrics()
    assert metrics is not None
    assert metrics.status == "http_error"
    assert metrics.images_attached == 0


def test_safe_error_detail_redacts_paths_uris_and_blobs() -> None:
    blob = "A" * 120
    body = json.dumps(
        {
            "error": {
                "message": (
                    f"bad /home/benja/secret.pdf data:image/png;base64,{blob} token {blob}"
                ),
                "type": "invalid_request_error",
            }
        }
    )
    response = _FakeResponse(status_code=400, body=body)

    detail = _safe_error_detail(response)

    assert "<path>" in detail
    assert "<data-uri>" in detail
    assert "<blob>" in detail
    assert "/home/benja" not in detail
    assert "data:image" not in detail
    assert blob not in detail
    assert len(detail) <= 512


def test_http_error_logs_redacted_server_detail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    adapter = _adapter()
    secret_path = "/home/benja/private/manual.pdf"
    body = json.dumps(
        {
            "error": {
                "message": f"invalid request for {secret_path}",
                "type": "invalid_request_error",
            }
        }
    )

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=400, body=body)

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    monkeypatch.setattr(local_llm_adapter, "TEXT_ONLY_RETRY_DELAY_SECONDS", 0.0)

    with caplog.at_level("ERROR"), pytest.raises(LLMInferenceError):
        adapter.generate_response(["texto tecnico"], "pregunta", attach_images=False)

    logged = caplog.text
    assert "server_detail=" in logged
    assert "<path>" in logged
    assert secret_path not in logged


def test_image_paths_are_read_from_header_line_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(_context_with_images([first, second]), "pregunta")

    content = captured["payload"]["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_unreadable_images_fall_back_to_text_only_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "broken.png"
    image.write_bytes(b"not-an-image")
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)
    monkeypatch.setattr(adapter, "_encode_image_to_base64", lambda _path: None)

    adapter.generate_response(_context_with_images([image]), "pregunta")

    system_prompt = captured["payload"]["messages"][0]["content"]
    content = captured["payload"]["messages"][1]["content"]
    assert isinstance(content, str)
    assert "pregunta" in content
    assert "IMAGE_PATHS" not in content
    assert str(image) not in content
    assert "imágenes adjuntas" not in content
    assert "imágenes/diagramas adjuntos" not in system_prompt
    assert "imágenes" not in system_prompt


def test_images_are_capped_per_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []
    for index in range(MAX_IMAGES_PER_REQUEST + 2):
        image = tmp_path / f"img_{index}.png"
        image.write_bytes(b"png")
        paths.append(image)
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(_context_with_images(paths), "pregunta")

    content = captured["payload"]["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == MAX_IMAGES_PER_REQUEST


def test_multimodal_rejection_falls_back_to_text_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"png")
    adapter = _adapter()
    payloads: list[dict[str, Any]] = []
    responses = [_FakeResponse(status_code=400), _FakeResponse(content="respuesta texto")]

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        payloads.append(json.loads(json.dumps(kwargs["json"])))
        return responses[len(payloads) - 1]

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    result = adapter.generate_response(_context_with_images([image]), "pregunta")

    assert result == "respuesta texto"
    assert len(payloads) == 2
    first_content = payloads[0]["messages"][1]["content"]
    second_content = payloads[1]["messages"][1]["content"]
    assert isinstance(first_content, list)
    assert isinstance(second_content, str)
    assert "pregunta" in second_content
    assert "imágenes adjuntas" not in second_content
    fallback_system = payloads[1]["messages"][0]["content"]
    assert "imágenes/diagramas adjuntos" not in fallback_system
    assert "imágenes" not in fallback_system


def test_non_multimodal_http_error_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"png")
    adapter = _adapter()
    calls: list[int] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append(1)
        return _FakeResponse(status_code=500)

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    with pytest.raises(LLMInferenceError):
        adapter.generate_response(_context_with_images([image]), "pregunta")

    assert len(calls) == 1


def test_router_is_bounded_and_disables_long_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(content='{"herramienta": "buscar_manual"}')

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    decision = adapter.get_orchestrator_decision("consulta tecnica", False)

    assert decision.herramienta == "buscar_manual"
    payload = captured["json"]
    assert 32 <= payload["max_tokens"] <= 48
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["herramienta"]["enum"] == [
        "buscar_manual",
        "analizar_imagen",
        "responder_directo",
    ]
    assert captured["timeout"][0] == 1.5
    assert captured["timeout"][1] == 10.0


def test_qwen3_router_adds_explicit_no_think_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LocalLLMAdapter(
        endpoint="http://llm.test/v1",
        model_chat="qwen3-14b",
        timeout=5,
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(content='{"herramienta": "buscar_manual"}')

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    adapter.get_orchestrator_decision("consulta tecnica", False)

    system_prompt = captured["json"]["messages"][0]["content"]
    assert "/no_think" in system_prompt
    assert captured["json"]["response_format"]["type"] == "json_schema"


def test_router_timeout_is_reported_as_transport_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        raise requests.Timeout("read budget exhausted")

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    with pytest.raises(LLMInferenceError):
        adapter.get_orchestrator_decision("consulta tecnica", False)

    metrics = adapter.consume_call_metrics()
    assert metrics is not None
    assert metrics.operation == "route"
    assert metrics.status == "timeout"


def test_router_error_never_leaks_model_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    sentinel = "CONTENIDO_SENSIBLE_DEL_MODELO"

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(content=sentinel)

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    with pytest.raises(LLMInferenceError) as exc_info:
        adapter.get_orchestrator_decision("consulta", False)

    assert sentinel not in str(exc_info.value)


def test_router_circuit_breaker_opens_after_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    adapter._router_breaker = _CircuitBreaker(failure_threshold=1, recovery_seconds=60)
    calls: list[int] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append(1)
        raise requests.ConnectionError("down")

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    with pytest.raises(LLMInferenceError):
        adapter.get_orchestrator_decision("consulta", False)
    with pytest.raises(LLMInferenceError):
        adapter.get_orchestrator_decision("consulta", False)

    assert len(calls) == 1


def test_health_check_uses_cached_state_between_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    calls: list[int] = []

    class _Session:
        @staticmethod
        def get(url: str, timeout: int) -> _FakeResponse:
            calls.append(1)
            return _FakeResponse(status_code=200)

    monkeypatch.setattr(adapter, "_session", _Session())

    assert adapter.check_health() is True
    assert adapter.check_health() is True
    assert len(calls) == 1
    assert adapter.check_health(force=True) is True
    assert len(calls) == 2


def test_images_outside_the_allowed_root_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        data_files_path=str(tmp_path / "data"),
        extracted_images_path="extracted_images",
    )
    settings.extracted_images_dir.mkdir(parents=True)
    inside = settings.extracted_images_dir / "ok.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\n")
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = LocalLLMAdapter(settings=settings, timeout=5)
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(_context_with_images([inside, outside]), "pregunta")

    content = captured["payload"]["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 1


def test_total_payload_budget_drops_images_and_reports_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter()
    adapter._max_total_payload_bytes = 5_000
    adapter._max_images = 3
    first = tmp_path / "large_a.png"
    second = tmp_path / "large_b.png"
    first.write_bytes(b"x" * 1_500)
    second.write_bytes(b"x" * 1_500)
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(_context_with_images([first, second]), "pregunta")

    content = captured["payload"]["messages"][1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    metrics = adapter.consume_call_metrics()
    assert len(image_parts) == 1
    assert metrics is not None
    assert metrics.operation == "generate"
    assert metrics.images_attached == 1
    assert metrics.images_skipped >= 1
    assert metrics.payload_bytes <= 5_000


def test_analyze_image_uses_detected_mime_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.bmp"
    image.write_bytes(b"BM" + b"0" * 64)
    adapter = _adapter()
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(content="diagrama")

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    assert adapter.analyze_image(str(image)) == "diagrama"
    url = captured["json"]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/bmp;base64,")


def test_analyze_image_uses_query_aware_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    adapter = _adapter()
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(content="diagrama")

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    assert adapter.analyze_image(str(image), question="Cadence Stratus HLS") == "diagrama"
    prompt = captured["json"]["messages"][0]["content"][0]["text"]
    assert "Cadence Stratus HLS" in prompt
    assert "consulta" in prompt.lower()


def test_router_prompt_allows_manual_visual_requests_without_uploaded_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(content='{"herramienta": "analizar_imagen"}')

    monkeypatch.setattr(adapter, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    decision = adapter.get_orchestrator_decision(
        "show me Block diagram of the generic training to inference pipeline", False
    )

    assert decision.herramienta == "analizar_imagen"
    system_prompt = captured["json"]["messages"][0]["content"].casefold()
    assert "diagrama" in system_prompt
    assert "pipeline" in system_prompt
    assert "aunque no haya subido una imagen" in system_prompt


def test_text_only_generation_prompt_forbids_the_generic_archive_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    captured = _capture_payload(monkeypatch, adapter)

    adapter.generate_response(
        ["=== FRAGMENTO 1 ===\nFUENTE: manual.pdf\nPAGINA: 3\n\nTexto tecnico"],
        "pregunta",
        attach_images=False,
    )

    system_prompt = captured["payload"]["messages"][0]["content"]
    user_prompt = captured["payload"]["messages"][1]["content"]
    assert "Nunca escribas 'Archivo'" in system_prompt
    assert "[Archivo, Pág]" not in system_prompt
    assert "[Archivo, Pág]" not in user_prompt
    assert "FUENTE y PAGINA" in user_prompt
