import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import requests

from src.domain.exceptions.core_errors import LLMInferenceError
from src.infrastructure.adapters.local_llm_adapter import (
    MAX_IMAGES_PER_REQUEST,
    LocalLLMAdapter,
    _CircuitBreaker,
)
from src.infrastructure.config.settings import Settings


class _FakeResponse:
    def __init__(self, content: str = "respuesta", status_code: int = 200) -> None:
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)  # type: ignore[arg-type]

    def json(self) -> dict[str, Any]:
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
    content = captured["payload"]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Cuerpo del fragmento" in content[0]["text"]
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

    content = captured["payload"]["messages"][1]["content"]
    assert isinstance(content, str)
    assert "pregunta" in content


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
    assert "imagenes adjuntas" not in second_content


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
    assert captured["timeout"][0] == 1.5
    assert captured["timeout"][1] == 10.0


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
