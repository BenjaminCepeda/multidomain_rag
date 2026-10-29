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
)


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
    adapter = LocalLLMAdapter(
        endpoint="http://llm.test/v1",
        model_chat="test-model",
        timeout=5,
    )
    adapter.check_health = lambda: True  # type: ignore[method-assign]
    return adapter


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
