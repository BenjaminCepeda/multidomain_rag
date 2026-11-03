import pytest
from pydantic import ValidationError

from src.infrastructure.config.settings import Settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_model_chat == "qwen3-14b"
    assert s.embedding_vector_dim == 1024
    assert s.embedding_batch_size == 32
    assert s.llm_context_max_chars_per_chunk == 900
    assert s.gradio_port == 7860
    assert s.llm_router_max_tokens == 48
    assert 32 <= s.llm_router_max_tokens <= 48
    assert s.llm_router_connect_timeout_seconds == 1.5
    assert s.llm_router_read_timeout_seconds == 10.0
    assert 8.0 <= s.llm_router_read_timeout_seconds <= 10.0
    assert s.llm_generation_read_timeout_seconds == 120.0
    assert s.llm_vision_read_timeout_seconds == 45.0
    assert s.llm_max_concurrency == 1
    assert s.vision_max_images_per_turn == 3
    assert s.vision_max_long_edge_px == 1600
    assert s.vision_jpeg_quality == 80
    assert s.vision_max_total_payload_bytes == 4_000_000
    assert s.vision_thumbnail_long_edge_px == 512


@pytest.mark.parametrize("value", [8.0, 9.0, 10.0])
def test_router_read_timeout_accepts_contract_bounds(value: float) -> None:
    s = Settings(_env_file=None, llm_router_read_timeout_seconds=value)
    assert s.llm_router_read_timeout_seconds == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_router_read_timeout_seconds", 7.9),
        ("llm_router_read_timeout_seconds", 10.1),
        ("llm_router_max_tokens", 31),
        ("llm_router_max_tokens", 49),
    ],
)
def test_router_settings_reject_values_outside_the_contract(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_CHAT", "test-model")
    monkeypatch.setenv("EMBEDDING_VECTOR_DIM", "512")
    s = Settings(_env_file=None)
    assert s.llm_model_chat == "test-model"
    assert s.embedding_vector_dim == 512
