from src.infrastructure.config.settings import Settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_model_chat == "qwen3-14b"
    assert s.embedding_vector_dim == 1024
    assert s.embedding_batch_size == 32
    assert s.llm_context_max_chars_per_chunk == 900
    assert s.gradio_port == 7860
    assert s.llm_router_max_tokens == 48
    assert s.llm_router_connect_timeout_seconds == 1.5
    assert s.llm_router_read_timeout_seconds == 6.0
    assert s.llm_generation_read_timeout_seconds == 120.0
    assert s.llm_vision_read_timeout_seconds == 45.0
    assert s.llm_max_concurrency == 1
    assert s.vision_max_images_per_turn == 3
    assert s.vision_max_long_edge_px == 1600
    assert s.vision_jpeg_quality == 80
    assert s.vision_max_total_payload_bytes == 4_000_000
    assert s.vision_thumbnail_long_edge_px == 512


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_CHAT", "test-model")
    monkeypatch.setenv("EMBEDDING_VECTOR_DIM", "512")
    s = Settings(_env_file=None)
    assert s.llm_model_chat == "test-model"
    assert s.embedding_vector_dim == 512
