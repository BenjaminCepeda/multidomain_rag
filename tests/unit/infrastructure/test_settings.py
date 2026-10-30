from src.infrastructure.config.settings import Settings


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_model_chat == "qwen3-14b"
    assert s.embedding_vector_dim == 1024
    assert s.embedding_batch_size == 32
    assert s.llm_context_max_chars_per_chunk == 900
    assert s.gradio_port == 7860


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_CHAT", "test-model")
    monkeypatch.setenv("EMBEDDING_VECTOR_DIM", "512")
    s = Settings(_env_file=None)
    assert s.llm_model_chat == "test-model"
    assert s.embedding_vector_dim == 512
