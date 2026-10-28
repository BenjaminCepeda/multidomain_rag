# src/infrastructure/config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed, centralized application settings.

    Values are loaded from environment variables and/or a local ``.env`` file.
    No other module in the codebase should read environment variables directly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM (LM Studio) ---
    llm_endpoint: str = Field(default="http://127.0.0.1:1234/v1")
    llm_model_chat: str = Field(default="qwen3-14b")
    llm_model_vision: str = Field(default="qwen3-14b")
    llm_timeout_seconds: int = Field(default=180, ge=1)

    # --- Vector store (Qdrant) ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="technical_docs")
    qdrant_timeout_seconds: float = Field(default=2.0, ge=0.1)

    # --- Embeddings ---
    embedding_model_name: str = Field(default="BAAI/bge-m3")
    embedding_vector_dim: int = Field(default=1024, ge=1)
    hf_model_cache: str = Field(default="./.cache/models")

    # --- Ingestion ---
    ingestion_normal_page_limit: int = Field(default=50, ge=1)
    ingestion_batch_size_extended: int = Field(default=10, ge=1)
    ingestion_max_retries: int = Field(default=3, ge=0)
    ingest_max_workers: int = Field(default=4, ge=1, le=32)

    # --- Speech-to-Text ---
    stt_model_size: str = Field(default="base")
    stt_device: str = Field(default="cpu")

    # --- Application / Presentation ---
    log_level: str = Field(default="INFO")
    gradio_host: str = Field(default="127.0.0.1")
    gradio_port: int = Field(default=7860, ge=1, le=65535)
