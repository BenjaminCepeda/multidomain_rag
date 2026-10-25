# src/infrastructure/adapters/embedding_adapter.py
import os
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from src.domain.models.embedding import EmbeddingVector
from src.domain.ports.embedding_service import IEmbeddingService
from src.infrastructure.config.settings import Settings


class SentenceTransformerEmbeddingAdapter(IEmbeddingService):
    """Embedding adapter backed by sentence-transformers."""

    def __init__(
        self,
        settings: Settings | None = None,
        model_name: str | None = None,
        cache_folder: str | None = None,
        expected_dim: int | None = None,
    ) -> None:
        if settings is not None:
            model_name = model_name or settings.embedding_model_name
            cache_folder = cache_folder or settings.hf_model_cache
            expected_dim = expected_dim or settings.embedding_vector_dim

        self._model_name = model_name or "BAAI/bge-m3"
        self._expected_dim = expected_dim
        cache = cache_folder or os.getenv("HF_MODEL_CACHE")
        self._model = SentenceTransformer(self._model_name, cache_folder=cache)

    def embed_text(self, text: str) -> EmbeddingVector:
        vector = self._model.encode(text)
        return EmbeddingVector(values=tuple(float(x) for x in vector))

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[EmbeddingVector]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), batch_size=batch_size)
        return [EmbeddingVector(values=tuple(float(x) for x in v)) for v in vectors]

    def dimension(self) -> int:
        if self._expected_dim is not None:
            return self._expected_dim
        return self._model.get_sentence_embedding_dimension() or 0

    def model_name(self) -> str:
        return self._model_name
