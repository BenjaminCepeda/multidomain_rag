# src/infrastructure/adapters/embedding_adapter.py
import logging
import os
from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from src.domain.models.embedding import EmbeddingVector
from src.domain.ports.embedding_service import IEmbeddingService
from src.infrastructure.config.settings import Settings
from src.infrastructure.hardware.device_factory import DeviceFactory

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingAdapter(IEmbeddingService):
    """Embedding adapter backed by sentence-transformers with hardware acceleration."""

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
            self._embedding_batch_size = settings.embedding_batch_size
        else:
            self._embedding_batch_size = 32

        self._model_name = model_name or "BAAI/bge-m3"
        self._expected_dim = expected_dim
        cache = cache_folder or os.getenv("HF_MODEL_CACHE")
        device = "cpu"
        try:
            device = DeviceFactory.configure_sentence_transformers_device()
        except Exception as exc:
            logger.warning("DeviceFactory failed for embeddings, fallback to cpu: %s", exc)
            device = "cpu"
        try:
            self._model = SentenceTransformer(self._model_name, cache_folder=cache, device=device)
            logger.info("Embeddings loaded on device=%s", device)
        except Exception as exc:
            logger.warning(
                "Failed to load embeddings on device=%s, falling back to cpu: %s", device, exc
            )
            try:
                self._model = SentenceTransformer(
                    self._model_name, cache_folder=cache, device="cpu"
                )
                logger.info("Embeddings loaded on device=cpu (fallback)")
            except Exception as exc2:
                raise exc2

    def embed_text(self, text: str) -> EmbeddingVector:
        vector = self._model.encode(text)
        return EmbeddingVector(values=tuple(float(x) for x in vector))

    def embed_batch(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[EmbeddingVector]:
        if not texts:
            return []
        resolved_batch_size = self._embedding_batch_size if batch_size is None else batch_size
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        vectors = self._model.encode(list(texts), batch_size=resolved_batch_size)
        return [EmbeddingVector(values=tuple(float(x) for x in v)) for v in vectors]

    def dimension(self) -> int:
        if self._expected_dim is not None:
            return self._expected_dim
        return self._model.get_sentence_embedding_dimension() or 0

    def model_name(self) -> str:
        return self._model_name
