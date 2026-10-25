# src/domain/ports/embedding_service.py
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.domain.models.embedding import EmbeddingVector


class IEmbeddingService(ABC):
    """Port for text embedding generation."""

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector: ...

    @abstractmethod
    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[EmbeddingVector]: ...

    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def model_name(self) -> str: ...
