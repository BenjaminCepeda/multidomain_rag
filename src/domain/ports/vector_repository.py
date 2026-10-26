# src/domain/ports/vector_repository.py
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector


class IVectorRepository(ABC):
    """Port for vector persistence and similarity search."""

    @abstractmethod
    def ensure_collection(self, vector_dim: int) -> None: ...

    @abstractmethod
    def upsert(self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]) -> None: ...

    @abstractmethod
    def search(
        self, query_vector: EmbeddingVector, limit: int, min_score: float
    ) -> list[ManualChunk]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def exists_by_id(self, chunk_id: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...
