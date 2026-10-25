# src/domain/models/document.py
from collections.abc import Mapping
from dataclasses import dataclass, field

from src.domain.models.ids import ChunkId, DocumentId


@dataclass(frozen=True)
class ManualChunk:
    """A single, immutable fragment of a document ready for indexing/retrieval."""

    chunk_id: ChunkId
    content: str
    page: int = 1
    section: str | None = None
    is_visual: bool = False
    document_id: DocumentId | None = None
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
