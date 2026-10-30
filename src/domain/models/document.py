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
    embedding_text: str | None = None

    @property
    def embedding_input(self) -> str:
        """Return the compact text used for dense embeddings.

        Visual chunks keep a rich ``content`` for payloads and the LLM while
        exposing a bounded ``embedding_text`` to avoid padding amplification.
        """
        if self.embedding_text and self.embedding_text.strip():
            return self.embedding_text
        return self.content
