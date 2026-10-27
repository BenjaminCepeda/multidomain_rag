# src/domain/models/ids.py
"""Domain value objects for stable identifiers."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentId:
    """Stable, unique identifier for a source document.

    Derived deterministically from the normalized source path so that
    re-ingestion of the same document yields the same identifier.
    """

    value: uuid.UUID

    @classmethod
    def from_source(cls, source: str | Path) -> DocumentId:
        normalized = str(Path(source).resolve()).lower()
        return cls(value=uuid.uuid5(uuid.NAMESPACE_URL, normalized))

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class ChunkId:
    """Stable identifier for a document chunk.

    Format: ``<document_id>:<section_path>:<page>:<index>``.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("ChunkId cannot be empty.")

    @classmethod
    def build(
        cls,
        document_id: DocumentId,
        section: str | None,
        page: int,
        index: int,
    ) -> ChunkId:
        if page < 1:
            raise ValueError("page must be >= 1")
        if index < 0:
            raise ValueError("index must be >= 0")
        section_part = (section or "root").strip().lower().replace(" ", "_")
        return cls(value=f"{document_id}:{section_part}:{page}:{index}")

    @classmethod
    def from_content(
        cls,
        document_id: DocumentId,
        content: str,
        page: int,
        section: str | None = None,
        image_paths: str | None = None,
    ) -> ChunkId:
        seed = f"{document_id}|{section or 'root'}|{page}|{content}|{image_paths or ''}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return cls(value=f"{document_id}:sha256:{digest}")

    def __str__(self) -> str:
        return self.value
