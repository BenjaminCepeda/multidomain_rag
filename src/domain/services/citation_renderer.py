"""Deterministic post-processing of residual citation placeholders.

Local models occasionally copy generic markers such as ``[Archivo, Pág 3]``
instead of the real source value injected in the prompt. This renderer
replaces those markers with the filename of the retrieved fragment that
belongs to the cited page. It never invents a source: when no retrieved
fragment carries a usable filename the text is left untouched.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from src.domain.models.document import ManualChunk

# Only generic markers are rewritten. A real filename in the same position is
# left untouched so already-correct citations stay immutable.
_PLACEHOLDER_PATTERN = re.compile(
    r"\[\s*(?P<label>archivo|documento|fuente|source|file|filename|document)\s*,\s*"
    r"(?:p[aá]g(?:ina)?\.?|page)\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)

_SOURCE_KEYS = ("citation_source", "document_name", "source", "filename")
_INVALID_SOURCES = frozenset({"desconocido", "unknown", "archivo", "documento", "fuente"})


@dataclass(frozen=True)
class CitationRenderStats:
    """Zero-Log-safe counters extracted from one rendering pass."""

    placeholders: int = 0
    replaced: int = 0
    used_fallback_source: bool = False


def _basename(value: str) -> str:
    clean = value.strip().replace("\\", "/")
    return PurePosixPath(clean).name


def _chunk_source(chunk: ManualChunk) -> str | None:
    for key in _SOURCE_KEYS:
        value = chunk.metadata.get(key)
        if isinstance(value, str):
            candidate = _basename(value)
            if candidate and candidate.casefold() not in _INVALID_SOURCES:
                return candidate
    return None


class CitationRenderer:
    """Render residual ``[Archivo, Pág N]`` markers using retrieved metadata."""

    def __init__(self, chunks: Sequence[ManualChunk]) -> None:
        self._page_sources: dict[int, str] = {}
        self._fallback_source: str | None = None

        for chunk in chunks:
            source = _chunk_source(chunk)
            if source is None:
                continue
            if self._fallback_source is None:
                self._fallback_source = source
            self._page_sources.setdefault(chunk.page, source)

    def render(self, text: str) -> tuple[str, CitationRenderStats]:
        if not text:
            return text, CitationRenderStats()

        placeholders = 0
        replaced = 0
        used_fallback = False

        def _replace(match: re.Match[str]) -> str:
            nonlocal placeholders, replaced, used_fallback
            placeholders += 1
            page = int(match.group("page"))
            source = self._page_sources.get(page)
            if source is None:
                source = self._fallback_source
                if source is not None:
                    used_fallback = True
            if source is None:
                return match.group(0)
            replaced += 1
            return f"[{source}, Pág {page}]"

        rendered = _PLACEHOLDER_PATTERN.sub(_replace, text)
        return rendered, CitationRenderStats(
            placeholders=placeholders,
            replaced=replaced,
            used_fallback_source=used_fallback,
        )
