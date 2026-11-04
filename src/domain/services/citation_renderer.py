"""Deterministic post-processing of residual citation placeholders.

Local models occasionally copy generic markers such as ``[Archivo, Pág 3]``
instead of the real source value injected in the prompt. This renderer
replaces those markers with the filename of the retrieved fragment that
belongs to the cited page. It also repairs the stochastic truncations a
local model leaves at the end of a line, where the citation opening bracket
survives but its label and closing bracket do not. It never invents a
source: when no retrieved fragment carries a usable filename the text is
cleaned without adding a citation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from src.domain.models.document import ManualChunk

# Labels that are never a real source. They cover the generic markers the
# local model copies from the prompt, including the literal angle-bracket
# template and the Spanish "nombre del archivo" variants.
_GENERIC_LABEL_PATTERN = (
    r"(?:archivo|documento|fuente|source|file|filename|document"
    r"|nombre(?:\s+real)?\s+del\s+archivo"
    r"|<[^>\n]{1,80}>)"
)

# Only generic markers are rewritten. A real filename in the same position is
# left untouched so already-correct citations stay immutable.
_PLACEHOLDER_PATTERN = re.compile(
    rf"\[\s*{_GENERIC_LABEL_PATTERN}\s*,\s*"
    r"(?:p[aá]g(?:ina)?\.?|page)\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)

# A truncated citation can end right after the opening bracket, after the
# generic label or after a partial page expression. It is anchored to the end
# of a line so a legitimate bracket inside prose is never touched.
_INCOMPLETE_TAIL_PATTERN = re.compile(
    rf"\[[ \t]*(?:{_GENERIC_LABEL_PATTERN}"
    r"(?:[ \t]*,[ \t]*(?:(?:p[aá]g(?:ina)?\.?|page)[ \t]*(?P<page>\d+)?)?)?)?"
    r"[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)

_SOURCE_KEYS = ("citation_source", "document_name", "source", "filename")
_INVALID_SOURCES = frozenset({"desconocido", "unknown", "archivo", "documento", "fuente"})


@dataclass(frozen=True)
class CitationRenderStats:
    """Zero-Log-safe counters extracted from one rendering pass."""

    placeholders: int = 0
    replaced: int = 0
    used_fallback_source: bool = False
    incomplete_tails: int = 0
    incomplete_replaced: int = 0
    incomplete_removed: int = 0


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
        self._fallback_page: int | None = None

        for chunk in chunks:
            source = _chunk_source(chunk)
            if source is None:
                continue
            if self._fallback_source is None:
                self._fallback_source = source
                self._fallback_page = chunk.page
            self._page_sources.setdefault(chunk.page, source)

    def render(self, text: str) -> tuple[str, CitationRenderStats]:
        if not text:
            return text, CitationRenderStats()

        placeholders = 0
        replaced = 0
        used_fallback = False
        incomplete_tails = 0
        incomplete_replaced = 0
        incomplete_removed = 0

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

        def _repair_tail(match: re.Match[str]) -> str:
            nonlocal used_fallback, incomplete_tails, incomplete_replaced, incomplete_removed
            incomplete_tails += 1
            raw_page = match.group("page")
            page = int(raw_page) if raw_page is not None else self._fallback_page
            source: str | None = None
            if page is not None:
                source = self._page_sources.get(page)
            if source is None:
                source = self._fallback_source
                if source is not None and page is not None:
                    used_fallback = True
            if source is None or page is None:
                # No traceable fragment backs this marker: drop the orphan
                # bracket instead of leaving a broken citation on screen.
                incomplete_removed += 1
                return ""
            incomplete_replaced += 1
            return f"[{source}, Pág {page}]"

        rendered = _PLACEHOLDER_PATTERN.sub(_replace, text)
        rendered = _INCOMPLETE_TAIL_PATTERN.sub(_repair_tail, rendered)
        return rendered, CitationRenderStats(
            placeholders=placeholders,
            replaced=replaced,
            used_fallback_source=used_fallback,
            incomplete_tails=incomplete_tails,
            incomplete_replaced=incomplete_replaced,
            incomplete_removed=incomplete_removed,
        )
