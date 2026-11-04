"""Canonical dense-embedding recipe for visual chunks.

The ingestion adapter and the query-time re-scoring path must build the exact
same compact representation. Keeping the recipe in the domain layer avoids
duplicating it across infrastructure and application.
"""

from __future__ import annotations

import re

MAX_VISUAL_EMBEDDING_CHARS = 400
MAX_VISUAL_EMBEDDING_TITLE_CHARS = 100
MAX_VISUAL_EMBEDDING_DOCUMENT_CHARS = 80
MAX_VISUAL_EMBEDDING_THEME_CHARS = 90

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _clip_embedding_part(text: str, limit: int) -> str:
    normalized = _normalize(text)
    if len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def build_visual_embedding_text(
    document_title: str,
    document_name: str,
    image_title: str,
    page_number: int,
    theme_summary: str,
    surrounding_text_context: str,
) -> str:
    """Build the bounded dense-embedding text for a visual asset.

    The rich ``content`` remains available for payloads and the LLM. Dense
    embeddings only receive the identity of the figure, its document, its
    theme and a short context window, capped at ``MAX_VISUAL_EMBEDDING_CHARS``
    to avoid padding amplification in mixed text/visual batches.
    """
    title = _clip_embedding_part(image_title, MAX_VISUAL_EMBEDDING_TITLE_CHARS)
    document = _clip_embedding_part(
        document_title or document_name, MAX_VISUAL_EMBEDDING_DOCUMENT_CHARS
    )
    theme = _clip_embedding_part(theme_summary, MAX_VISUAL_EMBEDDING_THEME_CHARS)
    header = f"[IMAGEN] {title}\nDocumento: {document} (pag. {page_number})\nTema: {theme}"
    prefix = f"{header}\nContexto: "
    remaining = MAX_VISUAL_EMBEDDING_CHARS - len(prefix)
    if remaining <= 0:
        return header[:MAX_VISUAL_EMBEDDING_CHARS]
    context = _clip_embedding_part(surrounding_text_context, remaining)
    if not context:
        return header
    return f"{prefix}{context}"[:MAX_VISUAL_EMBEDDING_CHARS]
