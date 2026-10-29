# src/infrastructure/adapters/pymupdf_adapter.py
"""PyMuPDF extraction adapter with semantic context enrichment for visual assets."""

import hashlib
import logging
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import fitz  # PyMuPDF

from src.domain.exceptions.core_errors import DataIncompleteError
from src.domain.models.document import ManualChunk
from src.domain.models.ids import ChunkId, DocumentId
from src.domain.ports.document_extractor import IDocumentExtractor
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 50
MAX_DOCUMENT_SUMMARY_CHARS = 600
MAX_SURROUNDING_CHARS = 700
MAX_TITLE_CHARS = 140
MAX_THEME_KEYWORDS = 8
PROFILE_PAGE_LIMIT = 3
CAPTION_MAX_CHARS = 90

type BlockInfo = tuple[float, float, float, float, str, int, int]


class ExtractedImage(NamedTuple):
    """An image written to disk together with its resolved caption and bbox."""

    path: str
    title: str
    top: float | None
    bottom: float | None


_TOKEN_PATTERN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_CAPTION_HINTS = (
    "figura",
    "fig.",
    "figure",
    "diagrama",
    "diagram",
    "esquema",
    "tabla",
    "ilustracion",
    "ilustración",
)

_STOPWORDS = frozenset(
    {
        "para",
        "como",
        "este",
        "esta",
        "esto",
        "estos",
        "estas",
        "desde",
        "donde",
        "cuando",
        "sobre",
        "entre",
        "hasta",
        "también",
        "tambien",
        "porque",
        "puede",
        "pueden",
        "debe",
        "deben",
        "todo",
        "todos",
        "toda",
        "todas",
        "otro",
        "otra",
        "otros",
        "otras",
        "cada",
        "solo",
        "sólo",
        "más",
        "mas",
        "pero",
        "sino",
        "ser",
        "son",
        "está",
        "estan",
        "están",
        "hay",
        "con",
        "sin",
        "los",
        "las",
        "del",
        "que",
        "por",
        "una",
        "uno",
        "unos",
        "unas",
        "sus",
        "según",
        "segun",
        "imagen",
        "image",
        "figura",
        "figure",
        "diagrama",
        "diagram",
        "tabla",
        "pagina",
        "página",
        "page",
        "documento",
        "document",
        "manual",
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "into",
        "over",
        "under",
        "when",
        "where",
        "which",
        "their",
        "there",
        "have",
        "has",
        "been",
        "will",
        "would",
        "could",
        "should",
        "than",
        "then",
        "also",
        "using",
        "used",
        "only",
        "such",
        "both",
        "each",
        "more",
        "most",
        "some",
        "very",
        "your",
        "about",
        "after",
        "before",
        "between",
        "through",
        "during",
    }
)


def _normalize(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _humanize_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    return _normalize(stem.replace("_", " ").replace("-", " "))


def _extract_keywords(text: str, limit: int = MAX_THEME_KEYWORDS) -> str:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    counts = Counter(token for token in tokens if token not in _STOPWORDS)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(word for word, _ in ranked[:limit])


def _vertical_distance(block: BlockInfo, top: float, bottom: float) -> float:
    if block[3] <= top:
        return top - block[3]
    if block[1] >= bottom:
        return block[1] - bottom
    return 0.0


def _nearest_caption(blocks: Sequence[BlockInfo], top: float, bottom: float) -> str:
    above = [
        (top - block[3], block) for block in blocks if block[3] <= top + 2 and block[4].strip()
    ]
    below = [
        (block[1] - bottom, block)
        for block in blocks
        if block[1] >= bottom - 2 and block[4].strip()
    ]
    candidates = sorted(above, key=lambda item: item[0]) + sorted(below, key=lambda item: item[0])

    fallback = ""
    for _, block in candidates:
        line = next((raw.strip() for raw in block[4].splitlines() if raw.strip()), "")
        if not line:
            continue
        if any(hint in line.lower() for hint in _CAPTION_HINTS):
            return line[:MAX_TITLE_CHARS]
        if not fallback and len(line) <= CAPTION_MAX_CHARS:
            fallback = line[:MAX_TITLE_CHARS]
    return fallback


def _surrounding_context(
    blocks: Sequence[BlockInfo],
    top: float | None,
    bottom: float | None,
    page_text: str,
    previous_text: str,
) -> str:
    if top is not None and bottom is not None:
        ranked = sorted(blocks, key=lambda block: _vertical_distance(block, top, bottom))
        parts: list[str] = []
        used = 0
        for block in ranked:
            excerpt = _normalize(block[4])
            if not excerpt:
                continue
            remaining = MAX_SURROUNDING_CHARS - used
            if remaining <= 0:
                break
            excerpt = excerpt[:remaining]
            parts.append(excerpt)
            used += len(excerpt) + 1
        if parts:
            return _normalize(" ".join(parts))
    if page_text:
        return page_text[:MAX_SURROUNDING_CHARS]
    return previous_text[-MAX_SURROUNDING_CHARS:]


def _build_visual_content(
    document_title: str,
    document_name: str,
    image_title: str,
    page_number: int,
    theme_summary: str,
    document_context_summary: str,
    surrounding_text_context: str,
) -> str:
    parts = [
        f"[IMAGEN] {image_title}",
        f"Documento: {document_title} ({document_name})",
        f"Pagina: {page_number}",
    ]
    if theme_summary:
        parts.append(f"Tema: {theme_summary}")
    if document_context_summary:
        parts.append(f"Contexto del documento: {document_context_summary}")
    if surrounding_text_context:
        parts.append(f"Texto circundante: {surrounding_text_context}")
    return "\n".join(parts)


class PyMuPDFAdapter(IDocumentExtractor):
    """Extracts text and images, enriching visuals with macro and micro context."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def extract_chunks(self, file_path: str) -> list[ManualChunk]:
        if not os.path.exists(file_path):
            raise DataIncompleteError(f"El extractor fisico no encontro el archivo en: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise DataIncompleteError(
                f"El motor PyMuPDF no pudo abrir el archivo: {file_path}", exc
            ) from exc

        try:
            return self._extract_document(doc, file_path)
        except DataIncompleteError:
            raise
        except Exception as exc:
            raise DataIncompleteError(
                f"El motor PyMuPDF colapso al leer la estructura binaria de: {file_path}", exc
            ) from exc
        finally:
            doc.close()

    def _extract_document(self, doc: fitz.Document, file_path: str) -> list[ManualChunk]:
        file_name = os.path.basename(file_path)
        document_id = DocumentId.from_source(file_path)
        doc_hash = hashlib.md5(str(document_id).encode()).hexdigest()[:12]

        document_title, document_context_summary, prefetched = self._build_document_profile(
            doc, file_name
        )

        images_dir = self._settings.extracted_images_dir
        images_dir.mkdir(parents=True, exist_ok=True)

        chunks: list[ManualChunk] = []
        previous_text = ""

        for page_num in range(len(doc)):
            page_number = page_num + 1
            page = doc[page_num]
            blocks = prefetched.get(page_num) or self._read_blocks(page)

            raw_text = "\n".join(block[4] for block in blocks if block[4].strip()).strip()
            page_text = _normalize(raw_text)
            page_heading = next(
                (
                    line.strip()
                    for line in raw_text.splitlines()
                    if MIN_TEXT_LENGTH > len(line.strip()) >= 3
                ),
                "",
            )

            images = self._extract_page_images(doc, page, blocks, images_dir, doc_hash, page_number)

            if len(page_text) < MIN_TEXT_LENGTH and not images:
                continue

            if len(page_text) >= MIN_TEXT_LENGTH:
                paragraphs = [
                    paragraph.strip()
                    for paragraph in raw_text.split("\n")
                    if len(paragraph.strip()) > MIN_TEXT_LENGTH
                ]
                if not paragraphs:
                    paragraphs = [page_text]
                for index, paragraph in enumerate(paragraphs):
                    metadata: dict[str, str | int | float | bool] = {
                        "page": page_number,
                        "page_number": page_number,
                        "source": file_name,
                        "document_name": file_name,
                        "document_title": document_title,
                    }
                    if images:
                        metadata["image_paths"] = "|".join(image.path for image in images)
                    chunks.append(
                        ManualChunk(
                            chunk_id=ChunkId.build(
                                document_id=document_id,
                                section=None,
                                page=page_number,
                                index=index,
                            ),
                            document_id=document_id,
                            content=paragraph,
                            page=page_number,
                            metadata=metadata,
                            is_visual=False,
                        )
                    )

            for image_index, image in enumerate(images):
                chunks.append(
                    self._build_visual_chunk(
                        document_id=document_id,
                        document_title=document_title,
                        document_name=file_name,
                        document_context_summary=document_context_summary,
                        image=image,
                        image_index=image_index,
                        page_number=page_number,
                        blocks=blocks,
                        page_text=page_text,
                        previous_text=previous_text,
                        page_heading=page_heading,
                    )
                )

            previous_text = page_text or previous_text

        return chunks

    def _build_document_profile(
        self, doc: fitz.Document, file_name: str
    ) -> tuple[str, str, dict[int, list[BlockInfo]]]:
        """Derive document title and macro summary, prefetching the first pages.

        Prefetched blocks are reused by the page loop, so the bounded profile pass
        does not re-parse text and adds no inference cost.
        """
        raw_title = _normalize(str((doc.metadata or {}).get("title") or ""))
        document_title = raw_title[:MAX_TITLE_CHARS] or _humanize_stem(file_name) or file_name

        prefetched: dict[int, list[BlockInfo]] = {}
        excerpts: list[str] = []
        remaining = MAX_DOCUMENT_SUMMARY_CHARS

        for page_index in range(min(PROFILE_PAGE_LIMIT, len(doc))):
            blocks = self._read_blocks(doc[page_index])
            prefetched[page_index] = blocks
            page_text = _normalize(" ".join(block[4] for block in blocks))
            if page_text:
                excerpt = page_text[:remaining]
                excerpts.append(excerpt)
                remaining -= len(excerpt)
            if remaining <= 0:
                break

        summary = _normalize(" ".join(excerpts))[:MAX_DOCUMENT_SUMMARY_CHARS]
        if not summary:
            summary = f"Documento tecnico: {document_title}"
        return document_title, summary, prefetched

    @staticmethod
    def _read_blocks(page: fitz.Page) -> list[BlockInfo]:
        raw_blocks = page.get_text("blocks")
        return [cast(BlockInfo, tuple(block)) for block in raw_blocks]

    def _extract_page_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        blocks: Sequence[BlockInfo],
        images_dir: Path,
        doc_hash: str,
        page_number: int,
    ) -> list[ExtractedImage]:
        images: list[ExtractedImage] = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:
            logger.warning("PDF_IMAGES | page=%d | action=skipped | error=%s", page_number, exc)
            return images

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue
                extension = base_image.get("ext", "png")
                image_name = f"{doc_hash}_p{page_number}_i{img_index}.{extension}"
                image_path = images_dir / image_name
                image_path.write_bytes(image_bytes)

                top, bottom = self._image_bounds(page, xref)
                title = (
                    _nearest_caption(blocks, top, bottom)
                    if top is not None and bottom is not None
                    else ""
                )
                images.append(
                    ExtractedImage(
                        path=str(image_path.resolve()),
                        title=title,
                        top=top,
                        bottom=bottom,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "PDF_IMAGES | page=%d | image=%d | action=skipped | error=%s",
                    page_number,
                    img_index,
                    exc,
                )
                continue
        return images

    @staticmethod
    def _image_bounds(page: fitz.Page, xref: int) -> tuple[float | None, float | None]:
        try:
            rects = page.get_image_rects(xref)
        except Exception as exc:
            logger.warning("PDF_IMAGES | action=skipped_bounds | error=%s", exc)
            return None, None
        if not rects:
            return None, None
        rect = rects[0]
        return float(rect.y0), float(rect.y1)

    @staticmethod
    def _build_visual_chunk(
        *,
        document_id: DocumentId,
        document_title: str,
        document_name: str,
        document_context_summary: str,
        image: ExtractedImage,
        image_index: int,
        page_number: int,
        blocks: Sequence[BlockInfo],
        page_text: str,
        previous_text: str,
        page_heading: str,
    ) -> ManualChunk:
        image_title = image.title or page_heading or f"Imagen en la pagina {page_number}"
        surrounding = _surrounding_context(
            blocks, image.top, image.bottom, page_text, previous_text
        )

        theme_summary = _extract_keywords(f"{image_title} {surrounding}")
        if not theme_summary:
            theme_summary = _extract_keywords(document_context_summary) or document_title

        content = _build_visual_content(
            document_title=document_title,
            document_name=document_name,
            image_title=image_title,
            page_number=page_number,
            theme_summary=theme_summary,
            document_context_summary=document_context_summary,
            surrounding_text_context=surrounding,
        )

        metadata: dict[str, str | int | float | bool] = {
            "page": page_number,
            "page_number": page_number,
            "source": document_name,
            "document_name": document_name,
            "document_title": document_title,
            "document_context_summary": document_context_summary,
            "image_title": image_title,
            "theme_summary": theme_summary,
            "surrounding_text_context": surrounding,
            "image_path": image.path,
            "image_paths": image.path,
        }

        return ManualChunk(
            chunk_id=ChunkId.build(
                document_id=document_id,
                section="visual",
                page=page_number,
                index=image_index,
            ),
            document_id=document_id,
            content=content,
            page=page_number,
            metadata=metadata,
            is_visual=True,
        )
