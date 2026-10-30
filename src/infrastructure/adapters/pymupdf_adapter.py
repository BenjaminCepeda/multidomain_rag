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
from src.domain.services.file_fingerprint import compute_file_sha256
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 50
MAX_DOCUMENT_SUMMARY_CHARS = 600
MAX_SURROUNDING_CHARS = 700
MAX_TITLE_CHARS = 140
MAX_THEME_KEYWORDS = 8
PROFILE_PAGE_LIMIT = 3
CAPTION_MAX_CHARS = 90
MAX_VISUAL_EMBEDDING_CHARS = 400
MAX_VISUAL_EMBEDDING_TITLE_CHARS = 100
MAX_VISUAL_EMBEDDING_DOCUMENT_CHARS = 80
MAX_VISUAL_EMBEDDING_THEME_CHARS = 90

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


def _clip_embedding_part(text: str, limit: int) -> str:
    normalized = _normalize(text)
    if len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def _build_visual_embedding_text(
    document_title: str,
    document_name: str,
    image_title: str,
    page_number: int,
    theme_summary: str,
    surrounding_text_context: str,
) -> str:
    """Build a bounded dense-embedding text for a visual asset.

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


class PyMuPDFAdapter(IDocumentExtractor):
    """Extracts text and images, enriching visuals with macro and micro context."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def extract_chunks(self, file_path: str, file_hash: str | None = None) -> list[ManualChunk]:
        if not os.path.exists(file_path):
            raise DataIncompleteError(f"El extractor fisico no encontro el archivo en: {file_path}")

        try:
            resolved_hash = file_hash or compute_file_sha256(file_path)
        except OSError as exc:
            raise DataIncompleteError(
                f"No se pudo calcular la huella del archivo: {file_path}", exc
            ) from exc

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise DataIncompleteError(
                f"El motor PyMuPDF no pudo abrir el archivo: {file_path}", exc
            ) from exc

        try:
            return self._extract_document(doc, file_path, resolved_hash)
        except DataIncompleteError:
            raise
        except Exception as exc:
            raise DataIncompleteError(
                f"El motor PyMuPDF colapso al leer la estructura binaria de: {file_path}", exc
            ) from exc
        finally:
            doc.close()

    def _extract_document(
        self, doc: fitz.Document, file_path: str, file_hash: str
    ) -> list[ManualChunk]:
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
                        "file_hash": file_hash,
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
                        file_hash=file_hash,
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
                loaded = self._load_renderable_image(doc, xref)
                if loaded is None:
                    continue
                image_bytes, extension = loaded
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
    def _load_renderable_image(doc: fitz.Document, xref: int) -> tuple[bytes, str] | None:
        """Load image bytes, forcing a safe RGB conversion when needed.

        ``Document.extract_image`` preserves the embedded colorspace, so CMYK
        or multi-channel pixmaps can be written as invalid image files that
        render as black boxes. Standard RGB/Gray images keep their original
        encoded bytes; any other colorspace is converted to RGB first.
        """
        try:
            pixmap = fitz.Pixmap(doc, xref)
        except Exception as exc:
            logger.debug("PDF_IMAGES | action=pixmap_load_failed | xref=%d | error=%s", xref, exc)
            pixmap = None

        if pixmap is not None:
            colorspace = pixmap.colorspace
            if colorspace is None:
                # Alpha-only masks are not standalone, renderable images.
                return None

            is_standard_gray = (
                colorspace.name == "DeviceGray" and pixmap.n == 1 and not pixmap.alpha
            )
            is_standard_rgb = colorspace.name == "DeviceRGB" and pixmap.n == 3 and not pixmap.alpha

            if is_standard_gray or is_standard_rgb:
                original = PyMuPDFAdapter._extract_original_image(doc, xref)
                if original is not None:
                    return original

            try:
                converted = (
                    pixmap
                    if (is_standard_gray or is_standard_rgb)
                    else fitz.Pixmap(fitz.csRGB, pixmap)
                )
                return converted.tobytes("png"), "png"
            except Exception as exc:
                logger.warning("PDF_IMAGES | action=color_conversion_failed | error=%s", exc)

        return PyMuPDFAdapter._extract_original_image(doc, xref)

    @staticmethod
    def _extract_original_image(doc: fitz.Document, xref: int) -> tuple[bytes, str] | None:
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.warning("PDF_IMAGES | action=extract_failed | error=%s", exc)
            return None

        image_bytes = base_image.get("image")
        if not image_bytes:
            return None
        return image_bytes, str(base_image.get("ext") or "png")

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
        file_hash: str,
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
        embedding_text = _build_visual_embedding_text(
            document_title=document_title,
            document_name=document_name,
            image_title=image_title,
            page_number=page_number,
            theme_summary=theme_summary,
            surrounding_text_context=surrounding,
        )

        metadata: dict[str, str | int | float | bool] = {
            "page": page_number,
            "page_number": page_number,
            "source": document_name,
            "document_name": document_name,
            "document_title": document_title,
            "document_context_summary": document_context_summary,
            "file_hash": file_hash,
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
            embedding_text=embedding_text,
        )
