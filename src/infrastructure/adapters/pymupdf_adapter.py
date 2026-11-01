# src/infrastructure/adapters/pymupdf_adapter.py
"""PyMuPDF extraction adapter with semantic context enrichment for visual assets."""

import hashlib
import io
import logging
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import fitz  # PyMuPDF
from PIL import Image, ImageOps

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
VISION_TRANSFORM_VERSION = 1

type BlockInfo = tuple[float, float, float, float, str, int, int]


class ExtractedImage(NamedTuple):
    """An image written to disk together with its resolved caption and bbox."""

    path: str
    title: str
    top: float | None
    bottom: float | None
    vision_path: str = ""
    thumb_path: str = ""
    vision_format: str = ""
    vision_bytes: int = 0
    vision_width: int = 0
    vision_height: int = 0
    image_sha256: str = ""
    vision_sha256: str = ""
    original_width: int = 0
    original_height: int = 0


class VisualDerivative(NamedTuple):
    """Optimized vision asset and thumbnail generated during ingestion."""

    vision_path: str
    thumb_path: str
    vision_format: str
    vision_bytes: int
    vision_width: int
    vision_height: int
    vision_sha256: str
    original_width: int
    original_height: int


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
            raise DataIncompleteError("El extractor fisico no encontro el archivo solicitado.")

        try:
            resolved_hash = file_hash or compute_file_sha256(file_path)
        except OSError as exc:
            raise DataIncompleteError("No se pudo calcular la huella del archivo.", exc) from exc

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise DataIncompleteError("El motor PyMuPDF no pudo abrir el archivo.", exc) from exc

        try:
            return self._extract_document(doc, file_path, resolved_hash)
        except DataIncompleteError:
            raise
        except Exception as exc:
            raise DataIncompleteError(
                "El motor PyMuPDF colapso al leer la estructura binaria.", exc
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
                        vision_paths = "|".join(
                            image.vision_path for image in images if image.vision_path
                        )
                        if vision_paths:
                            metadata["vision_paths"] = vision_paths
                        thumb_paths = "|".join(
                            image.thumb_path for image in images if image.thumb_path
                        )
                        if thumb_paths:
                            metadata["thumb_paths"] = thumb_paths
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
            logger.warning(
                "PDF_IMAGES | page=%d | action=skipped | error_type=%s",
                page_number,
                type(exc).__name__,
            )
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
                image_sha256 = hashlib.sha256(image_bytes).hexdigest()

                derivative = self._build_derivatives(
                    image_bytes,
                    extension,
                    images_dir,
                    doc_hash,
                    page_number,
                    img_index,
                )

                top, bottom = self._image_bounds(page, xref)
                title = (
                    _nearest_caption(blocks, top, bottom)
                    if top is not None and bottom is not None
                    else ""
                )
                extracted = ExtractedImage(
                    path=str(image_path.resolve()),
                    title=title,
                    top=top,
                    bottom=bottom,
                    image_sha256=image_sha256,
                )
                if derivative is not None:
                    extracted = extracted._replace(
                        vision_path=derivative.vision_path,
                        thumb_path=derivative.thumb_path,
                        vision_format=derivative.vision_format,
                        vision_bytes=derivative.vision_bytes,
                        vision_width=derivative.vision_width,
                        vision_height=derivative.vision_height,
                        vision_sha256=derivative.vision_sha256,
                        original_width=derivative.original_width,
                        original_height=derivative.original_height,
                    )
                images.append(extracted)
            except Exception as exc:
                logger.warning(
                    "PDF_IMAGES | page=%d | image=%d | action=skipped | error_type=%s",
                    page_number,
                    img_index,
                    type(exc).__name__,
                )
                continue
        return images

    def _build_derivatives(
        self,
        image_bytes: bytes,
        extension: str,
        images_dir: Path,
        doc_hash: str,
        page_number: int,
        image_index: int,
    ) -> VisualDerivative | None:
        """Generate a bounded vision asset and a UI thumbnail during ingestion.

        The original stays untouched for the interface. The LLM receives a
        normalized derivative whose long edge and encoded size are bounded by
        settings, which keeps Base64 payloads and visual prefill predictable.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                original_width, original_height = source.size
                original_mode = source.mode
                has_alpha = original_mode in {"RGBA", "LA"} or (
                    original_mode == "P" and "transparency" in source.info
                )
                prepared = ImageOps.exif_transpose(source)

            prepared = self._flatten_alpha(prepared, has_alpha)
            use_png = self._should_use_png(extension, original_mode, has_alpha, prepared)
            vision_data, vision_format, vision_width, vision_height = self._encode_vision(
                prepared, use_png
            )

            vision_name = (
                f"{doc_hash}_p{page_number}_i{image_index}"
                f"_v{VISION_TRANSFORM_VERSION}.{vision_format}"
            )
            vision_path = images_dir / vision_name
            vision_path.write_bytes(vision_data)

            thumb_data = self._encode_thumbnail(prepared)
            thumb_name = f"{doc_hash}_p{page_number}_i{image_index}_t.jpg"
            thumb_path = images_dir / thumb_name
            thumb_path.write_bytes(thumb_data)

            return VisualDerivative(
                vision_path=str(vision_path.resolve()),
                thumb_path=str(thumb_path.resolve()),
                vision_format=vision_format,
                vision_bytes=len(vision_data),
                vision_width=vision_width,
                vision_height=vision_height,
                vision_sha256=hashlib.sha256(vision_data).hexdigest(),
                original_width=int(original_width),
                original_height=int(original_height),
            )
        except Exception as exc:
            logger.warning(
                "PDF_IMAGES | page=%d | image=%d | action=derivative_skipped | error_type=%s",
                page_number,
                image_index,
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _flatten_alpha(image: Image.Image, has_alpha: bool) -> Image.Image:
        if not has_alpha:
            return image.convert("RGB")
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    @staticmethod
    def _should_use_png(
        extension: str,
        original_mode: str,
        has_alpha: bool,
        image: Image.Image,
    ) -> bool:
        """Prefer lossless output for line art and JPEG for photographic content."""
        if extension.lower() in {"jpg", "jpeg"}:
            return False
        if has_alpha or original_mode in {"1", "P", "L"}:
            return True
        try:
            sample = image.copy()
            sample.thumbnail((512, 512), Image.Resampling.BILINEAR)
            colors = sample.getcolors(maxcolors=2048)
            return colors is not None
        except Exception:
            return False

    def _encode_vision(self, image: Image.Image, use_png: bool) -> tuple[bytes, str, int, int]:
        """Resize and encode the derivative, tightening it until it fits budget."""
        max_edge = self._settings.vision_max_long_edge_px
        max_bytes = self._settings.vision_max_bytes_per_image
        quality = self._settings.vision_jpeg_quality

        edge = max_edge
        data = b""
        output_format = "png" if use_png else "jpg"
        width, height = image.size
        while True:
            candidate = self._resize_long_edge(image, edge)
            width, height = candidate.size
            buffer = io.BytesIO()
            if use_png:
                candidate.save(buffer, format="PNG", optimize=True)
            else:
                candidate.save(
                    buffer,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                )
            data = buffer.getvalue()
            if len(data) <= max_bytes or edge <= 960:
                break
            edge = max(960, edge // 2)

        if len(data) > max_bytes and use_png:
            # A detailed scan can stay large even after resizing; fall back to
            # lossy but bounded JPEG rather than shipping an oversized payload.
            buffer = io.BytesIO()
            self._resize_long_edge(image, edge).save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            data = buffer.getvalue()
            output_format = "jpg"

        return data, output_format, int(width), int(height)

    @staticmethod
    def _resize_long_edge(image: Image.Image, max_edge: int) -> Image.Image:
        if max(image.size) <= max_edge:
            return image
        resized = image.copy()
        resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return resized

    def _encode_thumbnail(self, image: Image.Image) -> bytes:
        thumb = image.copy()
        thumb.thumbnail(
            (
                self._settings.vision_thumbnail_long_edge_px,
                self._settings.vision_thumbnail_long_edge_px,
            ),
            Image.Resampling.LANCZOS,
        )
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=75, optimize=True, progressive=True)
        return buffer.getvalue()

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
            logger.debug(
                "PDF_IMAGES | action=pixmap_load_failed | xref=%d | error_type=%s",
                xref,
                type(exc).__name__,
            )
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
                logger.warning(
                    "PDF_IMAGES | action=color_conversion_failed | error_type=%s",
                    type(exc).__name__,
                )

        return PyMuPDFAdapter._extract_original_image(doc, xref)

    @staticmethod
    def _extract_original_image(doc: fitz.Document, xref: int) -> tuple[bytes, str] | None:
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.warning(
                "PDF_IMAGES | action=extract_failed | error_type=%s",
                type(exc).__name__,
            )
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
            logger.warning(
                "PDF_IMAGES | action=skipped_bounds | error_type=%s",
                type(exc).__name__,
            )
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
            "image_sha256": image.image_sha256,
            "original_width": image.original_width,
            "original_height": image.original_height,
        }
        if image.vision_path:
            metadata["vision_path"] = image.vision_path
            metadata["vision_paths"] = image.vision_path
            metadata["vision_format"] = image.vision_format
            metadata["vision_bytes"] = image.vision_bytes
            metadata["vision_width"] = image.vision_width
            metadata["vision_height"] = image.vision_height
            metadata["vision_sha256"] = image.vision_sha256
            metadata["vision_transform_version"] = VISION_TRANSFORM_VERSION
        if image.thumb_path:
            metadata["thumb_path"] = image.thumb_path
            metadata["thumb_paths"] = image.thumb_path

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
