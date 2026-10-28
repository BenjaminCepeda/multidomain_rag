# src/infrastructure/adapters/pymupdf_adapter.py
import hashlib
import os

import fitz  # PyMuPDF

from src.domain.exceptions.core_errors import DataIncompleteError
from src.domain.models.document import ManualChunk
from src.domain.models.ids import ChunkId, DocumentId
from src.domain.ports.document_extractor import IDocumentExtractor
from src.infrastructure.config.settings import Settings


class PyMuPDFAdapter(IDocumentExtractor):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._data_files_path = settings.data_files_path
        self._extracted_images_path = settings.extracted_images_path

    def extract_chunks(self, file_path: str) -> list[ManualChunk]:
        if not os.path.exists(file_path):
            raise DataIncompleteError(f"El extractor fisico no encontro el archivo en: {file_path}")

        chunks: list[ManualChunk] = []
        try:
            doc = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            document_id = DocumentId.from_source(file_path)
            doc_hash = hashlib.md5(str(document_id).encode()).hexdigest()[:12]
            images_dir = os.path.join(self._data_files_path, self._extracted_images_path)
            os.makedirs(images_dir, exist_ok=True)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                page_images: list[str] = []
                try:
                    image_list = page.get_images(full=True)
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        try:
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image.get("image")
                            if not image_bytes:
                                continue
                            ext = base_image.get("ext", "png")
                            img_name = f"{doc_hash}_p{page_num + 1}_i{img_idx}.{ext}"
                            img_path = os.path.join(images_dir, img_name)
                            with open(img_path, "wb") as f:
                                f.write(image_bytes)
                            page_images.append(os.path.abspath(img_path))
                        except Exception:
                            continue
                except Exception:
                    page_images = []

                if len(text) < 50 and not page_images:
                    continue

                if len(text) >= 50:
                    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
                    if not paragraphs and text.strip():
                        paragraphs = [text.strip()]
                    for idx, para in enumerate(paragraphs):
                        chunk_id = ChunkId.build(
                            document_id=document_id,
                            section=None,
                            page=page_num + 1,
                            index=idx,
                        )
                        metadata: dict[str, str | int | float | bool] = {
                            "page": page_num + 1,
                            "source": file_name,
                        }
                        if page_images:
                            metadata["image_paths"] = "|".join(page_images)
                        chunks.append(
                            ManualChunk(
                                chunk_id=chunk_id,
                                document_id=document_id,
                                content=para,
                                page=page_num + 1,
                                metadata=metadata,
                                is_visual=False,
                            )
                        )
                elif page_images:
                    chunk_id = ChunkId.build(
                        document_id=document_id,
                        section=None,
                        page=page_num + 1,
                        index=0,
                    )
                    img_meta: dict[str, str | int | float | bool] = {
                        "page": page_num + 1,
                        "source": file_name,
                        "image_paths": "|".join(page_images),
                    }
                    chunks.append(
                        ManualChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            content="[DIAGRAMA/IMAGEN EN PAGINA " + str(page_num + 1) + "]",
                            page=page_num + 1,
                            metadata=img_meta,
                            is_visual=True,
                        )
                    )
            doc.close()
            return chunks
        except Exception as exc:
            raise DataIncompleteError(
                f"El motor PyMuPDF colapso al leer la estructura binaria de: {file_path}", exc
            ) from exc
