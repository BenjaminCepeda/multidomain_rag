# src/infrastructure/adapters/pymupdf_adapter.py
import os

import fitz  # PyMuPDF

from src.domain.exceptions.core_errors import DataIncompleteError
from src.domain.models.document import ManualChunk
from src.domain.models.ids import ChunkId, DocumentId
from src.domain.ports.document_extractor import IDocumentExtractor


class PyMuPDFAdapter(IDocumentExtractor):
    def extract_chunks(self, file_path: str) -> list[ManualChunk]:
        if not os.path.exists(file_path):
            raise DataIncompleteError(f"El extractor fisico no encontro el archivo en: {file_path}")

        chunks: list[ManualChunk] = []
        try:
            doc = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            document_id = DocumentId.from_source(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()

                if len(text) < 50:
                    continue

                paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

                for idx, para in enumerate(paragraphs):
                    chunk_id = ChunkId.build(
                        document_id=document_id,
                        section=None,
                        page=page_num + 1,
                        index=idx,
                    )
                    chunks.append(
                        ManualChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            content=para,
                            page=page_num + 1,
                            metadata={
                                "page": page_num + 1,
                                "source": file_name,
                            },
                            is_visual=False,
                        )
                    )
            doc.close()
            return chunks
        except Exception as exc:
            raise DataIncompleteError(
                f"El motor PyMuPDF colapso al leer la estructura binaria de: {file_path}", exc
            ) from exc
