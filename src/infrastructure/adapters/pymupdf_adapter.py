# src/infrastructure/adapters/pymupdf_adapter.py
import os
import fitz  # PyMuPDF
from typing import List
from src.domain.ports.document_extractor import IDocumentExtractor
from src.domain.models.document import ManualChunk
from src.domain.exceptions.core_errors import DataIncompleteError

class PyMuPDFAdapter(IDocumentExtractor):
    def extract_chunks(self, file_path: str) -> List[ManualChunk]:
        if not os.path.exists(file_path):
            raise DataIncompleteError(f"El extractor físico no encontró el archivo en: {file_path}")

        chunks: List[ManualChunk] = []
        try:
            doc = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                if len(text) < 50:
                    continue
                
                # Segmentación por párrafos para mejorar la precisión semántica
                paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
                
                for idx, para in enumerate(paragraphs):
                    chunk_id = f"{file_name}_p{page_num}_s{idx}"
                    chunks.append(
                        ManualChunk(
                            chunk_id=chunk_id,
                            content=para,
                            metadata={
                                "page": page_num + 1,
                                "source": file_name
                            },
                            is_visual=False
                        )
                    )
            doc.close()
            return chunks
        except Exception as exc:
            raise DataIncompleteError(f"El motor PyMuPDF colapsó al leer la estructura binaria de: {file_path}", exc)