from abc import ABC, abstractmethod
from typing import List
from src.domain.models.document import ManualChunk

class IDocumentExtractor(ABC):
    """Contrato agnóstico para la extracción de texto y diagramas desde archivos físicos."""
    
    @abstractmethod
    def extract_chunks(self, file_path: str) -> List[ManualChunk]:
        """
        Analiza un archivo PDF técnico y lo segmenta en fragmentos lógicos.
        
        Raises:
            DataIncompleteError: Si el archivo está corrupto o ilegible.
        """
        pass