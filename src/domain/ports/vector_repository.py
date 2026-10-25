from abc import ABC, abstractmethod
from typing import List
from src.domain.models.document import ManualChunk

class IVectorRepository(ABC):
    """Contrato estricto para operaciones de indexación y búsqueda en base de datos vectorial."""
    
    @abstractmethod
    def save_chunks(self, chunks: List[ManualChunk]) -> None:
        """
        Indexa de forma masiva los fragmentos en el motor vectorial.
        
        Raises:
            VectorStorageError: Si falla la persistencia o la comunicación con el motor.
        """
        pass

    @abstractmethod
    def search_similar(self, query: str, limit: int = 2) -> List[ManualChunk]:
        """
        Realiza una búsqueda de similitud de coseno basada en texto o embeddings.
        
        Raises:
            VectorStorageError: Si el espacio de vectores o la colección es inaccesible.
        """
        pass