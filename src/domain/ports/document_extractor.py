from abc import ABC, abstractmethod

from src.domain.models.document import ManualChunk


class IDocumentExtractor(ABC):
    """Contrato agnóstico para la extracción de texto y diagramas desde archivos físicos."""

    @abstractmethod
    def extract_chunks(self, file_path: str, file_hash: str | None = None) -> list[ManualChunk]:
        """
        Analiza un archivo PDF técnico y lo segmenta en fragmentos lógicos.

        Args:
            file_path: Ruta física del documento.
            file_hash: Huella SHA-256 ya calculada por el caso de uso. Si se
                omite, el adaptador debe calcularla por su cuenta.

        Raises:
            DataIncompleteError: Si el archivo está corrupto o ilegible.
        """
        pass
