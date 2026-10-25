# src/application/use_cases/ingest_manual.py
import logging
import os
from typing import List, Callable, Optional
from src.domain.ports.document_extractor import IDocumentExtractor
from src.domain.ports.vector_repository import IVectorRepository

logger = logging.getLogger(__name__)

class IngestManualUseCase:
    """Caso de uso encargado de procesar e indexar múltiples manuales de ingeniería."""
    
    def __init__(self, extractor: IDocumentExtractor, repository: IVectorRepository) -> None:
        self._extractor: IDocumentExtractor = extractor
        self._repository: IVectorRepository = repository

    def execute(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> bool:
        """
        Procesa una lista de archivos y los indexa en el repositorio.
        Incluye un callback opcional para reportar progreso a la interfaz.
        """
        all_chunks = []
        total_files = len(file_paths)
        
        for i, path in enumerate(file_paths):
            try:
                # Reportar progreso si el callback está definido
                if progress_callback:
                    progress_callback((i / total_files), f"Extrayendo: {os.path.basename(path)}")
                
                logger.info(f"Iniciando extracción para: {path}")
                chunks = self._extractor.extract_chunks(path)
                
                if chunks:
                    all_chunks.extend(chunks)
                    logger.info(f"Extracción exitosa: {len(chunks)} chunks de {path}")
                else:
                    logger.warning(f"El archivo {path} no devolvió fragmentos.")
                    
            except Exception as e:
                logger.error(f"Error crítico procesando {path}: {str(e)}")
                # Mantenemos la resiliencia: si un archivo falla, continuamos con el siguiente
                continue
        
        if not all_chunks:
            logger.warning("No se pudieron extraer fragmentos de ningún archivo.")
            return False
            
        # Indexación en repositorio
        try:
            if progress_callback:
                progress_callback(0.9, "Persistiendo en base vectorial...")
            
            self._repository.save_chunks(all_chunks)
            logger.info(f"Indexación total: {len(all_chunks)} fragmentos procesados.")
            return True
        except Exception as e:
            logger.error(f"Error durante la persistencia vectorial: {str(e)}")
            return False