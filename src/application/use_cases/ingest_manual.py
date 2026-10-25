# src/application/use_cases/ingest_manual.py
import logging
import os
from collections.abc import Callable

from src.application.dtos.ingestion_dtos import IngestionRequestDTO, IngestionResultDTO
from src.domain.models.document import ManualChunk
from src.domain.ports.document_extractor import IDocumentExtractor
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.vector_repository import IVectorRepository

logger = logging.getLogger(__name__)


class IngestManualUseCase:
    """Use case responsible for processing and indexing technical manuals."""

    def __init__(
        self,
        extractor: IDocumentExtractor,
        repository: IVectorRepository,
        embedding: IEmbeddingService,
    ) -> None:
        self._extractor: IDocumentExtractor = extractor
        self._repository: IVectorRepository = repository
        self._embedding: IEmbeddingService = embedding

    def execute(
        self,
        request: IngestionRequestDTO,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> IngestionResultDTO:
        all_chunks: list[ManualChunk] = []
        failed_files: list[str] = []
        total_files = len(request.file_paths)

        for i, path in enumerate(request.file_paths):
            path_str = str(path)
            try:
                if progress_callback:
                    progress_callback(
                        (i / total_files) if total_files else 0.0,
                        f"Extrayendo: {os.path.basename(path_str)}",
                    )

                logger.info(f"Iniciando extraccion para: {path_str}")
                chunks = self._extractor.extract_chunks(path_str)

                if chunks:
                    all_chunks.extend(chunks)
                    logger.info(f"Extraccion exitosa: {len(chunks)} chunks de {path_str}")
                else:
                    logger.warning(f"El archivo {path_str} no devolvio fragmentos.")

            except Exception as e:
                logger.error(f"Error critico procesando {path_str}: {e!s}")
                failed_files.append(path_str)
                continue

        if not all_chunks:
            logger.warning("No se pudieron extraer fragmentos de ningun archivo.")
            return IngestionResultDTO(
                documents_processed=0,
                chunks_indexed=0,
                skipped_duplicates=0,
                failed_files=tuple(failed_files),
                duration_seconds=0.0,
            )

        try:
            if progress_callback:
                progress_callback(0.9, "Persistiendo en base vectorial...")

            vectors = self._embedding.embed_batch([c.content for c in all_chunks])
            self._repository.upsert(all_chunks, vectors)
            logger.info(f"Indexacion total: {len(all_chunks)} fragmentos procesados.")
        except Exception as e:
            logger.error(f"Error durante la persistencia vectorial: {e!s}")
            failed_files.append("vector_store")
            return IngestionResultDTO(
                documents_processed=0,
                chunks_indexed=0,
                skipped_duplicates=0,
                failed_files=tuple(failed_files),
                duration_seconds=0.0,
            )

        return IngestionResultDTO(
            documents_processed=total_files - len(failed_files),
            chunks_indexed=len(all_chunks),
            skipped_duplicates=0,
            failed_files=tuple(failed_files),
            duration_seconds=0.0,
        )
