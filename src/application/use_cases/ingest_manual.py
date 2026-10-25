# src/application/use_cases/ingest_manual.py
import logging
import os
from collections.abc import Callable

from src.application.dtos.ingestion_dtos import IngestionRequestDTO, IngestionResultDTO
from src.domain.ports.document_extractor import IDocumentExtractor
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.vector_repository import IVectorRepository

logger = logging.getLogger(__name__)

MICRO_BATCH_SIZE = 250


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
        failed_files: list[str] = []
        total_files = len(request.file_paths)
        indexed_chunks = 0

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
                    logger.info(f"Extraccion exitosa: {len(chunks)} chunks de {path_str}")
                    # Micro-batched embedding + upsert to avoid heavy monolithic transactions.
                    for start in range(0, len(chunks), MICRO_BATCH_SIZE):
                        end = min(start + MICRO_BATCH_SIZE, len(chunks))
                        batch = chunks[start:end]
                        vectors = self._embedding.embed_batch([c.content for c in batch])
                        self._repository.upsert(batch, vectors)
                        indexed_chunks += len(batch)
                        if progress_callback:
                            file_progress = (start + len(batch)) / len(chunks)
                            overall = (i + file_progress) / total_files if total_files else 1.0
                            progress_callback(
                                min(overall, 0.99),
                                f"Indexando: {os.path.basename(path_str)} ({end}/{len(chunks)})",
                            )
                else:
                    logger.warning(f"El archivo {path_str} no devolvio fragmentos.")

            except Exception as e:
                logger.error(f"Error critico procesando {path_str}: {e!s}")
                failed_files.append(path_str)
                continue

        if indexed_chunks == 0 and not failed_files:
            logger.warning("No se pudieron extraer fragmentos de ningun archivo.")
            return IngestionResultDTO(
                documents_processed=0,
                chunks_indexed=0,
                skipped_duplicates=0,
                failed_files=tuple(failed_files),
                duration_seconds=0.0,
            )

        if progress_callback:
            progress_callback(1.0, "Persistencia completada.")

        return IngestionResultDTO(
            documents_processed=total_files - len(failed_files),
            chunks_indexed=indexed_chunks,
            skipped_duplicates=0,
            failed_files=tuple(failed_files),
            duration_seconds=0.0,
        )
