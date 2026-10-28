# src/application/use_cases/ingest_manual.py
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.application.dtos.ingestion_dtos import IngestionRequestDTO, IngestionResultDTO
from src.domain.models.document import ManualChunk
from src.domain.ports.document_extractor import IDocumentExtractor
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MICRO_BATCH_SIZE = 250


class IngestManualUseCase:
    """Use case responsible for processing and indexing technical manuals."""

    def __init__(
        self,
        extractor: IDocumentExtractor,
        repository: IVectorRepository,
        embedding: IEmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self._extractor: IDocumentExtractor = extractor
        self._repository: IVectorRepository = repository
        self._embedding: IEmbeddingService = embedding
        self._settings = settings or Settings()
        self._max_workers = int(getattr(self._settings, "ingest_max_workers", 4))

    def _process_file(self, path_str: str) -> tuple[str, list[ManualChunk], int, int]:
        images_count = 0
        try:
            chunks = self._extractor.extract_chunks(path_str)
            for c in chunks:
                try:
                    img_paths = c.metadata.get("image_paths")
                    if img_paths and isinstance(img_paths, str):
                        images_count += len([p for p in img_paths.split("|") if p])
                except Exception:
                    pass
            return path_str, chunks, len(chunks), images_count
        except Exception as e:
            logger.error(f"Error critico procesando {path_str}: {e!s}")
            raise

    def execute(
        self,
        request: IngestionRequestDTO,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> IngestionResultDTO:
        failed_files: list[str] = []
        total_files = len(request.file_paths)
        indexed_chunks = 0
        total_images = 0
        controlled_exceptions = 0
        start_time = time.perf_counter()
        lock = Lock()

        all_chunks: list[ManualChunk] = []
        file_success: dict[str, bool] = {}
        points: list[object] = []

        if total_files == 0:
            return IngestionResultDTO(
                documents_processed=0,
                chunks_indexed=0,
                skipped_duplicates=0,
                failed_files=tuple(failed_files),
                duration_seconds=time.perf_counter() - start_time,
            )

        valid_paths: list[str] = []
        for p in request.file_paths:
            try:
                pstr = str(p)
                if not pstr:
                    controlled_exceptions += 1
                    failed_files.append(pstr)
                    continue
                if not os.path.exists(pstr):
                    logger.warning("Archivo huérfano o inexistente, se omite: %s", pstr)
                    controlled_exceptions += 1
                    failed_files.append(pstr)
                    continue
                if not os.path.isfile(pstr):
                    logger.warning("Ruta no es un archivo, se omite: %s", pstr)
                    controlled_exceptions += 1
                    failed_files.append(pstr)
                    continue
                valid_paths.append(pstr)
            except Exception as exc:
                logger.warning("Error al validar ruta de archivo: %s", exc)
                controlled_exceptions += 1
                continue

        total_files = len(valid_paths)
        if total_files == 0:
            return IngestionResultDTO(
                documents_processed=0,
                chunks_indexed=0,
                skipped_duplicates=0,
                failed_files=tuple(sorted(set(failed_files))),
                duration_seconds=time.perf_counter() - start_time,
            )

        max_workers = min(self._max_workers, total_files)
        logger.info(
            "INGEST_CONFIG | max_workers=%d (source=settings.ingest_max_workers)", max_workers
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_file, p): p for p in valid_paths}
            for fut in as_completed(futures):
                path_str = futures[fut]
                try:
                    path_res, chunks, c_count, i_count = fut.result()
                    with lock:
                        file_success[path_res] = True
                        if chunks:
                            all_chunks.extend(chunks)
                        total_images += i_count
                    logger.info(f"Extraccion exitosa: {c_count} chunks de {path_res}")
                except Exception:
                    with lock:
                        file_success[path_str] = False
                        failed_files.append(path_str)
                        controlled_exceptions += 1
                    continue

        if all_chunks:
            total_chunks = len(all_chunks)
            for start in range(0, total_chunks, MICRO_BATCH_SIZE):
                end = min(start + MICRO_BATCH_SIZE, total_chunks)
                batch = all_chunks[start:end]
                try:
                    vectors = self._embedding.embed_batch([c.content for c in batch])
                    self._repository.upsert(batch, vectors)
                    indexed_chunks += len(batch)
                    points.extend(batch)
                    if progress_callback:
                        overall = (end / total_chunks) * 0.99
                        msg = f"Indexando lote {end}/{total_chunks}"
                        progress_callback(min(overall, 0.99), msg)
                except Exception as e:
                    logger.error(f"Error en batch upsert: {e!s}")
                    controlled_exceptions += 1
                    continue

        if progress_callback:
            progress_callback(1.0, "Persistencia completada.")

        documents_processed = sum(1 for v in file_success.values() if v)
        duration_seconds = time.perf_counter() - start_time
        logger.info(
            "INGEST_METRICS | total_s=%.3f | files_ok=%d | files_fail=%d "
            "| chunks=%d | images=%d | exceptions=%d",
            duration_seconds,
            documents_processed,
            len(failed_files),
            indexed_chunks,
            total_images,
            controlled_exceptions,
        )

        return IngestionResultDTO(
            documents_processed=documents_processed,
            chunks_indexed=indexed_chunks,
            skipped_duplicates=0,
            failed_files=tuple(sorted(set(failed_files))),
            duration_seconds=duration_seconds,
        )
