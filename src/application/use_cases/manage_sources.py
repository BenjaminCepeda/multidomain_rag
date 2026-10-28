# src/application/use_cases/manage_sources.py
"""Cascade purge of indexed sources and their physical artifacts.

Vector state is always updated first; physical artifacts (original documents and
extracted visual assets) are then removed on a best-effort basis. Every file
operation is isolated so that a single locked or missing entry cannot abort the
whole cleanup, which keeps the cascade resilient on Windows and POSIX alike.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class ManageSourcesUseCase:
    """Use case for listing and deleting ingested sources with artifact cleanup."""

    def __init__(self, repository: IVectorRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def list_sources(self) -> list[str]:
        return self._repository.list_sources()

    def delete_source(self, source: str) -> None:
        self._repository.delete_by_source(source)
        logger.info("SOURCE_DELETE | source=%s | action=vectors_deleted", source)

        resolved = self._resolve_source_path(source)
        if resolved is None:
            logger.warning(
                "SOURCE_DELETE | source=%s | action=skipped | reason=path_out_of_scope", source
            )
            return
        self._unlink_file(resolved, event="SOURCE_DELETE")

    def clear_all(self) -> None:
        sources = self._repository.list_sources()
        logger.info("PHYSICAL_CLEAR_ALL | action=started | sources=%d", len(sources))

        self._repository.clear()
        logger.info("PHYSICAL_CLEAR_ALL | action=vectors_cleared | sources=%d", len(sources))

        for source in sources:
            resolved = self._resolve_source_path(source)
            if resolved is None:
                logger.warning(
                    "PHYSICAL_CLEAR_ALL | source=%s | action=skipped | reason=path_out_of_scope",
                    source,
                )
                continue
            self._unlink_file(resolved, event="PHYSICAL_CLEAR_ALL")

        self._purge_extracted_images()
        logger.info("PHYSICAL_CLEAR_ALL | action=completed | sources=%d", len(sources))

    def _resolve_source_path(self, source: str) -> Path | None:
        """Map a stored source reference to a path inside the configured data root.

        Relative references are reduced to their basename (path-traversal guard);
        absolute references are accepted only when they live under the data root.
        """
        candidate = Path(source)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            if not candidate.name:
                return None
            resolved = (self._settings.raw_files_dir / candidate.name).resolve()

        data_root = self._settings.data_root.resolve()
        if not resolved.is_relative_to(data_root):
            return None
        return resolved

    @staticmethod
    def _unlink_file(path: Path, event: str) -> None:
        try:
            if not path.exists():
                logger.warning("%s | resource=%s | action=skipped | reason=not_found", event, path)
                return
            if not path.is_file():
                logger.warning("%s | resource=%s | action=skipped | reason=not_a_file", event, path)
                return
            path.unlink()
            logger.info("%s | resource=%s | action=deleted", event, path)
        except OSError as exc:
            logger.warning("%s | resource=%s | action=failed | error=%s", event, path, exc)

    def _purge_extracted_images(self) -> None:
        images_dir = self._settings.extracted_images_dir
        if not images_dir.is_dir():
            logger.warning(
                "EXTRACTED_IMAGES_PURGE | dir=%s | action=skipped | reason=not_found", images_dir
            )
            return

        try:
            entries = list(images_dir.iterdir())
        except OSError as exc:
            logger.warning(
                "EXTRACTED_IMAGES_PURGE | dir=%s | action=failed | error=%s", images_dir, exc
            )
            return

        deleted = 0
        for entry in entries:
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                deleted += 1
                logger.info("EXTRACTED_IMAGES_PURGE | resource=%s | action=deleted", entry)
            except OSError as exc:
                logger.warning(
                    "EXTRACTED_IMAGES_PURGE | resource=%s | action=failed | error=%s", entry, exc
                )

        logger.info(
            "EXTRACTED_IMAGES_PURGE | dir=%s | action=completed | deleted=%d",
            images_dir,
            deleted,
        )
