# src/application/dtos/ingestion_dtos.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IngestionRequestDTO:
    file_paths: tuple[Path, ...]
    force_reindex: bool = False
    policy_override: str | None = None


@dataclass(frozen=True)
class IngestionResultDTO:
    documents_processed: int
    chunks_indexed: int
    skipped_duplicates: int
    failed_files: tuple[str, ...]
    duration_seconds: float
