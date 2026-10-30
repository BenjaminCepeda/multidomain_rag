import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from src.application.dtos.ingestion_dtos import IngestionRequestDTO
from src.application.use_cases.ingest_manual import IngestManualUseCase
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.models.ids import ChunkId, DocumentId
from src.domain.ports.document_extractor import IDocumentExtractor
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings


class StubExtractor(IDocumentExtractor):
    def __init__(self, chunks: list[ManualChunk]) -> None:
        self._chunks = chunks

    def extract_chunks(self, file_path: str) -> list[ManualChunk]:
        return list(self._chunks)


class RecordingEmbedding(IEmbeddingService):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=(0.1, 0.2))

    def embed_batch(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[EmbeddingVector]:
        self.seen.extend(texts)
        return [EmbeddingVector(values=(0.1, 0.2)) for _ in texts]

    def dimension(self) -> int:
        return 2

    def model_name(self) -> str:
        return "recording"


class RecordingRepository(IVectorRepository):
    def __init__(self) -> None:
        self.upserted: list[ManualChunk] = []

    def ensure_collection(self, vector_dim: int) -> None:
        return None

    def upsert(self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]) -> None:
        self.upserted.extend(chunks)

    def search(
        self, query_vector: EmbeddingVector, limit: int, min_score: float
    ) -> list[ManualChunk]:
        return []

    def count(self) -> int:
        return len(self.upserted)

    def exists_by_id(self, chunk_id: str) -> bool:
        return False

    def clear(self) -> None:
        self.upserted.clear()

    def list_sources(self) -> list[str]:
        return []

    def delete_by_source(self, source: str) -> None:
        return None


def _chunk(
    index: int, content: str, embedding_text: str | None = None, *, is_visual: bool = False
) -> ManualChunk:
    return ManualChunk(
        chunk_id=ChunkId.build(
            document_id=DocumentId.from_source("manual.pdf"),
            section=None,
            page=1,
            index=index,
        ),
        content=content,
        page=1,
        embedding_text=embedding_text,
        is_visual=is_visual,
    )


def test_ingestion_embeds_compact_text_and_logs_phase_metrics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"fake-pdf")

    secret_content = "CONTENIDO_DOCUMENTAL_SECRETO_QUE_NO_DEBE_APARECER_EN_LOGS"
    visual = _chunk(
        0,
        secret_content,
        embedding_text="[IMAGEN] Figura compacta del flujo",
        is_visual=True,
    )
    text = _chunk(1, "Linea tecnica del manual.")
    embedding = RecordingEmbedding()
    repository = RecordingRepository()
    settings = Settings(
        _env_file=None,
        data_files_path=str(tmp_path / "data"),
        ingest_max_workers=1,
        embedding_batch_size=16,
    )
    use_case = IngestManualUseCase(
        extractor=StubExtractor([visual, text]),
        repository=repository,
        embedding=embedding,
        settings=settings,
    )

    caplog.set_level(logging.INFO)
    result = use_case.execute(IngestionRequestDTO(file_paths=(source,)))

    assert result.chunks_indexed == 2
    assert embedding.seen == ["[IMAGEN] Figura compacta del flujo", "Linea tecnica del manual."]
    assert repository.upserted == [visual, text]
    expected_chars = sum(len(chunk.embedding_input) for chunk in (visual, text))

    metrics = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("INGEST_METRICS")
    )
    assert "extract_s=" in metrics
    assert "embed_s=" in metrics
    assert "upsert_s=" in metrics
    assert "chunks=2" in metrics
    assert "visuals=1" in metrics
    assert f"embedded_chars={expected_chars}" in metrics
    assert "micro_batch_size=250" in metrics
    assert "embed_batch_size=16" in metrics
    assert secret_content not in caplog.text
