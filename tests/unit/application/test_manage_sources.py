from collections.abc import Sequence
from pathlib import Path

import pytest

from src.application.use_cases.manage_sources import ManageSourcesUseCase
from src.domain.exceptions.core_errors import VectorStorageError
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings


class FakeVectorRepository(IVectorRepository):
    """In-memory stub that records every interaction of the use case."""

    def __init__(self, sources: list[str] | None = None) -> None:
        self.sources: list[str] = list(sources or [])
        self.cleared = False
        self.deleted_sources: list[str] = []
        self.clear_error: Exception | None = None

    def ensure_collection(self, vector_dim: int) -> None:
        return None

    def upsert(self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]) -> None:
        raise AssertionError("upsert must not be called by ManageSourcesUseCase")

    def search(
        self, query_vector: EmbeddingVector, limit: int, min_score: float
    ) -> list[ManualChunk]:
        raise AssertionError("search must not be called by ManageSourcesUseCase")

    def count(self) -> int:
        return len(self.sources)

    def exists_by_id(self, chunk_id: str) -> bool:
        return False

    def exists_by_file_hash(self, file_hash: str) -> bool:
        return False

    def clear(self) -> None:
        if self.clear_error is not None:
            raise self.clear_error
        self.cleared = True
        self.sources = []

    def list_sources(self) -> list[str]:
        return list(self.sources)

    def delete_by_source(self, source: str) -> None:
        self.deleted_sources.append(source)
        self.sources = [s for s in self.sources if s != source]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_files_path=str(tmp_path / "data"),
        raw_files_path="raw",
        extracted_images_path="extracted_images",
    )


def _make_artifacts(settings: Settings, sources: Sequence[str]) -> list[Path]:
    settings.raw_files_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name in sources:
        path = settings.raw_files_dir / name
        path.write_bytes(b"%PDF-1.7 fake")
        created.append(path)
    settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
    (settings.extracted_images_dir / "image_a.png").write_bytes(b"png")
    nested = settings.extracted_images_dir / "nested"
    nested.mkdir()
    (nested / "image_b.png").write_bytes(b"png")
    return created


def test_extracted_images_dir_is_derived_from_settings(settings: Settings, tmp_path: Path) -> None:
    assert settings.extracted_images_dir == tmp_path / "data" / "extracted_images"
    assert settings.raw_files_dir == tmp_path / "data" / "raw"


def test_clear_all_purges_vectors_originals_and_images(settings: Settings) -> None:
    originals = _make_artifacts(settings, ["manual_a.pdf", "manual_b.pdf"])
    repository = FakeVectorRepository(
        sources=["manual_a.pdf", "manual_b.pdf", "already_missing.pdf"]
    )
    use_case = ManageSourcesUseCase(repository=repository, settings=settings)

    use_case.clear_all()

    assert repository.cleared is True
    for original in originals:
        assert not original.exists()
    assert list(settings.extracted_images_dir.iterdir()) == []


def test_clear_all_keeps_artifacts_when_vector_purge_fails(settings: Settings) -> None:
    originals = _make_artifacts(settings, ["manual_a.pdf"])
    repository = FakeVectorRepository(sources=["manual_a.pdf"])
    repository.clear_error = VectorStorageError("qdrant down")
    use_case = ManageSourcesUseCase(repository=repository, settings=settings)

    with pytest.raises(VectorStorageError):
        use_case.clear_all()

    assert repository.cleared is False
    assert originals[0].exists()
    assert (settings.extracted_images_dir / "image_a.png").exists()


def test_delete_source_removes_vectors_and_original_file(settings: Settings) -> None:
    originals = _make_artifacts(settings, ["manual_a.pdf", "manual_b.pdf"])
    repository = FakeVectorRepository(sources=["manual_a.pdf", "manual_b.pdf"])
    use_case = ManageSourcesUseCase(repository=repository, settings=settings)

    use_case.delete_source("manual_a.pdf")

    assert repository.deleted_sources == ["manual_a.pdf"]
    assert not originals[0].exists()
    assert originals[1].exists()
    assert (settings.extracted_images_dir / "image_a.png").exists()


def test_delete_source_ignores_paths_outside_data_root(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "secret.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"do not delete")
    repository = FakeVectorRepository(sources=[str(outside)])
    use_case = ManageSourcesUseCase(repository=repository, settings=settings)

    use_case.delete_source(str(outside))
    use_case.delete_source("../../secret.pdf")

    assert outside.exists()
    assert repository.deleted_sources == [str(outside), "../../secret.pdf"]
