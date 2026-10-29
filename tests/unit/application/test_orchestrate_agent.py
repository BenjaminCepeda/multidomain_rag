from collections.abc import Sequence
from pathlib import Path

from src.application.dtos.agent_dtos import InboundPayloadDTO
from src.application.use_cases.orchestrate_agent import OrchestrateAgentUseCase
from src.domain.models.agent import OrchestratorDecision
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.models.ids import ChunkId
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings


class StubLLM(ILLMEngine):
    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        return OrchestratorDecision(razonamiento="requiere manual", herramienta="buscar_manual")

    def generate_response(self, context: Sequence[str], user_input: str) -> str:
        return "respuesta sintetizada"

    def analyze_image(self, image_path: str) -> str:
        return "analisis"

    def health_check(self) -> bool:
        return True


class StubEmbedding(IEmbeddingService):
    def embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=(0.1, 0.2))

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[EmbeddingVector]:
        return [EmbeddingVector(values=(0.1, 0.2)) for _ in texts]

    def dimension(self) -> int:
        return 2

    def model_name(self) -> str:
        return "stub"


class StubRepository(IVectorRepository):
    def __init__(self, chunks: list[ManualChunk]) -> None:
        self._chunks = chunks

    def ensure_collection(self, vector_dim: int) -> None:
        return None

    def upsert(self, chunks: Sequence[ManualChunk], vectors: Sequence[EmbeddingVector]) -> None:
        return None

    def search(
        self, query_vector: EmbeddingVector, limit: int, min_score: float
    ) -> list[ManualChunk]:
        return list(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    def exists_by_id(self, chunk_id: str) -> bool:
        return False

    def clear(self) -> None:
        self._chunks = []

    def list_sources(self) -> list[str]:
        return []

    def delete_by_source(self, source: str) -> None:
        return None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_files_path=str(tmp_path / "data"),
        extracted_images_path="extracted_images",
    )


def _visual_chunk(image_path: str, title: str = "Diagrama de flujo") -> ManualChunk:
    return ManualChunk(
        chunk_id=ChunkId("visual-1"),
        content="[IMAGEN] Diagrama de flujo de inferencia en el borde",
        page=7,
        is_visual=True,
        metadata={
            "image_path": image_path,
            "image_title": title,
            "document_title": "TinyML en el borde",
            "page_number": 7,
        },
    )


def _use_case(settings: Settings, chunks: list[ManualChunk]) -> OrchestrateAgentUseCase:
    return OrchestrateAgentUseCase(
        llm_engine=StubLLM(),
        vector_repo=StubRepository(chunks),
        embedding=StubEmbedding(),
        settings=settings,
    )


def test_visual_evidence_is_resolved_from_configured_assets_dir(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "diagram.png"
    image.write_bytes(b"png")
    use_case = _use_case(settings, [_visual_chunk(str(image))])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    assert len(result.visual_evidence) == 1
    evidence = result.visual_evidence[0]
    assert evidence.image_path == str(image.resolve())
    assert evidence.page == 7
    assert "Diagrama de flujo" in evidence.caption
    assert "TinyML en el borde" in evidence.caption


def test_visual_evidence_falls_back_to_basename_inside_assets_dir(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "diagram.png"
    image.write_bytes(b"png")
    stale_path = str(tmp_path / "old_workspace" / "diagram.png")
    use_case = _use_case(settings, [_visual_chunk(stale_path)])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    assert len(result.visual_evidence) == 1
    assert result.visual_evidence[0].image_path == str(image.resolve())


def test_visual_evidence_ignores_paths_outside_assets_dir(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    outside = tmp_path / "secrets.png"
    outside.write_bytes(b"png")
    use_case = _use_case(settings, [_visual_chunk(str(outside))])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    assert result.visual_evidence == ()


def test_visual_evidence_deduplicates_repeated_assets(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "diagram.png"
    image.write_bytes(b"png")
    duplicated = _visual_chunk(str(image))
    use_case = _use_case(settings, [duplicated, duplicated])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    assert len(result.visual_evidence) == 1
