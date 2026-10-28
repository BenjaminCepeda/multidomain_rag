# src/infrastructure/composition/root.py
from dataclasses import dataclass

from src.application.use_cases.ingest_manual import IngestManualUseCase
from src.application.use_cases.manage_sources import ManageSourcesUseCase
from src.application.use_cases.orchestrate_agent import OrchestrateAgentUseCase
from src.domain.ports.embedding_service import IEmbeddingService
from src.infrastructure.adapters.embedding_adapter import SentenceTransformerEmbeddingAdapter
from src.infrastructure.adapters.local_llm_adapter import LocalLLMAdapter
from src.infrastructure.adapters.pymupdf_adapter import PyMuPDFAdapter
from src.infrastructure.adapters.qdrant_adapter import QdrantAdapter
from src.infrastructure.adapters.stt_adapter import STTAdapter
from src.infrastructure.config.settings import Settings


@dataclass
class ApplicationContext:
    settings: Settings
    llm: LocalLLMAdapter
    vector_repo: QdrantAdapter
    embedding: IEmbeddingService
    extractor: PyMuPDFAdapter
    stt: STTAdapter
    ingest_use_case: IngestManualUseCase
    orchestrate_use_case: OrchestrateAgentUseCase
    manage_sources_use_case: ManageSourcesUseCase


class CompositionRoot:
    """Single construction site for the application object graph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def build(self) -> ApplicationContext:
        embedding: IEmbeddingService = SentenceTransformerEmbeddingAdapter(settings=self.settings)
        vector_repo = QdrantAdapter(settings=self.settings, embedding=embedding)
        llm = LocalLLMAdapter(settings=self.settings)
        extractor = PyMuPDFAdapter(settings=self.settings)
        stt = STTAdapter(
            model_size=self.settings.stt_model_size,
            device=self.settings.stt_device,
        )

        ingest_uc = IngestManualUseCase(
            extractor=extractor,
            repository=vector_repo,
            embedding=embedding,
            settings=self.settings,
        )
        orchestrate_uc = OrchestrateAgentUseCase(
            llm_engine=llm,
            vector_repo=vector_repo,
            embedding=embedding,
        )
        manage_sources_uc = ManageSourcesUseCase(repository=vector_repo, settings=self.settings)

        return ApplicationContext(
            settings=self.settings,
            llm=llm,
            vector_repo=vector_repo,
            embedding=embedding,
            extractor=extractor,
            stt=stt,
            ingest_use_case=ingest_uc,
            orchestrate_use_case=orchestrate_uc,
            manage_sources_use_case=manage_sources_uc,
        )
