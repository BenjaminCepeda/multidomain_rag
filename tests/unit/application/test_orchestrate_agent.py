import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from src.application.dtos.agent_dtos import InboundPayloadDTO
from src.application.use_cases.orchestrate_agent import OrchestrateAgentUseCase
from src.domain.exceptions.core_errors import LLMInferenceError, UnsupportedVisionModelError
from src.domain.models.agent import OrchestratorDecision
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.models.ids import ChunkId
from src.domain.models.telemetry import LLMCallMetrics
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings


class StubLLM(ILLMEngine):
    def __init__(self) -> None:
        self.last_context: list[str] = []
        self.last_attach_images: bool | None = None
        self.analyzed_images: list[str] = []
        self.last_vision_question: str | None = None

    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        return OrchestratorDecision(razonamiento="requiere manual", herramienta="buscar_manual")

    def generate_response(
        self, context: Sequence[str], user_input: str, *, attach_images: bool = True
    ) -> str:
        self.last_context = list(context)
        self.last_attach_images = attach_images
        return "respuesta sintetizada"

    def analyze_image(self, image_path: str, question: str | None = None) -> str:
        self.analyzed_images.append(image_path)
        self.last_vision_question = question
        return "analisis"

    def health_check(self) -> bool:
        return True


class StubEmbedding(IEmbeddingService):
    def embed_text(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(values=(0.1, 0.2))

    def embed_batch(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[EmbeddingVector]:
        return [EmbeddingVector(values=(0.1, 0.2)) for _ in texts]

    def dimension(self) -> int:
        return 2

    def model_name(self) -> str:
        return "stub"


class KeywordEmbedding(IEmbeddingService):
    """Two-axis embedding that separates in-domain from unrelated content."""

    def __init__(self, keyword: str) -> None:
        self._keyword = keyword.casefold()

    def _embed(self, text: str) -> EmbeddingVector:
        if self._keyword in text.casefold():
            return EmbeddingVector(values=(1.0, 0.0))
        return EmbeddingVector(values=(0.0, 1.0))

    def embed_text(self, text: str) -> EmbeddingVector:
        return self._embed(text)

    def embed_batch(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[EmbeddingVector]:
        return [self._embed(text) for text in texts]

    def dimension(self) -> int:
        return 2

    def model_name(self) -> str:
        return "keyword-stub"


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

    def exists_by_file_hash(self, file_hash: str) -> bool:
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


def _text_chunk(image_paths: str, page: int = 2) -> ManualChunk:
    return ManualChunk(
        chunk_id=ChunkId("text-1"),
        content="The basic training and inference process of a DL model",
        page=page,
        metadata={
            "document_title": "TinyML en el borde",
            "document_name": "tinyml.pdf",
            "page_number": page,
            "image_paths": image_paths,
        },
    )


def _use_case(
    settings: Settings,
    chunks: list[ManualChunk],
    llm: StubLLM | None = None,
    embedding: IEmbeddingService | None = None,
) -> OrchestrateAgentUseCase:
    return OrchestrateAgentUseCase(
        llm_engine=llm or StubLLM(),
        vector_repo=StubRepository(chunks),
        embedding=embedding or StubEmbedding(),
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


def test_text_fragments_render_the_figures_on_their_page(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    first = settings.extracted_images_dir / "page2_fig1.png"
    second = settings.extracted_images_dir / "page2_fig2.png"
    first.write_bytes(b"png")
    second.write_bytes(b"png")
    text_chunk = _text_chunk(f"{first}|{second}")
    use_case = _use_case(settings, [text_chunk])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el proceso"))

    assert len(result.visual_evidence) == 2
    assert all(evidence.page == 2 for evidence in result.visual_evidence)
    assert "TinyML en el borde" in result.visual_evidence[0].caption


def test_text_and_visual_chunks_referencing_same_asset_are_deduplicated(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "shared.png"
    image.write_bytes(b"png")
    use_case = _use_case(settings, [_text_chunk(str(image)), _visual_chunk(str(image))])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el proceso"))

    assert len(result.visual_evidence) == 1


def test_prompt_context_is_clipped_per_chunk_and_keeps_source_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(
        _env_file=None,
        data_files_path=str(tmp_path / "data"),
        extracted_images_path="extracted_images",
        llm_context_max_chars_per_chunk=200,
    )
    long_content = "PRIMERA FRASE. " + ("relleno " * 400)
    chunk = ManualChunk(
        chunk_id=ChunkId("long-1"),
        content=long_content,
        page=4,
        metadata={
            "source": "manual.pdf",
            "page_number": 4,
            "document_title": "Manual de inferencia",
        },
    )
    llm = StubLLM()
    use_case = _use_case(settings, [chunk], llm=llm)

    caplog.set_level(logging.INFO)
    use_case.execute(InboundPayloadDTO(user_input="explica el proceso"))

    assert len(llm.last_context) == 1
    prompt = llm.last_context[0]
    assert "FUENTE: manual.pdf" in prompt
    assert "PAGINA: 4" in prompt
    assert "DOCUMENTO: Manual de inferencia" in prompt

    _header, body = prompt.split("\n\n", 1)
    assert len(body) <= settings.llm_context_max_chars_per_chunk
    assert body.startswith("PRIMERA FRASE.")
    assert body.endswith("…")
    assert long_content not in prompt

    metrics = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("PROMPT_CONTEXT")
    )
    assert "chunks=1" in metrics
    assert "max_chars_per_chunk=200" in metrics
    assert "truncated=1" in metrics
    assert "relleno relleno relleno" not in caplog.text


class FailingRouterLLM(StubLLM):
    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        raise LLMInferenceError("router down with sensitive detail")


class TimeoutRouterLLM(FailingRouterLLM):
    def __init__(self) -> None:
        super().__init__()
        self._metrics_consumed = False

    def consume_call_metrics(self) -> LLMCallMetrics | None:
        if self._metrics_consumed:
            return None
        self._metrics_consumed = True
        return LLMCallMetrics(operation="route", status="timeout", total_seconds=9.0)


class FailingGenerateLLM(StubLLM):
    def generate_response(
        self, context: Sequence[str], user_input: str, *, attach_images: bool = True
    ) -> str:
        raise LLMInferenceError("generation boom with sensitive detail")


class UnsupportedVisionLLM(StubLLM):
    def analyze_image(self, image_path: str, question: str | None = None) -> str:
        self.analyzed_images.append(image_path)
        self.last_vision_question = question
        raise UnsupportedVisionModelError("vision model rejected the image")


class TextOnlyMetricsLLM(StubLLM):
    """Mimics the adapter modality metrics without touching the filesystem."""

    def __init__(self) -> None:
        super().__init__()
        self._last_metrics: LLMCallMetrics | None = None

    def generate_response(
        self, context: Sequence[str], user_input: str, *, attach_images: bool = True
    ) -> str:
        self.last_context = list(context)
        self.last_attach_images = attach_images
        candidates = sum(1 for item in context if "IMAGE_PATHS:" in item)
        self._last_metrics = LLMCallMetrics(
            operation="generate",
            images_attached=0,
            images_skipped=0 if attach_images else candidates,
            image_prep_seconds=0.0,
        )
        return "respuesta sintetizada"

    def consume_call_metrics(self) -> LLMCallMetrics | None:
        metrics = self._last_metrics
        self._last_metrics = None
        return metrics


def test_manual_search_requests_text_only_generation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "diagram.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    llm = TextOnlyMetricsLLM()
    use_case = _use_case(settings, [_text_chunk(str(image))], llm=llm)

    caplog.set_level(logging.INFO)
    result = use_case.execute(InboundPayloadDTO(user_input="explica el proceso"))

    assert result.used_tool == "buscar_manual"
    assert llm.last_attach_images is False
    trace = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("TURN_TRACE")
    )
    assert "images=0" in trace
    assert "images_skipped=1" in trace
    assert "image_bytes=0" in trace
    assert "image_prep_ms=0.0" in trace


def test_router_failure_fails_open_to_deterministic_search(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    llm = FailingRouterLLM()
    use_case = _use_case(settings, [_text_chunk("")], llm=llm)

    caplog.set_level(logging.INFO)
    result = use_case.execute(InboundPayloadDTO(user_input="explica el proceso"))

    assert result.used_tool == "buscar_manual"
    assert result.is_degraded is True
    assert result.error_log == "LLMInferenceError"
    assert result.response_text
    assert result.timings is not None
    assert result.timings.total_seconds >= result.timings.routing_seconds
    assert llm.last_context
    assert "ROUTER_FALLBACK" in caplog.text
    assert "TURN_TRACE" in caplog.text
    assert "sensitive detail" not in caplog.text


def test_router_fail_open_reports_transport_root_cause(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    use_case = _use_case(settings, [_text_chunk("")], llm=TimeoutRouterLLM())

    caplog.set_level(logging.INFO)
    result = use_case.execute(InboundPayloadDTO(user_input="explica el proceso"))

    assert result.used_tool == "buscar_manual"
    assert result.is_degraded is True
    assert result.error_log == "router_timeout"
    assert result.response_text
    assert "reason=router_timeout" in caplog.text
    assert "error_code=router_timeout" in caplog.text
    assert "sensitive detail" not in caplog.text


def test_generation_failure_returns_partial_timings_and_sanitized_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    use_case = _use_case(settings, [_text_chunk("")], llm=FailingGenerateLLM())

    caplog.set_level(logging.INFO)
    result = use_case.execute(InboundPayloadDTO(user_input="explica el proceso"))

    assert result.is_degraded is True
    assert result.timings is not None
    assert result.timings.total_seconds > 0
    assert result.error_log == "LLMInferenceError"
    assert "sensitive detail" not in result.response_text
    assert "sensitive detail" not in caplog.text
    assert "ORCHESTRATION_ERROR" in caplog.text


def test_turn_trace_never_logs_query_or_document_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    secret_query = "CONSULTA_SECRETA_DEL_OPERARIO"
    secret_document = "CONTENIDO_DOCUMENTAL_CONFIDENCIAL"
    chunk = ManualChunk(
        chunk_id=ChunkId("secret-1"),
        content=secret_document,
        page=1,
        metadata={"source": "manual.pdf", "page_number": 1},
    )
    use_case = _use_case(settings, [chunk])

    caplog.set_level(logging.DEBUG)
    result = use_case.execute(InboundPayloadDTO(user_input=secret_query))

    assert result.response_text
    assert secret_query not in caplog.text
    assert secret_document not in caplog.text
    assert "TURN_TRACE" in caplog.text


def test_visual_evidence_exposes_thumbnail_and_original_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    original = settings.extracted_images_dir / "diagram.png"
    thumbnail = settings.extracted_images_dir / "diagram_t.jpg"
    original.write_bytes(b"png")
    thumbnail.write_bytes(b"jpg")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-thumb"),
        content="[IMAGEN] Diagrama",
        page=3,
        is_visual=True,
        metadata={
            "image_path": str(original),
            "thumb_path": str(thumbnail),
            "image_sha256": "abc123",
            "image_title": "Diagrama",
            "document_title": "Manual",
            "page_number": 3,
        },
    )
    use_case = _use_case(settings, [chunk])

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    assert len(result.visual_evidence) == 1
    evidence = result.visual_evidence[0]
    assert evidence.display_path == str(thumbnail.resolve())
    assert evidence.original_path == str(original.resolve())


@pytest.mark.parametrize(
    "query",
    [
        "muéstrame el gráfico relacionado con HLS",
        "necesito un diagrama del pipeline",
        "show me the chart for inference latency",
        "revisa la figura 20",
        "describe el esquema de memoria",
        "cual es la imagen del procesador",
    ],
)
def test_visual_intent_detection_accepts_explicit_visual_requests(query: str) -> None:
    assert OrchestrateAgentUseCase._has_visual_intent(query) is True


@pytest.mark.parametrize(
    "query", ["explica el proceso de inferencia", "que es un acelerador", "", "  "]
)
def test_visual_intent_detection_ignores_text_only_queries(query: str) -> None:
    assert OrchestrateAgentUseCase._has_visual_intent(query) is False


def test_visual_intent_forces_image_analysis_over_router_choice(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "hls.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-hls"),
        content="[IMAGEN] Fig. 20. Cadence Stratus HLS for RTL synthesis",
        page=29,
        is_visual=True,
        metadata={
            "image_path": str(image),
            "image_title": "Fig. 20.",
            "document_title": "Edge inference review",
            "page_number": 29,
        },
    )
    llm = StubLLM()
    use_case = _use_case(settings, [chunk], llm=llm, embedding=KeywordEmbedding(keyword="hls"))
    query = "muestrame el grafico de Cadence Stratus HLS"

    result = use_case.execute(InboundPayloadDTO(user_input=query))

    assert result.used_tool == "analizar_imagen"
    assert llm.last_attach_images is True
    assert llm.analyzed_images == [str(image.resolve())]
    assert llm.last_vision_question == query
    assert [Path(item.image_path).name for item in result.visual_evidence] == ["hls.png"]


def test_visual_analysis_uses_the_optimized_vision_derivative(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    original = settings.extracted_images_dir / "diagram.png"
    derivative = settings.extracted_images_dir / "diagram_v1.jpg"
    thumbnail = settings.extracted_images_dir / "diagram_t.jpg"
    original.write_bytes(b"\x89PNG\r\n\x1a\n" + b"original")
    derivative.write_bytes(b"\xff\xd8\xff" + b"derivative")
    thumbnail.write_bytes(b"\xff\xd8\xff" + b"thumb")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-derivative"),
        content="[IMAGEN] Diagrama de arquitectura",
        page=7,
        is_visual=True,
        metadata={
            "image_path": str(original),
            "vision_path": str(derivative),
            "thumb_path": str(thumbnail),
            "image_title": "Diagrama",
            "document_title": "Manual",
            "document_name": "manual.pdf",
            "page_number": 7,
        },
    )
    llm = StubLLM()
    use_case = _use_case(settings, [chunk], llm=llm)

    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    evidence = result.visual_evidence[0]
    assert evidence.image_path == str(original.resolve())
    assert evidence.original_path == str(original.resolve())
    assert evidence.vision_path == str(derivative.resolve())
    assert evidence.display_path == str(thumbnail.resolve())
    assert evidence.analysis_path == str(derivative.resolve())
    assert llm.analyzed_images == [str(derivative.resolve())]


def test_visual_intent_degrades_to_text_when_model_rejects_images(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    original = settings.extracted_images_dir / "diagram.png"
    derivative = settings.extracted_images_dir / "diagram_v1.jpg"
    original.write_bytes(b"\x89PNG\r\n\x1a\n")
    derivative.write_bytes(b"\xff\xd8\xff")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-unsupported"),
        content="[IMAGEN] Diagrama de arquitectura de un detector de objetos",
        page=7,
        is_visual=True,
        metadata={
            "image_path": str(original),
            "vision_path": str(derivative),
            "image_title": "Diagrama",
            "document_title": "Manual",
            "document_name": "manual.pdf",
            "page_number": 7,
        },
    )
    llm = UnsupportedVisionLLM()
    use_case = _use_case(settings, [chunk], llm=llm)

    caplog.set_level(logging.INFO)
    result = use_case.execute(InboundPayloadDTO(user_input="muestrame el diagrama"))

    assert result.used_tool == "analizar_imagen"
    assert result.is_degraded is True
    assert result.error_log == "vision_unsupported"
    assert result.response_text == "respuesta sintetizada"
    assert result.visual_evidence
    assert result.visual_evidence[0].image_path == str(original.resolve())
    assert llm.analyzed_images == [str(derivative.resolve())]
    assert llm.last_context
    assert "VISION_FALLBACK" in caplog.text
    assert "degraded_phase=vision" in caplog.text
    assert "error_code=vision_unsupported" in caplog.text


def test_uploaded_image_vision_rejection_falls_back_to_manual_search(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploaded = tmp_path / "upload.png"
    uploaded.write_bytes(b"\x89PNG\r\n\x1a\n")
    chunk = ManualChunk(
        chunk_id=ChunkId("text-fallback"),
        content="La inferencia se ejecuta en microcontroladores de baja potencia.",
        page=7,
        metadata={
            "source": "manual_principal.pdf",
            "page_number": 7,
            "document_title": "TinyML edge deployment",
        },
    )
    llm = UnsupportedVisionLLM()
    use_case = _use_case(settings, [chunk], llm=llm)

    result = use_case.execute(
        InboundPayloadDTO(user_input="muestrame el proceso de inferencia", image_path=str(uploaded))
    )

    assert result.used_tool == "buscar_manual"
    assert result.is_degraded is True
    assert result.error_log == "vision_unsupported"
    assert result.response_text == "respuesta sintetizada"
    assert result.visual_evidence == ()
    assert llm.last_attach_images is False
    assert llm.last_context
    assert "FUENTE: manual_principal.pdf" in llm.last_context[0]


def test_visual_evidence_filters_unrelated_graphics_from_same_page(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    relevant = settings.extracted_images_dir / "hls.png"
    irrelevant = settings.extracted_images_dir / "tensorflow.png"
    relevant.write_bytes(b"\x89PNG\r\n\x1a\n")
    irrelevant.write_bytes(b"\x89PNG\r\n\x1a\n")

    relevant_chunk = ManualChunk(
        chunk_id=ChunkId("visual-hls"),
        content="[IMAGEN] Fig. 20. Cadence Stratus HLS RTL synthesis",
        page=29,
        is_visual=True,
        metadata={
            "image_path": str(relevant),
            "image_title": "Fig. 20.",
            "document_title": "Edge inference review",
            "page_number": 29,
        },
    )
    irrelevant_chunk = ManualChunk(
        chunk_id=ChunkId("visual-tensorflow"),
        content="[IMAGEN] TensorFlow training graph for mobile deployment",
        page=29,
        is_visual=True,
        metadata={
            "image_path": str(irrelevant),
            "image_title": "Fig. 21.",
            "document_title": "Edge inference review",
            "page_number": 29,
        },
    )
    page_chunk = ManualChunk(
        chunk_id=ChunkId("text-page-29"),
        content="Cadence Stratus HLS for RTL synthesis and Snapdragon inference",
        page=29,
        metadata={
            "document_title": "Edge inference review",
            "page_number": 29,
            "image_paths": f"{relevant}|{irrelevant}",
        },
    )
    use_case = _use_case(
        settings,
        [page_chunk, relevant_chunk, irrelevant_chunk],
        embedding=KeywordEmbedding(keyword="hls"),
    )

    result = use_case.execute(
        InboundPayloadDTO(user_input="muestrame el grafico de Cadence Stratus HLS para Snapdragon")
    )

    assert [Path(item.image_path).name for item in result.visual_evidence] == ["hls.png"]


def test_visual_intent_falls_back_to_text_when_no_visual_is_relevant(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    unrelated = settings.extracted_images_dir / "tensorflow.png"
    unrelated.write_bytes(b"\x89PNG\r\n\x1a\n")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-tensorflow"),
        content="[IMAGEN] TensorFlow training graph for mobile deployment",
        page=12,
        is_visual=True,
        metadata={
            "image_path": str(unrelated),
            "image_title": "Fig. 3.",
            "document_title": "Edge inference review",
            "page_number": 12,
        },
    )
    llm = StubLLM()
    use_case = _use_case(settings, [chunk], llm=llm, embedding=KeywordEmbedding(keyword="hls"))

    result = use_case.execute(
        InboundPayloadDTO(user_input="muestrame el grafico de Cadence Stratus HLS")
    )

    assert result.used_tool == "buscar_manual"
    assert result.visual_evidence == ()
    assert llm.analyzed_images == []
    assert llm.last_attach_images is False


class RecipeAwareEmbedding(IEmbeddingService):
    """Separates the compact visual recipe from the rich fallback content."""

    def _embed(self, text: str) -> EmbeddingVector:
        lowered = text.casefold()
        if "pipeline" in lowered and "diagram" in lowered:
            return EmbeddingVector(values=(1.0, 0.0))
        if "[imagen]" in lowered and "tema: pipeline" in lowered:
            return EmbeddingVector(values=(1.0, 0.0))
        return EmbeddingVector(values=(0.0, 1.0))

    def embed_text(self, text: str) -> EmbeddingVector:
        return self._embed(text)

    def embed_batch(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[EmbeddingVector]:
        return [self._embed(text) for text in texts]

    def dimension(self) -> int:
        return 2

    def model_name(self) -> str:
        return "recipe-aware-stub"


class PlaceholderCitationLLM(StubLLM):
    def generate_response(
        self, context: Sequence[str], user_input: str, *, attach_images: bool = True
    ) -> str:
        self.last_context = list(context)
        self.last_attach_images = attach_images
        return "La inferencia ocurre en el dispositivo [Archivo, Pág 7]."


def test_visual_evidence_uses_the_compact_recipe_for_legacy_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "p24_i0.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-pipeline"),
        content="[IMAGEN] Fig. 15. Accelerator design flow.",
        page=24,
        is_visual=True,
        metadata={
            "image_path": str(image),
            "image_title": "Fig. 15.",
            "document_title": "Efficient Acceleration Review",
            "document_name": "Efficient_Acceleration_Review.pdf",
            "page_number": 24,
            "theme_summary": "pipeline accelerator",
            "surrounding_text_context": "generic training to inference pipeline",
        },
    )
    use_case = _use_case(
        settings,
        [chunk],
        embedding=RecipeAwareEmbedding(),
    )

    result = use_case.execute(
        InboundPayloadDTO(
            user_input="show me Block diagram of the generic training to inference pipeline"
        )
    )

    assert result.used_tool == "analizar_imagen"
    assert [Path(item.image_path).name for item in result.visual_evidence] == ["p24_i0.png"]


def test_text_only_search_never_returns_visual_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.extracted_images_dir.mkdir(parents=True)
    image = settings.extracted_images_dir / "tiny_related.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    chunk = ManualChunk(
        chunk_id=ChunkId("visual-tiny"),
        content="[IMAGEN] tinyML benchmark",
        page=6,
        is_visual=True,
        metadata={
            "image_path": str(image),
            "image_title": "Fig. 1.",
            "document_title": "TinyML review",
            "page_number": 6,
        },
    )
    use_case = _use_case(settings, [chunk], embedding=KeywordEmbedding(keyword="tinyml"))

    result = use_case.execute(InboundPayloadDTO(user_input="¿Qué es tinyML?"))

    assert result.used_tool == "buscar_manual"
    assert result.visual_evidence == ()


def test_citation_placeholder_is_rendered_with_the_real_source(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    chunk = ManualChunk(
        chunk_id=ChunkId("text-7"),
        content="La inferencia se ejecuta en microcontroladores.",
        page=7,
        metadata={
            "source": "Deploying_TinyML_EdgeAI.pdf",
            "page_number": 7,
            "document_title": "TinyML edge deployment",
        },
    )
    use_case = _use_case(settings, [chunk], llm=PlaceholderCitationLLM())

    result = use_case.execute(InboundPayloadDTO(user_input="explica el proceso de inferencia"))

    assert result.response_text == (
        "La inferencia ocurre en el dispositivo [Deploying_TinyML_EdgeAI.pdf, Pág 7]."
    )
