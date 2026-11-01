# src/application/use_cases/orchestrate_agent.py
import logging
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from src.application.dtos.agent_dtos import (
    InboundPayloadDTO,
    OutboundResponseDTO,
    VisualEvidenceDTO,
)
from src.domain.exceptions.core_errors import DataIncompleteError, DomainBaseError
from src.domain.models.document import ManualChunk
from src.domain.models.telemetry import LLMCallMetrics, PhaseTimings, TurnTrace
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = frozenset({"buscar_manual", "analizar_imagen", "responder_directo"})
ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})


def _clip_context(text: str, max_chars: int) -> str:
    """Bound one retrieved fragment before it enters the LLM prompt.

    The cut prefers a natural boundary near the limit so a local model does
    not receive a mid-token fragment. Source, page and visual metadata travel
    in the prompt header and are therefore never discarded.
    """
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    window = stripped[: max_chars - 1]
    cut = max(
        window.rfind("\n"),
        window.rfind(". "),
        window.rfind("; "),
        window.rfind(" "),
    )
    if cut < max_chars // 2:
        cut = len(window)
    return window[:cut].rstrip() + "…"


class OrchestrateAgentUseCase:
    """Use case coordinating the multimodal agent tools."""

    def __init__(
        self,
        llm_engine: ILLMEngine,
        vector_repo: IVectorRepository,
        embedding: IEmbeddingService,
        settings: Settings,
    ) -> None:
        self._llm: ILLMEngine = llm_engine
        self._repo: IVectorRepository = vector_repo
        self._embedding: IEmbeddingService = embedding
        self._settings = settings

    def execute(self, payload: InboundPayloadDTO) -> OutboundResponseDTO:
        turn_id = uuid.uuid4().hex
        has_image: bool = payload.image_path is not None
        start_time = time.perf_counter()

        routing_seconds = 0.0
        embedding_seconds = 0.0
        retrieval_seconds = 0.0
        evidence_seconds = 0.0
        prompt_seconds = 0.0
        image_prep_seconds = 0.0
        vision_analysis_seconds = 0.0
        inference_seconds = 0.0
        generation_started_at: float | None = None

        context_chars = 0
        truncated_chunks = 0
        visual_evidence: tuple[VisualEvidenceDTO, ...] = ()
        used_tool = "orquestador"
        current_phase = "router"
        router_degraded = False
        error_code: str | None = None

        try:
            routing_start = time.perf_counter()
            try:
                decision = self._llm.get_orchestrator_decision(payload.user_input, has_image)
                used_tool = (
                    decision.herramienta
                    if decision.herramienta in ALLOWED_TOOLS
                    else "buscar_manual"
                )
            except DomainBaseError as exc:
                # Fail-open: a broken or slow router never blocks the turn.
                router_degraded = True
                error_code = type(exc).__name__
                used_tool = (
                    "analizar_imagen"
                    if has_image and not payload.user_input.strip()
                    else "buscar_manual"
                )
                logger.warning(
                    "ROUTER_FALLBACK | turn_id=%s | tool=%s | error_code=%s",
                    turn_id,
                    used_tool,
                    error_code,
                )
            finally:
                routing_seconds = time.perf_counter() - routing_start
                self._llm.consume_call_metrics()

            logger.info(
                "AGENT_ROUTE | turn_id=%s | tool=%s | fallback=%s",
                turn_id,
                used_tool,
                router_degraded,
            )

            contexto_sintesis: list[str] = []

            if used_tool == "buscar_manual":
                current_phase = "embedding"
                embedding_start = time.perf_counter()
                query_vector = self._embedding.embed_text(payload.user_input)
                embedding_seconds = time.perf_counter() - embedding_start

                current_phase = "retrieval"
                logger.info(
                    "MANUAL_SEARCH | turn_id=%s | query_chars=%d",
                    turn_id,
                    len(payload.user_input),
                )
                retrieval_start = time.perf_counter()
                chunks: list[ManualChunk] = self._repo.search(
                    query_vector=query_vector, limit=3, min_score=0.20
                )
                retrieval_seconds = time.perf_counter() - retrieval_start

                if not chunks:
                    logger.warning(
                        "MANUAL_SEARCH | turn_id=%s | action=no_results | query_chars=%d",
                        turn_id,
                        len(payload.user_input),
                    )
                else:
                    logger.info("MANUAL_SEARCH | turn_id=%s | fragments=%d", turn_id, len(chunks))
                    current_phase = "evidence"
                    evidence_start = time.perf_counter()
                    visual_evidence = self._collect_visual_evidence(chunks)
                    evidence_seconds = time.perf_counter() - evidence_start

                current_phase = "prompt"
                prompt_start = time.perf_counter()
                max_chars = self._settings.llm_context_max_chars_per_chunk
                for i, c in enumerate(chunks):
                    src = c.metadata.get("source") or c.metadata.get("filename") or "Desconocido"
                    pval = c.metadata.get("page") or c.metadata.get("page_number") or c.page
                    header = f"=== FRAGMENTO {i + 1} ===\nFUENTE: {src}\nPAGINA: {pval}\n"
                    document_title = c.metadata.get("document_title")
                    image_title = c.metadata.get("image_title")
                    if document_title:
                        header += f"DOCUMENTO: {document_title}\n"
                    if image_title:
                        header += f"IMAGEN: {image_title}\n"
                    # The LLM consumes the optimized derivative, never the
                    # full-resolution original kept for the UI.
                    img_paths = c.metadata.get("vision_paths") or c.metadata.get("image_paths")
                    if img_paths:
                        header += f"IMAGE_PATHS: {img_paths}\n"

                    body = _clip_context(c.content, max_chars)
                    if len(body) < len(c.content.strip()):
                        truncated_chunks += 1
                    context_chars += len(header) + 1 + len(body)
                    contexto_sintesis.append(header + "\n" + body)
                prompt_seconds = time.perf_counter() - prompt_start

                logger.info(
                    "PROMPT_CONTEXT | turn_id=%s | chunks=%d | chars=%d | "
                    "max_chars_per_chunk=%d | truncated=%d",
                    turn_id,
                    len(contexto_sintesis),
                    context_chars,
                    max_chars,
                    truncated_chunks,
                )

            elif used_tool == "analizar_imagen":
                if not payload.image_path:
                    raise DataIncompleteError("No se proporciono imagen.")
                current_phase = "vision"
                vision_start = time.perf_counter()
                vision_analysis = self._llm.analyze_image(payload.image_path)
                vision_analysis_seconds = time.perf_counter() - vision_start
                vision_metrics = self._llm.consume_call_metrics()
                if vision_metrics is not None:
                    vision_analysis_seconds = max(
                        0.0, vision_analysis_seconds - vision_metrics.queue_wait_seconds
                    )
                contexto_sintesis = [f"[FUENTE: imagen]\n{vision_analysis}"]

            elif used_tool == "responder_directo":
                contexto_sintesis = ["No se requieren herramientas externas."]

            current_phase = "inference"
            generation_started_at = time.perf_counter()
            respuesta_final = self._llm.generate_response(contexto_sintesis, payload.user_input)
            generation_wall = time.perf_counter() - generation_started_at
            generation_metrics = self._llm.consume_call_metrics()
            image_prep_seconds, inference_seconds, image_stats = self._split_generation(
                generation_wall, generation_metrics
            )

            total_seconds = time.perf_counter() - start_time
            current_phase = "complete"
            timings = PhaseTimings(
                routing_seconds=routing_seconds,
                embedding_seconds=embedding_seconds,
                retrieval_seconds=retrieval_seconds,
                evidence_seconds=evidence_seconds,
                prompt_seconds=prompt_seconds,
                image_prep_seconds=image_prep_seconds,
                vision_analysis_seconds=vision_analysis_seconds,
                inference_seconds=inference_seconds,
                total_seconds=total_seconds,
            )
            trace = TurnTrace(
                turn_id=turn_id,
                tool=used_tool,
                timings=timings,
                images_attached=image_stats.images_attached,
                images_skipped=image_stats.images_skipped,
                image_bytes=image_stats.image_bytes,
                image_pixels=image_stats.image_pixels,
                payload_bytes=image_stats.payload_bytes,
                context_chars=context_chars,
                truncated_chunks=truncated_chunks,
                degraded=router_degraded,
                degraded_phase="router" if router_degraded else None,
                error_code=error_code,
                status="degraded" if router_degraded else "ok",
            )
            logger.info(trace.format_line())

            return OutboundResponseDTO(
                response_text=respuesta_final,
                is_degraded=router_degraded,
                used_tool=used_tool,
                error_log=error_code,
                elapsed_seconds=total_seconds,
                timings=timings,
                visual_evidence=visual_evidence,
                turn_id=turn_id,
            )

        except DomainBaseError as exc:
            code = type(exc).__name__
            # Attribute the wall time already spent in generation so a failed
            # turn still reports where the budget went.
            if generation_started_at is not None and inference_seconds == 0.0:
                inference_seconds = max(0.0, time.perf_counter() - generation_started_at)
            total_seconds = time.perf_counter() - start_time
            timings = PhaseTimings(
                routing_seconds=routing_seconds,
                embedding_seconds=embedding_seconds,
                retrieval_seconds=retrieval_seconds,
                evidence_seconds=evidence_seconds,
                prompt_seconds=prompt_seconds,
                image_prep_seconds=image_prep_seconds,
                vision_analysis_seconds=vision_analysis_seconds,
                inference_seconds=inference_seconds,
                total_seconds=total_seconds,
            )
            logger.error(
                "ORCHESTRATION_ERROR | turn_id=%s | phase=%s | error_code=%s",
                turn_id,
                current_phase,
                code,
            )
            logger.info(
                TurnTrace(
                    turn_id=turn_id,
                    tool=used_tool,
                    timings=timings,
                    context_chars=context_chars,
                    truncated_chunks=truncated_chunks,
                    degraded=True,
                    degraded_phase=current_phase,
                    error_code=code,
                    status="error",
                ).format_line()
            )
            return self._handle_graceful_degradation(
                herramienta=used_tool,
                error_code=code,
                elapsed_seconds=total_seconds,
                timings=timings,
                visual_evidence=visual_evidence,
            )
        except Exception as exc:
            code = type(exc).__name__
            if generation_started_at is not None and inference_seconds == 0.0:
                inference_seconds = max(0.0, time.perf_counter() - generation_started_at)
            total_seconds = time.perf_counter() - start_time
            timings = PhaseTimings(
                routing_seconds=routing_seconds,
                embedding_seconds=embedding_seconds,
                retrieval_seconds=retrieval_seconds,
                evidence_seconds=evidence_seconds,
                prompt_seconds=prompt_seconds,
                image_prep_seconds=image_prep_seconds,
                vision_analysis_seconds=vision_analysis_seconds,
                inference_seconds=inference_seconds,
                total_seconds=total_seconds,
            )
            logger.error(
                "ORCHESTRATION_ERROR | turn_id=%s | phase=%s | error_code=%s",
                turn_id,
                current_phase,
                code,
            )
            logger.info(
                TurnTrace(
                    turn_id=turn_id,
                    tool=used_tool,
                    timings=timings,
                    context_chars=context_chars,
                    truncated_chunks=truncated_chunks,
                    degraded=True,
                    degraded_phase=current_phase,
                    error_code=code,
                    status="error",
                ).format_line()
            )
            return self._handle_graceful_degradation(
                herramienta=used_tool,
                error_code=code,
                elapsed_seconds=total_seconds,
                timings=timings,
                visual_evidence=visual_evidence,
            )

    @staticmethod
    def _split_generation(
        generation_wall: float, metrics: LLMCallMetrics | None
    ) -> tuple[float, float, LLMCallMetrics]:
        if metrics is None:
            return 0.0, generation_wall, LLMCallMetrics(operation="generate")
        image_prep = max(0.0, metrics.image_prep_seconds)
        queue_wait = max(0.0, metrics.queue_wait_seconds)
        inference = max(0.0, generation_wall - image_prep - queue_wait)
        return image_prep, inference, metrics

    def _collect_visual_evidence(
        self, chunks: Sequence[ManualChunk]
    ) -> tuple[VisualEvidenceDTO, ...]:
        """Resolve indexed visual assets to renderable, in-scope local paths.

        Both dedicated visual chunks and text fragments that reference the
        figures on their page are considered, because top-k retrieval often
        ranks the surrounding prose above the image chunk. Assets are
        deduplicated by content hash when available and capped to the
        configured multimodal budget.
        """
        assets_dir = self._settings.extracted_images_dir.resolve()
        limit = self._settings.vision_max_images_per_turn
        evidence: list[VisualEvidenceDTO] = []
        seen_paths: set[str] = set()
        seen_hashes: set[str] = set()

        for chunk in chunks:
            originals = self._chunk_image_paths(chunk)
            thumbnails = self._chunk_thumb_paths(chunk)
            for index, raw_path in enumerate(originals):
                resolved = self._resolve_asset_path(raw_path, assets_dir)
                if resolved is None:
                    logger.warning(
                        "VISUAL_EVIDENCE | chunk_id=%s | action=skipped | reason=unresolved_path",
                        chunk.chunk_id,
                    )
                    continue

                resolved_key = str(resolved)
                digest = chunk.metadata.get("image_sha256")
                digest_key = str(digest) if digest else ""
                if resolved_key in seen_paths or (digest_key and digest_key in seen_hashes):
                    continue
                seen_paths.add(resolved_key)
                if digest_key:
                    seen_hashes.add(digest_key)

                thumbnail_raw = thumbnails[index] if index < len(thumbnails) else ""
                thumbnail = (
                    self._resolve_asset_path(thumbnail_raw, assets_dir) if thumbnail_raw else None
                )
                evidence.append(
                    VisualEvidenceDTO(
                        image_path=str(resolved),
                        image_title=str(chunk.metadata.get("image_title") or ""),
                        document_title=str(
                            chunk.metadata.get("document_title")
                            or chunk.metadata.get("source")
                            or ""
                        ),
                        page=chunk.page,
                        thumbnail_path=str(thumbnail or resolved),
                        original_path=str(resolved),
                    )
                )
                if len(evidence) >= limit:
                    return tuple(evidence)

        return tuple(evidence)

    @staticmethod
    def _chunk_image_paths(chunk: ManualChunk) -> list[str]:
        single = chunk.metadata.get("image_path")
        if isinstance(single, str) and single.strip():
            return [single.strip()]
        multiple = chunk.metadata.get("image_paths")
        if isinstance(multiple, str):
            return [path.strip() for path in multiple.split("|") if path.strip()]
        return []

    @staticmethod
    def _chunk_thumb_paths(chunk: ManualChunk) -> list[str]:
        single = chunk.metadata.get("thumb_path")
        if isinstance(single, str) and single.strip():
            return [single.strip()]
        multiple = chunk.metadata.get("thumb_paths")
        if isinstance(multiple, str):
            return [path.strip() for path in multiple.split("|") if path.strip()]
        return []

    @staticmethod
    def _resolve_asset_path(raw_path: str, assets_dir: Path) -> Path | None:
        candidate = Path(raw_path)
        for path in (candidate, assets_dir / candidate.name):
            if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved.is_file() and resolved.is_relative_to(assets_dir):
                return resolved
        return None

    def _handle_graceful_degradation(
        self,
        *,
        herramienta: str,
        error_code: str,
        elapsed_seconds: float,
        timings: PhaseTimings,
        visual_evidence: tuple[VisualEvidenceDTO, ...],
    ) -> OutboundResponseDTO:
        return OutboundResponseDTO(
            response_text=(
                f"Aviso del sistema: no se pudo completar el turno (codigo: {error_code})."
            ),
            is_degraded=True,
            used_tool=herramienta,
            error_log=error_code,
            elapsed_seconds=elapsed_seconds,
            timings=timings,
            visual_evidence=visual_evidence,
        )
