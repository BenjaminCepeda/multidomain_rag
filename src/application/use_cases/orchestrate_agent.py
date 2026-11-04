# src/application/use_cases/orchestrate_agent.py
import logging
import math
import re
import time
import unicodedata
import uuid
from collections.abc import Sequence
from pathlib import Path

from src.application.dtos.agent_dtos import (
    InboundPayloadDTO,
    OutboundResponseDTO,
    VisualEvidenceDTO,
)
from src.domain.exceptions.core_errors import DomainBaseError
from src.domain.models.document import ManualChunk
from src.domain.models.embedding import EmbeddingVector
from src.domain.models.telemetry import LLMCallMetrics, PhaseTimings, TurnTrace
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = frozenset({"buscar_manual", "analizar_imagen", "responder_directo"})
ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})

# Deterministic pre-routing signal: an explicit request for a visual artefact
# always needs the multimodal tool, regardless of the router's text bias.
VISUAL_INTENT_PATTERN = re.compile(
    r"\b(?:grafic[oa]s?|diagramas?|imagen(?:es)?|charts?|figuras?|esquemas?|"
    r"ilustracion(?:es)?|infografias?|visual(?:es)?)\b",
    re.IGNORECASE,
)
VISUAL_INTENT_PAGE_CHUNKS = 3


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


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
        visual_intent = self._has_visual_intent(payload.user_input)
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
            router_metrics: LLMCallMetrics | None = None
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
            finally:
                routing_seconds = time.perf_counter() - routing_start
                router_metrics = self._llm.consume_call_metrics()

            if router_degraded:
                # Report the transport root cause (timeout, unavailable,
                # circuit open, ...) instead of only the exception class.
                error_code = self._router_root_cause(router_metrics, error_code)
                logger.warning(
                    "ROUTER_FALLBACK | turn_id=%s | tool=%s | reason=%s",
                    turn_id,
                    used_tool,
                    error_code,
                )

            if visual_intent and used_tool != "analizar_imagen":
                logger.info(
                    "VISUAL_INTENT | turn_id=%s | router_tool=%s | forced_tool=analizar_imagen",
                    turn_id,
                    used_tool,
                )
                used_tool = "analizar_imagen"

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
                candidates: list[ManualChunk] = self._repo.search(
                    query_vector=query_vector,
                    limit=self._settings.vision_evidence_candidate_limit,
                    min_score=0.20,
                )
                retrieval_seconds = time.perf_counter() - retrieval_start
                chunks = candidates[:VISUAL_INTENT_PAGE_CHUNKS]

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
                    visual_evidence = self._collect_visual_evidence(query_vector, candidates)
                    evidence_seconds = time.perf_counter() - evidence_start

                current_phase = "prompt"
                prompt_start = time.perf_counter()
                contexto_sintesis, context_chars, truncated_chunks = self._build_prompt_context(
                    chunks, turn_id
                )
                prompt_seconds = time.perf_counter() - prompt_start

            elif used_tool == "analizar_imagen":
                if payload.image_path:
                    current_phase = "vision"
                    vision_start = time.perf_counter()
                    vision_analysis = self._llm.analyze_image(
                        payload.image_path, question=payload.user_input
                    )
                    vision_analysis_seconds = time.perf_counter() - vision_start
                    vision_metrics = self._llm.consume_call_metrics()
                    if vision_metrics is not None:
                        vision_analysis_seconds = max(
                            0.0, vision_analysis_seconds - vision_metrics.queue_wait_seconds
                        )
                    contexto_sintesis = [f"[FUENTE: imagen]\n{vision_analysis}"]
                else:
                    # No uploaded image: the visual intent is satisfied by the
                    # most semantically relevant indexed figure, if any.
                    current_phase = "embedding"
                    embedding_start = time.perf_counter()
                    query_vector = self._embedding.embed_text(payload.user_input)
                    embedding_seconds = time.perf_counter() - embedding_start

                    current_phase = "retrieval"
                    logger.info(
                        "VISUAL_SEARCH | turn_id=%s | query_chars=%d",
                        turn_id,
                        len(payload.user_input),
                    )
                    retrieval_start = time.perf_counter()
                    candidates = self._repo.search(
                        query_vector=query_vector,
                        limit=self._settings.vision_evidence_candidate_limit,
                        min_score=0.20,
                    )
                    retrieval_seconds = time.perf_counter() - retrieval_start

                    current_phase = "evidence"
                    evidence_start = time.perf_counter()
                    visual_evidence = self._collect_visual_evidence(query_vector, candidates)
                    evidence_seconds = time.perf_counter() - evidence_start

                    if visual_evidence:
                        target = visual_evidence[0]
                        current_phase = "vision"
                        vision_start = time.perf_counter()
                        vision_analysis = self._llm.analyze_image(
                            target.original_path or target.image_path,
                            question=payload.user_input,
                        )
                        vision_analysis_seconds = time.perf_counter() - vision_start
                        vision_metrics = self._llm.consume_call_metrics()
                        if vision_metrics is not None:
                            vision_analysis_seconds = max(
                                0.0,
                                vision_analysis_seconds - vision_metrics.queue_wait_seconds,
                            )

                        current_phase = "prompt"
                        prompt_start = time.perf_counter()
                        text_context, context_chars, truncated_chunks = self._build_prompt_context(
                            candidates[:VISUAL_INTENT_PAGE_CHUNKS],
                            turn_id,
                            include_image_paths=False,
                        )
                        prompt_seconds = time.perf_counter() - prompt_start
                        contexto_sintesis = [
                            f"[FUENTE: imagen]\n{vision_analysis}",
                            *text_context,
                        ]
                        logger.info(
                            "VISUAL_ANALYSIS | turn_id=%s | evidence=%d | analyzed=%d",
                            turn_id,
                            len(visual_evidence),
                            1,
                        )
                    else:
                        # Avoid rendering unrelated graphics: degrade to a
                        # strictly text-only manual search instead.
                        logger.info(
                            "VISUAL_INTENT | turn_id=%s | action=fallback_to_text | "
                            "reason=no_relevant_visual",
                            turn_id,
                        )
                        used_tool = "buscar_manual"
                        current_phase = "prompt"
                        prompt_start = time.perf_counter()
                        contexto_sintesis, context_chars, truncated_chunks = (
                            self._build_prompt_context(
                                candidates[:VISUAL_INTENT_PAGE_CHUNKS], turn_id
                            )
                        )
                        prompt_seconds = time.perf_counter() - prompt_start

            elif used_tool == "responder_directo":
                contexto_sintesis = ["No se requieren herramientas externas."]

            current_phase = "inference"
            generation_started_at = time.perf_counter()
            respuesta_final = self._llm.generate_response(
                contexto_sintesis,
                payload.user_input,
                attach_images=(used_tool == "analizar_imagen"),
            )
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

    @staticmethod
    def _router_root_cause(metrics: LLMCallMetrics | None, fallback: str | None) -> str | None:
        """Translate the adapter transport status into a precise degradation code.

        The code is Zero-Log safe (no query, prompt or model output) and lets
        the trace distinguish a slow router from an unavailable or open circuit.
        """
        if metrics is None:
            return fallback
        mapping = {
            "timeout": "router_timeout",
            "unavailable": "router_unavailable",
            "circuit_open": "router_circuit_open",
            "http_error": "router_http_error",
            "transport_error": "router_transport_error",
            "protocol_error": "router_protocol_error",
        }
        return mapping.get(metrics.status, fallback)

    @staticmethod
    def _has_visual_intent(query: str) -> bool:
        """Return True when the operator explicitly asks for a visual artefact."""
        if not query.strip():
            return False
        return VISUAL_INTENT_PATTERN.search(_strip_accents(query)) is not None

    def _build_prompt_context(
        self,
        chunks: Sequence[ManualChunk],
        turn_id: str,
        *,
        include_image_paths: bool = True,
    ) -> tuple[list[str], int, int]:
        """Clip retrieved fragments and keep their source metadata in the prompt."""
        max_chars = self._settings.llm_context_max_chars_per_chunk
        context: list[str] = []
        context_chars = 0
        truncated = 0

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
            if include_image_paths:
                img_paths = c.metadata.get("vision_paths") or c.metadata.get("image_paths")
                if img_paths:
                    header += f"IMAGE_PATHS: {img_paths}\n"

            body = _clip_context(c.content, max_chars)
            if len(body) < len(c.content.strip()):
                truncated += 1
            context_chars += len(header) + 1 + len(body)
            context.append(header + "\n" + body)

        logger.info(
            "PROMPT_CONTEXT | turn_id=%s | chunks=%d | chars=%d | "
            "max_chars_per_chunk=%d | truncated=%d",
            turn_id,
            len(context),
            context_chars,
            max_chars,
            truncated,
        )
        return context, context_chars, truncated

    @staticmethod
    def _cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
        if left.dimension != right.dimension:
            return 0.0
        dot = sum(a * b for a, b in zip(left.values, right.values, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left.values))
        right_norm = math.sqrt(sum(value * value for value in right.values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _collect_visual_evidence(
        self,
        query_vector: EmbeddingVector,
        chunks: Sequence[ManualChunk],
    ) -> tuple[VisualEvidenceDTO, ...]:
        """Return only visual assets that are semantically relevant to the query.

        Dedicated visual chunks are authoritative because they describe one
        specific figure. Page-level text chunks are used as a fallback only
        when the retrieved pool contains no visual chunk at all: mixing both
        sources would leak every figure drawn on a page regardless of topic.
        Assets are deduplicated and capped to the multimodal budget.
        """
        assets_dir = self._settings.extracted_images_dir.resolve()
        limit = self._settings.vision_max_images_per_turn
        min_similarity = self._settings.vision_evidence_min_similarity

        visual_chunks = [c for c in chunks if c.is_visual and self._chunk_image_paths(c)]
        text_chunks = (
            []
            if visual_chunks
            else [
                chunk for chunk in chunks if not chunk.is_visual and self._chunk_image_paths(chunk)
            ]
        )
        scored_chunks = [*visual_chunks, *text_chunks]

        if not scored_chunks:
            logger.info(
                "VISUAL_EVIDENCE | candidates=0 | selected=0 | min_similarity=%.2f",
                min_similarity,
            )
            return ()

        vectors = self._embedding.embed_batch([chunk.embedding_input for chunk in scored_chunks])
        candidates: list[tuple[float, ManualChunk, Path, Path | None]] = []

        for chunk, vector in zip(scored_chunks, vectors, strict=True):
            score = self._cosine_similarity(query_vector, vector)
            if score < min_similarity:
                continue

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

                thumbnail_raw = thumbnails[index] if index < len(thumbnails) else ""
                thumbnail = (
                    self._resolve_asset_path(thumbnail_raw, assets_dir) if thumbnail_raw else None
                )
                candidates.append((score, chunk, resolved, thumbnail))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        evidence: list[VisualEvidenceDTO] = []
        seen_paths: set[str] = set()
        seen_hashes: set[str] = set()
        for _score, chunk, resolved, thumbnail in candidates:
            resolved_key = str(resolved)
            digest = chunk.metadata.get("image_sha256")
            digest_key = str(digest) if digest else ""
            if resolved_key in seen_paths or (digest_key and digest_key in seen_hashes):
                continue
            seen_paths.add(resolved_key)
            if digest_key:
                seen_hashes.add(digest_key)

            evidence.append(
                VisualEvidenceDTO(
                    image_path=str(resolved),
                    image_title=str(chunk.metadata.get("image_title") or ""),
                    document_title=str(
                        chunk.metadata.get("document_title") or chunk.metadata.get("source") or ""
                    ),
                    page=chunk.page,
                    thumbnail_path=str(thumbnail or resolved),
                    original_path=str(resolved),
                )
            )
            if len(evidence) >= limit:
                break

        logger.info(
            "VISUAL_EVIDENCE | candidates=%d | selected=%d | min_similarity=%.2f",
            len(scored_chunks),
            len(evidence),
            min_similarity,
        )
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
