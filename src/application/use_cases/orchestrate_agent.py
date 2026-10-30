# src/application/use_cases/orchestrate_agent.py
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from src.application.dtos.agent_dtos import (
    InboundPayloadDTO,
    OutboundResponseDTO,
    VisualEvidenceDTO,
)
from src.domain.exceptions.core_errors import DataIncompleteError, DomainBaseError
from src.domain.models.document import ManualChunk
from src.domain.models.telemetry import PhaseTimings
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository
from src.infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

MAX_VISUAL_EVIDENCE = 4
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
        has_image: bool = payload.image_path is not None
        start_time = time.perf_counter()

        retrieval_seconds = 0.0
        inference_seconds = 0.0
        visual_evidence: tuple[VisualEvidenceDTO, ...] = ()

        try:
            decision = self._llm.get_orchestrator_decision(payload.user_input, has_image)

            allowed_tools = ["buscar_manual", "analizar_imagen", "responder_directo"]
            used_tool = (
                decision.herramienta if decision.herramienta in allowed_tools else "buscar_manual"
            )

            logger.info(f"Orquestador usando: {used_tool}")

            contexto_sintesis: list[str] = []

            if used_tool == "buscar_manual":
                query_target = payload.user_input
                logger.info("MANUAL_SEARCH | query_chars=%d", len(query_target))

                query_vector = self._embedding.embed_text(query_target)
                retrieval_start = time.perf_counter()
                chunks: list[ManualChunk] = self._repo.search(
                    query_vector=query_vector, limit=3, min_score=0.20
                )
                retrieval_seconds = time.perf_counter() - retrieval_start

                if not chunks:
                    logger.warning(
                        "MANUAL_SEARCH | action=no_results | query_chars=%d", len(query_target)
                    )
                else:
                    logger.info(f"Recuperados {len(chunks)} fragmentos.")
                    visual_evidence = self._collect_visual_evidence(chunks)

                contexto_sintesis = []
                max_chars = self._settings.llm_context_max_chars_per_chunk
                context_chars = 0
                truncated_chunks = 0
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
                    img_paths = c.metadata.get("image_paths")
                    if img_paths:
                        header += f"IMAGE_PATHS: {img_paths}\n"

                    body = _clip_context(c.content, max_chars)
                    if len(body) < len(c.content.strip()):
                        truncated_chunks += 1
                    context_chars += len(header) + 1 + len(body)
                    contexto_sintesis.append(header + "\n" + body)

                logger.info(
                    "PROMPT_CONTEXT | chunks=%d | chars=%d | max_chars_per_chunk=%d | truncated=%d",
                    len(contexto_sintesis),
                    context_chars,
                    max_chars,
                    truncated_chunks,
                )

            elif used_tool == "analizar_imagen":
                if not payload.image_path:
                    raise DataIncompleteError("No se proporciono imagen.")
                vision_analysis = self._llm.analyze_image(payload.image_path)
                contexto_sintesis = [f"[FUENTE: imagen]\n{vision_analysis}"]

            elif used_tool == "responder_directo":
                contexto_sintesis = ["No se requieren herramientas externas."]

            inference_start = time.perf_counter()
            respuesta_final = self._llm.generate_response(contexto_sintesis, payload.user_input)
            inference_seconds = time.perf_counter() - inference_start

            total_seconds = time.perf_counter() - start_time
            timings = PhaseTimings(
                retrieval_seconds=retrieval_seconds,
                inference_seconds=inference_seconds,
                total_seconds=total_seconds,
            )
            logger.info(f"PHASE_TIMINGS | {timings.format_summary()}")

            return OutboundResponseDTO(
                response_text=respuesta_final,
                is_degraded=False,
                used_tool=used_tool,
                elapsed_seconds=total_seconds,
                timings=timings,
                visual_evidence=visual_evidence,
            )

        except DomainBaseError as e:
            logger.exception(f"Error en orquestacion: {e!s}")
            return self._handle_graceful_degradation(
                payload.user_input, "orquestador", str(e), time.perf_counter() - start_time
            )
        except Exception as e:
            logger.exception(f"Error inesperado en orquestacion: {e!s}")
            return self._handle_graceful_degradation(
                payload.user_input, "orquestador", str(e), time.perf_counter() - start_time
            )

    def _collect_visual_evidence(
        self, chunks: Sequence[ManualChunk]
    ) -> tuple[VisualEvidenceDTO, ...]:
        """Resolve indexed visual assets to renderable, in-scope local paths.

        Both dedicated visual chunks and text fragments that reference the
        figures on their page are considered, because top-k retrieval often
        ranks the surrounding prose above the image chunk. Assets are
        deduplicated by resolved path and capped to keep the chat compact.
        """
        assets_dir = self._settings.extracted_images_dir.resolve()
        evidence: list[VisualEvidenceDTO] = []
        seen: set[str] = set()

        for chunk in chunks:
            for raw_path in self._chunk_image_paths(chunk):
                resolved = self._resolve_asset_path(raw_path, assets_dir)
                if resolved is None:
                    logger.warning(
                        "VISUAL_EVIDENCE | chunk_id=%s | action=skipped | reason=unresolved_path",
                        chunk.chunk_id,
                    )
                    continue

                key = str(resolved)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    VisualEvidenceDTO(
                        image_path=key,
                        image_title=str(chunk.metadata.get("image_title") or ""),
                        document_title=str(
                            chunk.metadata.get("document_title")
                            or chunk.metadata.get("source")
                            or ""
                        ),
                        page=chunk.page,
                    )
                )
                if len(evidence) >= MAX_VISUAL_EVIDENCE:
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
    def _resolve_asset_path(raw_path: str, assets_dir: Path) -> Path | None:
        candidate = Path(raw_path)
        for path in (candidate, assets_dir / candidate.name):
            if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                continue
            try:
                resolved = path.resolve()
            except OSError as exc:
                logger.warning("VISUAL_EVIDENCE | path=%s | action=skipped | error=%s", path, exc)
                continue
            if resolved.is_file() and resolved.is_relative_to(assets_dir):
                return resolved
        return None

    def _handle_graceful_degradation(
        self, query: str, herramienta: str, motivo_error: str, elapsed_seconds: float = 0.0
    ) -> OutboundResponseDTO:
        return OutboundResponseDTO(
            response_text=f"Aviso del sistema: {motivo_error}",
            is_degraded=True,
            used_tool=herramienta,
            error_log=motivo_error,
            elapsed_seconds=elapsed_seconds,
        )
