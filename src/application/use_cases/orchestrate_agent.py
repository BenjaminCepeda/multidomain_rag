# src/application/use_cases/orchestrate_agent.py
import logging
import time

from src.application.dtos.agent_dtos import InboundPayloadDTO, OutboundResponseDTO
from src.domain.exceptions.core_errors import DataIncompleteError, DomainBaseError
from src.domain.models.document import ManualChunk
from src.domain.models.telemetry import PhaseTimings
from src.domain.ports.embedding_service import IEmbeddingService
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository

logger = logging.getLogger(__name__)


class OrchestrateAgentUseCase:
    """Use case coordinating the multimodal agent tools."""

    def __init__(
        self,
        llm_engine: ILLMEngine,
        vector_repo: IVectorRepository,
        embedding: IEmbeddingService,
    ) -> None:
        self._llm: ILLMEngine = llm_engine
        self._repo: IVectorRepository = vector_repo
        self._embedding: IEmbeddingService = embedding

    def execute(self, payload: InboundPayloadDTO) -> OutboundResponseDTO:
        has_image: bool = payload.image_path is not None
        start_time = time.perf_counter()

        retrieval_seconds = 0.0
        inference_seconds = 0.0

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
                logger.info(f"Buscando en manuales con query: '{query_target}'")

                query_vector = self._embedding.embed_text(query_target)
                retrieval_start = time.perf_counter()
                chunks: list[ManualChunk] = self._repo.search(
                    query_vector=query_vector, limit=3, min_score=0.20
                )
                retrieval_seconds = time.perf_counter() - retrieval_start

                if not chunks:
                    logger.warning(f"No se encontraron resultados para: '{query_target}'")
                else:
                    logger.info(f"Recuperados {len(chunks)} fragmentos.")

                contexto_sintesis = []
                for i, c in enumerate(chunks):
                    src = c.metadata.get("source") or c.metadata.get("filename") or "Desconocido"
                    pval = c.metadata.get("page", "N/A")
                    header = f"=== FRAGMENTO {i + 1} ===\nFUENTE: {src}\nPAGINA: {pval}\n"
                    img_paths = c.metadata.get("image_paths")
                    if img_paths:
                        header += f"IMAGE_PATHS: {img_paths}\n"
                    contexto_sintesis.append(header + "\n" + c.content)

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
