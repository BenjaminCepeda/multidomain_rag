# src/application/use_cases/orchestrate_agent.py
import logging
from typing import List, Dict, Any
from src.application.dtos.agent_dtos import InboundPayloadDTO, OutboundResponseDTO
from src.domain.ports.llm_engine import ILLMEngine
from src.domain.ports.vector_repository import IVectorRepository
from src.domain.exceptions.core_errors import DomainBaseError, DataIncompleteError

logger = logging.getLogger(__name__)

class OrchestrateAgentUseCase:
    """Caso de uso principal para coordinar las herramientas del Agente de Fusión Multimodal."""

    def __init__(self, llm_engine: ILLMEngine, vector_repo: IVectorRepository) -> None:
        self._llm: ILLMEngine = llm_engine
        self._repo: IVectorRepository = vector_repo

    def execute(self, payload: InboundPayloadDTO) -> OutboundResponseDTO:
        has_image: bool = payload.image_path is not None
        
        try:
            # 1. Decisión del orquestador
            decision = self._llm.get_orchestrator_decision(payload.user_input, has_image)
            
            # Normalización de herramientas
            herramientas_permitidas = ["buscar_manual", "analizar_imagen", "responder_directo"]
            used_tool = decision.herramienta if decision.herramienta in herramientas_permitidas else "buscar_manual"
            
            logger.info(f"Orquestador usando: {used_tool}")
            
            contexto_sintesis: List[Dict[str, Any]] = []

            # 2. Enrutamiento dinámico (CORREGIDO: Forzamos payload.user_input siempre)
            if used_tool == "buscar_manual":
                # Ignoramos completamente decision.parametro para evitar que pase 'False' o datos erróneos
                query_target = payload.user_input
                logger.info(f"Buscando en manuales con query: '{query_target}'")
                
                chunks_recuperados = self._repo.search_similar(query_target, limit=3)
                
                if not chunks_recuperados:
                    logger.warning(f"No se encontraron resultados para: '{query_target}'")
                else:
                    logger.info(f"Recuperados {len(chunks_recuperados)} fragmentos.")
                
                contexto_sintesis = [
                    {"content": c.content, "metadata": c.metadata} 
                    for c in chunks_recuperados
                ]
                
            elif used_tool == "analizar_imagen":
                if not payload.image_path:
                    raise DataIncompleteError("No se proporcionó imagen.")
                vision_analysis = self._llm.analyze_image(payload.image_path)
                contexto_sintesis = [{"content": vision_analysis, "metadata": {"source": "imagen"}}]
                
            elif used_tool == "responder_directo":
                contexto_sintesis = [{"content": "No se requieren herramientas externas.", "metadata": {"source": "sistema"}}]

            # 3. Síntesis
            respuesta_final = self._llm.generate_response(contexto_sintesis, payload.user_input)
            
            return OutboundResponseDTO(
                response_text=respuesta_final,
                is_degraded=False,
                used_tool=used_tool
            )

        except Exception as e:
            logger.exception(f"Error en orquestación: {str(e)}")
            return self._handle_graceful_degradation(payload.user_input, "orquestador", str(e))

    def _handle_graceful_degradation(self, query: str, herramienta: str, motivo_error: str) -> OutboundResponseDTO:
        return OutboundResponseDTO(
            response_text=f"Aviso del sistema: {motivo_error}",
            is_degraded=True,
            used_tool=herramienta,
            error_log=motivo_error
        )