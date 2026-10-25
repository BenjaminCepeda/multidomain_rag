# src/domain/ports/llm_engine.py
from abc import ABC, abstractmethod
from typing import Optional
from src.domain.models.agent import OrchestratorDecision

class ILLMEngine(ABC):
    """Contrato para la ejecución de inferencia lingüística y multimodal local."""
    
    @abstractmethod
    def get_orchestrator_decision(self, user_input: str, has_image: bool) -> OrchestratorDecision:
        """
        Consulta al modelo de razonamiento para determinar el flujo de ejecución del agente.
        
        Raises:
            LLMInferenceError: Si LM Studio devuelve un JSON inválido o agota el timeout.
        """
        pass

    @abstractmethod
    def generate_response(self, context: str, user_input: str) -> str:
        """
        Sintetiza una respuesta técnica basada exclusivamente en el contexto recuperado.
        
        Raises:
            LLMInferenceError: Si el modelo de síntesis local falla o se queda sin VRAM.
        """
        pass

    @abstractmethod
    def analyze_image(self, image_path: str) -> str:
        """
        Ejecuta inferencia visual sobre un diagrama o captura de hardware.
        
        Raises:
            DataIncompleteError: Si no se localiza la imagen física.
            LLMInferenceError: Si el modelo de visión decae.
        """
        pass