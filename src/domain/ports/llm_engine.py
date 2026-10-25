# src/domain/ports/llm_engine.py
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.domain.models.agent import OrchestratorDecision


class ILLMEngine(ABC):
    """Port for local LLM inference (chat + vision)."""

    @abstractmethod
    def get_orchestrator_decision(
        self, user_input: str, has_image: bool
    ) -> OrchestratorDecision: ...

    @abstractmethod
    def generate_response(self, context: Sequence[str], user_input: str) -> str: ...

    @abstractmethod
    def analyze_image(self, image_path: str) -> str: ...

    @abstractmethod
    def health_check(self) -> bool: ...
