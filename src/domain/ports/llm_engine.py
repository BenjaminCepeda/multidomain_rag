# src/domain/ports/llm_engine.py
from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.domain.models.agent import OrchestratorDecision
from src.domain.models.telemetry import LLMCallMetrics


class ILLMEngine(ABC):
    """Port for local LLM inference (chat + vision).

    Implementations may expose the metrics of their most recent call through
    ``consume_call_metrics``. The default implementation keeps simple stubs
    and test doubles valid without forcing them to report transport details.
    """

    @abstractmethod
    def get_orchestrator_decision(
        self, user_input: str, has_image: bool
    ) -> OrchestratorDecision: ...

    @abstractmethod
    def generate_response(
        self,
        context: Sequence[str],
        user_input: str,
        *,
        attach_images: bool = True,
    ) -> str:
        """Synthesize an answer from text context.

        ``attach_images`` is the explicit modality gate: callers set it to
        ``False`` when the active tool only needs the retrieved text, so the
        adapter must not attach vision derivatives from the fragments.
        """
        ...

    @abstractmethod
    def analyze_image(self, image_path: str) -> str: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    def consume_call_metrics(self) -> LLMCallMetrics | None:
        """Return and clear the metrics of the last call made by this thread."""
        return None
