from dataclasses import dataclass
from typing import Any

from src.domain.models.telemetry import PhaseTimings


@dataclass(frozen=True)
class InboundPayloadDTO:
    """DTO inmutable para capturar la entrada multimedia del operador."""

    user_input: str
    image_path: str | None = None
    telemetry_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class VisualEvidenceDTO:
    """Immutable reference to an indexed visual asset rendered alongside an answer."""

    image_path: str
    image_title: str
    document_title: str
    page: int
    thumbnail_path: str | None = None
    original_path: str | None = None

    @property
    def display_path(self) -> str:
        """Small asset used in the chat history; the original stays on demand."""
        return self.thumbnail_path or self.image_path

    @property
    def caption(self) -> str:
        label = self.image_title or "Imagen"
        source = self.document_title or "Documento"
        return f"{label} - {source} (pag. {self.page})"


@dataclass(frozen=True)
class OutboundResponseDTO:
    """DTO de salida que incluye la respuesta final y banderas de diagnóstico del sistema."""

    response_text: str
    is_degraded: bool
    used_tool: str
    error_log: str | None = None
    elapsed_seconds: float = 0.0
    timings: PhaseTimings | None = None
    visual_evidence: tuple[VisualEvidenceDTO, ...] = ()
    turn_id: str | None = None
