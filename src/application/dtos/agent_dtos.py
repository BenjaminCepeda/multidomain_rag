from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InboundPayloadDTO:
    """DTO inmutable para capturar la entrada multimedia del operador."""

    user_input: str
    image_path: str | None = None
    telemetry_meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class OutboundResponseDTO:
    """DTO de salida que incluye la respuesta final y banderas de diagnóstico del sistema."""

    response_text: str
    is_degraded: bool
    used_tool: str
    error_log: str | None = None
