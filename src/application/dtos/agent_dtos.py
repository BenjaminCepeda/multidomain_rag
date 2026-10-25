from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class InboundPayloadDTO:
    """DTO inmutable para capturar la entrada multimedia del operador."""
    user_input: str
    image_path: Optional[str] = None
    telemetry_meta: Dict[str, Any] = Optional[Dict[str, Any]]

@dataclass(frozen=True)
class OutboundResponseDTO:
    """DTO de salida que incluye la respuesta final y banderas de diagnóstico del sistema."""
    response_text: str
    is_degraded: bool
    used_tool: str
    error_log: Optional[str] = None