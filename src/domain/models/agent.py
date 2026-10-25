from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class OrchestratorDecision:
    razonamiento: str
    herramienta: str
    parametro: Optional[str] = None