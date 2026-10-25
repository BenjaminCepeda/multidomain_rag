from dataclasses import dataclass


@dataclass(frozen=True)
class OrchestratorDecision:
    razonamiento: str
    herramienta: str
    parametro: str | None = None
