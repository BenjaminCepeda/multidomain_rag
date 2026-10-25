# src/presentation/ui_state_presenter.py
"""Presentation helper that aggregates service health into a view-friendly string."""

from __future__ import annotations

from src.infrastructure.composition.root import ApplicationContext


class UIStatePresenter:
    """Aggregates health/status of backend services for the UI."""

    def __init__(self, context: ApplicationContext) -> None:
        self._ctx = context

    def build_status(self) -> str:
        lines: list[str] = []

        # LLM
        try:
            llm_ok = self._ctx.llm.health_check()
            lines.append(f"LM Studio: {'Disponible' if llm_ok else 'No disponible'}")
        except Exception as exc:  # pragma: no cover - defensive
            lines.append(f"LM Studio: Error ({exc})")

        # Qdrant
        try:
            if self._ctx.vector_repo.has_data():
                count = self._ctx.vector_repo.count()
                lines.append(f"Qdrant: Conectado ({count} fragmentos)")
            else:
                lines.append("Qdrant: Conectado (sin datos)")
        except Exception as exc:  # pragma: no cover - defensive
            lines.append(f"Qdrant: Error ({exc})")

        # Embeddings
        try:
            lines.append(
                f"Embeddings: {self._ctx.embedding.model_name()} "
                f"({self._ctx.embedding.dimension()} dims)"
            )
        except Exception as exc:  # pragma: no cover - defensive
            lines.append(f"Embeddings: Error ({exc})")

        return "\n".join(lines)
