# src/domain/models/embedding.py
"""Domain value object for an embedding vector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingVector:
    """Immutable embedding vector with basic validation."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("EmbeddingVector cannot be empty.")
        if any(not isinstance(v, (int, float)) for v in self.values):
            raise TypeError("EmbeddingVector values must be numeric.")

    @property
    def dimension(self) -> int:
        return len(self.values)

    def to_list(self) -> list[float]:
        return [float(v) for v in self.values]
