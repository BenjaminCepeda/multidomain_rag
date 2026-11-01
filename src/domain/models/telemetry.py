"""Immutable telemetry value objects shared across the application boundary.

Every field is numeric, boolean or a short enum-like string. No field is
allowed to carry document content, user queries, transcriptions, Base64
payloads or filesystem paths, which keeps ``TURN_TRACE`` compliant with the
project Zero-Log policy by construction.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseTimings:
    """Semantic phase durations (seconds) for one agent turn."""

    routing_seconds: float = 0.0
    embedding_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    evidence_seconds: float = 0.0
    prompt_seconds: float = 0.0
    image_prep_seconds: float = 0.0
    vision_analysis_seconds: float = 0.0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def measured_seconds(self) -> float:
        """Sum of the individually attributed phases (excluding ``total``)."""
        return (
            self.routing_seconds
            + self.embedding_seconds
            + self.retrieval_seconds
            + self.evidence_seconds
            + self.prompt_seconds
            + self.image_prep_seconds
            + self.vision_analysis_seconds
            + self.inference_seconds
        )

    @property
    def unaccounted_seconds(self) -> float:
        """Wall-clock time not attributed to a known phase (never negative)."""
        return max(0.0, self.total_seconds - self.measured_seconds)

    def format_summary(self) -> str:
        return (
            f"Routing: {self.routing_seconds:.3f}s | "
            f"Embedding: {self.embedding_seconds:.3f}s | "
            f"Retrieval: {self.retrieval_seconds:.3f}s | "
            f"Evidence: {self.evidence_seconds:.3f}s | "
            f"Prompt: {self.prompt_seconds:.3f}s | "
            f"ImagePrep: {self.image_prep_seconds:.3f}s | "
            f"Vision: {self.vision_analysis_seconds:.3f}s | "
            f"Inference: {self.inference_seconds:.3f}s | "
            f"Total: {self.total_seconds:.3f}s"
        )


@dataclass(frozen=True)
class LLMCallMetrics:
    """Transport and payload metrics captured by an LLM adapter call.

    ``connect_seconds`` and ``ttfb_seconds`` remain ``0`` until a streaming
    HTTP client is adopted; the field is kept so the contract does not change
    when that happens.
    """

    operation: str
    status: str = "ok"
    total_seconds: float = 0.0
    connect_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    ttfb_seconds: float = 0.0
    image_prep_seconds: float = 0.0
    images_attached: int = 0
    images_skipped: int = 0
    image_bytes: int = 0
    image_pixels: int = 0
    payload_bytes: int = 0
    output_tokens: int | None = None


@dataclass(frozen=True)
class TurnTrace:
    """Structured, Zero-Log-safe summary of a single agent turn."""

    turn_id: str
    tool: str
    timings: PhaseTimings
    images_attached: int = 0
    images_skipped: int = 0
    image_bytes: int = 0
    image_pixels: int = 0
    payload_bytes: int = 0
    context_chars: int = 0
    truncated_chunks: int = 0
    degraded: bool = False
    degraded_phase: str | None = None
    error_code: str | None = None
    status: str = "ok"

    def format_line(self) -> str:
        t = self.timings
        return (
            f"TURN_TRACE | turn_id={self.turn_id} | tool={self.tool} | "
            f"route_ms={t.routing_seconds * 1000:.1f} | "
            f"embed_ms={t.embedding_seconds * 1000:.1f} | "
            f"retrieval_ms={t.retrieval_seconds * 1000:.1f} | "
            f"evidence_ms={t.evidence_seconds * 1000:.1f} | "
            f"prompt_ms={t.prompt_seconds * 1000:.1f} | "
            f"image_prep_ms={t.image_prep_seconds * 1000:.1f} | "
            f"vision_ms={t.vision_analysis_seconds * 1000:.1f} | "
            f"inference_ms={t.inference_seconds * 1000:.1f} | "
            f"total_ms={t.total_seconds * 1000:.1f} | "
            f"unaccounted_ms={t.unaccounted_seconds * 1000:.1f} | "
            f"images={self.images_attached} | images_skipped={self.images_skipped} | "
            f"image_bytes={self.image_bytes} | pixels={self.image_pixels} | "
            f"payload_bytes={self.payload_bytes} | context_chars={self.context_chars} | "
            f"truncated={self.truncated_chunks} | degraded={self.degraded} | "
            f"degraded_phase={self.degraded_phase or 'none'} | "
            f"error_code={self.error_code or 'none'} | status={self.status}"
        )
