from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseTimings:
    retrieval_seconds: float = 0.0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0

    def format_summary(self) -> str:
        return (
            f"Retrieval: {self.retrieval_seconds:.3f}s | "
            f"Inference: {self.inference_seconds:.3f}s | "
            f"Total: {self.total_seconds:.3f}s"
        )
