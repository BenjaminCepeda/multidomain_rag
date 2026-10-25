# src/domain/models/document.py
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class ManualChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_visual: bool = False