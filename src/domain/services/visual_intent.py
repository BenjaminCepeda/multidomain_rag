"""Deterministic, multilingual visual-intent detection.

The detector is intentionally independent from any infrastructure concern: it
only classifies the operator query so the application layer can decide whether
to force the multimodal flow. It combines explicit visual artefacts, structural
concepts (pipeline, flow, architecture, ...) and request verbs, and suppresses
pure conceptual definitions such as "what is a data pipeline?".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class VisualIntentDecision:
    """Immutable result of the visual-intent classification."""

    is_visual: bool
    confidence: float
    signals: tuple[str, ...] = ()
    reason: str = "none"


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


# Unambiguous visual artefacts: asking for one of these always implies an image.
_VISUAL_ARTEFACT_PATTERN = re.compile(
    r"\b(?:"
    r"diagram(?:a|as|s)?|"
    r"esquema(?:s)?|"
    r"grafic(?:o|a|os|as)|"
    r"figura(?:s)?|figures?|"
    r"charts?|graphs?|"
    r"ilustracion(?:es)?|"
    r"infografia(?:s)?|"
    r"imagen(?:es)?|images?|pictures?|"
    r"flujograma(?:s)?|flowcharts?|"
    r"schematic(?:s)?|blueprint(?:s)?|"
    r"planos?|mapas?|"
    r"visual(?:es|ization|isation)?"
    r")\b",
    re.IGNORECASE,
)

# Structural / flow concepts. They are visual only when the operator asks to
# see/display them, or when the phrasing asks for a concrete architecture.
_STRUCTURAL_PATTERN = re.compile(
    r"\b(?:"
    r"pipelines?|"
    r"flows?|flujos?|"
    r"workflows?|"
    r"architectures?|arquitecturas?|"
    r"topolog(?:y|ia|ias|ies)|"
    r"block\s+diagrams?|"
    r"data\s+flows?|process\s+flows?|"
    r"training[\s-]+to[\s-]+inference"
    r")\b",
    re.IGNORECASE,
)

# Explicit request for a visual representation.
_VISUAL_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    r"show\s+me|display|render|draw|visuali[sz]e|visuali[sz]ation|"
    r"muestrame|muestrenme|muestra|mostrar|"
    r"ensen(?:a|ame|enme)|"
    r"dibuja(?:me)?|"
    r"quiero\s+ver|puedes\s+mostrar|podrias\s+mostrar|"
    r"cual\s+es\s+la\s+imagen|"
    r"ver\s+(?:la|el|los|las)\s+(?:figura|diagrama|grafico|esquema|imagen)"
    r")\b",
    re.IGNORECASE,
)

# Review verbs only imply a visual artefact when paired with a structural term.
_STRUCTURAL_REVIEW_PATTERN = re.compile(
    r"\b(?:revisa(?:r)?|analiza(?:r)?|compara(?:r)?|review|analy[sz]e|compare)\b",
    re.IGNORECASE,
)

# Pure conceptual questions. They suppress structural terms but never
# unambiguous visual artefacts ("what is figure 15?" still points to an image).
_DEFINITION_PATTERN = re.compile(
    r"\b(?:"
    r"que\s+es|que\s+son|"
    r"what\s+is|what\s+are|"
    r"define|definition|definicion|"
    r"explica(?:me)?|explain|"
    r"como\s+funciona|how\s+does|how\s+do|"
    r"para\s+que\s+sirve"
    r")\b",
    re.IGNORECASE,
)

# A concrete architecture/topology inquiry is usually a request for the diagram.
_ARCHITECTURE_INQUIRY_PATTERN = re.compile(
    r"\b(?:"
    r"cual\s+es\s+la\s+(?:arquitectura|topologia)|"
    r"which\s+is\s+the\s+(?:architecture|topology)|"
    r"what\s+is\s+the\s+(?:architecture|topology)"
    r")\b",
    re.IGNORECASE,
)


def detect_visual_intent(query: str) -> VisualIntentDecision:
    """Classify whether ``query`` explicitly asks for a visual artefact.

    The result is deterministic and Zero-Log safe: ``signals`` only contains
    fixed vocabulary labels, never fragments of the operator query.
    """
    normalized = _strip_accents(query)
    if not normalized.strip():
        return VisualIntentDecision(is_visual=False, confidence=0.0, reason="empty")

    signals: list[str] = []
    if _VISUAL_ARTEFACT_PATTERN.search(normalized):
        signals.append("visual_artefact")
    if _STRUCTURAL_PATTERN.search(normalized):
        signals.append("structural")
    if _VISUAL_REQUEST_PATTERN.search(normalized):
        signals.append("visual_request")
    if _STRUCTURAL_REVIEW_PATTERN.search(normalized):
        signals.append("structural_review")
    if _ARCHITECTURE_INQUIRY_PATTERN.search(normalized):
        signals.append("architecture_inquiry")

    definition = _DEFINITION_PATTERN.search(normalized) is not None

    if "visual_artefact" in signals:
        confidence = 0.95 if "visual_request" in signals else 0.85
        return VisualIntentDecision(
            is_visual=True,
            confidence=confidence,
            signals=tuple(signals),
            reason="visual_artefact",
        )

    if "structural" in signals:
        if "visual_request" in signals:
            return VisualIntentDecision(
                is_visual=True,
                confidence=0.80,
                signals=tuple(signals),
                reason="structural_request",
            )
        if "structural_review" in signals and not definition:
            return VisualIntentDecision(
                is_visual=True,
                confidence=0.70,
                signals=tuple(signals),
                reason="structural_review",
            )
        if "architecture_inquiry" in signals and not definition:
            return VisualIntentDecision(
                is_visual=True,
                confidence=0.70,
                signals=tuple(signals),
                reason="architecture_inquiry",
            )
        return VisualIntentDecision(
            is_visual=False,
            confidence=0.30,
            signals=tuple(signals),
            reason="conceptual_structural",
        )

    if "visual_request" in signals and not definition:
        return VisualIntentDecision(
            is_visual=True,
            confidence=0.60,
            signals=tuple(signals),
            reason="visual_request",
        )

    return VisualIntentDecision(is_visual=False, confidence=0.0, signals=tuple(signals))
