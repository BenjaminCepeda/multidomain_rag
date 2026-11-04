import pytest

from src.domain.services.visual_intent import VisualIntentDecision, detect_visual_intent


@pytest.mark.parametrize(
    "query",
    [
        "show me Block diagram of the generic training to inference pipeline",
        "display the system architecture",
        "draw the inference pipeline",
        "¿cuál es la arquitectura del sistema?",
        "muéstrame el pipeline de inferencia",
        "necesito un diagrama de flujo del entrenamiento",
        "describe el esquema de memoria",
        "revisa la topología de la red",
        "what is figure 15?",
    ],
)
def test_visual_intent_accepts_structural_and_visual_requests(query: str) -> None:
    decision = detect_visual_intent(query)

    assert isinstance(decision, VisualIntentDecision)
    assert decision.is_visual is True
    assert decision.confidence > 0.0
    assert decision.signals


@pytest.mark.parametrize(
    "query",
    [
        "¿Qué es un pipeline de datos?",
        "what is a data pipeline",
        "explica el proceso de inferencia",
        "how to configure the pipeline",
        "que es un acelerador",
        "",
        "   ",
    ],
)
def test_visual_intent_ignores_conceptual_text_questions(query: str) -> None:
    assert detect_visual_intent(query).is_visual is False


def test_visual_intent_decision_is_zero_log_safe() -> None:
    query = "muéstrame el diagrama confidencial de la planta secreta"
    decision = detect_visual_intent(query)

    assert decision.is_visual is True
    assert all("confidencial" not in signal for signal in decision.signals)
    assert all("planta" not in signal for signal in decision.signals)
