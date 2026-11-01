from src.domain.models.telemetry import LLMCallMetrics, PhaseTimings, TurnTrace


def test_phase_timings_exposes_attributed_and_unaccounted_time() -> None:
    timings = PhaseTimings(
        routing_seconds=0.5,
        retrieval_seconds=0.25,
        inference_seconds=1.0,
        total_seconds=2.0,
    )

    assert timings.measured_seconds == 1.75
    assert timings.unaccounted_seconds == 0.25
    summary = timings.format_summary()
    assert "Routing: 0.500s" in summary
    assert "Inference: 1.000s" in summary
    assert "Total: 2.000s" in summary


def test_unaccounted_time_is_never_negative() -> None:
    timings = PhaseTimings(inference_seconds=2.0, total_seconds=1.0)

    assert timings.unaccounted_seconds == 0.0


def test_turn_trace_line_is_structured_and_zero_log() -> None:
    trace = TurnTrace(
        turn_id="turn-123",
        tool="buscar_manual",
        timings=PhaseTimings(routing_seconds=0.125, total_seconds=0.5),
        images_attached=2,
        images_skipped=1,
        image_bytes=1234,
        image_pixels=640000,
        payload_bytes=2345,
        context_chars=900,
        truncated_chunks=1,
        degraded=True,
        degraded_phase="router",
        error_code="LLMInferenceError",
        status="degraded",
    )

    line = trace.format_line()

    assert "TURN_TRACE" in line
    assert "turn_id=turn-123" in line
    assert "route_ms=125.0" in line
    assert "total_ms=500.0" in line
    assert "unaccounted_ms=375.0" in line
    assert "images=2" in line
    assert "images_skipped=1" in line
    assert "degraded_phase=router" in line
    assert "error_code=LLMInferenceError" in line


def test_llm_call_metrics_defaults_are_safe() -> None:
    metrics = LLMCallMetrics(operation="route")

    assert metrics.status == "ok"
    assert metrics.images_attached == 0
    assert metrics.payload_bytes == 0
    assert metrics.output_tokens is None
