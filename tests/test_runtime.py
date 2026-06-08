import pytest

from double_ok_gesture.runtime import RuntimeMetrics


def test_runtime_metrics_reports_window_fps_and_latency():
    metrics = RuntimeMetrics(window_size=3)

    first = metrics.update(0.90, 1.00)
    second = metrics.update(1.45, 1.50)
    third = metrics.update(1.95, 2.00)

    assert first.fps == 0.0
    assert second.fps == pytest.approx(2.0)
    assert third.fps == pytest.approx(2.0)
    assert third.processing_ms == pytest.approx(50.0)
    assert third.frame_count == 3


def test_runtime_metrics_rejects_reversed_timestamps():
    metrics = RuntimeMetrics()

    with pytest.raises(ValueError, match="must not precede"):
        metrics.update(2.0, 1.0)
