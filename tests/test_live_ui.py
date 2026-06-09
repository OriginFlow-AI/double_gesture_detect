import numpy as np
import pytest

from double_ok_gesture.capture_gate import CaptureGateDecision, GateReason
from double_ok_gesture.live_ui import LiveDashboardRenderer, gate_progress
from double_ok_gesture.recognizer import DoubleOKResult, HandPrediction
from double_ok_gesture.runtime import RuntimeSnapshot


def make_decision(*, ready: bool) -> CaptureGateDecision:
    return CaptureGateDecision(
        ready=ready,
        reason=GateReason.READY if ready else GateReason.NEED_DOUBLE_OK,
        prompt="条件满足，开始采集" if ready else "请双手做出 OK 手势",
        glasses_pose_ok=True,
        hands_centered=True,
        hands_visible=True,
        hands_separated=True,
        double_ok=ready,
        gesture_ok=ready,
        hand_count=2,
    )


def make_result() -> DoubleOKResult:
    landmarks = np.full((21, 3), 0.5, dtype=np.float32)
    hand = HandPrediction("Left", 0.91, True, landmarks)
    return DoubleOKResult(
        hands=[hand],
        double_ok=False,
        stable_double_ok=False,
        ok_count=1,
    )


def test_dashboard_renderer_outputs_requested_size():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    renderer = LiveDashboardRenderer(1280, 720)

    rendered = renderer.render(
        frame,
        make_result(),
        make_decision(ready=False),
        RuntimeSnapshot(fps=29.8, processing_ms=17.2, frame_count=120),
        camera_label="/dev/video0 1280x720",
        target_fps=25.0,
    )

    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8
    assert np.any(rendered)


def test_gate_progress_counts_passed_checks():
    assert gate_progress(make_decision(ready=True)) == (5, 5)
    assert gate_progress(make_decision(ready=False)) == (4, 5)
    assert gate_progress(None) == (0, 5)


def test_dashboard_rejects_tiny_layout():
    with pytest.raises(ValueError, match="at least"):
        LiveDashboardRenderer(800, 500)
