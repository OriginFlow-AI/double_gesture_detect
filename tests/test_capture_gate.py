import numpy as np
import pytest

from double_ok_gesture.capture_gate import (
    CaptureGateConfig,
    GateReason,
    GlassesPose,
    evaluate_capture_gate,
    evaluate_stereo_capture_gate,
    load_glasses_pose,
)
from double_ok_gesture.recognizer import DoubleOKResult, HandPrediction


def make_hand(center_x: float, center_y: float, is_ok: bool = True) -> HandPrediction:
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 0] = center_x + np.linspace(-0.02, 0.02, 21, dtype=np.float32)
    pts[:, 1] = center_y
    return HandPrediction("Left", 0.9 if is_ok else 0.1, is_ok, pts)


def make_result(*hands: HandPrediction, stable: bool = True) -> DoubleOKResult:
    return DoubleOKResult(
        hands=list(hands),
        double_ok=sum(hand.is_ok for hand in hands) >= 2,
        stable_double_ok=stable,
        ok_count=sum(hand.is_ok for hand in hands),
    )


def test_gate_ready_when_pose_fov_and_double_ok_pass():
    result = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5))

    decision = evaluate_capture_gate(
        result,
        CaptureGateConfig(require_glasses_pose=True),
        GlassesPose(pitch=0.0, roll=0.0, yaw=0.0),
    )

    assert decision.ready
    assert decision.reason == GateReason.READY


def test_gate_blocks_when_required_pose_is_missing():
    result = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5))

    decision = evaluate_capture_gate(result, CaptureGateConfig(require_glasses_pose=True))

    assert not decision.ready
    assert decision.reason == GateReason.GLASSES_POSE_MISSING


def test_gate_blocks_when_required_pose_is_incomplete():
    result = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5))

    decision = evaluate_capture_gate(
        result,
        CaptureGateConfig(require_glasses_pose=True),
        GlassesPose(pitch=0.0, roll=None, yaw=0.0),
    )

    assert not decision.ready
    assert decision.reason == GateReason.GLASSES_POSE_MISSING


def test_gate_blocks_when_hands_are_not_centered():
    result = make_result(make_hand(0.08, 0.5), make_hand(0.6, 0.5))

    decision = evaluate_capture_gate(result)

    assert not decision.ready
    assert decision.reason == GateReason.HANDS_NOT_CENTERED


def test_gate_blocks_until_double_ok_is_stable():
    result = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5), stable=False)

    decision = evaluate_capture_gate(result)

    assert not decision.ready
    assert decision.reason == GateReason.NEED_DOUBLE_OK


def test_stereo_gate_left_mode_uses_left_view_only():
    left = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5))
    right = make_result(make_hand(0.08, 0.5), make_hand(0.6, 0.5))

    decision = evaluate_stereo_capture_gate(left, right, mode="left")

    assert decision.ready
    assert decision.reason == GateReason.READY
    assert decision.right is None


def test_stereo_gate_both_mode_requires_right_view():
    left = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5))
    right = make_result(make_hand(0.08, 0.5), make_hand(0.6, 0.5))

    decision = evaluate_stereo_capture_gate(left, right, mode="both")

    assert not decision.ready
    assert decision.reason == GateReason.HANDS_NOT_CENTERED
    assert decision.right is not None
    assert decision.right.reason == GateReason.HANDS_NOT_CENTERED


def test_stereo_gate_both_mode_ready_when_both_views_pass():
    left = make_result(make_hand(0.4, 0.5), make_hand(0.6, 0.5))
    right = make_result(make_hand(0.42, 0.5), make_hand(0.62, 0.5))

    decision = evaluate_stereo_capture_gate(left, right, mode="both")

    assert decision.ready
    assert decision.reason == GateReason.READY


def test_load_glasses_pose_returns_none_for_transient_invalid_json(tmp_path):
    pose_path = tmp_path / "pose.json"
    pose_path.write_text('{"pitch": 0,', encoding="utf-8")

    assert load_glasses_pose(pose_path) is None


def test_capture_gate_config_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="center_x"):
        CaptureGateConfig(center_x_min=0.8, center_x_max=0.2)
