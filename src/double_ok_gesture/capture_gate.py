"""Pre-capture gate for glasses-side double OK collection."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .recognizer import DoubleOKResult


class GateReason(str, Enum):
    READY = "ready"
    GLASSES_POSE_MISSING = "glasses_pose_missing"
    GLASSES_POSE_BAD = "glasses_pose_bad"
    NEED_TWO_HANDS = "need_two_hands"
    HANDS_OUT_OF_FRAME = "hands_out_of_frame"
    HANDS_NOT_CENTERED = "hands_not_centered"
    HANDS_TOO_CLOSE = "hands_too_close"
    NEED_DOUBLE_OK = "need_double_ok"
    AVOID_DOUBLE_OK = "avoid_double_ok"


class StereoGateMode(str, Enum):
    LEFT = "left"
    BOTH = "both"


@dataclass(frozen=True)
class GlassesPose:
    """Glasses pose angles in degrees from the glasses IMU or device pose stack."""

    pitch: float | None = None
    roll: float | None = None
    yaw: float | None = None


@dataclass(frozen=True)
class CaptureGateConfig:
    """Tunable capture-start constraints in normalized camera coordinates."""

    require_glasses_pose: bool = False
    pitch_min: float = -20.0
    pitch_max: float = 20.0
    roll_min: float = -12.0
    roll_max: float = 12.0
    yaw_min: float = -25.0
    yaw_max: float = 25.0
    frame_margin: float = 0.03
    center_x_min: float = 0.20
    center_x_max: float = 0.80
    center_y_min: float = 0.18
    center_y_max: float = 0.82
    min_hand_separation: float = 0.16
    use_stable_double_ok: bool = True
    require_double_ok: bool = True

    def __post_init__(self) -> None:
        _validate_range("pitch", self.pitch_min, self.pitch_max)
        _validate_range("roll", self.roll_min, self.roll_max)
        _validate_range("yaw", self.yaw_min, self.yaw_max)
        _validate_normalized_range("center_x", self.center_x_min, self.center_x_max)
        _validate_normalized_range("center_y", self.center_y_min, self.center_y_max)
        if not math.isfinite(self.frame_margin) or not 0.0 <= self.frame_margin < 0.5:
            raise ValueError("frame_margin must be finite and in [0.0, 0.5)")
        if not math.isfinite(self.min_hand_separation) or self.min_hand_separation < 0.0:
            raise ValueError("min_hand_separation must be finite and non-negative")
        if not isinstance(self.use_stable_double_ok, bool):
            raise ValueError("use_stable_double_ok must be a boolean")
        if not isinstance(self.require_double_ok, bool):
            raise ValueError("require_double_ok must be a boolean")


@dataclass(frozen=True)
class CaptureGateDecision:
    ready: bool
    reason: GateReason
    prompt: str
    glasses_pose_ok: bool
    hands_centered: bool
    hands_visible: bool
    hands_separated: bool
    double_ok: bool
    gesture_ok: bool
    hand_count: int


@dataclass(frozen=True)
class StereoCaptureGateDecision:
    ready: bool
    reason: GateReason
    prompt: str
    left: CaptureGateDecision
    right: CaptureGateDecision | None = None


PROMPTS = {
    GateReason.READY: "条件满足，开始采集",
    GateReason.GLASSES_POSE_MISSING: "等待眼镜姿态数据",
    GateReason.GLASSES_POSE_BAD: "请调整眼镜角度，保持视野正对双手",
    GateReason.NEED_TWO_HANDS: "请把双手放入相机画面",
    GateReason.HANDS_OUT_OF_FRAME: "请把双手完整放入画面",
    GateReason.HANDS_NOT_CENTERED: "请把双手移到画面中心",
    GateReason.HANDS_TOO_CLOSE: "请将双手分开一些",
    GateReason.NEED_DOUBLE_OK: "请双手分开并做出 OK 手势",
    GateReason.AVOID_DOUBLE_OK: "负样本采集中，请不要同时做双手 OK",
}

OVERLAY_LABELS = {
    GateReason.READY: "READY_TO_CAPTURE",
    GateReason.GLASSES_POSE_MISSING: "WAITING_GLASSES_POSE",
    GateReason.GLASSES_POSE_BAD: "ADJUST_GLASSES_ANGLE",
    GateReason.NEED_TWO_HANDS: "SHOW_TWO_HANDS",
    GateReason.HANDS_OUT_OF_FRAME: "HANDS_OUT_OF_FRAME",
    GateReason.HANDS_NOT_CENTERED: "MOVE_HANDS_TO_CENTER",
    GateReason.HANDS_TOO_CLOSE: "SEPARATE_HANDS",
    GateReason.NEED_DOUBLE_OK: "MAKE_DOUBLE_OK",
    GateReason.AVOID_DOUBLE_OK: "AVOID_DOUBLE_OK",
}


def load_glasses_pose(path: str | Path | None) -> GlassesPose | None:
    if not path:
        return None
    pose_path = Path(path)
    if not pose_path.exists():
        return None
    try:
        with pose_path.open("r", encoding="utf-8") as file:
            data: Any = json.load(file)
        if not isinstance(data, dict):
            return None
        return GlassesPose(
            pitch=_optional_float(data.get("pitch")),
            roll=_optional_float(data.get("roll")),
            yaw=_optional_float(data.get("yaw")),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        # The glasses process may be replacing this file while a frame is read.
        return None


def evaluate_capture_gate(
    result: DoubleOKResult,
    config: CaptureGateConfig | None = None,
    glasses_pose: GlassesPose | None = None,
) -> CaptureGateDecision:
    cfg = config or CaptureGateConfig()
    pose_ok, pose_reason = _check_glasses_pose(glasses_pose, cfg)
    hand_count = len(result.hands)
    hands_visible = False
    hands_centered = False
    hands_separated = False
    double_ok = result.double_ok and (result.stable_double_ok if cfg.use_stable_double_ok else True)

    if not pose_ok:
        return _decision(
            pose_reason,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )

    if hand_count < 2:
        return _decision(
            GateReason.NEED_TWO_HANDS,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )

    landmarks = [_landmark_xy(hand.landmarks) for hand in result.hands[:2]]
    if any(points is None for points in landmarks):
        return _decision(
            GateReason.HANDS_OUT_OF_FRAME,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )
    landmarks = [points for points in landmarks if points is not None]
    hands_visible = all(_is_hand_visible(points, cfg) for points in landmarks)
    if not hands_visible:
        return _decision(
            GateReason.HANDS_OUT_OF_FRAME,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )

    centers = np.asarray([points.mean(axis=0) for points in landmarks], dtype=np.float32)
    hands_centered = all(_is_point_centered(center, cfg) for center in centers)
    if not hands_centered:
        return _decision(
            GateReason.HANDS_NOT_CENTERED,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )

    hands_separated = float(np.linalg.norm(centers[0] - centers[1])) >= cfg.min_hand_separation
    if not hands_separated:
        return _decision(
            GateReason.HANDS_TOO_CLOSE,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )

    if cfg.require_double_ok and not double_ok:
        return _decision(
            GateReason.NEED_DOUBLE_OK,
            cfg,
            pose_ok,
            hands_centered,
            hands_visible,
            hands_separated,
            double_ok,
            hand_count,
        )

    return _decision(
        GateReason.READY,
        cfg,
        pose_ok,
        hands_centered,
        hands_visible,
        hands_separated,
        double_ok,
        hand_count,
    )


def evaluate_labeled_capture_gate(
    result: DoubleOKResult,
    label: str,
    config: CaptureGateConfig | None = None,
    glasses_pose: GlassesPose | None = None,
) -> CaptureGateDecision:
    """Evaluate geometry plus the gesture condition required by a sample label."""

    cfg = config or CaptureGateConfig()
    if label == "double_ok":
        return evaluate_capture_gate(result, cfg, glasses_pose)
    if label != "not_double_ok":
        raise ValueError(f"Unsupported capture label: {label}")

    geometry_cfg = replace(cfg, require_double_ok=False)
    geometry_decision = evaluate_capture_gate(result, geometry_cfg, glasses_pose)
    if not geometry_decision.ready:
        return replace(geometry_decision, gesture_ok=not result.double_ok)

    common = (
        geometry_cfg,
        geometry_decision.glasses_pose_ok,
        geometry_decision.hands_centered,
        geometry_decision.hands_visible,
        geometry_decision.hands_separated,
        result.double_ok,
        geometry_decision.hand_count,
    )
    if result.double_ok:
        return _decision(GateReason.AVOID_DOUBLE_OK, *common, gesture_ok=False)
    return _decision(GateReason.READY, *common, gesture_ok=True)


def evaluate_stereo_capture_gate(
    left_result: DoubleOKResult,
    right_result: DoubleOKResult | None = None,
    config: CaptureGateConfig | None = None,
    glasses_pose: GlassesPose | None = None,
    mode: StereoGateMode | str = StereoGateMode.LEFT,
) -> StereoCaptureGateDecision:
    """Combine capture gates for a binocular camera pair.

    LEFT mode keeps the early prototype path simple by using the left image as
    the gate authority. BOTH mode requires both views to satisfy the same gate.
    """

    stereo_mode = StereoGateMode(mode)
    left_decision = evaluate_capture_gate(left_result, config, glasses_pose)
    if stereo_mode == StereoGateMode.LEFT:
        return StereoCaptureGateDecision(
            ready=left_decision.ready,
            reason=left_decision.reason,
            prompt=left_decision.prompt,
            left=left_decision,
        )

    if right_result is None:
        raise ValueError("right_result is required when stereo gate mode is 'both'")

    right_decision = evaluate_capture_gate(right_result, config, glasses_pose)
    if left_decision.ready and right_decision.ready:
        return StereoCaptureGateDecision(
            ready=True,
            reason=GateReason.READY,
            prompt=PROMPTS[GateReason.READY],
            left=left_decision,
            right=right_decision,
        )

    if not left_decision.ready and not right_decision.ready:
        prompt = (
            f"双目: {left_decision.prompt}"
            if left_decision.reason == right_decision.reason
            else f"左眼: {left_decision.prompt}; 右眼: {right_decision.prompt}"
        )
        return StereoCaptureGateDecision(
            ready=False,
            reason=left_decision.reason,
            prompt=prompt,
            left=left_decision,
            right=right_decision,
        )

    blocked_view = "左眼" if not left_decision.ready else "右眼"
    blocked_decision = left_decision if not left_decision.ready else right_decision
    return StereoCaptureGateDecision(
        ready=False,
        reason=blocked_decision.reason,
        prompt=f"{blocked_view}: {blocked_decision.prompt}",
        left=left_decision,
        right=right_decision,
    )


class PromptSpeaker:
    """Throttle repeated prompts and optionally speak them through espeak."""

    def __init__(self, enabled: bool = False, min_interval_sec: float = 2.0) -> None:
        if not math.isfinite(min_interval_sec) or min_interval_sec < 0.0:
            raise ValueError("min_interval_sec must be finite and non-negative")
        self.enabled = enabled
        self.min_interval_sec = min_interval_sec
        self._last_prompt: str | None = None
        self._last_time = 0.0
        self._tts = shutil.which("espeak") if enabled else None

    def emit(self, prompt: str) -> None:
        now = time.monotonic()
        if prompt == self._last_prompt and now - self._last_time < self.min_interval_sec:
            return
        self._last_prompt = prompt
        self._last_time = now
        print(prompt)
        if self._tts:
            subprocess.Popen(
                [self._tts, "-v", "zh", prompt],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def draw_gate_overlay(frame_bgr: np.ndarray, decision: CaptureGateDecision, config: CaptureGateConfig) -> None:
    import cv2

    h, w = frame_bgr.shape[:2]
    x1 = int(config.center_x_min * w)
    x2 = int(config.center_x_max * w)
    y1 = int(config.center_y_min * h)
    y2 = int(config.center_y_max * h)
    color = (40, 200, 40) if decision.ready else (40, 160, 255)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame_bgr, OVERLAY_LABELS[decision.reason], (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    checks = [
        ("POSE", decision.glasses_pose_ok),
        ("VISIBLE", decision.hands_visible),
        ("CENTER", decision.hands_centered),
        ("SEPARATE", decision.hands_separated),
        ("GESTURE", decision.gesture_ok),
    ]
    for index, (name, passed) in enumerate(checks):
        check_color = (40, 200, 40) if passed else (40, 160, 255)
        marker = "OK" if passed else "--"
        cv2.putText(
            frame_bgr,
            f"{name}:{marker}",
            (24, 112 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            check_color,
            1,
        )


def _check_glasses_pose(
    glasses_pose: GlassesPose | None,
    config: CaptureGateConfig,
) -> tuple[bool, GateReason]:
    if not config.require_glasses_pose:
        return True, GateReason.READY
    if glasses_pose is None or any(
        value is None for value in (glasses_pose.pitch, glasses_pose.roll, glasses_pose.yaw)
    ):
        return False, GateReason.GLASSES_POSE_MISSING
    checks = [
        _in_optional_range(glasses_pose.pitch, config.pitch_min, config.pitch_max),
        _in_optional_range(glasses_pose.roll, config.roll_min, config.roll_max),
        _in_optional_range(glasses_pose.yaw, config.yaw_min, config.yaw_max),
    ]
    if all(checks):
        return True, GateReason.READY
    return False, GateReason.GLASSES_POSE_BAD


def _decision(
    reason: GateReason,
    _config: CaptureGateConfig,
    glasses_pose_ok: bool,
    hands_centered: bool,
    hands_visible: bool,
    hands_separated: bool,
    double_ok: bool,
    hand_count: int,
    *,
    gesture_ok: bool | None = None,
) -> CaptureGateDecision:
    if gesture_ok is None:
        gesture_ok = double_ok if _config.require_double_ok else True
    return CaptureGateDecision(
        ready=reason == GateReason.READY,
        reason=reason,
        prompt=PROMPTS[reason],
        glasses_pose_ok=glasses_pose_ok,
        hands_centered=hands_centered,
        hands_visible=hands_visible,
        hands_separated=hands_separated,
        double_ok=double_ok,
        gesture_ok=gesture_ok,
        hand_count=hand_count,
    )


def _is_hand_visible(points: np.ndarray, config: CaptureGateConfig) -> bool:
    margin = config.frame_margin
    return bool(
        np.all(points[:, 0] >= margin)
        and np.all(points[:, 0] <= 1.0 - margin)
        and np.all(points[:, 1] >= margin)
        and np.all(points[:, 1] <= 1.0 - margin)
    )


def _landmark_xy(landmarks: object) -> np.ndarray | None:
    try:
        points = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] != 21 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    if not np.isfinite(points).all():
        return None
    return points


def _is_point_centered(point: np.ndarray, config: CaptureGateConfig) -> bool:
    x, y = float(point[0]), float(point[1])
    return config.center_x_min <= x <= config.center_x_max and config.center_y_min <= y <= config.center_y_max


def _in_optional_range(value: float | None, minimum: float, maximum: float) -> bool:
    if value is None:
        return True
    return minimum <= value <= maximum


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("pose values must be finite")
    return result


def _validate_range(name: str, minimum: float, maximum: float) -> None:
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ValueError(f"{name}_min and {name}_max must be finite and ordered")


def _validate_normalized_range(name: str, minimum: float, maximum: float) -> None:
    _validate_range(name, minimum, maximum)
    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError(f"{name}_min and {name}_max must satisfy 0 <= min < max <= 1")
