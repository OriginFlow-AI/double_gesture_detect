"""Realtime camera demo for double OK gesture recognition."""

from __future__ import annotations

import argparse
import logging
import math
import time

from .camera import (
    CameraSettings,
    CameraStream,
    format_camera_source,
    format_video_devices,
    open_camera,
)
from .capture_gate import (
    PromptSpeaker,
    draw_gate_overlay,
    evaluate_capture_gate,
    evaluate_stereo_capture_gate,
    load_glasses_pose,
)
from .config import capture_gate_config, load_config, recognizer_config
from .recognizer import DoubleOKRecognizer
from .runtime import RuntimeMetrics, configure_logging, draw_runtime_overlay

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime double OK gesture demo.")
    parser.add_argument("--camera", default="/dev/video0", help="Camera index or device path.")
    parser.add_argument("--left-camera", default=None, help="Left camera index or path for binocular input.")
    parser.add_argument("--right-camera", default=None, help="Right camera index or path for binocular input.")
    parser.add_argument(
        "--stereo-gate",
        choices=["left", "both"],
        default="left",
        help="Use the left view as gate authority, or require both stereo views to pass.",
    )
    parser.add_argument("--model", default=None, help="Optional trained model path.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--open-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--warmup-reads", type=int, default=5)
    parser.add_argument("--read-failure-limit", type=int, default=5)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true", help="Print results without opening a window.")
    parser.add_argument("--list-cameras", action="store_true", help="List Linux video devices and exit.")
    parser.add_argument("--capture-gate", action="store_true", help="Show pre-capture readiness prompts.")
    parser.add_argument("--require-glasses-pose", action="store_true", help="Require a valid glasses pose input.")
    parser.add_argument("--glasses-pose", default=None, help="JSON file with pitch/roll/yaw in degrees.")
    parser.add_argument("--voice-prompts", action="store_true", help="Speak readiness prompts if espeak is installed.")
    parser.add_argument("--prompt-interval", type=float, default=2.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def camera_settings(args: argparse.Namespace, source: str | int) -> CameraSettings:
    return CameraSettings(
        source=source,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        fourcc=args.fourcc,
        open_retries=args.open_retries,
        retry_delay_sec=args.retry_delay,
        warmup_reads=args.warmup_reads,
        read_failure_limit=args.read_failure_limit,
    )


def describe_result(prefix: str, result) -> str:
    scores = ",".join(f"{hand.ok_score:.3f}" for hand in result.hands) or "-"
    return (
        f"{prefix}_hands={len(result.hands)} {prefix}_ok_count={result.ok_count} "
        f"{prefix}_scores={scores} {prefix}_double_ok={result.double_ok} "
        f"{prefix}_stable={result.stable_double_ok}"
    )


def stack_stereo_frames(left_frame, right_frame):
    import cv2
    import numpy as np

    if left_frame.shape[0] != right_frame.shape[0]:
        target_h = left_frame.shape[0]
        scale = target_h / right_frame.shape[0]
        right_frame = cv2.resize(right_frame, (int(right_frame.shape[1] * scale), target_h))
    return np.hstack([left_frame, right_frame])


def _camera_overlay_label(stream: CameraStream) -> str:
    return f"{format_camera_source(stream.info.source)} {stream.info.width}x{stream.info.height}"


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    if args.list_cameras:
        print(format_video_devices())
        return
    if not math.isfinite(args.target_fps) or args.target_fps < 0.0:
        raise ValueError("--target-fps must be finite and non-negative")
    if not math.isfinite(args.status_interval) or args.status_interval < 0.0:
        raise ValueError("--status-interval must be finite and non-negative")

    import cv2

    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg["ok_threshold"] = args.threshold

    gate_cfg = capture_gate_config(cfg, args.require_glasses_pose)
    speaker = PromptSpeaker(enabled=args.voice_prompts, min_interval_sec=args.prompt_interval)
    left_camera_source = args.left_camera if args.left_camera is not None else args.camera
    stereo_enabled = args.right_camera is not None
    if args.stereo_gate == "both" and not stereo_enabled:
        raise RuntimeError("--stereo-gate both requires --right-camera")

    left_recognizer = None
    right_recognizer = None
    left_camera = None
    right_camera = None
    metrics = RuntimeMetrics()
    last_status_time = 0.0
    try:
        left_camera = open_camera(camera_settings(args, left_camera_source))
        if stereo_enabled:
            right_camera = open_camera(camera_settings(args, args.right_camera))

        left_recognizer = DoubleOKRecognizer(model_path=args.model, **recognizer_config(cfg))
        if stereo_enabled:
            right_recognizer = DoubleOKRecognizer(model_path=args.model, **recognizer_config(cfg))

        while True:
            left_frame = left_camera.read()
            if left_frame is None:
                continue
            right_frame = None
            if right_camera:
                right_frame = right_camera.read()
                if right_frame is None:
                    continue

            frame_started = time.monotonic()
            left_result = left_recognizer.process_bgr(left_frame)
            right_result = (
                right_recognizer.process_bgr(right_frame) if right_recognizer and right_frame is not None else None
            )
            decision = None
            if args.capture_gate:
                pose = load_glasses_pose(args.glasses_pose)
                if stereo_enabled:
                    decision = evaluate_stereo_capture_gate(
                        left_result,
                        right_result,
                        gate_cfg,
                        pose,
                        mode=args.stereo_gate,
                    )
                else:
                    decision = evaluate_capture_gate(left_result, gate_cfg, pose)
                speaker.emit(decision.prompt)

            snapshot = metrics.update(frame_started)
            if args.headless:
                now = time.monotonic()
                if args.status_interval == 0.0 or now - last_status_time >= args.status_interval:
                    line = describe_result("left", left_result)
                    if right_result:
                        line += " " + describe_result("right", right_result)
                    if decision:
                        line += f" gate_ready={decision.ready} reason={decision.reason.value}"
                    line += f" fps={snapshot.fps:.1f} processing_ms={snapshot.processing_ms:.1f}"
                    print(line, flush=True)
                    last_status_time = now
                continue

            left_recognizer.draw(left_frame, left_result)
            if decision:
                left_decision = decision.left if stereo_enabled else decision
                draw_gate_overlay(left_frame, left_decision, gate_cfg)

            if stereo_enabled and right_frame is not None and right_result is not None:
                right_recognizer.draw(right_frame, right_result)
                if decision and decision.right:
                    draw_gate_overlay(right_frame, decision.right, gate_cfg)
                frame = stack_stereo_frames(left_frame, right_frame)
                if decision:
                    color = (40, 200, 40) if decision.ready else (40, 160, 255)
                    cv2.putText(
                        frame,
                        f"STEREO_GATE:{decision.reason.value}",
                        (24, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        2,
                    )
                camera_label = f"{_camera_overlay_label(left_camera)} + {_camera_overlay_label(right_camera)}"
            else:
                frame = left_frame
                camera_label = _camera_overlay_label(left_camera)

            draw_runtime_overlay(
                frame,
                snapshot,
                camera_label,
                target_fps=args.target_fps,
            )
            cv2.imshow("double_ok_gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if left_camera:
            left_camera.close()
        if right_camera:
            right_camera.close()
        if left_recognizer:
            left_recognizer.close()
        if right_recognizer:
            right_recognizer.close()
        if not args.headless:
            cv2.destroyAllWindows()
        LOGGER.info("Demo stopped")


if __name__ == "__main__":
    main()
