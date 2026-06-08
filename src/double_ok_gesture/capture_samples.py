"""Capture local validation images from a camera."""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path

from .camera import CameraSettings, format_camera_source, open_camera
from .capture_gate import (
    PromptSpeaker,
    draw_gate_overlay,
    evaluate_labeled_capture_gate,
    load_glasses_pose,
)
from .config import capture_gate_config, load_config, recognizer_config
from .recognizer import DoubleOKRecognizer
from .runtime import RuntimeMetrics, configure_logging, draw_runtime_overlay

LOGGER = logging.getLogger(__name__)


def save_frame(frame, label_dir: Path) -> Path:
    import cv2

    label_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    path = label_dir / name
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Failed to write captured frame: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture local gesture images for validation.")
    parser.add_argument("--camera", default="/dev/video0", help="Camera index or device path.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--open-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--warmup-reads", type=int, default=5)
    parser.add_argument("--read-failure-limit", type=int, default=5)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--label", required=True, choices=["double_ok", "not_double_ok"])
    parser.add_argument("--output-dir", default="data/raw/local_validation")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", default=None, help="Optional trained model path for gated capture.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gate", action="store_true", help="Require valid pose, framing, spacing, and label.")
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="Save frames automatically while the label-specific gate is ready.",
    )
    parser.add_argument("--auto-capture-cooldown", type=float, default=1.0)
    parser.add_argument("--require-glasses-pose", action="store_true", help="Require a valid glasses pose input.")
    parser.add_argument("--glasses-pose", default=None, help="JSON file with pitch/roll/yaw in degrees.")
    parser.add_argument("--voice-prompts", action="store_true", help="Speak readiness prompts if espeak is installed.")
    parser.add_argument("--prompt-interval", type=float, default=2.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    import cv2

    args = parse_args()
    configure_logging(args.log_level)
    if args.auto_capture and not args.gate:
        raise ValueError("--auto-capture requires --gate")
    if not math.isfinite(args.auto_capture_cooldown) or args.auto_capture_cooldown < 0.0:
        raise ValueError("--auto-capture-cooldown must be finite and non-negative")
    if not math.isfinite(args.target_fps) or args.target_fps < 0.0:
        raise ValueError("--target-fps must be finite and non-negative")

    label_dir = Path(args.output_dir) / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg["ok_threshold"] = args.threshold

    gate_cfg = capture_gate_config(cfg, args.require_glasses_pose)
    speaker = PromptSpeaker(enabled=args.voice_prompts, min_interval_sec=args.prompt_interval)
    last_auto_capture = 0.0
    settings = CameraSettings(
        source=args.camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        fourcc=args.fourcc,
        open_retries=args.open_retries,
        retry_delay_sec=args.retry_delay,
        warmup_reads=args.warmup_reads,
        read_failure_limit=args.read_failure_limit,
    )

    recognizer = None
    camera = None
    metrics = RuntimeMetrics()
    try:
        camera = open_camera(settings)
        if args.gate:
            recognizer = DoubleOKRecognizer(model_path=args.model, **recognizer_config(cfg))

        LOGGER.info("Press SPACE to save a frame; press Q or ESC to quit")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            frame_started = time.monotonic()
            raw_frame = frame.copy()
            decision = None
            if recognizer:
                result = recognizer.process_bgr(frame)
                decision = evaluate_labeled_capture_gate(
                    result,
                    args.label,
                    gate_cfg,
                    load_glasses_pose(args.glasses_pose),
                )
                speaker.emit(decision.prompt)
                recognizer.draw(frame, result)
                draw_gate_overlay(frame, decision, gate_cfg)
                if args.auto_capture and decision.ready:
                    now = time.monotonic()
                    if now - last_auto_capture >= args.auto_capture_cooldown:
                        path = save_frame(raw_frame, label_dir)
                        last_auto_capture = now
                        print(path, flush=True)

            snapshot = metrics.update(frame_started)
            h, _w = frame.shape[:2]
            cv2.putText(
                frame,
                f"LABEL:{args.label}",
                (24, max(24, h - 48)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 220, 40),
                2,
            )
            draw_runtime_overlay(
                frame,
                snapshot,
                f"{format_camera_source(camera.info.source)} {camera.info.width}x{camera.info.height}",
                target_fps=args.target_fps,
            )
            cv2.imshow("capture_samples", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if decision and not decision.ready:
                    speaker.emit(decision.prompt)
                    continue
                path = save_frame(raw_frame, label_dir)
                print(path, flush=True)
    finally:
        if camera:
            camera.close()
        if recognizer:
            recognizer.close()
        cv2.destroyAllWindows()
        LOGGER.info("Sample capture stopped")


if __name__ == "__main__":
    main()
