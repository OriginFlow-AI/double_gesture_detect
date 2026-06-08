"""Capture small local validation sets from a webcam."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from .capture_gate import (
    PromptSpeaker,
    draw_gate_overlay,
    evaluate_capture_gate,
    load_glasses_pose,
)
from .config import capture_gate_config, load_config, recognizer_config
from .recognizer import DoubleOKRecognizer


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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--label", required=True, choices=["double_ok", "not_double_ok"])
    parser.add_argument("--output-dir", default="data/raw/local_validation")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", default=None, help="Optional trained model path for gated capture.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gate", action="store_true", help="Only capture after glasses/FOV/double-OK checks pass.")
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="Save frames automatically while the gate is ready.",
    )
    parser.add_argument("--auto-capture-cooldown", type=float, default=1.0)
    parser.add_argument("--require-glasses-pose", action="store_true", help="Require a valid glasses pose input.")
    parser.add_argument("--glasses-pose", default=None, help="JSON file with pitch/roll/yaw in degrees.")
    parser.add_argument("--voice-prompts", action="store_true", help="Speak readiness prompts if espeak is installed.")
    parser.add_argument("--prompt-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    import cv2

    args = parse_args()
    if args.auto_capture and not args.gate:
        raise ValueError("--auto-capture requires --gate")
    if args.auto_capture_cooldown < 0.0:
        raise ValueError("--auto-capture-cooldown must be non-negative")

    label_dir = Path(args.output_dir) / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg["ok_threshold"] = args.threshold

    gate_cfg = capture_gate_config(cfg, args.require_glasses_pose)
    speaker = PromptSpeaker(enabled=args.voice_prompts, min_interval_sec=args.prompt_interval)
    last_auto_capture = 0.0

    recognizer = None
    cap = None
    try:
        if args.gate:
            recognizer = DoubleOKRecognizer(model_path=args.model, **recognizer_config(cfg))
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}")

        print("Press SPACE to save a frame, q or ESC to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_frame = frame.copy()
            decision = None
            if recognizer:
                result = recognizer.process_bgr(frame)
                decision = evaluate_capture_gate(result, gate_cfg, load_glasses_pose(args.glasses_pose))
                speaker.emit(decision.prompt)
                recognizer.draw(frame, result)
                draw_gate_overlay(frame, decision, gate_cfg)
                if args.auto_capture and decision.ready:
                    now = time.monotonic()
                    if now - last_auto_capture >= args.auto_capture_cooldown:
                        path = save_frame(raw_frame, label_dir)
                        last_auto_capture = now
                        print(path)

            cv2.putText(frame, args.label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 220, 40), 2)
            cv2.imshow("capture_samples", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if decision and not decision.ready:
                    speaker.emit(decision.prompt)
                    continue
                path = save_frame(raw_frame, label_dir)
                print(path)
    finally:
        if cap:
            cap.release()
        if recognizer:
            recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
