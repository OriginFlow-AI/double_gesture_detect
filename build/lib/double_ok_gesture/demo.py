"""Webcam demo for double OK gesture recognition."""

from __future__ import annotations

import argparse

from .capture_gate import (
    PromptSpeaker,
    draw_gate_overlay,
    evaluate_capture_gate,
    evaluate_stereo_capture_gate,
    load_glasses_pose,
)
from .config import capture_gate_config, load_config, recognizer_config
from .recognizer import DoubleOKRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime double OK gesture demo.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--left-camera", type=int, default=None, help="Left camera id for binocular input.")
    parser.add_argument("--right-camera", type=int, default=None, help="Right camera id for binocular input.")
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
    parser.add_argument("--headless", action="store_true", help="Print results without opening a window.")
    parser.add_argument("--capture-gate", action="store_true", help="Show pre-capture readiness prompts.")
    parser.add_argument("--require-glasses-pose", action="store_true", help="Require a valid glasses pose input.")
    parser.add_argument("--glasses-pose", default=None, help="JSON file with pitch/roll/yaw in degrees.")
    parser.add_argument("--voice-prompts", action="store_true", help="Speak readiness prompts if espeak is installed.")
    parser.add_argument("--prompt-interval", type=float, default=2.0)
    return parser.parse_args()


def open_camera(camera_id: int, width: int, height: int):
    import cv2

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    return cap


def describe_result(prefix: str, result) -> str:
    return (
        f"{prefix}_hands={len(result.hands)} {prefix}_ok_count={result.ok_count} "
        f"{prefix}_double_ok={result.double_ok} {prefix}_stable={result.stable_double_ok}"
    )


def stack_stereo_frames(left_frame, right_frame):
    import cv2
    import numpy as np

    if left_frame.shape[0] != right_frame.shape[0]:
        target_h = left_frame.shape[0]
        scale = target_h / right_frame.shape[0]
        right_frame = cv2.resize(right_frame, (int(right_frame.shape[1] * scale), target_h))
    return np.hstack([left_frame, right_frame])


def main() -> None:
    import cv2

    args = parse_args()
    cfg = load_config(args.config)
    if args.threshold is not None:
        cfg["ok_threshold"] = args.threshold

    gate_cfg = capture_gate_config(cfg, args.require_glasses_pose)
    speaker = PromptSpeaker(enabled=args.voice_prompts, min_interval_sec=args.prompt_interval)
    left_camera_id = args.left_camera if args.left_camera is not None else args.camera
    stereo_enabled = args.right_camera is not None
    if args.stereo_gate == "both" and not stereo_enabled:
        raise RuntimeError("--stereo-gate both requires --right-camera")

    left_recognizer = None
    right_recognizer = None
    left_cap = None
    right_cap = None
    try:
        left_recognizer = DoubleOKRecognizer(model_path=args.model, **recognizer_config(cfg))
        if stereo_enabled:
            right_recognizer = DoubleOKRecognizer(model_path=args.model, **recognizer_config(cfg))
        left_cap = open_camera(left_camera_id, args.width, args.height)
        if stereo_enabled:
            right_cap = open_camera(args.right_camera, args.width, args.height)

        while True:
            ok, left_frame = left_cap.read()
            if not ok:
                break
            right_frame = None
            if right_cap:
                right_ok, right_frame = right_cap.read()
                if not right_ok:
                    break

            left_result = left_recognizer.process_bgr(left_frame)
            right_result = (
                right_recognizer.process_bgr(right_frame)
                if right_recognizer and right_frame is not None
                else None
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

            if args.headless:
                line = describe_result("left", left_result)
                if right_result:
                    line += " " + describe_result("right", right_result)
                if decision:
                    line += f" gate_ready={decision.ready} reason={decision.reason.value}"
                print(line)
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
                        (24, 122),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        color,
                        2,
                    )
            else:
                frame = left_frame

            cv2.imshow("double_ok_gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if left_cap:
            left_cap.release()
        if right_cap:
            right_cap.release()
        if left_recognizer:
            left_recognizer.close()
        if right_recognizer:
            right_recognizer.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
