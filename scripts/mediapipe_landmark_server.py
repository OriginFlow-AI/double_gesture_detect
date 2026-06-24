#!/usr/bin/env python3
"""Length-prefixed MediaPipe Hands sidecar for the C++ live demo."""

from __future__ import annotations

import json
import struct
import sys

import cv2
import mediapipe as mp
import numpy as np


def _read_exact(size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _handedness_labels(result: object) -> list[str]:
    labels: list[str] = []
    multi_handedness = getattr(result, "multi_handedness", None)
    if not multi_handedness:
        return labels
    for item in multi_handedness:
        classifications = getattr(item, "classification", [])
        if classifications:
            labels.append(classifications[0].label)
    return labels


def _detect(hands: object, jpeg_bytes: bytes) -> dict[str, object]:
    encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {"hands": [], "error": "decode_failed"}

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    result = hands.process(frame_rgb)
    labels = _handedness_labels(result)
    detected: list[dict[str, object]] = []

    if result.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label = labels[index] if index < len(labels) else "Unknown"
            detected.append(
                {
                    "handedness": label,
                    "landmarks": [
                        {"x": lm.x, "y": lm.y, "z": lm.z}
                        for lm in hand_landmarks.landmark
                    ],
                }
            )
    return {"hands": detected}


def main() -> int:
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    print(json.dumps({"ready": True}, separators=(",", ":")), flush=True)
    try:
        while True:
            header = _read_exact(4)
            if header is None:
                break
            (length,) = struct.unpack("<I", header)
            if length == 0:
                print(json.dumps({"hands": []}, separators=(",", ":")), flush=True)
                continue
            payload = _read_exact(length)
            if payload is None:
                break
            print(json.dumps(_detect(hands, payload), separators=(",", ":")), flush=True)
    finally:
        hands.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
