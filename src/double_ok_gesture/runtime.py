"""Runtime logging, frame-rate measurement, and display helpers."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuntimeSnapshot:
    fps: float
    processing_ms: float
    frame_count: int


class RuntimeMetrics:
    def __init__(self, window_size: int = 60) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._frame_count = 0

    def update(
        self,
        frame_started: float,
        frame_finished: float | None = None,
    ) -> RuntimeSnapshot:
        finished = time.monotonic() if frame_finished is None else frame_finished
        if not math.isfinite(frame_started) or not math.isfinite(finished):
            raise ValueError("frame timestamps must be finite")
        if finished < frame_started:
            raise ValueError("frame_finished must not precede frame_started")

        self._timestamps.append(finished)
        self._frame_count += 1
        fps = 0.0
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0.0:
                fps = (len(self._timestamps) - 1) / elapsed
        return RuntimeSnapshot(
            fps=fps,
            processing_ms=(finished - frame_started) * 1000.0,
            frame_count=self._frame_count,
        )


def configure_logging(level: str = "INFO") -> None:
    normalized = level.upper()
    numeric_level = getattr(logging, normalized, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def draw_runtime_overlay(
    frame_bgr: np.ndarray,
    snapshot: RuntimeSnapshot,
    camera_label: str,
    *,
    target_fps: float = 0.0,
) -> np.ndarray:
    import cv2

    h, _w = frame_bgr.shape[:2]
    fps_ok = target_fps <= 0.0 or snapshot.frame_count < 10 or snapshot.fps >= target_fps
    color = (220, 220, 220) if fps_ok else (40, 160, 255)
    label = f"FPS:{snapshot.fps:5.1f}  PROC:{snapshot.processing_ms:5.1f}ms  CAM:{camera_label}"
    cv2.putText(
        frame_bgr,
        label,
        (24, max(24, h - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    return frame_bgr
