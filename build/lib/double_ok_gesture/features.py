"""Hand landmark features for OK gesture classification.

The package uses 21 MediaPipe-style hand landmarks. Features are deliberately
simple: normalized landmark coordinates plus a small set of geometry distances
and finger extension scores. This keeps training fast and works well for OK vs
not-OK classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_CHAINS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}


@dataclass(frozen=True)
class GeometryScores:
    pinch: float
    middle_extension: float
    ring_extension: float
    pinky_extension: float
    open_finger_mean: float
    ok_score: float


def as_landmark_array(landmarks: Iterable[object] | np.ndarray) -> np.ndarray:
    """Convert MediaPipe landmarks, dicts, tuples, or arrays to shape (21, 3)."""

    if isinstance(landmarks, np.ndarray):
        arr = landmarks.astype(np.float32, copy=False)
    else:
        rows = []
        for item in landmarks:
            if hasattr(item, "x") and hasattr(item, "y"):
                rows.append([float(item.x), float(item.y), float(getattr(item, "z", 0.0))])
            elif isinstance(item, dict):
                rows.append([float(item["x"]), float(item["y"]), float(item.get("z", 0.0))])
            else:
                values = list(item)  # type: ignore[arg-type]
                if len(values) == 2:
                    values.append(0.0)
                rows.append([float(values[0]), float(values[1]), float(values[2])])
        arr = np.asarray(rows, dtype=np.float32)

    if arr.shape == (21, 2):
        zeros = np.zeros((21, 1), dtype=np.float32)
        arr = np.concatenate([arr, zeros], axis=1)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 2/3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Landmarks must contain only finite values")
    return arr


def normalize_landmarks(
    landmarks: Iterable[object] | np.ndarray,
    handedness: str | None = None,
) -> np.ndarray:
    """Translate to wrist, scale by palm size, and mirror right hands."""

    points = as_landmark_array(landmarks).copy()
    points -= points[WRIST]

    palm_refs = [
        np.linalg.norm(points[MIDDLE_MCP]),
        np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP]),
        np.max(np.linalg.norm(points[:, :2], axis=1)),
    ]
    scale = max(float(x) for x in palm_refs if np.isfinite(x))
    if scale < 1e-6:
        scale = 1.0
    points /= scale

    if handedness and handedness.lower().startswith("right"):
        points[:, 0] *= -1.0
    return points.astype(np.float32, copy=False)


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    ab = points[a] - points[b]
    cb = points[c] - points[b]
    denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return math.acos(cosine) / math.pi


def finger_extension(points: np.ndarray, finger: str) -> float:
    """Return a soft extension score for one finger in normalized space."""

    mcp, pip, dip, tip = FINGER_CHAINS[finger]
    tip_gain = _distance(points, tip, WRIST) - _distance(points, pip, WRIST)
    straightness = (_angle(points, mcp, pip, dip) + _angle(points, pip, dip, tip)) * 0.5
    return float(np.clip(0.65 * _sigmoid(8.0 * tip_gain) + 0.35 * straightness, 0.0, 1.0))


def geometry_scores(
    landmarks: Iterable[object] | np.ndarray,
    handedness: str | None = None,
) -> GeometryScores:
    points = normalize_landmarks(landmarks, handedness)

    pinch = _distance(points, THUMB_TIP, INDEX_TIP)
    pinch_score = math.exp(-((pinch / 0.28) ** 2))

    middle_ext = finger_extension(points, "middle")
    ring_ext = finger_extension(points, "ring")
    pinky_ext = finger_extension(points, "pinky")
    open_mean = (middle_ext + ring_ext + pinky_ext) / 3.0

    index_thumb_mcp_gap = _distance(points, THUMB_MCP, INDEX_MCP)
    circle_gap_score = math.exp(-((pinch / max(index_thumb_mcp_gap, 0.15)) ** 2))

    ok_score = 0.58 * pinch_score + 0.30 * open_mean + 0.12 * circle_gap_score
    return GeometryScores(
        pinch=pinch,
        middle_extension=middle_ext,
        ring_extension=ring_ext,
        pinky_extension=pinky_ext,
        open_finger_mean=open_mean,
        ok_score=float(np.clip(ok_score, 0.0, 1.0)),
    )


def rule_ok_score(
    landmarks: Iterable[object] | np.ndarray,
    handedness: str | None = None,
) -> float:
    """Rule-based OK score used before a trained classifier exists."""

    return geometry_scores(landmarks, handedness).ok_score


def feature_vector(
    landmarks: Iterable[object] | np.ndarray,
    handedness: str | None = None,
) -> np.ndarray:
    """Build a numeric vector for ML classifiers."""

    points = normalize_landmarks(landmarks, handedness)
    flat = points.reshape(-1)

    distances = [
        _distance(points, THUMB_TIP, INDEX_TIP),
        _distance(points, THUMB_TIP, MIDDLE_TIP),
        _distance(points, INDEX_TIP, MIDDLE_TIP),
        _distance(points, INDEX_TIP, RING_TIP),
        _distance(points, INDEX_TIP, PINKY_TIP),
        _distance(points, THUMB_MCP, INDEX_MCP),
        _distance(points, INDEX_MCP, PINKY_MCP),
    ]
    tip_distances = [
        _distance(points, THUMB_TIP, WRIST),
        _distance(points, INDEX_TIP, WRIST),
        _distance(points, MIDDLE_TIP, WRIST),
        _distance(points, RING_TIP, WRIST),
        _distance(points, PINKY_TIP, WRIST),
    ]
    extensions = [finger_extension(points, name) for name in FINGER_CHAINS]
    angles = []
    for mcp, pip, dip, tip in FINGER_CHAINS.values():
        angles.extend([_angle(points, mcp, pip, dip), _angle(points, pip, dip, tip)])

    geom = geometry_scores(points)
    extras = np.asarray(
        distances
        + tip_distances
        + extensions
        + angles
        + [
            geom.pinch,
            geom.middle_extension,
            geom.ring_extension,
            geom.pinky_extension,
            geom.open_finger_mean,
            geom.ok_score,
        ],
        dtype=np.float32,
    )
    return np.concatenate([flat.astype(np.float32), extras], axis=0)


def feature_names() -> list[str]:
    names = []
    for i in range(21):
        names.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
    extra_count = len(feature_vector(np.zeros((21, 3), dtype=np.float32))) - len(names)
    names.extend([f"geom_{i}" for i in range(extra_count)])
    return names
