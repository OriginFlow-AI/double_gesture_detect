"""Runtime recognizer for two-hand OK gestures."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .features import as_landmark_array, feature_names, feature_vector, rule_ok_score
from .model_io import load_model_artifact


@dataclass
class HandPrediction:
    handedness: str
    ok_score: float
    is_ok: bool
    landmarks: np.ndarray


@dataclass
class DoubleOKResult:
    hands: list[HandPrediction]
    double_ok: bool
    stable_double_ok: bool
    ok_count: int


class OKHandClassifier:
    """OK vs not-OK classifier with rule fallback."""

    def __init__(self, model_path: str | None = None, threshold: float = 0.68) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")

        self.model = None
        if model_path is not None:
            artifact = load_model_artifact(model_path)
            expected_features = artifact.metadata.get("feature_columns")
            if expected_features and list(expected_features) != feature_names():
                raise ValueError("Model feature schema does not match the runtime feature schema")
            self.model = artifact.model
        self.threshold = threshold

    def score(self, landmarks: Iterable[object] | np.ndarray, handedness: str | None = None) -> float:
        if self.model is None:
            return rule_ok_score(landmarks, handedness)

        vector = feature_vector(landmarks, handedness).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vector)[0]
            classes = list(getattr(self.model, "classes_", [0, 1]))
            if 1 in classes:
                score = float(proba[classes.index(1)])
            else:
                score = float(proba[-1])
            return _bounded_score(score)
        if hasattr(self.model, "decision_function"):
            raw = float(self.model.decision_function(vector)[0])
            return _bounded_score(float(1.0 / (1.0 + np.exp(-raw))))
        return _bounded_score(float(self.model.predict(vector)[0]))

    def predict(
        self,
        landmarks: Iterable[object] | np.ndarray,
        handedness: str | None = None,
    ) -> HandPrediction:
        points = as_landmark_array(landmarks)
        score = self.score(points, handedness)
        return HandPrediction(
            handedness=handedness or "Unknown",
            ok_score=score,
            is_ok=score >= self.threshold,
            landmarks=points,
        )


class DoubleOKRecognizer:
    """Detect hands, classify each hand, and decide whether both are OK."""

    def __init__(
        self,
        model_path: str | None = None,
        ok_threshold: float = 0.68,
        max_num_hands: int = 2,
        stable_window: int = 5,
        stable_min_positive: int = 3,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        if max_num_hands < 2:
            raise ValueError("max_num_hands must be at least 2")
        if stable_window < 1:
            raise ValueError("stable_window must be at least 1")
        if not 1 <= stable_min_positive <= stable_window:
            raise ValueError("stable_min_positive must be between 1 and stable_window")
        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")

        self.classifier = OKHandClassifier(model_path=model_path, threshold=ok_threshold)
        self.stable_min_positive = stable_min_positive
        self.history: deque[bool] = deque(maxlen=stable_window)

        import mediapipe as mp

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "DoubleOKRecognizer":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def process_rgb(self, frame_rgb: np.ndarray) -> DoubleOKResult:
        result = self._hands.process(frame_rgb)
        predictions: list[HandPrediction] = []

        if result.multi_hand_landmarks:
            handedness_labels = []
            if result.multi_handedness:
                for item in result.multi_handedness:
                    handedness_labels.append(item.classification[0].label)

            for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = handedness_labels[index] if index < len(handedness_labels) else "Unknown"
                points = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                predictions.append(self.classifier.predict(points, handedness))

        ok_count = sum(1 for hand in predictions if hand.is_ok)
        double_ok = ok_count >= 2
        self.history.append(double_ok)
        stable_double_ok = sum(self.history) >= self.stable_min_positive

        return DoubleOKResult(
            hands=predictions,
            double_ok=double_ok,
            stable_double_ok=stable_double_ok,
            ok_count=ok_count,
        )

    def process_bgr(self, frame_bgr: np.ndarray) -> DoubleOKResult:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.process_rgb(frame_rgb)

    def draw(self, frame_bgr: np.ndarray, result: DoubleOKResult) -> np.ndarray:
        from .live_ui import draw_hand_tracking

        return draw_hand_tracking(frame_bgr, result)


def _bounded_score(score: float) -> float:
    if not np.isfinite(score):
        raise ValueError("Classifier produced a non-finite score")
    return float(np.clip(score, 0.0, 1.0))
