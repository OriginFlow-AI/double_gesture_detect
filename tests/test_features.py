import numpy as np
import pytest

from double_ok_gesture.features import feature_vector, geometry_scores, rule_ok_score


def make_ok_landmarks():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]
    pts[1] = [0.35, -0.35, 0.0]
    pts[2] = [0.28, -0.52, 0.0]
    pts[3] = [0.18, -0.62, 0.0]
    pts[4] = [0.08, -0.60, 0.0]
    pts[5] = [-0.35, -0.75, 0.0]
    pts[6] = [-0.20, -0.65, 0.0]
    pts[7] = [-0.05, -0.60, 0.0]
    pts[8] = [0.08, -0.60, 0.0]
    pts[9] = [-0.05, -1.00, 0.0]
    pts[10] = [-0.08, -1.65, 0.0]
    pts[11] = [-0.10, -2.25, 0.0]
    pts[12] = [-0.12, -2.90, 0.0]
    pts[13] = [0.22, -0.92, 0.0]
    pts[14] = [0.30, -1.50, 0.0]
    pts[15] = [0.36, -2.02, 0.0]
    pts[16] = [0.42, -2.55, 0.0]
    pts[17] = [0.48, -0.80, 0.0]
    pts[18] = [0.60, -1.22, 0.0]
    pts[19] = [0.70, -1.58, 0.0]
    pts[20] = [0.80, -1.95, 0.0]
    return pts


def make_open_palm_landmarks():
    pts = make_ok_landmarks()
    pts[4] = [0.75, -0.85, 0.0]
    pts[8] = [-0.38, -2.45, 0.0]
    return pts


def test_feature_vector_has_stable_shape():
    vector = feature_vector(make_ok_landmarks(), "Left")
    assert vector.ndim == 1
    assert vector.shape[0] > 80
    assert np.isfinite(vector).all()


def test_rule_score_prefers_ok_over_open_palm():
    ok_score = rule_ok_score(make_ok_landmarks(), "Left")
    palm_score = rule_ok_score(make_open_palm_landmarks(), "Left")
    assert ok_score > 0.65
    assert palm_score < ok_score


def test_geometry_scores_are_bounded():
    scores = geometry_scores(make_ok_landmarks(), "Right")
    assert 0.0 <= scores.ok_score <= 1.0
    assert scores.open_finger_mean > 0.5


def test_feature_vector_rejects_non_finite_landmarks():
    landmarks = make_ok_landmarks()
    landmarks[4, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        feature_vector(landmarks)
