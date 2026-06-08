import pytest

from double_ok_gesture.recognizer import DoubleOKRecognizer, OKHandClassifier


def test_classifier_rejects_missing_explicit_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        OKHandClassifier(model_path=str(tmp_path / "missing.pkl"))


def test_recognizer_rejects_impossible_stability_config():
    with pytest.raises(ValueError, match="stable_min_positive"):
        DoubleOKRecognizer(stable_window=3, stable_min_positive=4)


def test_classifier_predict_accepts_landmark_dictionaries():
    landmarks = [
        {"x": index * 0.01, "y": index * -0.02, "z": 0.0}
        for index in range(21)
    ]

    prediction = OKHandClassifier().predict(landmarks, "Left")

    assert prediction.landmarks.shape == (21, 3)
    assert 0.0 <= prediction.ok_score <= 1.0
