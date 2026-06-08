import pytest

from double_ok_gesture.model_io import load_model, load_model_artifact, save_model


def test_load_model_raises_for_explicit_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pkl")


def test_model_artifact_round_trip_preserves_metadata(tmp_path):
    path = tmp_path / "nested" / "model.pkl"
    model = {"weights": [1, 2, 3]}

    save_model(path, model, {"feature_columns": ["a", "b"]})
    artifact = load_model_artifact(path)

    assert artifact.model == model
    assert artifact.metadata == {"feature_columns": ["a", "b"]}
    assert load_model(path) == model
