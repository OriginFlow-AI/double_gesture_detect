"""Model loading helpers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    metadata: dict[str, Any]


def load_model(path: str | Path | None) -> Any | None:
    if path is None:
        return None
    return load_model_artifact(path).model


def load_model_artifact(path: str | Path) -> ModelArtifact:
    """Load a trusted local model artifact.

    Joblib and pickle files can execute code while loading. Only load artifacts
    produced by this project or another trusted source.
    """

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    try:
        import joblib
    except ModuleNotFoundError:
        import pickle

        with model_path.open("rb") as f:
            loaded = pickle.load(f)
    else:
        loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        metadata = loaded.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Model metadata must be a dictionary: {model_path}")
        return ModelArtifact(model=loaded["model"], metadata=dict(metadata))
    return ModelArtifact(model=loaded, metadata={})


def save_model(path: str | Path, model: Any, metadata: dict[str, Any] | None = None) -> None:
    payload = {"model": model, "metadata": metadata or {}}
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=model_path.parent,
        prefix=f".{model_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        try:
            import joblib
        except ModuleNotFoundError:
            import pickle

            with temporary_path.open("wb") as file:
                pickle.dump(payload, file)
        else:
            joblib.dump(payload, temporary_path)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
