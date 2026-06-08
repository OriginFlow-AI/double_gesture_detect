"""Shared configuration loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .capture_gate import CaptureGateConfig

RECOGNIZER_CONFIG_KEYS = frozenset(
    {
        "max_num_hands",
        "ok_threshold",
        "stable_window",
        "stable_min_positive",
        "min_detection_confidence",
        "min_tracking_confidence",
    }
)


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    return config


def recognizer_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in RECOGNIZER_CONFIG_KEYS if key in config}


def capture_gate_config(
    config: dict[str, Any],
    require_glasses_pose: bool = False,
) -> CaptureGateConfig:
    raw_gate_config = config.get("capture_gate", {})
    if not isinstance(raw_gate_config, dict):
        raise ValueError("capture_gate config must be a JSON object")

    gate_config = dict(raw_gate_config)
    if require_glasses_pose:
        gate_config["require_glasses_pose"] = True
    return CaptureGateConfig(**gate_config)
