import json

import pytest

from double_ok_gesture.config import capture_gate_config, load_config, recognizer_config


def test_load_config_requires_json_object(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_config(path)


def test_config_helpers_filter_recognizer_and_build_gate_options():
    config = {
        "ok_threshold": 0.7,
        "ignored": "value",
        "capture_gate": {"min_hand_separation": 0.2},
    }

    assert recognizer_config(config) == {"ok_threshold": 0.7}
    assert capture_gate_config(config, require_glasses_pose=True).require_glasses_pose
