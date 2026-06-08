import csv
import json

from double_ok_gesture.gui import collect_dashboard_summary, render_dashboard, write_dashboard


def test_collect_dashboard_summary_scans_feature_csv(tmp_path):
    csv_path = tmp_path / "features.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "source_json", "image_id", "gesture_label", "target", "handedness", "lm_0_x"])
        writer.writerow(["train", "ok.json", "1", "ok", "1", "Left", "0.1"])
        writer.writerow(["train", "palm.json", "2", "palm", "0", "Right", "0.2"])
        writer.writerow(["val", "fist.json", "3", "fist", "0", "Left", "0.3"])

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "ok_threshold": 0.7,
                "stable_window": 5,
                "stable_min_positive": 3,
                "max_num_hands": 2,
                "capture_gate": {"min_hand_separation": 0.16},
            }
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")

    summary = collect_dashboard_summary(config_path=config_path, csv_path=csv_path, model_path=model_path)

    assert summary.csv.row_count == 3
    assert summary.csv.positive_count == 1
    assert summary.csv.negative_count == 2
    assert summary.csv.feature_count == 1
    assert summary.csv.split_counts == {"train": 2, "val": 1}
    assert summary.model.exists


def test_write_dashboard_outputs_html(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "split,source_json,image_id,gesture_label,target,handedness,lm_0_x\n"
        "train,ok.json,1,ok,1,Left,0.1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text('{"capture_gate": {"require_glasses_pose": false}}', encoding="utf-8")
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")
    output_path = tmp_path / "gui" / "index.html"

    output = write_dashboard(
        output_path=output_path,
        config_path=config_path,
        csv_path=csv_path,
        model_path=model_path,
    )

    html = output.read_text(encoding="utf-8")
    assert output == output_path
    assert "Double OK GUI" in html
    assert "门控模拟器" in html
    assert "训练用 landmark CSV" in html


def test_render_dashboard_escapes_template_literals(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "split,source_json,image_id,gesture_label,target,handedness,lm_0_x\n"
        "train,ok.json,1,ok,1,Left,0.1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    summary = collect_dashboard_summary(
        config_path=config_path,
        csv_path=csv_path,
        model_path=tmp_path / "missing.pkl",
    )

    html = render_dashboard(summary)

    assert "${result.values.handCount}" in html


def test_render_dashboard_escapes_script_closing_sequences(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "split,source_json,image_id,gesture_label,target,handedness,lm_0_x\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text('{"note": "</script><script>alert(1)</script>"}', encoding="utf-8")
    summary = collect_dashboard_summary(config_path=config_path, csv_path=csv_path, model_path=tmp_path / "model.pkl")

    html = render_dashboard(summary)

    assert '"note": "</script>' not in html
    assert "\\u003c/script\\u003e" in html
