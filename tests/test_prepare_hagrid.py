import json

from double_ok_gesture.prepare_hagrid import iter_hagrid_hands


def _write_annotation(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "one": {"label": label, "hand_landmarks": [[[0.0, 0.0, 0.0]] * 21]},
                "two": {"label": label, "hand_landmarks": [[[0.0, 0.0, 0.0]] * 21]},
            }
        ),
        encoding="utf-8",
    )


def test_negative_limit_is_applied_per_dataset_split(tmp_path):
    for split in ("train", "val", "test"):
        _write_annotation(tmp_path / split / "palm.json", "palm")

    samples = list(
        iter_hagrid_hands(
            annotations_dir=tmp_path,
            positive_label="ok",
            negative_labels={"palm"},
            max_negative_per_class=1,
        )
    )

    assert [sample["split"] for sample in samples] == ["test", "train", "val"]
