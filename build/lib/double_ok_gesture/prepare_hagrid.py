"""Prepare OK/not-OK landmark features from HaGRID annotation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .features import feature_names, feature_vector


DEFAULT_NEGATIVE_LABELS = {
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "no_gesture",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
    "grabbing",
    "grip",
    "little_finger",
    "point",
    "take_picture",
    "timeout",
}


def _infer_split(path: Path) -> str:
    for part in path.parts:
        if part in {"train", "val", "valid", "validation", "test"}:
            return "val" if part in {"valid", "validation"} else part
    return "unknown"


def _item_labels(item: dict, fallback_label: str) -> list[str]:
    labels = item.get("labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    label = item.get("label")
    if label:
        return [str(label)]
    return [fallback_label]


def _item_handedness(item: dict, index: int) -> str:
    leading = item.get("leading_hand")
    if isinstance(leading, list) and index < len(leading):
        return str(leading[index])
    if isinstance(leading, str):
        return leading
    return ""


def iter_hagrid_hands(
    annotations_dir: Path,
    positive_label: str,
    negative_labels: set[str],
    max_negative_per_class: int,
):
    if max_negative_per_class < 0:
        raise ValueError("max_negative_per_class must be non-negative")

    negative_counts: dict[tuple[str, str], int] = {}
    positive_label = positive_label.lower()
    negative_labels = {label.lower() for label in negative_labels}

    for json_path in sorted(annotations_dir.rglob("*.json")):
        if ".ipynb_checkpoints" in json_path.parts:
            continue
        fallback_label = json_path.stem.lower()
        split = _infer_split(json_path)
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            continue

        for image_id, item in payload.items():
            if not isinstance(item, dict):
                continue
            landmarks_list = item.get("hand_landmarks") or item.get("landmarks")
            if not isinstance(landmarks_list, list):
                continue

            labels = _item_labels(item, fallback_label)
            if len(labels) == 1 and len(landmarks_list) > 1:
                labels = labels * len(landmarks_list)

            for index, landmarks in enumerate(landmarks_list):
                if index >= len(labels):
                    label = fallback_label
                else:
                    label = labels[index].lower()

                if label == positive_label:
                    target = 1
                elif label in negative_labels:
                    count_key = (split, label)
                    count = negative_counts.get(count_key, 0)
                    if count >= max_negative_per_class:
                        continue
                    negative_counts[count_key] = count + 1
                    target = 0
                else:
                    continue

                yield {
                    "split": split,
                    "source_json": str(json_path),
                    "image_id": str(image_id),
                    "gesture_label": label,
                    "target": target,
                    "handedness": _item_handedness(item, index),
                    "landmarks": landmarks,
                }


def build_csv(args: argparse.Namespace) -> int:
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.is_dir():
        raise NotADirectoryError(annotations_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")

    names = feature_names()
    negative_labels = set(args.negative_labels or DEFAULT_NEGATIVE_LABELS)
    rows_written = 0
    skipped_invalid = 0

    try:
        with temporary_output.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "split",
                "source_json",
                "image_id",
                "gesture_label",
                "target",
                "handedness",
                *names,
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for sample in iter_hagrid_hands(
                annotations_dir=annotations_dir,
                positive_label=args.positive_label,
                negative_labels=negative_labels,
                max_negative_per_class=args.max_negative_per_class,
            ):
                try:
                    vector = feature_vector(sample["landmarks"], sample["handedness"])
                except (TypeError, ValueError):
                    skipped_invalid += 1
                    continue
                row = {
                    "split": sample["split"],
                    "source_json": sample["source_json"],
                    "image_id": sample["image_id"],
                    "gesture_label": sample["gesture_label"],
                    "target": sample["target"],
                    "handedness": sample["handedness"],
                }
                row.update({name: float(value) for name, value in zip(names, vector)})
                writer.writerow(row)
                rows_written += 1
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    if skipped_invalid:
        print(f"Skipped {skipped_invalid} invalid landmark rows", file=sys.stderr)
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HaGRID landmark features for OK hand training.")
    parser.add_argument("--annotations-dir", required=True, help="Path containing HaGRID annotation JSON files.")
    parser.add_argument("--output", default="data/processed/hagrid_ok_features.csv")
    parser.add_argument("--positive-label", default="ok")
    parser.add_argument("--negative-labels", nargs="*", default=None)
    parser.add_argument("--max-negative-per-class", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_csv(args)
    print(f"Wrote {rows} rows to {args.output}")


if __name__ == "__main__":
    main()
