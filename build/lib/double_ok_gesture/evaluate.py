"""Evaluate a trained OK hand classifier on a feature CSV."""

from __future__ import annotations

import argparse

import numpy as np

from .model_io import load_model_artifact
from .train import load_feature_csv, print_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an OK hand classifier.")
    parser.add_argument("--input", default="data/processed/hagrid_ok_features.csv")
    parser.add_argument("--model", default="models/ok_hand_numpy_logreg.pkl")
    parser.add_argument(
        "--split",
        choices=["auto", "train", "val", "test", "all"],
        default="auto",
        help="Dataset split to evaluate. auto prefers test, then val, then all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_model_artifact(args.model)
    x, y, splits, feature_columns = load_feature_csv(args.input)
    expected_features = artifact.metadata.get("feature_columns")
    if expected_features and list(expected_features) != feature_columns:
        raise ValueError("Model feature schema does not match the evaluation CSV")

    eval_x, eval_y, split_name = select_evaluation_split(x, y, splits, args.split)
    print(f"Evaluating split: {split_name} ({len(eval_y)} rows)")
    pred = artifact.model.predict(eval_x)
    print_metrics(eval_y, pred)


def select_evaluation_split(
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    requested_split: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    split_name = requested_split
    if requested_split == "auto":
        split_name = next(
            (candidate for candidate in ("test", "val") if np.any(splits == candidate)),
            "all",
        )
    if split_name == "all":
        return x, y, "all"

    mask = splits == split_name
    if not mask.any():
        raise ValueError(f"Requested split is not present in the CSV: {split_name}")
    return x[mask], y[mask], split_name


if __name__ == "__main__":
    main()
