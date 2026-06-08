"""Train an OK/not-OK hand classifier from prepared feature CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .model_io import save_model
from .simple_models import NumpyLogisticClassifier


def load_feature_csv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        feature_columns = [name for name in fieldnames if name.startswith(("lm_", "geom_"))]
        if not feature_columns:
            raise ValueError(f"No feature columns found in {csv_path}")
        if "target" not in fieldnames:
            raise ValueError(f"Missing required target column in {csv_path}")
        row_count = sum(1 for row in reader if _has_values(row))

    if row_count == 0:
        raise ValueError(f"No rows found in {path}")

    x = np.empty((row_count, len(feature_columns)), dtype=np.float32)
    y = np.empty(row_count, dtype=np.int64)
    splits = np.empty(row_count, dtype=object)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        row_index = 0
        for csv_line, row in enumerate(reader, start=2):
            if not _has_values(row):
                continue
            try:
                x[row_index] = [float(row[name]) for name in feature_columns]
                y[row_index] = int(row["target"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid feature row at CSV line {csv_line}") from exc
            if y[row_index] not in (0, 1):
                raise ValueError(f"Target must be 0 or 1 at CSV line {csv_line}")
            if not np.isfinite(x[row_index]).all():
                raise ValueError(f"Features must be finite at CSV line {csv_line}")
            splits[row_index] = (row.get("split") or "unknown").lower()
            row_index += 1

    return x, y, splits, feature_columns


def make_model(model_type: str, max_iter: int, random_state: int):
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if model_type == "numpy_logreg":
        return NumpyLogisticClassifier(max_iter=max_iter)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for mlp/logreg. Use --model numpy_logreg without scikit-learn."
        ) from exc

    if model_type == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            batch_size=256,
            early_stopping=True,
            max_iter=max_iter,
            random_state=random_state,
        )
    elif model_type == "logreg":
        classifier = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([("scale", StandardScaler()), ("clf", classifier)])


def split_data(x: np.ndarray, y: np.ndarray, splits: np.ndarray, random_state: int):
    train_mask = splits == "train"
    validation_mask = splits == "val"
    if (
        train_mask.any()
        and validation_mask.any()
        and _contains_both_binary_labels(y[train_mask])
        and _contains_both_binary_labels(y[validation_mask])
    ):
        return x[train_mask], x[validation_mask], y[train_mask], y[validation_mask]

    # A missing or single-class validation split is not useful. Build a
    # stratified holdout from train and leave the independent test split alone.
    if train_mask.any():
        train_x = x[train_mask]
        train_y = y[train_mask]
        train_indices, validation_indices = _stratified_split_indices(
            train_y,
            random_state,
        )
        return (
            train_x[train_indices],
            train_x[validation_indices],
            train_y[train_indices],
            train_y[validation_indices],
        )

    train_indices, test_indices = _stratified_split_indices(y, random_state)
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def confusion_matrix_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("confusion_matrix_2x2 only supports binary labels 0 and 1")
    matrix = np.zeros((2, 2), dtype=np.int64)
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[int(true), int(pred)] += 1
    return matrix


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix_2x2(y_true, y_pred)
    print("Confusion matrix:")
    print(matrix)
    print()
    print("label precision recall f1 support")
    for label, name in [(0, "not_ok"), (1, "ok")]:
        tp = float(matrix[label, label])
        fp = float(matrix[:, label].sum() - tp)
        fn = float(matrix[label, :].sum() - tp)
        support = int(matrix[label, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        print(f"{name:6s} {precision:.4f} {recall:.4f} {f1:.4f} {support}")


def train(args: argparse.Namespace) -> None:
    x, y, splits, feature_columns = load_feature_csv(args.input)
    x_train, x_test, y_train, y_test = split_data(x, y, splits, args.random_state)

    model = make_model(args.model, args.max_iter, args.random_state)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    print_metrics(y_test, pred)

    metadata = {
        "model_type": args.model,
        "feature_columns": feature_columns,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "positive_label": "ok",
    }
    output = args.output or f"models/ok_hand_{args.model}.pkl"
    save_model(output, model, metadata)
    print(f"Saved model to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OK hand classifier.")
    parser.add_argument("--input", default="data/processed/hagrid_ok_features.csv")
    parser.add_argument("--output", default=None, help="Defaults to models/ok_hand_<model>.pkl.")
    parser.add_argument("--model", choices=["mlp", "logreg", "numpy_logreg"], default="numpy_logreg")
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


def _has_values(row: dict[str | None, Any]) -> bool:
    return any(value not in (None, "") for value in row.values())


def _contains_both_binary_labels(y: np.ndarray) -> bool:
    return set(np.unique(y)) == {0, 1}


def _stratified_split_indices(
    y: np.ndarray,
    random_state: int,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("At least two target rows are required for a train/test split")

    generator = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        if len(label_indices) < 2:
            raise ValueError(f"Label {label!r} needs at least two rows for a stratified split")
        generator.shuffle(label_indices)
        test_count = max(1, int(round(len(label_indices) * test_fraction)))
        test_count = min(test_count, len(label_indices) - 1)
        test_parts.append(label_indices[:test_count])
        train_parts.append(label_indices[test_count:])

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return train_indices, test_indices


if __name__ == "__main__":
    main()
