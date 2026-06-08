"""Small NumPy-only models for environments without scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyLogisticClassifier:
    """Binary logistic classifier with sklearn-like predict APIs."""

    max_iter: int = 160
    learning_rate: float = 0.25
    l2: float = 1e-4

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticClassifier":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        _validate_training_data(x, y)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        x_scaled = (x - self.mean_) / self.scale_

        positives = max(float(y.sum()), 1.0)
        negatives = max(float(len(y) - y.sum()), 1.0)
        sample_weight = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
        weight_sum = float(sample_weight.sum())

        self.coef_ = np.zeros(x.shape[1], dtype=np.float32)
        prior = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        self.intercept_ = float(np.log(prior / (1.0 - prior)))

        for step in range(self.max_iter):
            proba = _sigmoid(x_scaled @ self.coef_ + self.intercept_)
            error = (proba - y) * sample_weight
            grad = (x_scaled.T @ error) / weight_sum + self.l2 * self.coef_
            intercept_grad = float(error.sum() / weight_sum)
            rate = self.learning_rate / np.sqrt(1.0 + step * 0.05)
            self.coef_ -= rate * grad.astype(np.float32)
            self.intercept_ -= rate * intercept_grad
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if not hasattr(self, "coef_"):
            raise RuntimeError("Classifier must be fitted before prediction")
        if x.ndim != 2 or x.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"Expected prediction data with shape (n, {self.coef_.shape[0]}), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("Prediction features must be finite")
        x_scaled = (x - self.mean_) / self.scale_
        positive = _sigmoid(x_scaled @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.int64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _validate_training_data(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("Training features must have shape (n >= 2, features >= 1)")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("Training targets must be one-dimensional and match feature rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training data must contain only finite values")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Training targets must contain both binary labels 0 and 1")
