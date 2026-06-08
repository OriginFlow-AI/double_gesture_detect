import numpy as np
import pytest

from double_ok_gesture.simple_models import NumpyLogisticClassifier


def test_numpy_classifier_rejects_single_class_training_data():
    x = np.asarray([[0.0], [1.0]], dtype=np.float32)
    y = np.asarray([1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="both binary labels"):
        NumpyLogisticClassifier(max_iter=1).fit(x, y)


def test_numpy_classifier_requires_fit_before_prediction():
    with pytest.raises(RuntimeError, match="fitted"):
        NumpyLogisticClassifier().predict_proba(np.asarray([[0.0]], dtype=np.float32))
