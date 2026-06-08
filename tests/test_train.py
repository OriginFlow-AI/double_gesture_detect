import numpy as np
import pytest

from double_ok_gesture.evaluate import select_evaluation_split
from double_ok_gesture.train import load_feature_csv, split_data


def test_load_feature_csv_requires_feature_columns(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("target,split\n1,train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="feature columns"):
        load_feature_csv(path)


def test_split_data_has_numpy_stratified_fallback():
    x = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.asarray([0] * 20 + [1] * 20)
    splits = np.asarray(["unknown"] * 40)

    x_train, x_test, y_train, y_test = split_data(x, y, splits, random_state=7)

    assert len(x_train) + len(x_test) == 40
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}


def test_split_data_prefers_validation_to_keep_test_independent():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    splits = np.asarray(["train", "train", "val", "val", "test", "test"])

    x_train, x_holdout, _y_train, y_holdout = split_data(
        x,
        y,
        splits,
        random_state=7,
    )

    assert x_train.tolist() == x[:2].tolist()
    assert x_holdout.tolist() == x[2:4].tolist()
    assert y_holdout.tolist() == y[2:4].tolist()


def test_split_data_uses_train_holdout_when_validation_has_one_class():
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1])
    splits = np.asarray(["train"] * 6 + ["val", "val"] + ["test", "test", "test", "test"])

    x_train, x_holdout, y_train, y_holdout = split_data(
        x,
        y,
        splits,
        random_state=7,
    )

    test_rows = {tuple(row) for row in x[8:]}
    assert not test_rows.intersection(tuple(row) for row in x_train)
    assert not test_rows.intersection(tuple(row) for row in x_holdout)
    assert set(y_train) == {0, 1}
    assert set(y_holdout) == {0, 1}


def test_evaluation_auto_split_prefers_test():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    splits = np.asarray(["train", "train", "val", "val", "test", "test"])

    selected_x, selected_y, split_name = select_evaluation_split(x, y, splits, "auto")

    assert split_name == "test"
    assert selected_x.tolist() == x[4:].tolist()
    assert selected_y.tolist() == y[4:].tolist()
