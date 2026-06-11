# HaGRID Workflow

当前仓库已经切换为 C++ 工程。

## 当前可用路径

如果已有处理好的 landmark CSV：

```bash
scripts/train_numpy_logreg.sh
scripts/evaluate_numpy_logreg.sh
```

默认 CSV：

```text
data/processed/hagrid_ok_features.csv
```

报告和 `dev_` 内容展示使用的历史模型：

```text
models/ok_hand_numpy_logreg.pkl
```

C++ 训练、评估和实时运行使用的文本模型：

```text
models/ok_hand_numpy_logreg.txt
```

## HaGRID JSON 转换

当前 `double-ok-prepare-hagrid` 是 C++ 实现：

```bash
scripts/prepare_hagrid.sh /path/to/hagrid_annotations
```

转换流程是 `annotation JSON -> feature_vector -> hagrid_ok_features.csv`。

## 本地样本

当前 C++ 采样入口支持手动保存摄像头帧：

```bash
build/double-ok-capture --label double_ok --camera /dev/video0
```
