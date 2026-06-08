# Models

本目录放本地训练模型，不提交大文件。

当前可用模型：

```text
ok_hand_numpy_logreg.pkl
```

模型采用 joblib/pickle 序列化，只加载本项目生成或其他可信来源的文件。

训练方式：

```bash
PYTHONPATH=src python -m double_ok_gesture.train \
  --input data/processed/hagrid_ok_features.csv \
  --output models/ok_hand_numpy_logreg.pkl \
  --model numpy_logreg
```

独立测试集评估：

```bash
scripts/evaluate_numpy_logreg.sh
```
