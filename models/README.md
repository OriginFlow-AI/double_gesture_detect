# Models

本目录放本地训练模型，不提交大文件。

当前 C++ 默认模型路径：

```text
models/ok_hand_numpy_logreg.txt
```

模型采用项目自定义的纯文本线性模型格式，不再使用 Python `joblib` / `pickle`。

训练方式：

```bash
scripts/train_numpy_logreg.sh
```

独立测试集评估：

```bash
scripts/evaluate_numpy_logreg.sh
```
