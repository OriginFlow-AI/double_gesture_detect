# Models

本目录放本地训练模型，不提交大文件。

报告和 `dev_` 内容口径使用的模型：

```text
models/ok_hand_numpy_logreg.pkl
```

这个文件是历史 Python/joblib 产物，只加载本项目生成或其他可信来源的文件。

当前 `main` 的必要 C++ 实现无法直接反序列化 `.pkl`；本地 C++ 训练和实时运行会使用纯文本模型：

```text
models/ok_hand_numpy_logreg.txt
```

训练方式：

```bash
scripts/train_numpy_logreg.sh
```

独立测试集评估：

```bash
scripts/evaluate_numpy_logreg.sh
```
