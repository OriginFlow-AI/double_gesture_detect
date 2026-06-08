# Processed Data

本目录放本地生成的训练特征，不提交大文件。

当前训练文件：

```text
hagrid_ok_features.csv
```

生成方式：

```bash
PYTHONPATH=src python -m double_ok_gesture.prepare_hagrid \
  --annotations-dir data/raw/hagrid/annotations \
  --output data/processed/hagrid_ok_features.csv
```
