# 双 OK 采集门控实现路径

## 目标

在 GLASSES 端触发采集前，必须同时满足：

1. 眼镜姿态角度合格。
2. 双手完整进入相机 FOV 中心区域。
3. 左手和右手都稳定做出 OK 手势。

## 第一性原理拆分

双手 OK 不是一个必须单独训练的类别。更短路径是：

```text
单手是否 OK -> 左手 OK + 右手 OK -> 双手 OK 采集触发
```

原因：

1. 开源数据集中单手 OK 数据更多。
2. 左右手可以复用同一个单手分类器。
3. 双手同时出现、是否居中、是否分开，是运行时几何门控问题，不是分类问题。

## 数据路线

当前代码训练的是 MediaPipe/HaGRID landmarks 特征，不训练原图 CNN。

因此只需要下载 HaGRIDv2 annotations：

```text
data/raw/hagrid/annotations_zip/annotations.zip
data/raw/hagrid/annotations/
```

暂时不下载每个手势的原始图片包。`ok.zip` 单类约 42.5GB，完整 HaGRIDv2 约 1.5TB；对当前 landmark 训练没有必要。

官方 annotations 地址：

```text
https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
```

## 执行步骤

### 1. 下载 annotations

```bash
mkdir -p data/raw/hagrid/annotations_zip
curl -L -C - \
  -o data/raw/hagrid/annotations_zip/annotations.zip \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
unzip -t data/raw/hagrid/annotations_zip/annotations.zip
```

### 2. 解压 annotations

```bash
mkdir -p data/raw/hagrid/annotations
unzip -o data/raw/hagrid/annotations_zip/annotations.zip -d data/raw/hagrid/annotations
```

### 3. 生成训练特征

```bash
python -m double_ok_gesture.prepare_hagrid \
  --annotations-dir data/raw/hagrid/annotations \
  --output data/processed/hagrid_ok_features.csv
```

输出 CSV 每一行是一只手：

```text
landmarks + geometry features + target
target=1 -> ok
target=0 -> not_ok
```

### 4. 训练单手 OK 分类器

当前环境没有 `scikit-learn` 时，先用纯 NumPy 逻辑回归跑通闭环：

```bash
python -m double_ok_gesture.train \
  --input data/processed/hagrid_ok_features.csv \
  --output models/ok_hand_numpy_logreg.pkl \
  --model numpy_logreg
```

如果已安装 `scikit-learn`，再训练 MLP：

```bash
python -m double_ok_gesture.train \
  --input data/processed/hagrid_ok_features.csv \
  --output models/ok_hand_mlp.joblib \
  --model mlp
```

### 5. 验证模型

```bash
python -m double_ok_gesture.evaluate \
  --input data/processed/hagrid_ok_features.csv \
  --model models/ok_hand_numpy_logreg.pkl
```

### 6. 接入双手采集门控

```bash
python -m double_ok_gesture.demo \
  --camera 0 \
  --model models/ok_hand_numpy_logreg.pkl \
  --capture-gate
```

## 验收标准

最低验收：

1. `unzip -t` 通过。
2. `prepare_hagrid` 生成非空 CSV。
3. `train` 输出所选模型文件。
4. `evaluate` 能输出 confusion matrix 和 classification report。
5. demo 中只有两只手都 OK、在 FOV 中心、稳定后才显示 ready。

## 后续再做

如果实机误判高，再采集本地眼镜视角样本：

```bash
python -m double_ok_gesture.capture_samples --label double_ok --gate --auto-capture
python -m double_ok_gesture.capture_samples --label not_double_ok --gate
```

本地数据用于调阈值或二次训练，不替代 HaGRID 初始训练。

## 历史数据产物

已完成：

```text
annotations.zip: 686MB，unzip -t 通过
annotations 解压目录: data/raw/hagrid/annotations/
训练 CSV: data/processed/hagrid_ok_features.csv
CSV 样本数: 303,697
正样本 ok: 31,041
负样本 not_ok: 272,656
跳过空 landmark: 15,456
模型: models/ok_hand_numpy_logreg.pkl
```

历史版本曾在完整 CSV 上输出以下结果：

```text
Confusion matrix:
[[269942   2714]
 [    78  30963]]

label precision recall f1 support
not_ok 0.9997 0.9900 0.9949 272656
ok     0.9194 0.9975 0.9569 31041
```

完整 CSV 包含训练样本，因此该结果只能作为历史记录，不能作为独立泛化指标。当前评估命令默认选择 `test` split；代码也支持 NumPy 逻辑回归、scikit-learn 逻辑回归和 MLP。

数据准备器现按 `split + gesture_label` 分别限制负样本数量。已有 CSV 是旧版本生成的，如需重新训练模型，应先重新运行 `prepare_hagrid`，确保 val/test 中保留负样本。

当前模型在独立 `test` split（130,756 行）上的结果：

```text
Confusion matrix:
[[123493   2274]
 [     5   4984]]

label precision recall f1 support
not_ok 1.0000 0.9819 0.9909 125767
ok     0.6867 0.9990 0.8139 4989
```

OK 召回率很高，但精确率偏低；上线前应使用眼镜实机视角的负样本校准阈值并验证误触发率。
