# Project Structure

## 目录职责

```text
.
├── README.md                 # 最短使用说明
├── pyproject.toml            # Python 包配置和命令入口
├── requirements.txt          # 运行依赖
├── configs/                  # 可调参数
├── data/                     # 本地数据，不提交大文件
├── docs/                     # 设计、流程、数据、实现和 AI 协作规则
├── models/                   # 本地训练模型，不提交大文件
├── scripts/                  # 常用命令封装
├── src/double_ok_gesture/    # 业务源码
└── tests/                    # 单元测试
```

## 源码模块

```text
src/double_ok_gesture/features.py        # 手部 landmark 特征
src/double_ok_gesture/config.py          # JSON 配置加载与参数转换
src/double_ok_gesture/camera.py          # 摄像头打开、重试、断流和设备诊断
src/double_ok_gesture/recognizer.py      # 单手 OK 和双手 OK 识别
src/double_ok_gesture/capture_gate.py    # 采集前门控：FOV、双手分开、姿态、稳定 OK
src/double_ok_gesture/runtime.py         # FPS、处理延迟、日志和实时叠加层
src/double_ok_gesture/live_ui.py         # 实时仪表盘、手部骨架和门控状态面板
src/double_ok_gesture/prepare_hagrid.py  # HaGRID JSON -> 特征 CSV
src/double_ok_gesture/train.py           # 训练模型
src/double_ok_gesture/evaluate.py        # 评估模型
src/double_ok_gesture/demo.py            # 摄像头 demo
src/double_ok_gesture/capture_samples.py # 本地样本采集
src/double_ok_gesture/model_io.py        # 模型保存/加载
src/double_ok_gesture/simple_models.py   # 无 sklearn 时的 NumPy 模型
```

## 数据约定

当前项目走 landmark 路线，不走原图训练路线。

```text
data/raw/hagrid/annotations_zip/annotations.zip # HaGRID landmarks 标注压缩包
data/raw/hagrid/annotations/                    # 解压后的 JSON 标注
data/processed/hagrid_ok_features.csv           # 训练特征
models/ok_hand_numpy_logreg.pkl                 # 当前可用模型
```

这些都是本地生成/下载文件，不进入版本库。

## 不应提交

```text
.venv/
.pytest_cache/
__pycache__/
*.pyc
*.egg-info/
data/raw/*
data/processed/*
models/*
```

`data/` 和 `models/` 下只保留说明文件与 `.gitkeep`。

## 常用命令

```bash
scripts/test.sh

scripts/prepare_hagrid.sh data/raw/hagrid/annotations
scripts/train_numpy_logreg.sh
scripts/evaluate_numpy_logreg.sh
scripts/check_camera.sh /dev/video0
scripts/run_demo.sh /dev/video0
```
