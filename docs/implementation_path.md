# Implementation Path

本项目已从 Python 迁移到 C++。当前 `main` 的实现保留 C++ 主线；用户可见功能、门控流程和报告内容以 `dev_` 的双手 OK 采集门控为主。

## 已完成

1. CMake 工程骨架。
2. C++ 核心特征、门控、识别器、配置、相机、运行指标。
3. C++ 逻辑回归训练、评估和文本模型保存/加载。
4. C++ 命令行入口。
5. C++ 单元测试和 `scripts/test.sh`。
6. 参考 Allan calibrator 的 Qt Widgets 暗色工作台风格，重建实时 demo 界面。
7. 移除 Python 包、pytest 测试、`pyproject.toml`、`requirements.txt`。

## 当前命令

```bash
scripts/test.sh
scripts/train_numpy_logreg.sh
scripts/evaluate_numpy_logreg.sh
scripts/check_camera.sh /dev/video0
scripts/run_demo.sh /dev/video0
```

## 下一步

1. 接入 RV1126 RKNN/RKNPU hand landmark provider，恢复 Python/MediaPipe 版 21 点关键点精度。
2. 准备 hand landmark ONNX/RKNN 模型和代表性量化数据。
3. 在板端验证 USB UVC 摄像头、FPS、延迟、采集保存路径。
4. 针对眼镜视角采集样本校准检测阈值、OK 阈值和中心区域。
5. 增加 CI、覆盖率和真实性能 smoke test。
6. 在接入真实 21 点 hand-landmark provider 前，继续明确区分生产目标后端和 `opencv-heuristic` 调试后端。
