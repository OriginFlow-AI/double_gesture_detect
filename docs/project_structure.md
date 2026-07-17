# Project Structure

```text
.
├── CMakeLists.txt
├── apps/                         # Qt 实时界面和相机检查入口
├── configs/                      # 严格 JSON 运行配置和测试点
├── data/raw/                     # 本地采集数据
├── docs/                         # 架构、配置和精度说明
├── include/double_ok_gesture/    # 公共 C++ 接口
├── models/opencv_zoo/            # 默认 FP32 ONNX 双模型和许可证
├── models/rk3588/                 # RK3588 NPU FP16 RKNN 双模型和清单
├── scripts/                      # 运行、测试、相机检查、RKNN 离线转换
├── src/                          # 推理、识别、门控、设备和 UI 实现
├── third_party/rknn/             # RKNN 2.3.2 AArch64 最小运行 SDK
└── tests/cpp/                    # C++ 自动测试
```

核心模块：

```text
onnx_mediapipe_provider  共享 MediaPipe 流水线及 ONNX/RKNN runner
features                 21 点几何和 OK 分数
recognizer               单手结果、双手 OK、时序稳定
capture_gate             入框、居中、间距、姿态和手势门控
capture_writer           原始帧与 metadata 成对保存
runtime_pipeline         后端装配和单帧流水线
qt_dashboard/live_ui     实时界面、骨架和状态
camera/config/json       设备与严格配置解析
```

正式运行目标只有 `double-ok-demo`；`double-ok-camera-check` 仅用于设备检查。
