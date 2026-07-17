# Project Structure

```text
.
├── CMakeLists.txt
├── apps/                         # Qt、Headless、相机检查入口
├── configs/                      # 严格 JSON 运行配置和测试点
├── data/raw/                     # 本地采集数据
├── docs/                         # 架构、配置和精度说明
├── include/double_ok_gesture/    # 公共 C++ 接口
├── models/opencv_zoo/            # 默认 FP32 ONNX 双模型和许可证
├── scripts/                      # 运行、测试、相机检查
├── src/                          # 推理、识别、门控、设备和 UI 实现
└── tests/cpp/                    # C++ 自动测试
```

核心模块：

```text
onnx_mediapipe_provider  掌心检测、旋转裁剪、21 点 ONNX 推理
features                 21 点几何和 OK 分数
recognizer               单手结果、双手 OK、时序稳定
capture_gate             入框、居中、间距、姿态和手势门控
capture_writer           原始帧与 metadata 成对保存
runtime_pipeline         后端装配和单帧流水线
qt_dashboard/live_ui     实时界面、骨架和状态
camera/config/json       设备与严格配置解析
```

可执行目标：`double-ok-demo`、`double-ok-headless`、`double-ok-camera-check`。
