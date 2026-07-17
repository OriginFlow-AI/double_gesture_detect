# Double OK Gesture Detect

这是一个 C++20 / OpenCV / Qt 的双手 OK 实时识别项目。当前主链路只使用 ONNX：

```text
相机 BGR 帧
-> MediaPipe FP32 ONNX 掌心检测
-> 旋转手部 ROI
-> MediaPipe FP32 ONNX 21 点、手别和存在置信度
-> OK 几何评分与时序稳定
-> 采集门控
-> Qt 界面或 Headless 输出
```

## 默认模型

效果优先版本是 OpenCV Zoo 的 MediaPipe FP32 双模型：

- `models/opencv_zoo/palm_detection_mediapipe_2023feb.onnx`
- `models/opencv_zoo/handpose_estimation_mediapipe_2023feb_opencv46.onnx`

两个文件必须一起使用。第二个文件是与官方 FP32 计算等价的 OpenCV 4.6 兼容导出。
来源、许可证和 SHA-256 见 [模型说明](models/opencv_zoo/README.md)。

## 运行

先检查相机节点：

```bash
scripts/check_camera.sh /dev/video6
```

运行 Qt 界面；ONNX 是默认后端和默认模型，无需再传模型参数：

```bash
scripts/run_demo.sh /dev/video6 \
  --disable-auto-capture \
  --target-fps 15 \
  --log-level INFO
```

按 `Q` 或 `Esc` 退出，按 `S` 保存界面截图。需要门控通过后自动保存时，去掉
`--disable-auto-capture`。

Headless 运行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DDOUBLE_OK_BUILD_QT_DEMO=OFF
cmake --build build -j2
build/double-ok-headless \
  --camera /dev/video6 \
  --disable-auto-capture \
  --max-frames 100 \
  --status-interval 0
```

自定义模型时，掌心和手部模型必须成对覆盖：

```bash
scripts/run_demo.sh /dev/video6 \
  --palm-model /path/to/palm.onnx \
  --hand-model /path/to/hand_landmark.onnx
```

## 构建与测试

依赖：CMake 3.20+、C++20 编译器、OpenCV 4（core/dnn/imgproc/imgcodecs/videoio）
以及 Qt5 Widgets（仅界面需要）。

```bash
scripts/test.sh
DOUBLE_OK_BUILD_QT_DEMO=ON BUILD_DIR=build-full scripts/test.sh
```

## RK3588

这套 ONNX 可在 RK3588 的 AArch64 Linux 上通过 OpenCV DNN CPU 运行，不依赖专用
运行时。当前仓库不包含 NPU 转换链或板级运行库；如需 NPU 加速，需要另行把两个
ONNX 模型转换并重新实现对应后端。

配置字段见 [运行配置](docs/runtime_configuration.md)，模块边界见
[架构说明](docs/architecture.md)，精度验收见
[识别精度](docs/recognition_accuracy.md)。
