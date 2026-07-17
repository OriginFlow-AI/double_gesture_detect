# Double OK Gesture Detect

版本：`0.3.0-rk3588-npu`（RK3588 NPU 双手 OK 手势识别部署版）

这是一个 C++20 / OpenCV / Qt 的双手 OK 实时识别项目。RK3588 部署以 NPU 为主：

```text
相机 BGR 帧
-> MediaPipe FP16 RKNN 掌心检测（RK3588 NPU）
-> 旋转手部 ROI
-> MediaPipe FP16 RKNN 21 点、手别和存在置信度（RK3588 NPU）
-> OK 几何评分与时序稳定
-> 采集门控
-> Qt 实时界面
```

## 默认模型

RK3588 默认使用由官方 FP32 ONNX 转换得到的非量化 FP16 双模型：

- `models/rk3588/palm_detection_mediapipe_2023feb_fp16.rknn`
- `models/rk3588/handpose_estimation_mediapipe_2023feb_fp16.rknn`

桌面开发和对照测试保留 OpenCV DNN FP32 双模型：

- `models/opencv_zoo/palm_detection_mediapipe_2023feb.onnx`
- `models/opencv_zoo/handpose_estimation_mediapipe_2023feb_opencv46.onnx`

每种后端的两个文件必须成对使用。OpenCV 的第二个文件是与官方 FP32 计算等价的
OpenCV 4.6 兼容导出。
来源、许可证和 SHA-256 见 [模型说明](models/opencv_zoo/README.md)。

## 唯一运行方式

在 PC 或 RK3588 板端都执行同一个命令：

```bash
scripts/run_demo.sh
```

默认相机是 `/dev/video6`。如果设备号不同，只传一个相机参数：

```bash
scripts/run_demo.sh /dev/video0
```

脚本自动完成 CMake 配置、编译和启动：x86_64 PC 使用 ONNX CPU 进行开发验证；
RK3588（AArch64）强制使用 RKNN NPU 和 FP16 双模型，不会静默回退到 CPU。两种硬件
都显示 Qt 实时界面。按 `Q` 或 `Esc` 退出，按 `S` 保存界面截图。

仓库已包含 RKNN Runtime 2.3.2 的最小 AArch64 SDK，正常情况下无需另外设置路径。
板端仍需安装与 Runtime 兼容的 RK3588 BSP/NPU 内核驱动，以及 OpenCV 4、Qt5
Widgets 和 C++ 构建工具。若板端 BSP 要求使用其他版本的 RKNN Runtime，可覆盖路径：

```bash
export RKNN_SDK_ROOT=/path/to/rknn-toolkit2/rknpu2
```

首次部署建议确认：

```bash
uname -m                    # 应输出 aarch64 或 arm64
sha256sum -c models/rk3588/SHA256SUMS
scripts/verify_rk3588.sh    # 强制校验并编译 RKNN NPU 版本
scripts/run_demo.sh         # 默认相机 /dev/video6
```

需要交付不含测试、采集数据和 PC 模型的精简部署包时执行：

```bash
scripts/package_rk3588.sh
```

生成的 `dist/double-ok-rk3588-0.3.0-rk3588-npu.tar.gz` 解压后仍使用
`scripts/run_demo.sh` 启动。

同事部署步骤见 [RK3588 部署说明](DEPLOY_RK3588.md)。

## 构建与测试

依赖：CMake 3.20+、C++20 编译器、OpenCV 4（core/dnn/imgproc/imgcodecs/videoio）
以及 Qt5 Widgets（仅界面需要）。

```bash
scripts/test.sh
DOUBLE_OK_BUILD_QT_DEMO=ON BUILD_DIR=build-full scripts/test.sh
```

## 开发维护

需要重新生成模型时，在 x86_64 Python 3.12 环境安装 RKNN Toolkit2 2.3.2：

```bash
python3 -m venv .venv-rknn
.venv-rknn/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0
.venv-rknn/bin/pip install -r scripts/requirements-rknn.txt
.venv-rknn/bin/python scripts/convert_rknn_models.py
```

Python 只用于离线模型转换，板端运行仍为纯 C++。

配置字段见 [运行配置](docs/runtime_configuration.md)，模块边界见
[架构说明](docs/architecture.md)，精度验收见
[识别精度](docs/recognition_accuracy.md)。
