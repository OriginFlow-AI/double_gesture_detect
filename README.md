# Double OK Gesture Capture Gate

本分支以 `261902a4`（0612）为用户可见基准：奥比相机、Qt 深色 dashboard、
21 点骨架、时序稳定、五项采集门控、截图/采集流程和 CLI 操作保持原有方式；
模型路径替换为：

```text
Orbbec BGR
-> YOLOv8n-Pose RKNN (box + 21 x/y/visibility)
-> Left/Right + OK dual-output hand attribute model
-> stable Double-OK
-> existing capture gate and Qt GUI
```

完整合同与当前阻塞见 [0612 双模型替换说明](docs/model_replacement_0612.md)，分层边界见
[架构说明](docs/architecture.md)。

## 当前真实状态

- 模型一已存在：`models/rk3588/hand_pose_640_fp.rknn`，RK3588、640×640、FP。
- 模型一 size/SHA/manifest 和纯 CPU 后处理已检查；真实 tensor 仍须在 RK3588 上
  通过 RKNN Runtime query 验证。
- 模型二的双输出接口已实现，但仓库没有训练权重，也没有足够的 Gemini 335
  目标域标注数据。
- 生产 `yolov8-rknn` 初始化会因模型二缺失明确失败，不会回退到 JSON、MediaPipe、
  OpenCV heuristic 或几何规则。
- 未完成真板和目标域评估，不能宣称生产可用或精度达标。

## 依赖和本机构建

- CMake 3.20+
- C++20 编译器（保持 0612 工程基准）
- OpenCV 4（包含 DNN 模块）
- Qt5 Widgets（只用于实时 Demo）
- RK3588 板端需要 BSP 匹配的 RKNN Runtime

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

无 Qt 构建：

```bash
cmake -S . -B build-headless -DCMAKE_BUILD_TYPE=Release \
  -DDOUBLE_OK_BUILD_QT_DEMO=OFF \
  -DDOUBLE_OK_BUILD_CAPTURE_TOOL=OFF
cmake --build build-headless -j 2
ctest --test-dir build-headless --output-on-failure
```

日常回归建议直接使用无界面测试入口；它默认使用独立的 `build-test/`，并关闭 Qt
Demo 和 HighGUI 采集工具，避免 CI/板端环境被桌面依赖阻塞或覆盖 `build/` 中的
桌面目标：

```bash
scripts/test.sh
```

需要同时验证完整桌面目标时：

```bash
DOUBLE_OK_BUILD_QT_DEMO=ON \
DOUBLE_OK_BUILD_CAPTURE_TOOL=ON \
BUILD_DIR=build-full scripts/test.sh
```

运行配置采用严格 JSON 类型与嵌套作用域，字段范围、稳定窗口和采集目录会在启动前
统一校验；完整字段说明见 [运行配置说明](docs/runtime_configuration.md)。运行日志写入
标准错误，Demo/Headless 可用 `--log-level DEBUG|INFO|WARNING|ERROR` 控制级别。

## x86 桌面端 YOLOv8 Pose ONNX

桌面端使用 OpenCV DNN 直接运行手部 YOLOv8 Pose ONNX。没有属性模型时会明确标为
“实验几何规则”，用于现场调试；此时 `Unknown` 是预期手别，OK 数值是匹配分，
不是概率。若已有 `double_ok_hand_attribute_v1`，可追加 `--model` 接入模型二。

```bash
scripts/run_demo.sh /dev/video6 \
  --landmark-backend yolov8-onnx \
  --pose-model models/hand_pose.onnx \
  --disable-auto-capture \
  --log-level INFO
```

本轮几何 fallback 已改为使用原图等距坐标，关键手势点 visibility 不足时安全判为
非 OK，并在 NMS 限制两手之前过滤低质量 pose。界面分别显示 Pose 检测分、手别分
和 OK 匹配分，不再把“未知手 45.8%”误写成手别置信度。但现场对照也确认当前
`hand_pose.onnx` 会把真 OK 与张掌输出成近似甚至反向的关键点，不能靠阈值同时解决
召回和误报；它只保留作 Pose PoC。

桌面精度调试优先使用 MediaPipe Full（必须显式选择，不是生产 fallback）：

```bash
DOUBLE_OK_MEDIAPIPE_PYTHON=/path/to/mediapipe/venv/bin/python \
scripts/run_demo.sh /dev/video6 \
  --landmark-backend mediapipe \
  --disable-auto-capture \
  --log-level INFO
```

评分原理、诊断命令和精度验收边界见
[识别精度说明](docs/recognition_accuracy.md)。

当前工作区已生成 `models/hand_pose.onnx`（16,610,827 字节，SHA-256
`5dadc44325569fb7c1901b9fd8eaee2275d1636e51f75ce4f22f1e2ada7122fc`）。该大文件
受 `.gitignore` 管理，不会随源码提交；重新克隆后仍需单独复制。模型是未经
end-to-end NMS 的标准单输出 Ultralytics Pose 导出，单类别 `hand`、21 个关键点；
输出支持 `[1,68,candidates]` 和 `[1,candidates,68]`。程序执行 letterbox、
OpenCV DNN 推理、置信度过滤、NMS、坐标还原，并把关键点送入现有 Qt 骨架显示。
模型缺失或输出合同不匹配时会直接报错，不会回退到 JSON、MediaPipe 或启发式后端。
模型来源、固定提交、校验值及许可边界见 [模型说明](models/README.md)。

## 奥比相机和 GUI

先检查 Gemini 335 彩色节点：

```bash
scripts/check_camera.sh /dev/video6
```

RK3588 上使用真实双模型运行；`--model` 是模型二，不是 pose 模型：

```bash
scripts/run_demo.sh /dev/video6 \
  --width 640 --height 480 --camera-fps 30 --fourcc MJPG \
  --landmark-backend yolov8-rknn \
  --pose-model models/rk3588/hand_pose_640_fp.rknn \
  --pose-manifest models/rk3588/hand_pose_640_fp.rknn.manifest.json \
  --model /path/to/trained_hand_attribute_v1.json
```

按 `Q` 或 `Esc` 退出，按 `S` 保存 dashboard 截图，`--fullscreen` 全屏，
`--dashboard-width/--dashboard-height` 调整窗口渲染尺寸。

显式 JSON 仅验证相机、GUI 和 21 点显示链，窗口会标注“测试后端 / 非 YOLOv8 推理”：

```bash
scripts/run_demo.sh /dev/video6 \
  --landmark-backend landmarks-json \
  --landmarks-json configs/debug_landmarks.json
```

静态 HTML 报告由 `scripts/gui_report.sh` 生成，它不是实时推理程序。

## RK3588 交叉编译

```bash
export RK3588_TOOLCHAIN_PREFIX=/opt/rk3588-toolchain/bin/aarch64-linux-gnu-
export RK3588_SYSROOT=/opt/rk3588-sysroot
scripts/build_rk3588.sh
```

工具链、sysroot、OpenCV 和板端 RKNN driver/Runtime 必须来自兼容的 RK3588 BSP。

## 旧工具边界

历史 MediaPipe/JSON/OpenCV/OK-only 训练工具仍保留，以免破坏 0612 的显式调试和
离线入口；它们不是 `yolov8-rknn` 的自动 fallback。历史 OK-only 模型使用的特征
合同不兼容 YOLOv8 的二维关键点，不能冒充模型二。
