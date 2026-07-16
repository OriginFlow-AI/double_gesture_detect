# Architecture

本文档定义当前工程按目的驱动后的目标架构。第一性原理是：本项目不是单纯识别手势，而是在采集前做质量门控，只有画面、手势、位置和姿态都满足时才允许采集。

## 目标流水线

```text
Camera
-> YOLOv8-Pose Landmark Provider
-> Hand Attribute Classification
-> Double-OK Recognition
-> Capture Gate
-> Capture / Output
-> UI / App
```

含义：

1. 相机提供 BGR 帧。
2. YOLOv8 provider 为每只手输出框、0-20 共 21 个二维关键点及 visibility。
3. 有模型二时，属性分类器基于 21 组 x/y/visibility 输出可信 Left/Right 和 OK
   score；桌面 ONNX 缺模型二时才显式使用实验几何 fallback。
4. recognizer 只在恰好一左一右且两手均 OK 时产生即时双手 OK，并用时间窗口稳定。
5. capture gate 判断眼镜姿态、完整入框、中心区域、双手分离和手势条件。
6. gate ready 后才允许保存原始帧和 metadata。
7. GUI 和 CLI 只展示结果、提示用户或执行保存，不重新定义业务规则。

## 分层边界

```text
core
  landmarks / features / recognizer / capture_gate
device
  camera / glasses_pose
inference
  provider / opencv_debug / mediapipe / onnx / rknn
runtime
  pipeline / metrics / app_config
capture
  capture_writer / capture_session
ui
  live overlay / dashboard
report
  data summary / html renderer
apps
  demo_app / demo / gui_report / camera_check / train / evaluate
scripts
  build / run / data / deploy helpers
```

核心约束：

- `core` 不依赖 Qt、不依赖具体相机、不依赖 RKNN SDK。
- `inference` 封装后端差异；`opencv-heuristic` 只用于本机调试，不是生产 21 点后端。
- `runtime` 串联单帧处理，但不负责界面布局。
- `capture` 保存原始帧和 metadata，不保存画了骨架的展示帧。
- `ui` 负责可视化，不改变门控结果。
- `report` 负责离线说明和验收，不伪装成实时推理。

## 当前文件映射

```text
core:
  include/double_ok_gesture/features.hpp
  include/double_ok_gesture/recognizer.hpp
  include/double_ok_gesture/capture_gate.hpp
  src/features.cpp
  src/recognizer.cpp
  src/capture_gate.cpp

device:
  include/double_ok_gesture/camera.hpp
  src/camera.cpp

inference:
  include/double_ok_gesture/landmark_provider.hpp
  include/double_ok_gesture/hand_detector.hpp
  include/double_ok_gesture/yolov8_pose_postprocess.hpp
  include/double_ok_gesture/hand_attribute_classifier.hpp
  include/double_ok_gesture/model_contract.hpp
  src/landmark_provider.cpp
  src/hand_detector.cpp
  src/onnx_yolov8_pose_provider.cpp
  src/rknn_yolov8_pose_provider.cpp
  src/yolov8_pose_postprocess.cpp
  src/hand_attribute_classifier.cpp
  src/model_contract.cpp
  include/double_ok_gesture/runtime_pipeline.hpp
  src/runtime_pipeline.cpp

runtime:
  include/double_ok_gesture/runtime.hpp
  include/double_ok_gesture/runtime_pipeline.hpp
  src/runtime.cpp
  src/runtime_pipeline.cpp

app:
  include/double_ok_gesture/demo_app.hpp
  src/demo_app.cpp
  apps/headless.cpp

capture:
  include/double_ok_gesture/capture_writer.hpp
  src/capture_writer.cpp

ui:
  include/double_ok_gesture/live_ui.hpp
  include/double_ok_gesture/qt_dashboard.hpp
  src/live_ui.cpp
  src/qt_dashboard.cpp
  apps/demo.cpp

report:
  include/double_ok_gesture/report.hpp
  src/report.cpp
  apps/gui_report.cpp
  reports/gui/index.html
```

当前目录仍保持平铺头文件和源文件，避免一次性大搬家；新增代码先按职责命名，后续再按目录分层迁移。

## 后端口径

```text
yolov8-onnx       桌面 Pose 后端；可接模型二，缺失时明确标记实验几何模式
yolov8-rknn       生产目标后端，面向 RK3588/RKNN；rknn 是兼容别名
mediapipe         显式桌面调试后端，不是生产 fallback
landmarks-json    外部 21 点 JSON 输入，用于验证显示链路和后端契约
opencv-heuristic  本机调试候选检测，只用于试看 GUI 和相机链路
none              关闭检测
```

真实产品目标是 YOLOv8-Pose 的框、21 点二维坐标和 visibility，再接手属性双输出模型。
任何调试后端都必须在 GUI、日志和文档中明确标注，不得当成生产结果，也不得在生产
初始化失败时自动启用。

## 双 GUI 边界

- 实时 Qt GUI：现场运行界面，显示摄像头、骨架/候选框、目标区域、门控状态、FPS、延迟和采集事件。
- 静态 HTML 报告：工程验收界面，展示数据分布、模型产物、配置、21 点示意和门控模拟。

两者共享门控语义、后端口径和状态文案；不强制共享 UI 框架。

## 模型口径

- `models/ok_hand_numpy_logreg.pkl`：报告展示和历史 `dev_` 口径。
- `models/ok_hand_numpy_logreg.txt`：C++ 运行、训练和评估口径。

C++ 主线不得直接加载 pickle/joblib。

## 分阶段落地

1. 统一契约、架构文档、脚本和 README。
2. 从 `apps/demo.cpp` 抽出 capture writer、runtime bundle、单帧 pipeline、CLI/headless 和 Qt dashboard。
3. 增加 `HandLandmarkProvider` 抽象，统一 `rknn / mediapipe / landmarks-json / opencv-heuristic / none`。
4. 按高风险边界拆分测试：核心流程、配置/I/O、YOLOv8 后处理、模型替换合同。
5. 保持 0612 GUI/CLI 边界，接入 RK3588 YOLOv8 RKNN 和属性分类器。
6. 模型二及目标域数据齐备后完成 RK3588 板端精度、性能和长稳验收，再继续规整 target。

Pose provider 同时保留两套坐标：按宽高归一化的点只供 UI/门控，原图像素等距点供
几何 fallback，避免 1280×720 画面把角度和距离拉伸。低可靠 pose 在 NMS 和
`max_hands` 前过滤；fallback 还要求拇/食指尖及三根开放手指的远端点可靠。

## 验收矩阵

```bash
bash -n scripts/*.sh
scripts/test.sh
cmake --build build --target double-ok-demo double-ok-headless double-ok-gui double_ok_gesture_tests -j 2
ctest --test-dir build --output-on-failure
scripts/gui_report.sh
scripts/check_camera.sh
scripts/run_demo.sh --headless --max-frames 1 --landmark-backend landmarks-json --landmarks-json configs/debug_landmarks.json
cmake -S . -B build-noqt -DCMAKE_BUILD_TYPE=Release -DDOUBLE_OK_BUILD_QT_DEMO=OFF -DDOUBLE_OK_BUILD_CAPTURE_TOOL=OFF
cmake --build build-noqt --target double-ok-headless double-ok-camera-check double_ok_gesture_tests -j 2
git diff --check
```

RK3588 交叉编译、Runtime query、真板摄像头和性能测试属于条件验收，不放入普通主机验收。
