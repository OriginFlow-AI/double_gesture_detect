# Current Main Contract

本文档定义当前 `main` 分支的统一口径，避免 C++ 主线、`dev_` 功能内容、模型文件和 GUI 报告相互打架。

工程分层和长期演进以 [architecture.md](architecture.md) 为准。

## 功能目标

用户可见的功能目标以 `dev_` 为主：

```text
摄像头 BGR 帧
-> MediaPipe 风格的两只手 21 点 landmarks
-> 每只手提取归一化特征
-> 模型或几何规则判断单手 OK
-> 时间窗口判断稳定双手 OK
-> 姿态、完整入框、中心位置、双手距离门控
-> ready / 阻断原因
```

当前 `main` 保留 C++/Qt 实现，这是部署和工程化约束；不要把 Python 重新作为用户入口。桌面对齐真实 21 点时允许 C++ provider 启动 MediaPipe sidecar，后续生产部署再替换为 RKNN provider。GUI 视觉允许使用 `dev_` 的深色边框、chip、角框和仪表盘风格。

实时 Qt GUI 的控件创建、状态刷新、截图和事件日志在 `qt_dashboard` UI 模块内维护；CLI 解析和 headless smoke 在非 Qt 的 `demo_app` 模块内维护；入口 `apps/demo.cpp` 只负责 Qt main 装配和退出清理，`apps/headless.cpp` 负责无 Qt 运行入口。用户可见内容仍按 `dev_` 的双手 OK 采集门控口径对齐。

## 模型口径

报告和 `dev_` 内容展示使用历史模型产物：

```text
models/ok_hand_numpy_logreg.pkl
```

C++ 运行、训练和评估不能直接加载 `.pkl`，当前使用项目自定义文本模型：

```text
models/ok_hand_numpy_logreg.txt
```

因此：

- `scripts/gui_report.sh` 展示 `.pkl`，保证报告内容贴近 `dev_`。
- `scripts/train_numpy_logreg.sh` 和 `scripts/evaluate_numpy_logreg.sh` 使用 `.txt`，保证 C++ 主线可构建、可测试。
- `scripts/run_demo.sh` 如存在 `.txt` 则加载它；不存在时使用几何规则。

## 后端口径

真实产品目标是 MediaPipe 等价的 21 点 hand-landmark 输入。当前 C++ 入口保留多个后端选项：

- `rknn`：生产目标后端，当前等待 SDK/模型接入。
- `mediapipe`：桌面对齐后端，C++ provider 通过 Python sidecar 调用 MediaPipe Hands，输出真实 21 点关键点。
- `landmarks-json`：外部 21 点 JSON 输入，用于验证 C++ GUI 骨架显示链路，不是摄像头检测模型。
- `opencv-heuristic`：本机调试候选框检测，只用于试看界面，不等价于 `dev_` 的真实 21 点结果，也不显示关键点骨架。
- `none`：关闭检测。

默认脚本在本机存在 `.venv/bin/python` 和 sidecar 脚本时优先启用 `mediapipe`，用于复现 `dev_` 的实时 21 点效果；否则回退到候选框调试。需要显式指定时使用：

```bash
scripts/run_demo.sh --landmark-backend mediapipe
```

只需要试用 GUI 候选框效果时显式传入：

```bash
scripts/run_demo.sh --landmark-backend opencv-heuristic
```

脚本未显式传相机时会优先探测 Orbbec Gemini 335，并选择第一个能打开、能出帧的彩色视频节点。当前机器上可用节点是 `/dev/video2`；内置摄像头仍可通过 `/dev/video0` 显式选择。

## 日常验收

代码与脚本验收：

```bash
cmake --build build --target double-ok-demo double-ok-headless double-ok-gui double_ok_gesture_tests -j 2
ctest --test-dir build --output-on-failure
scripts/gui_report.sh
```

视觉试用：

```bash
xdg-open reports/gui/index.html
scripts/run_demo.sh
```

## 规范化原则

1. 先统一契约、脚本和文档，再做代码结构重构。
2. 重构不得改变采集门控结果、阈值、CLI 参数含义和报告内容。
3. GUI 改动必须保持深色仪表盘美观、信息密度和可读性。
4. 每次改动后至少运行构建、CTest 和报告生成。
