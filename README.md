# Double OK Gesture Capture Gate

在 GLASSES 端开始采集前，确认以下条件同时满足：

1. 眼镜姿态在允许范围内。
2. 两只手完整进入相机 FOV 的中心区域。
3. 两只手保持足够距离。
4. 两只手稳定做出 OK 手势。

全部满足时门控返回 `ready=true`，采集流程才保存画面或触发下一步动作。

## 当前说明

当前 `main` 分支保留 C++/Qt 实现，这是为了部署和运行环境需要；功能目标、门控逻辑、报告内容和用户可见流程以 `dev_` 的双手 OK 采集门控为主。GUI 边框和深色仪表盘样式沿用 `dev_` 的视觉方向。

统一口径见 [docs/current_main_contract.md](docs/current_main_contract.md)，架构分层见 [docs/architecture.md](docs/architecture.md)。

## 环境安装

项目当前使用 CMake 构建 C++ 程序：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

依赖：

- CMake 3.20+
- C++20 编译器
- OpenCV 4
- Qt5 Widgets

## 核心流程

```text
摄像头 BGR 帧
-> MediaPipe 风格的两只手 21 点 landmarks
-> 每只手提取 21 点归一化特征
-> 模型或几何规则判断单手 OK
-> 时间窗口判断稳定双手 OK
-> 姿态、完整入框、中心位置、双手距离门控
-> ready / 阻断原因
```

## 实时运行

先确认 Orbbec 设备可以打开：

```bash
scripts/check_camera.sh
```

启动实时界面：

```bash
scripts/run_demo.sh
```

当前本机如果还没有接入真实 21 点 hand-landmark 后端，可以显式启用调试候选检测查看界面效果：

```bash
scripts/run_demo.sh --landmark-backend opencv-heuristic
```

脚本会优先选择 Orbbec Gemini 335 的彩色视频节点；当前机器检测到的是 `/dev/video8`，稳定路径是 `/dev/v4l/by-id/usb-Orbbec_R__Orbbec_Gemini_335_CP0HC5300092-video-index0`。需要改用内置摄像头时，显式传入 `/dev/video0`。

实时窗口采用仪表盘布局：

- 左侧显示实时画面、手部骨架或候选框、目标区域和操作提示。
- 右侧显示五项门控进度、左右手置信度、模型和设备状态。
- 顶部显示最终状态、FPS、处理延迟和摄像头。

按 `Q` 或 `Esc` 退出，按 `S` 保存界面截图到 `reports/live/`。使用 `--fullscreen` 可进入全屏，使用 `--dashboard-width` 和 `--dashboard-height` 可调整渲染尺寸。

## 姿态输入

GLASSES 端可持续写入包含完整角度的 JSON 文件：

```json
{"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
```

启用姿态门控：

```bash
scripts/run_demo.sh \
  --capture-gate \
  --require-glasses-pose \
  --glasses-pose /path/to/glasses_pose.json
```

当文件正在被替换、JSON 暂时不完整或缺少任一角度时，门控返回“等待眼镜姿态数据”，不会误判为姿态合格。

## 训练与评估

```bash
scripts/prepare_hagrid.sh data/raw/hagrid/annotations
scripts/train_numpy_logreg.sh
scripts/evaluate_numpy_logreg.sh
```

负样本上限按 `split + gesture_label` 分别计算，避免某个 split 抢占全部负样本。评估默认选择 `test`，没有 `test` 时选择 `val`，不会默认在训练全集上报告指标。

## 本地采集

当前 `double-ok-capture` 是手动采集工具，按空格保存原始帧：

```bash
build/double-ok-capture --label double_ok --camera /dev/video0
```

需要门控自动采集时，使用实时 demo；只有门控 ready 时才会写盘：

```bash
scripts/run_demo.sh --capture-gate
```

采集负样本时，门控应要求双手完整、居中并保持间距，同时明确阻止双手 OK，避免标签污染。

## 测试与报告

```bash
scripts/test.sh
scripts/gui_report.sh
xdg-open reports/gui/index.html
```

静态数据与门控模拟报告输出到 `reports/gui/index.html`；实时测试界面由 `scripts/run_demo.sh` 启动。

所有识别与门控阈值集中在 `configs/default.json`。
