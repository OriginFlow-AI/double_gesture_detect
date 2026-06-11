# Production Readiness

## 系统边界

```text
hand landmarks
-> C++ feature extraction
-> C++ rule score or linear model
-> double OK stability window
-> pose / full-frame / center / separation / gesture gate
-> ready or blocking reason
```

工程负责“是否允许开始采集”的判断，不负责眼镜 IMU 的生产、相机标定、原始数据持久化服务或下游业务动作。

## C++ 生产化约定

1. 构建入口是 CMake。
2. C++ 运行模型格式是项目自定义文本模型；静态报告可展示 `dev_` 历史 `.pkl` 产物。
3. 摄像头层使用 OpenCV `VideoCapture`，默认 `/dev/video0`、MJPG、1280x720、30 FPS。
4. 连续读帧失败达到上限时终止，不继续使用陈旧画面。
5. 正样本必须满足稳定双 OK、完整入框、中心区域、双手分离，以及配置要求的姿态门控。
6. `val` 同时包含正负类时用于验证，否则从 `train` 分层留出；独立 `test` 由评估命令使用。
7. 实时界面使用 Qt Widgets，视觉风格对齐 Allan calibrator 的暗色工作台，而不是 OpenCV HighGUI 临时窗口。
8. 自动采集只允许在 `gate_ready=1` 时写盘；`gate_ready` 同时要求稳定双 OK、完整入框、居中、双手分离，以及配置要求的姿态门控。

## 日常运行

```bash
scripts/test.sh
scripts/check_camera.sh /dev/video0
scripts/run_demo.sh /dev/video0
```

## 验收指标

代码验收：

- CMake 配置通过。
- C++ 编译通过。
- CTest 通过。
- `git diff --check` 无空白错误。

设备验收：

- `/dev/video0` 能连续返回图像。
- 实际分辨率和格式符合日志。
- 拔出设备或连续读帧失败时明确退出。

## 已知风险

1. 本机未发现 RKNN SDK、RKNN runtime 或 `.rknn` 手部关键点模型；生产级结果必须接入 RKNN/RKNPU hand landmark provider。
2. C++ 实时运行不能直接加载旧 `.pkl` 模型；需要训练或转换为文本模型，报告展示仍可使用 `.pkl`。
3. 未经本地眼镜视角数据验证，不应仅通过调整阈值宣称降低了误触发。
