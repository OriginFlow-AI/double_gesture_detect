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
9. 运行配置、模型文本和 JSON 合同在使用前做严格类型、范围、重复字段和有限数校验；CLI 数值不接受尾随垃圾字符或 `NaN/Inf`。
10. 采集图片与 metadata 先写临时文件，只有两者都成功时才提交，避免磁盘或权限错误留下可见的半成品。
11. `double_ok_capture_v2` metadata 保存逐手框、21 点、visibility 和各分类分；
    `label_source=runtime_prediction` 明确表示它不是人工真值。

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

1. 仓库已有 RK3588 Pose RKNN 产物和 manifest，但仍需在目标板用匹配 BSP/RKNN Runtime 查询真实 tensor，并完成长稳与性能验证。
2. 生产链要求的 Left/Right + OK 属性模型权重尚未入库；缺失时初始化会明确失败，不会退回几何规则。
3. C++ 历史 OK-only 运行模型不能直接加载旧 `.pkl`；需使用文本模型，静态报告仍可展示 `.pkl`。
4. 尚无足够 Gemini 335 目标域标注集；未经真机数据验证，不应仅通过调整阈值宣称精度或误触发达标。
5. MediaPipe sidecar 和 JSON/OpenCV 后端仅用于显式调试；sidecar 属于本地子进程依赖，不在 RK3588 生产链内。
6. 当前 ONNX Pose 权重不是 Gemini 335/OK 指尖接触专项模型；实验几何修复提高了
   遮挡容忍度并补上负类否决条件，但仍需人工标注正负视频统计 PR/误触发率。
