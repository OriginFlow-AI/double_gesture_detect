# Architecture

## 实时流水线

```text
camera frame
-> palm_detection_mediapipe FP16 RKNN / FP32 ONNX (192x192)
-> palm NMS, at most two hands
-> oriented hand crop
-> handpose_estimation_mediapipe FP16 RKNN / FP32 ONNX (224x224)
-> 21 screen landmarks + handedness + presence
-> OK geometry score
-> Double-OK temporal stability
-> capture gate
-> Qt live output
```

唯一启动脚本按硬件选择 runner：RK3588 构建由 RKNN Runtime 在 NPU 上执行两个网络；
x86_64 桌面构建由 OpenCV DNN 在 CPU 上执行。两者共享 RGB 预处理、anchors、NMS、
旋转裁剪和坐标还原。
模型或输出合同不匹配时启动失败，不做静默回退。`landmarks-json` 只用于确定性测试，
`none` 只用于关闭关键点。

## 分层

```text
core        features / recognizer / capture_gate
device      camera
inference   shared MediaPipe pipeline / RKNN runner / OpenCV runner / JSON
runtime     config / runtime_pipeline / metrics / demo_app
capture     capture_writer
ui          live_ui / qt_dashboard
apps        demo / camera_check
```

UI 和门控使用按图像宽高归一化的坐标；OK 几何评分使用原图像素等距坐标，避免
16:9 画面分别归一化 x/y 后造成角度和距离失真。provider 负责模型前后处理，
recognizer 负责 OK 与时序结论，UI 不重新计算识别结果。

## 验证

```bash
bash -n scripts/*.sh
scripts/test.sh
DOUBLE_OK_BUILD_QT_DEMO=ON BUILD_DIR=build-full scripts/test.sh
scripts/check_camera.sh /dev/video6
scripts/run_demo.sh /dev/video6
git diff --check
```
