# Architecture

## 实时流水线

```text
camera frame
-> palm_detection_mediapipe FP32 ONNX (192x192 NHWC)
-> palm NMS, at most two hands
-> oriented hand crop
-> handpose_estimation_mediapipe FP32 ONNX (224x224 NHWC)
-> 21 screen landmarks + handedness + presence
-> OK geometry score
-> Double-OK temporal stability
-> capture gate
-> Qt/headless output
```

两个网络都由 OpenCV DNN 在 CPU 上运行。模型或输出合同不匹配时启动失败，不做静默
回退。`landmarks-json` 只用于确定性测试，`none` 只用于关闭关键点。

## 分层

```text
core        features / recognizer / capture_gate
device      camera
inference   onnx_mediapipe_provider / landmark_provider(JSON)
runtime     config / runtime_pipeline / metrics / demo_app
capture     capture_writer
ui          live_ui / qt_dashboard
apps        demo / headless / camera_check
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
scripts/run_demo.sh --list-cameras
git diff --check
```
