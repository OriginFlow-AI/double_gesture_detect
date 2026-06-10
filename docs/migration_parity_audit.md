# Migration Parity Audit

## 基准版本

行为基准使用 Git 历史里的 Python/MediaPipe 版本：

```text
dd0fc42 0609
```

该版本的实时识别链路是：

```text
OpenCV camera frame
-> RGB conversion
-> mediapipe.solutions.hands.Hands
-> real 21-point hand landmarks
-> OKHandClassifier rule/model score
-> DoubleOKRecognizer stable double OK window
-> capture gate and overlay
```

## 当前 C++ 保留情况

已保留：

- C++ camera open/read/retry path.
- 21 点 landmark feature schema.
- Geometry OK score and optional text linear model.
- Per-hand OK prediction and stable double OK window.
- Capture gate checks: pose, full-frame visibility, center bounds, hand separation, gesture.
- HaGRID landmark CSV conversion, training, evaluation, report, and C++ unit tests.
- Qt Widgets live dashboard style aligned with the Allan calibrator tool.

未完成：

- C++ MediaPipe-equivalent hand landmark provider.

Important boundary: the OpenCV skin-region detector is not the parity backend. The strict RV1126/parity backend is:

```bash
scripts/run_demo.sh /dev/video0 --landmark-backend rknn
```

The live view must label heuristic boxes as temporary candidates and must not present them as MediaPipe hands.

## 验收命令

Normal integrated run:

```bash
scripts/run_demo.sh /dev/video0
```

This should be the only command needed for the actual live experience after the C++ landmark backend is available.
The commands below are checks for development and troubleshooting.

Code validation:

```bash
scripts/test.sh
git diff --check
```

Camera validation:

```bash
scripts/check_camera.sh /dev/video0
scripts/run_demo.sh /dev/video0 --headless --max-frames 3
scripts/run_demo.sh /dev/video0 --max-frames 1
```

Parity validation after the C++ landmark backend is added:

```bash
scripts/run_demo.sh /dev/video0
```

Expected parity behavior: the live view shows real 21-point hand skeletons and OK scores comparable to the `dd0fc42`
Python/MediaPipe version, not skin-color candidate boxes.
