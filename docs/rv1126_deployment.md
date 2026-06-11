# RV1126 Deployment

## Runtime Goal

The board runtime must be C++ only and does not require Qt:

```text
USB UVC camera
-> RKNN/RKNPU hand landmark backend
-> 21-point landmark features
-> per-hand OK score
-> stable double OK
-> full-frame / center / separation / optional pose gate
-> save capture only when every gate condition passes
```

Capture is allowed only when:

- two hands are detected;
- both hands are OK;
- both hands are fully visible in frame;
- both hands are inside the configured center area;
- both hands keep the configured minimum separation;
- glasses pose passes if pose gate is enabled;
- the double OK decision is stable for the configured window.

## Required External Artifacts

These files are not present in this workstation yet and must be provided before real RV1126 inference can run:

- RV1126 cross compiler;
- RV1126 sysroot with OpenCV for UVC capture;
- Rockchip RKNN runtime headers and libraries;
- a MediaPipe-equivalent hand landmark model converted to `models/hand_landmark.rknn`;
- representative calibration images listed in `data/processed/rknn_calibration.txt` for quantization.

## Host Conversion

Conversion can use Python on the host machine. The RV1126 runtime must not depend on Python.

```bash
scripts/convert_hand_landmark_to_rknn.sh models/hand_landmark.onnx models/hand_landmark.rknn
```

The script intentionally fails if RKNN Toolkit2, the ONNX model, or calibration data are missing.

## Cross Build

```bash
export RV1126_TOOLCHAIN_PREFIX=arm-linux-gnueabihf-
export RV1126_SYSROOT=/path/to/rv1126/sysroot
scripts/build_rv1126.sh
```

The script configures `DOUBLE_OK_BUILD_QT_DEMO=OFF` and builds the non-Qt `double-ok-headless` runtime for the board.

## Package

```bash
RKNN_MODEL=models/hand_landmark.rknn scripts/package_rv1126.sh
```

Copy `deploy/rv1126/double_ok_gate` to the board.

## Board Run

```bash
cd /userdata/double_ok_gate
scripts/run_rv1126.sh /dev/video0
```

Expected log format:

```text
hands=2 ok_count=2 double_ok=1 stable=1 gate_ready=1 reason=ready fps=...
capture_saved=/userdata/double_ok/captures/double_ok_centered_....jpg
```

Blocking examples:

```text
hands=2 ok_count=2 double_ok=1 stable=1 gate_ready=0 reason=hands_not_centered
hands=2 ok_count=1 double_ok=0 stable=0 gate_ready=0 reason=need_double_ok
```

## Current Blocker

This repository now has the C++ gate, tests, packaging scripts, and deployment contract. Real RV1126 parity still needs
the RKNN hand landmark backend implementation and the converted `.rknn` model. OpenCV skin-region detection must not be
used as the production landmark provider.
