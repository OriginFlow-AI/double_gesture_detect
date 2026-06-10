# Double OK Gesture Capture Gate

C++ implementation for two-hand OK gesture capture gating.

The project no longer uses Python. Core recognition math, capture gate logic, model training/evaluation, camera probing,
sample capture, and report generation are built with CMake.

## Requirements

- CMake 3.20+
- A C++20 compiler
- OpenCV 4 with `core`, `imgproc`, `imgcodecs`, `videoio`, and `highgui`
- Qt5 development libraries when building OpenCV highgui targets

On the current development machine the verified toolchain is:

```bash
g++ --version
cmake --version
pkg-config --modversion opencv4
```

## Build And Test

```bash
scripts/test.sh
```

Equivalent manual commands:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

## Project Layout

```text
include/double_ok_gesture/  Public C++ headers
src/                        Core C++ library implementation
apps/                       CLI executables
tests/cpp/                  C++ unit tests
configs/                    Runtime and calibration configuration
scripts/                    Build-and-run helper scripts
docs/                       Notes and production guidance
data/                       Raw and processed datasets
models/                     Generated model artifacts
reports/                    Generated reports and screenshots
```

## Core Runtime Flow

```text
hand landmarks
-> 21-point normalization
-> geometry features and optional linear model score
-> per-hand OK prediction
-> sliding-window stable double OK
-> pose / frame / center / separation / gesture capture gate
-> ready / blocking reason
```

The live camera target is the previous Python MediaPipe behavior: real 21-point hand landmarks feed the same OK scoring,
stability, and capture-gate path. Production deployment targets RV1126 with an RKNN/RKNPU hand-landmark backend. OpenCV
skin-region detection is available only as an explicit local debug fallback and must not be treated as the product model.

## Commands

Normal integrated run:

```bash
scripts/run_demo.sh /dev/video0
```

This single command builds the C++ demo if needed, opens the camera, updates the Qt dashboard, runs the current hand
detection backend, computes OK scores, applies the stable double-OK window, and evaluates the capture gate. The extra
commands below are diagnostics and validation helpers, not separate product steps.

Camera probe:

```bash
scripts/check_camera.sh /dev/video0
```

Strict RV1126/parity backend selection:

```bash
scripts/run_demo.sh /dev/video0 --landmark-backend rknn
```

Automatic capture writes frames only when `gate_ready=1`, meaning stable double OK and centered hands both pass. The
output path and save cooldown are configured in `data_capture`, and can be overridden locally:

```bash
scripts/run_demo.sh /dev/video0 --capture-output-dir data/raw/session_001 --capture-cooldown 1.0
scripts/run_demo.sh /dev/video0 --disable-auto-capture
```

Local debug fallback only:

```bash
scripts/run_demo.sh /dev/video0 --landmark-backend opencv-heuristic
```

Train the built-in C++ logistic model from a prepared landmark CSV:

```bash
scripts/train_numpy_logreg.sh
```

Evaluate the model:

```bash
scripts/evaluate_numpy_logreg.sh
```

Generate the static report:

```bash
scripts/gui_report.sh
```

Capture local images:

```bash
build/double-ok-capture --label double_ok --camera /dev/video0
```

RV1126 deployment:

```bash
scripts/convert_hand_landmark_to_rknn.sh models/hand_landmark.onnx models/hand_landmark.rknn
scripts/build_rv1126.sh
scripts/package_rv1126.sh
```

See [docs/rv1126_deployment.md](docs/rv1126_deployment.md).

## Model Format

The C++ build writes plain text model artifacts such as:

```text
models/ok_hand_numpy_logreg.txt
```

Python `joblib` / `pickle` artifacts are no longer loaded.

## Current Migration Notes

- Python package files, pytest tests, `pyproject.toml`, and `requirements.txt` have been removed.
- C++ unit tests cover features, recognizer stability, capture gate decisions, config parsing, runtime metrics, training,
  and model save/load.
- HaGRID JSON conversion is implemented in C++ with the repository's small JSON parser.
- `double-ok-demo` is now a Qt Widgets dashboard using the same dark operational style as the Allan calibrator tool.
- The target live result is the previous Python MediaPipe Hands result: real 21-point hand landmarks, skeleton overlay,
  OK scoring, and capture gate. The RV1126 production backend is expected to be RKNN/RKNPU; the OpenCV detector is only
  a temporary explicit debug fallback and is not equivalent.
