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

The C++ live camera executable currently opens and displays camera frames, but a MediaPipe C++ hand-landmark provider is
not linked into this repository yet. The core recognizer accepts landmark arrays directly, and all gate/math/model logic is
C++.

## Commands

Camera probe:

```bash
scripts/check_camera.sh /dev/video0
```

Run live camera demo:

```bash
scripts/run_demo.sh /dev/video0
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
- The live C++ demo still needs a real C++ landmark provider to match the old MediaPipe-based frame-to-landmark path.
