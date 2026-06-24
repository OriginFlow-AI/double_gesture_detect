# Agents Notes

This repository is now a C++/CMake project.

## Useful Commands

```bash
scripts/test.sh
scripts/check_camera.sh /dev/video0
scripts/train_numpy_logreg.sh
scripts/evaluate_numpy_logreg.sh
scripts/run_demo.sh /dev/video0
```

Manual build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

## Boundaries

- Do not reintroduce Python entrypoints.
- Keep core gate and feature logic testable without a camera.
- Use C++ text model artifacts under `models/*.txt` for runtime/training/evaluation.
- Keep static GUI reports aligned with the `dev_` content contract and historical `.pkl` display path when requested.
- The live demo target is the old Python MediaPipe Hands behavior deployed as a C++/RV1126 RKNN pipeline: real 21-point
  hand landmarks, skeleton overlay, OK scoring, centered-field gate, and capture. OpenCV heuristic detection is only an
  explicit debug fallback and must be labeled as non-parity.
