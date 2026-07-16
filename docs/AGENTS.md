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
- The production live path is C++ on RK3588: YOLOv8-Pose RKNN outputs hand boxes and 21 x/y/visibility points, followed
  by a separately trained Left/Right + OK attribute model. Missing assets or backend mismatch must fail initialization.
- MediaPipe, landmark JSON and OpenCV heuristic are explicit test/debug backends only. They must be labeled as non-production
  and must never be selected as an automatic fallback.
