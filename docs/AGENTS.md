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
- Use C++ text model artifacts under `models/*.txt`.
- MediaPipe C++ landmark integration is still pending.
