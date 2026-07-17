# Agent Notes

This repository is a C++20/CMake project.

```bash
scripts/test.sh
scripts/check_camera.sh /dev/video0
scripts/run_demo.sh /dev/video0
```

Boundaries:

- Keep the default live path fully in C++ with OpenCV DNN.
- The only production inference path is the two-stage MediaPipe FP32 ONNX pipeline.
- Keep gate and geometry logic testable without a camera.
- Landmark JSON is an explicit deterministic test backend, never an automatic fallback.
- Do not add a Python runtime dependency.
