# Agent Notes

This repository is a C++20/CMake project.

```bash
scripts/test.sh
scripts/check_camera.sh /dev/video0
scripts/run_demo.sh /dev/video0
```

Boundaries:

- Keep the default live path fully in C++; Python is host-side conversion only.
- On RK3588, the production path is the two-stage MediaPipe FP16 RKNN NPU pipeline.
- Keep OpenCV DNN FP32 ONNX as the desktop reference and parity backend.
- Share MediaPipe preprocessing and postprocessing between RKNN and OpenCV runners.
- Keep gate and geometry logic testable without a camera.
- Landmark JSON is an explicit deterministic test backend, never an automatic fallback.
- Do not add a Python runtime dependency.
