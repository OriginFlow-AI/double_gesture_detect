# MediaPipe FP32 ONNX models

The default runtime uses these two models together:

- `palm_detection_mediapipe_2023feb.onnx`: palm boxes and seven palm keypoints.
- `handpose_estimation_mediapipe_2023feb_opencv46.onnx`: 21 hand landmarks, presence and handedness for each rotated palm crop.

Both originate from the Apache-2.0 licensed OpenCV Zoo MediaPipe models:

- https://github.com/opencv/opencv_zoo/tree/main/models/palm_detection_mediapipe
- https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe

The hand-pose file is an inference-equivalent OpenCV 4.6 compatibility export of the official model with SHA-256 `db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c`. Its `Clip`, `Squeeze` and `Gemm` encodings were changed without changing the FP32 calculations. ONNX validation passed, and all four outputs matched the official model exactly for the same ONNX Runtime input.

SHA-256:

```text
78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c  palm_detection_mediapipe_2023feb.onnx
30f633bfa4a6f20ea79762f21b07efa51ff30744c4910e0d44e98687483683d4  handpose_estimation_mediapipe_2023feb_opencv46.onnx
```
