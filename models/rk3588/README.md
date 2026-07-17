# RK3588 NPU models

These two non-quantized FP16 models are the production RK3588 pair:

- `palm_detection_mediapipe_2023feb_fp16.rknn`
- `handpose_estimation_mediapipe_2023feb_fp16.rknn`

They were generated with RKNN-Toolkit2 2.3.2 by
`scripts/convert_rknn_models.py`. The converter wraps the original NHWC ONNX
input in an NCHW-to-NHWC adapter, allowing RKNN Runtime to accept contiguous
RGB uint8 NHWC input and perform `/255` normalization. Outputs remain FP32 at
the C++ API boundary through `want_float=1`.

`manifest.json` records source and artifact SHA-256 values plus the exact
input/output contract. The hand outputs must be selected by name because RKNN
may reorder `Identity`, `Identity_1`, `Identity_2`, and `Identity_3`.

No Python package is required on the RK3588 target. The target needs only the
matching RKNPU2 driver and `librknnrt.so`.

The RKNN 2.3.2 PC simulator was compared with ONNX Runtime on deterministic
random RGB input. Output cosine similarity was at least `0.99999968` for palm
regression/scores and at least `0.99999204` across all four hand outputs.
Final performance and driver compatibility still require an RK3588 board test.
