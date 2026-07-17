# Minimal RKNN Runtime SDK

This directory is the minimal AArch64 subset needed to compile and package the
RK3588 backend. It was extracted from Rockchip `rknn_model_zoo` commit
`bad6c7334531becaf90a561988519b7bec34d0ab`.

- `include/rknn_api.h`: SHA-256
  `c17dbc8454b91af5eb1fd8c4a52c73b508eca1f3b530d3bbdb043bbb4a80187c`
- `aarch64/librknnrt.so`: RKNN Runtime 2.3.2, SHA-256
  `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`

The board kernel driver/BSP must be ABI-compatible with this runtime. Replace
the directory or pass an external `RKNN_SDK_ROOT` when the target image requires a
different BSP-matched runtime. See `LICENSE.rknn_model_zoo`.
