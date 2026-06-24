#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT_MODEL="${1:-models/hand_landmark.onnx}"
OUTPUT_MODEL="${2:-models/hand_landmark.rknn}"
TARGET_PLATFORM="${TARGET_PLATFORM:-rv1126}"
DATASET="${RKNN_DATASET:-data/processed/rknn_calibration.txt}"

if ! python3 - <<'PY' >/dev/null 2>&1
import rknn
PY
then
  echo "Missing rknn-toolkit2 Python package on the conversion host." >&2
  echo "Install Rockchip RKNN Toolkit2 on the x86 conversion machine, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "$INPUT_MODEL" ]]; then
  echo "Missing input model: $INPUT_MODEL" >&2
  echo "Export/obtain a MediaPipe-equivalent hand landmark ONNX model first." >&2
  exit 1
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Missing quantization dataset list: $DATASET" >&2
  echo "Create a text file containing representative calibration image paths." >&2
  exit 1
fi

python3 - "$INPUT_MODEL" "$OUTPUT_MODEL" "$TARGET_PLATFORM" "$DATASET" <<'PY'
import sys
from rknn.api import RKNN

input_model, output_model, target_platform, dataset = sys.argv[1:5]

rknn = RKNN(verbose=True)
rknn.config(target_platform=target_platform)
ret = rknn.load_onnx(model=input_model)
if ret != 0:
    raise SystemExit(f"load_onnx failed: {ret}")
ret = rknn.build(do_quantization=True, dataset=dataset)
if ret != 0:
    raise SystemExit(f"build failed: {ret}")
ret = rknn.export_rknn(output_model)
if ret != 0:
    raise SystemExit(f"export_rknn failed: {ret}")
rknn.release()
PY

echo "$OUTPUT_MODEL"
