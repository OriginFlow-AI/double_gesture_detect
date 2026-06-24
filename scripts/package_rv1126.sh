#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build-rv1126}"
PACKAGE_DIR="${PACKAGE_DIR:-deploy/rv1126/double_ok_gate}"
RKNN_MODEL="${RKNN_MODEL:-models/hand_landmark.rknn}"

if [[ ! -x "$BUILD_DIR/double-ok-headless" ]]; then
  echo "Missing $BUILD_DIR/double-ok-headless. Run scripts/build_rv1126.sh first." >&2
  exit 1
fi

if [[ ! -f "$RKNN_MODEL" ]]; then
  echo "Missing $RKNN_MODEL. Convert the hand landmark model to RKNN first." >&2
  exit 1
fi

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/bin" "$PACKAGE_DIR/configs" "$PACKAGE_DIR/models" "$PACKAGE_DIR/scripts"

cp "$BUILD_DIR/double-ok-headless" "$PACKAGE_DIR/bin/"
cp "$BUILD_DIR/double-ok-camera-check" "$PACKAGE_DIR/bin/"
cp configs/rv1126_uvc.json "$PACKAGE_DIR/configs/"
cp "$RKNN_MODEL" "$PACKAGE_DIR/models/"
cp scripts/run_rv1126.sh "$PACKAGE_DIR/scripts/"

cat > "$PACKAGE_DIR/README.txt" <<'EOF'
RV1126 board run:

  cd double_ok_gate
  scripts/run_rv1126.sh /dev/video0

Capture starts only when the configured gate is ready: two hands, stable double OK,
full-frame visibility, centered hands, minimum separation, and pose if enabled.
EOF

echo "$PACKAGE_DIR"
