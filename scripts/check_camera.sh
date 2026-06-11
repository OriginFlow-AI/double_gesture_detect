#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"

detect_default_camera() {
  local candidate=""
  while IFS= read -r candidate; do
    if [[ -e "$candidate" ]]; then
      printf '%s' "$candidate"
      return
    fi
  done < <(compgen -G '/dev/v4l/by-id/usb-Orbbec*Gemini*video-index0' | sort)
  printf '%s' "/dev/video0"
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  CAMERA_SOURCE="$1"
else
  CAMERA_SOURCE="$(detect_default_camera)"
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-camera-check -j "$JOBS" >/dev/null

"$BUILD_DIR/double-ok-camera-check" \
  --list \
  --probe \
  --camera "$CAMERA_SOURCE"
