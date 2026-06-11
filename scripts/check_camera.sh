#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
EXPLICIT_CAMERA=0

detect_default_camera() {
  local candidate=""
  local candidates=()
  while IFS= read -r candidate; do
    candidates+=("$candidate")
  done < <(
    for node in /sys/class/video4linux/video*; do
      [[ -r "$node/name" ]] || continue
      if grep -qi 'Orbbec.*Gemini' "$node/name"; then
        printf '/dev/%s\n' "$(basename "$node")"
      fi
    done | sort -V
  )
  while IFS= read -r candidate; do
    candidates+=("$candidate")
  done < <(printf '%s\n' /dev/video* 2>/dev/null | sort -V)

  for candidate in "${candidates[@]}"; do
    if [[ -e "$candidate" ]] && timeout 8s "$BUILD_DIR/double-ok-camera-check" --probe --camera "$candidate" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return
    fi
  done
  printf '%s' "/dev/video0"
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  CAMERA_SOURCE="$1"
  EXPLICIT_CAMERA=1
else
  CAMERA_SOURCE=""
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-camera-check -j "$JOBS" >/dev/null

if [[ "$EXPLICIT_CAMERA" -eq 0 ]]; then
  CAMERA_SOURCE="$(detect_default_camera)"
fi

"$BUILD_DIR/double-ok-camera-check" \
  --list

timeout 15s "$BUILD_DIR/double-ok-camera-check" \
  --probe \
  --camera "$CAMERA_SOURCE"
