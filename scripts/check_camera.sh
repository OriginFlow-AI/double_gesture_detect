#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
EXPLICIT_CAMERA=0

video_devices() {
  { compgen -G '/dev/video*' || true; } | sort -V
}

device_name() {
  local device="$1"
  local node="/sys/class/video4linux/$(basename "$device")/name"
  [[ -r "$node" ]] && cat "$node"
}

is_orbbec_device() {
  device_name "$1" | grep -qi 'Orbbec.*Gemini'
}

has_color_format() {
  local device="$1"
  command -v v4l2-ctl >/dev/null 2>&1 || return 0
  v4l2-ctl -d "$device" --list-formats 2>/dev/null | grep -Eq "'(MJPG|YUYV|UYVY|RGB3|BGR3)'"
}

camera_candidates() {
  local device=""
  while IFS= read -r device; do
    [[ -e "$device" ]] || continue
    is_orbbec_device "$device" && has_color_format "$device" && printf '%s\n' "$device"
  done < <(video_devices)
  while IFS= read -r device; do
    [[ -e "$device" ]] || continue
    is_orbbec_device "$device" && continue
    has_color_format "$device" && printf '%s\n' "$device"
  done < <(video_devices)
}

detect_default_camera() {
  local candidate=""
  while IFS= read -r candidate; do
    if [[ -e "$candidate" ]] && timeout 8s "$BUILD_DIR/double-ok-camera-check" --probe --camera "$candidate" >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return
    fi
  done < <(camera_candidates)
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
