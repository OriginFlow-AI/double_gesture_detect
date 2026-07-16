#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
LIST_CAMERAS=0
HAS_LANDMARK_BACKEND=0
HAS_ATTRIBUTE_MODEL=0
EXPLICIT_CAMERA=0

for arg in "$@"; do
  if [[ "$arg" == "--list-cameras" ]]; then
    LIST_CAMERAS=1
    break
  fi
  if [[ "$arg" == "--landmark-backend" ]]; then
    HAS_LANDMARK_BACKEND=1
  fi
  if [[ "$arg" == "--model" ]]; then
    HAS_ATTRIBUTE_MODEL=1
  fi
done

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

if [[ "$LIST_CAMERAS" -eq 1 ]]; then
  CAMERA_SOURCE=""
elif [[ $# -gt 0 && "$1" != --* ]]; then
  CAMERA_SOURCE="$1"
  EXPLICIT_CAMERA=1
  shift
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
  -DDOUBLE_OK_BUILD_QT_DEMO=ON
  -DDOUBLE_OK_BUILD_CAPTURE_TOOL=OFF
)
if [[ "$(uname -m)" == "aarch64" ]]; then
  RKNN_ROOT="${RKNN_ROOT:-$PWD/third_party/rknn_runtime}"
  CMAKE_ARGS+=(
    -DDOUBLE_OK_ENABLE_RKNN=ON
    -DRKNN_ROOT="$RKNN_ROOT"
  )
fi
cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-demo -j "$JOBS" >/dev/null

clean_ld_library_path() {
  local cleaned=""
  local entry=""
  IFS=':' read -r -a entries <<< "${LD_LIBRARY_PATH:-}"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      /opt/sogoupinyin/*) continue ;;
    esac
    cleaned="${cleaned:+$cleaned:}$entry"
  done
  printf '%s' "$cleaned"
}

clean_qt_plugin_path() {
  local original="${1:-}"
  local cleaned=""
  local entry=""
  IFS=':' read -r -a entries <<< "$original"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      /opt/sogoupinyin/*) continue ;;
    esac
    cleaned="${cleaned:+$cleaned:}$entry"
  done
  printf '%s' "$cleaned"
}

if [[ -z "${SYSTEM_QT_PLUGIN_PATH:-}" ]]; then
  if command -v qmake >/dev/null 2>&1; then
    SYSTEM_QT_PLUGIN_PATH="$(qmake -query QT_INSTALL_PLUGINS)"
  elif [[ "$(uname -m)" == "aarch64" ]]; then
    SYSTEM_QT_PLUGIN_PATH="/usr/lib/aarch64-linux-gnu/qt5/plugins"
  else
    SYSTEM_QT_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
  fi
fi
SYSTEM_QT_PLATFORM_PLUGIN_PATH="${SYSTEM_QT_PLATFORM_PLUGIN_PATH:-$SYSTEM_QT_PLUGIN_PATH/platforms}"

QT_PLUGIN_PATH="$(clean_qt_plugin_path "${QT_PLUGIN_PATH:-}")"
QT_PLUGIN_PATH="${QT_PLUGIN_PATH:+$QT_PLUGIN_PATH:}$SYSTEM_QT_PLUGIN_PATH"

run_double_ok_demo() {
  local runtime_library_path="$(clean_ld_library_path)"
  if [[ "$(uname -m)" == "aarch64" ]]; then
    runtime_library_path="$PWD/third_party/rknn_runtime/aarch64${runtime_library_path:+:$runtime_library_path}"
  fi
  LD_LIBRARY_PATH="$runtime_library_path" \
  QT_PLUGIN_PATH="$QT_PLUGIN_PATH" \
  QT_QPA_PLATFORM_PLUGIN_PATH="$SYSTEM_QT_PLATFORM_PLUGIN_PATH" \
  "$BUILD_DIR/double-ok-demo" "$@"
}

if [[ "$LIST_CAMERAS" -eq 1 ]]; then
  run_double_ok_demo --list-cameras "$@"
  exit $?
fi

MODEL_ARGS=()
if [[ "$HAS_ATTRIBUTE_MODEL" -eq 0 && -n "${DOUBLE_OK_HAND_ATTRIBUTE_MODEL:-}" ]]; then
  MODEL_ARGS+=(--model "$DOUBLE_OK_HAND_ATTRIBUTE_MODEL")
fi

BACKEND_ARGS=()
if [[ "$HAS_LANDMARK_BACKEND" -eq 0 ]]; then
  BACKEND_ARGS+=(--landmark-backend yolov8-rknn)
fi

demo_args_for_camera() {
  local camera_source="$1"
  DEMO_ARGS=(
    --camera "$camera_source"
    --config configs/default.json
    --capture-gate
    "${MODEL_ARGS[@]}"
    "${BACKEND_ARGS[@]}"
  )
}

if [[ "$EXPLICIT_CAMERA" -eq 1 ]]; then
  demo_args_for_camera "$CAMERA_SOURCE"
  run_double_ok_demo "${DEMO_ARGS[@]}" "$@"
  exit $?
fi

status=1
while IFS= read -r CAMERA_SOURCE; do
  [[ -e "$CAMERA_SOURCE" ]] || continue
  demo_args_for_camera "$CAMERA_SOURCE"
  if run_double_ok_demo "${DEMO_ARGS[@]}" "$@"; then
    exit 0
  fi
  status=$?
  echo "Camera $CAMERA_SOURCE failed; trying next video device." >&2
done < <(camera_candidates)

exit "$status"
