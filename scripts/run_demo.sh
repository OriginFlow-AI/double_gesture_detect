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
  shift
else
  CAMERA_SOURCE="$(detect_default_camera)"
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
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

SYSTEM_QT_PLUGIN_PATH="${SYSTEM_QT_PLUGIN_PATH:-/usr/lib/x86_64-linux-gnu/qt5/plugins}"
SYSTEM_QT_PLATFORM_PLUGIN_PATH="${SYSTEM_QT_PLATFORM_PLUGIN_PATH:-$SYSTEM_QT_PLUGIN_PATH/platforms}"

DEMO_ARGS=(
  --camera "$CAMERA_SOURCE" \
  --config configs/default.json \
  --capture-gate
)

if [[ -f models/ok_hand_numpy_logreg.txt ]]; then
  DEMO_ARGS+=(--model models/ok_hand_numpy_logreg.txt)
else
  echo "models/ok_hand_numpy_logreg.txt not found; using geometry rules." >&2
fi

QT_PLUGIN_PATH="$(clean_qt_plugin_path "${QT_PLUGIN_PATH:-}")"
QT_PLUGIN_PATH="${QT_PLUGIN_PATH:+$QT_PLUGIN_PATH:}$SYSTEM_QT_PLUGIN_PATH"

LD_LIBRARY_PATH="$(clean_ld_library_path)" \
QT_PLUGIN_PATH="$QT_PLUGIN_PATH" \
QT_QPA_PLATFORM_PLUGIN_PATH="$SYSTEM_QT_PLATFORM_PLUGIN_PATH" \
"$BUILD_DIR/double-ok-demo" "${DEMO_ARGS[@]}" "$@"
