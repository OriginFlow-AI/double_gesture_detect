#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
LIST_CAMERAS=0
HAS_LANDMARK_BACKEND=0
EXPLICIT_CAMERA=0

for arg in "$@"; do
  if [[ "$arg" == "--list-cameras" ]]; then
    LIST_CAMERAS=1
    break
  fi
  if [[ "$arg" == "--landmark-backend" ]]; then
    HAS_LANDMARK_BACKEND=1
  fi
done

detect_default_camera() {
  local candidate=""
  while IFS= read -r candidate; do
    if [[ -e "$candidate" ]]; then
      printf '%s' "$candidate"
      return
    fi
  done < <(
    for node in /sys/class/video4linux/video*; do
      [[ -r "$node/name" ]] || continue
      if grep -qi 'Orbbec.*Gemini' "$node/name"; then
        printf '/dev/%s\n' "$(basename "$node")"
      fi
    done | sort -V
  )
  while IFS= read -r candidate; do
    if [[ -e "$candidate" ]]; then
      printf '%s' "$candidate"
      return
    fi
  done < <(printf '%s\n' /dev/video* 2>/dev/null | sort -V)
  printf '%s' "/dev/video0"
}

if [[ "$LIST_CAMERAS" -eq 1 ]]; then
  CAMERA_SOURCE=""
elif [[ $# -gt 0 && "$1" != --* ]]; then
  CAMERA_SOURCE="$1"
  EXPLICIT_CAMERA=1
  shift
else
  CAMERA_SOURCE=""
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-demo -j "$JOBS" >/dev/null

if [[ "$LIST_CAMERAS" -eq 0 && "$EXPLICIT_CAMERA" -eq 0 ]]; then
  CAMERA_SOURCE="$(detect_default_camera)"
fi

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

if [[ "$LIST_CAMERAS" -eq 1 ]]; then
  DEMO_ARGS=(--list-cameras)
else
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
  if [[ "$HAS_LANDMARK_BACKEND" -eq 0 ]]; then
    if [[ -x .venv/bin/python && -f scripts/mediapipe_landmark_server.py ]]; then
      DEMO_ARGS+=(--landmark-backend mediapipe)
    else
      echo "MediaPipe sidecar not found; using OpenCV heuristic boxes without 21-point landmarks." >&2
      DEMO_ARGS+=(--landmark-backend opencv-heuristic)
    fi
  fi
fi

QT_PLUGIN_PATH="$(clean_qt_plugin_path "${QT_PLUGIN_PATH:-}")"
QT_PLUGIN_PATH="${QT_PLUGIN_PATH:+$QT_PLUGIN_PATH:}$SYSTEM_QT_PLUGIN_PATH"

LD_LIBRARY_PATH="$(clean_ld_library_path)" \
QT_PLUGIN_PATH="$QT_PLUGIN_PATH" \
QT_QPA_PLATFORM_PLUGIN_PATH="$SYSTEM_QT_PLATFORM_PLUGIN_PATH" \
"$BUILD_DIR/double-ok-demo" "${DEMO_ARGS[@]}" "$@"
