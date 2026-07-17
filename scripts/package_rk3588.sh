#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION_NAME="$(tr -d '[:space:]' < VERSION)"
PACKAGE_NAME="double-ok-rk3588-${VERSION_NAME}"
OUTPUT_DIR="${1:-dist}"
STAGING_DIR="$(mktemp -d)"
PACKAGE_ROOT="$STAGING_DIR/$PACKAGE_NAME"

cleanup() {
  case "$STAGING_DIR" in
    /tmp/tmp.*) rm -r -- "$STAGING_DIR" ;;
    *) echo "拒绝清理非临时目录：$STAGING_DIR" >&2 ;;
  esac
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
ARCHIVE_PATH="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"

mkdir -p \
  "$PACKAGE_ROOT/apps" \
  "$PACKAGE_ROOT/configs" \
  "$PACKAGE_ROOT/models" \
  "$PACKAGE_ROOT/scripts" \
  "$PACKAGE_ROOT/third_party"

copy_tracked() {
  local source_path=""
  while IFS= read -r -d '' source_path; do
    mkdir -p "$PACKAGE_ROOT/$(dirname "$source_path")"
    cp -a -- "$source_path" "$PACKAGE_ROOT/$source_path"
  done < <(git ls-files -z -- "$@")
}

cp CMakeLists.txt VERSION "$PACKAGE_ROOT/"
cp DEPLOY_RK3588.md "$PACKAGE_ROOT/README.md"
cp apps/demo.cpp "$PACKAGE_ROOT/apps/"
cp configs/default.json "$PACKAGE_ROOT/configs/"
copy_tracked include src models/rk3588 third_party/rknn
cp scripts/run_demo.sh "$PACKAGE_ROOT/scripts/"

tar -C "$STAGING_DIR" -czf "$ARCHIVE_PATH" "$PACKAGE_NAME"

(
  cd "$OUTPUT_DIR"
  sha256sum "$PACKAGE_NAME.tar.gz" > "$PACKAGE_NAME.tar.gz.sha256"
)

echo "部署包：$ARCHIVE_PATH"
echo "校验文件：$ARCHIVE_PATH.sha256"
