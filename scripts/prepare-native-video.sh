#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(uname -s)" in
  Linux) ;;
  *) exit 0 ;;
esac

for tool in cmake rustc; do
  command -v "$tool" >/dev/null || {
    echo "prepare-native-video: missing required tool: $tool" >&2
    exit 1
  }
done

TARGET="$(rustc -vV | sed -n 's/^host: //p')"
if [[ -z "$TARGET" ]]; then
  echo "prepare-native-video: unable to determine Rust host target" >&2
  exit 1
fi

BUILD_DIR="$ROOT/native-video/build-package"
OUTPUT_DIR="$ROOT/src-tauri/binaries"
cmake -S "$ROOT/native-video" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF
cmake --build "$BUILD_DIR" --target localbooru-native-video -j2
mkdir -p "$OUTPUT_DIR"
install -m 0755 \
  "$BUILD_DIR/localbooru-native-video" \
  "$OUTPUT_DIR/localbooru-native-video-$TARGET"

echo "Prepared native video helper: src-tauri/binaries/localbooru-native-video-$TARGET"
