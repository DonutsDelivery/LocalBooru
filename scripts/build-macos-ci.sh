#!/usr/bin/env bash
# Build and verify LocalBooru's unsigned universal macOS artifacts on native macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="universal-apple-darwin"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/src-tauri/target}"
export CARGO_TARGET_DIR="$TARGET_DIR"
BUNDLE_DIR="$TARGET_DIR/$TARGET/release/bundle"
DIST_DIR="${LOCALBOORU_DIST_MACOS_DIR:-$ROOT/dist-macos}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: macOS artifacts must be built on native macOS" >&2
  exit 1
fi

command -v cargo >/dev/null
command -v npm >/dev/null
command -v lipo >/dev/null
command -v hdiutil >/dev/null
command -v plutil >/dev/null
command -v ditto >/dev/null
cargo tauri --version

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

npm --prefix "$ROOT/frontend" ci
npm --prefix "$ROOT/frontend" test
npm --prefix "$ROOT/frontend" run build

cargo test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib
cargo check --locked --manifest-path "$ROOT/src-tauri/Cargo.toml"
(
  cd "$ROOT"
  cargo tauri build --ci --target "$TARGET" --bundles app,dmg
)
git -C "$ROOT" diff --exit-code -- Cargo.lock

APP="$BUNDLE_DIR/macos/LocalBooru.app"
DMG="$(find "$BUNDLE_DIR/dmg" -maxdepth 1 -type f -name '*.dmg' -print -quit)"
BINARY="$APP/Contents/MacOS/localbooru"
INFO_PLIST="$APP/Contents/Info.plist"

[[ -d "$APP" ]]
[[ -n "$DMG" && -s "$DMG" ]]
[[ -x "$BINARY" ]]
[[ -f "$INFO_PLIST" ]]

ARCHS="$(lipo -archs "$BINARY")"
[[ " $ARCHS " == *" arm64 "* ]]
[[ " $ARCHS " == *" x86_64 "* ]]
[[ "$(plutil -extract CFBundleShortVersionString raw "$INFO_PLIST")" == "2.0.0" ]]
[[ "$(plutil -extract CFBundleIdentifier raw "$INFO_PLIST")" == "com.localbooru.app" ]]
[[ "$(plutil -extract LSMinimumSystemVersion raw "$INFO_PLIST")" == "11.0" ]]
hdiutil verify "$DMG"

# Signing credentials are intentionally not required for CI validation. Report
# the actual signature state so a future notarized publisher can fail closed.
if codesign --verify --deep --strict "$APP"; then
  echo "macOS app signature verification passed"
else
  echo "macOS app is unsigned/ad-hoc, as expected for this CI artifact"
fi

cp "$DMG" "$DIST_DIR/LocalBooru-macOS-universal.dmg"
ditto -c -k --sequesterRsrc --keepParent \
  "$APP" "$DIST_DIR/LocalBooru-macOS-universal.zip"

(
  cd "$DIST_DIR"
  unzip -t LocalBooru-macOS-universal.zip
  shasum -a 256 \
    LocalBooru-macOS-universal.dmg \
    LocalBooru-macOS-universal.zip \
    > SHA256SUMS-macOS
  shasum -a 256 -c SHA256SUMS-macOS
)

printf 'macOS universal artifacts verified (%s):\n' "$ARCHS"
find "$DIST_DIR" -maxdepth 1 -type f -exec stat -f '  %N (%z bytes)' {} \; | sort
