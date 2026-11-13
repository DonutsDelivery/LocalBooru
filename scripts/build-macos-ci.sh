#!/usr/bin/env bash
# Build and verify LocalBooru's ad-hoc-signed universal macOS artifacts on native macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/scripts/check-release-version.py"
TARGET="universal-apple-darwin"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/src-tauri/target}"
export CARGO_TARGET_DIR="$TARGET_DIR"
BUNDLE_DIR="$TARGET_DIR/$TARGET/release/bundle"
DIST_DIR="${LOCALBOORU_DIST_MACOS_DIR:-$ROOT/dist-macos}"
EDITION="${LOCALBOORU_MACOS_EDITION:-full}"
FEATURE_ARGS=()
TAURI_ARGS=()
APP_NAME="DonutMediaCenter"
APP_ID="com.localbooru.app"
ARTIFACT_NAME="DonutMediaCenter-macOS-universal"
case "$EDITION" in
  full) ;;
  store-lite)
    FEATURE_ARGS=(--features store-lite)
    TAURI_ARGS=(--config "$ROOT/src-tauri/tauri.store-lite.conf.json")
    APP_NAME="DonutMediaCenter Lite"
    APP_ID="com.localbooru.app.lite"
    ARTIFACT_NAME="DonutMediaCenter-Lite-macOS-universal"
    TARGET_DIR="$TARGET_DIR/store-lite"
    export CARGO_TARGET_DIR="$TARGET_DIR"
    BUNDLE_DIR="$TARGET_DIR/$TARGET/release/bundle"
    DIST_DIR="${LOCALBOORU_DIST_MACOS_DIR:-$ROOT/dist-macos-lite}"
    ;;
  *) echo "ERROR: LOCALBOORU_MACOS_EDITION must be full or store-lite" >&2; exit 2 ;;
esac

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

"$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib
if [[ "$EDITION" == store-lite ]]; then
  "$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib "${FEATURE_ARGS[@]}" edition::tests
  "$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib "${FEATURE_ARGS[@]}" store_lite_policy
fi
"$ROOT/scripts/run-cargo.sh" check --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" "${FEATURE_ARGS[@]}"
(
  cd "$ROOT"
  "$ROOT/scripts/run-cargo.sh" tauri build --ci --target "$TARGET" --bundles app,dmg "${FEATURE_ARGS[@]}" "${TAURI_ARGS[@]}"
)
git -C "$ROOT" diff --exit-code -- Cargo.lock

APP="$BUNDLE_DIR/macos/$APP_NAME.app"
DMG="$(find "$BUNDLE_DIR/dmg" -maxdepth 1 -type f -name '*.dmg' -print -quit)"
BINARY="$APP/Contents/MacOS/donutmediacenter"
INFO_PLIST="$APP/Contents/Info.plist"

[[ -d "$APP" ]]
[[ -n "$DMG" && -s "$DMG" ]]
[[ -x "$BINARY" ]]
[[ -f "$INFO_PLIST" ]]

ARCHS="$(lipo -archs "$BINARY")"
[[ " $ARCHS " == *" arm64 "* ]]
[[ " $ARCHS " == *" x86_64 "* ]]
[[ "$(plutil -extract CFBundleShortVersionString raw "$INFO_PLIST")" == "2.1.1" ]]
[[ "$(plutil -extract CFBundleIdentifier raw "$INFO_PLIST")" == "$APP_ID" ]]
[[ "$(plutil -extract LSMinimumSystemVersion raw "$INFO_PLIST")" == "11.0" ]]
hdiutil verify "$DMG"

codesign --verify --deep --strict "$APP"
SIGNATURE_INFO="$(codesign -dvv "$APP" 2>&1)"
printf '%s\n' "$SIGNATURE_INFO"
[[ "$SIGNATURE_INFO" == *"Signature=adhoc"* ]] || {
  echo "ERROR: macOS app is not ad-hoc signed" >&2
  exit 1
}
echo "macOS app ad-hoc signature verification passed"

cp "$DMG" "$DIST_DIR/$ARTIFACT_NAME.dmg"
ditto -c -k --sequesterRsrc --keepParent \
  "$APP" "$DIST_DIR/$ARTIFACT_NAME.zip"

(
  cd "$DIST_DIR"
  unzip -t "$ARTIFACT_NAME.zip"
  shasum -a 256 \
    "$ARTIFACT_NAME.dmg" \
    "$ARTIFACT_NAME.zip" \
    > SHA256SUMS-macOS
  shasum -a 256 -c SHA256SUMS-macOS
)

printf 'macOS universal artifacts verified (%s):\n' "$ARCHS"
if [[ "$EDITION" == store-lite ]]; then
  echo "Lite capability inspection build only: App Sandbox bookmarks, store signing and real-Mac acceptance remain required."
fi
find "$DIST_DIR" -maxdepth 1 -type f -exec stat -f '  %N (%z bytes)' {} \; | sort
