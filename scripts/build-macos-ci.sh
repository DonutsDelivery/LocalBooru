#!/usr/bin/env bash
# Build and verify LocalBooru's ad-hoc-signed universal macOS artifacts on native macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == Darwin && "${DMC_MAC_BUILD_LOCK_HELD:-0}" != 1 ]]; then
  exec python3 "$ROOT/scripts/macos-build-lock.py" \
    --wait "${LOCALBOORU_MACOS_BUILD_WAIT_SECONDS:-0}" -- "$0" "$@"
fi
python3 "$ROOT/scripts/check-release-version.py"
CHECK_ONLY="${LOCALBOORU_MACOS_CHECK_ONLY:-0}"
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || { echo 'LOCALBOORU_MACOS_CHECK_ONLY must be 0 or 1' >&2; exit 2; }
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

npm --prefix "$ROOT/frontend" ci
if [[ "$CHECK_ONLY" != 1 ]]; then npm --prefix "$ROOT/frontend" test; fi
npm --prefix "$ROOT/frontend" run build

if [[ "$CHECK_ONLY" == 1 ]]; then
  "$ROOT/scripts/run-cargo.sh" check --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" "${FEATURE_ARGS[@]}"
  git -C "$ROOT" diff --exit-code -- Cargo.lock
  echo "Native macOS $EDITION check passed; no package was built or signed."
  exit 0
fi

DIST_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).absolute())' "$DIST_DIR")"
mkdir -p "$DIST_DIR"
[[ -z "$(find "$DIST_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo "ERROR: use a fresh macOS artifact output directory; existing files were preserved" >&2
  exit 1
}

"$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib
if [[ "$EDITION" == store-lite ]]; then
  "$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib "${FEATURE_ARGS[@]}" edition::tests
  "$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib "${FEATURE_ARGS[@]}" store_lite_policy
  "$ROOT/scripts/run-cargo.sh" test --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" --lib "${FEATURE_ARGS[@]}" macos_grants::tests
fi
"$ROOT/scripts/run-cargo.sh" check --locked --manifest-path "$ROOT/src-tauri/Cargo.toml" "${FEATURE_ARGS[@]}"

# Resources use absolute source paths in an external generated config. Nothing is
# copied into the checkout, and a fresh output prevents accepting a partial build.
HELPER_CACHE="${LOCALBOORU_MACOS_HELPER_CACHE:-${XDG_CACHE_HOME:-$HOME/Library/Caches}/dmc-macos-media}"
HELPER_PAYLOAD="$DIST_DIR/media-helpers"
python3 "$ROOT/scripts/build-macos-media.py" --cache "$HELPER_CACHE" --output "$HELPER_PAYLOAD"
RESOURCE_CONFIG="$DIST_DIR/media-resource-config.json"
python3 - "$ROOT" "$EDITION" "$HELPER_PAYLOAD" "$RESOURCE_CONFIG" <<'PY'
import json, sys
from pathlib import Path
root, edition, payload, output = sys.argv[1:]
config = json.loads((Path(root) / 'src-tauri/tauri.store-lite.conf.json').read_text()) if edition == 'store-lite' else {}
resources = json.loads((Path(root) / 'src-tauri/tauri.conf.json').read_text())['bundle']['resources']
resources = {str((Path(root) / 'src-tauri' / source).resolve()) + ('/' if source.endswith('/') else ''): destination
             for source, destination in resources.items()}
resources.update({
    str(Path(payload).resolve() / 'bin/ffmpeg'): 'bin/ffmpeg',
    str(Path(payload).resolve() / 'bin/ffprobe'): 'bin/ffprobe',
    str(Path(payload).resolve() / 'licenses') + '/': 'licenses/media/',
    str(Path(payload).resolve() / 'media-build-receipt.json'): 'licenses/media-build-receipt.json',
})
config.setdefault('bundle', {})['resources'] = resources
Path(output).write_text(json.dumps(config, indent=2) + '\n')
PY
TAURI_ARGS=(--config "$RESOURCE_CONFIG")
(
  cd "$ROOT"
  "$ROOT/scripts/run-cargo.sh" tauri build --ci --target "$TARGET" --bundles app "${FEATURE_ARGS[@]}" "${TAURI_ARGS[@]}"
)
git -C "$ROOT" diff --exit-code -- Cargo.lock

APP="$BUNDLE_DIR/macos/$APP_NAME.app"
BINARY="$APP/Contents/MacOS/donutmediacenter"
INFO_PLIST="$APP/Contents/Info.plist"

[[ -d "$APP" ]]
[[ -x "$BINARY" ]]
[[ -f "$INFO_PLIST" ]]

ARCHS="$(lipo -archs "$BINARY")"
[[ " $ARCHS " == *" arm64 "* ]]
[[ " $ARCHS " == *" x86_64 "* ]]
[[ "$(plutil -extract CFBundleShortVersionString raw "$INFO_PLIST")" == "2.1.1" ]]
[[ "$(plutil -extract CFBundleIdentifier raw "$INFO_PLIST")" == "$APP_ID" ]]
[[ "$(plutil -extract LSMinimumSystemVersion raw "$INFO_PLIST")" == "11.0" ]]

python3 "$ROOT/scripts/package-macos-media.py" --app "$APP" --payload "$HELPER_PAYLOAD" \
  --edition "$EDITION" --receipt "$DIST_DIR/media-signature-receipt.json"

# Tauri's earlier DMG would predate the explicit nested-helper signatures. Create
# it only after the final app has passed its payload and signature gates.
DMG_STAGE="$(mktemp -d "$DIST_DIR/dmg-stage.XXXXXX")"
ditto "$APP" "$DMG_STAGE/$APP_NAME.app"
ln -s /Applications "$DMG_STAGE/Applications"
DMG="$DIST_DIR/$ARTIFACT_NAME.dmg"
hdiutil create -ov -format UDZO -volname "$APP_NAME" -srcfolder "$DMG_STAGE" "$DMG"
rm -rf "$DMG_STAGE"
hdiutil verify "$DMG"

codesign --verify --deep --strict "$APP"
SIGNATURE_INFO="$(codesign -dvv "$APP" 2>&1)"
printf '%s\n' "$SIGNATURE_INFO"
[[ "$SIGNATURE_INFO" == *"Signature=adhoc"* ]] || {
  echo "ERROR: macOS app is not ad-hoc signed" >&2
  exit 1
}
echo "macOS app ad-hoc signature verification passed"
if [[ "$EDITION" == store-lite ]]; then
  codesign -d --entitlements :- "$APP" > "$DIST_DIR/store-lite-entitlements.plist"
  python3 - "$DIST_DIR/store-lite-entitlements.plist" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'rb') as stream:
    entitlements = plistlib.load(stream)
for key in ('app-sandbox', 'network.client', 'network.server',
            'files.user-selected.read-write', 'files.bookmarks.app-scope'):
    assert entitlements.get('com.apple.security.' + key) is True, key
PY
fi

cp "$HELPER_PAYLOAD/DonutMediaCenter-macOS-media-sources.tar.xz" "$DIST_DIR/"
cp "$HELPER_PAYLOAD/media-build-receipt.json" "$DIST_DIR/"
ditto -c -k --sequesterRsrc --keepParent \
  "$APP" "$DIST_DIR/$ARTIFACT_NAME.zip"

(
  cd "$DIST_DIR"
  unzip -t "$ARTIFACT_NAME.zip"
  shasum -a 256 \
    "$ARTIFACT_NAME.dmg" \
    "$ARTIFACT_NAME.zip" \
    DonutMediaCenter-macOS-media-sources.tar.xz \
    media-build-receipt.json media-signature-receipt.json \
    > SHA256SUMS-macOS
  shasum -a 256 -c SHA256SUMS-macOS
)

printf 'macOS universal artifacts verified (%s):\n' "$ARCHS"
if [[ "$EDITION" == store-lite ]]; then
  echo "Sandboxed lite inspection build only: store signing, real-Mac folder access and helper sandbox acceptance remain required."
fi
find "$DIST_DIR" -maxdepth 1 -type f -exec stat -f '  %N (%z bytes)' {} \; | sort
