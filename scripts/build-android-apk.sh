#!/usr/bin/env bash
# Build a signed Android APK and clean up the build cache afterward.
#
# Why the cleanup: cargo tauri android build leaves ~10GB of intermediate
# artifacts in src-tauri/gen/android/{app/build,build,.gradle} and another few
# GB in src-tauri/target/aarch64-linux-android. Once the APK is signed and
# copied to the project root, none of that is useful — the next build needs
# to compile against current source anyway.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KEYSTORE="${ANDROID_KEYSTORE:-$HOME/.android/debug.keystore}"
KEY_ALIAS="${ANDROID_KEY_ALIAS:-androiddebugkey}"
KEY_PASS="${ANDROID_KEY_PASS:-android}"
STORE_PASS="${ANDROID_STORE_PASS:-android}"
OUTPUT_APK="$PROJECT_ROOT/LocalBooru.apk"

KEEP_CACHE=0
SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --keep-cache) KEEP_CACHE=1 ;;
    --sign-only) SKIP_BUILD=1 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [--keep-cache] [--sign-only]

  --keep-cache  Don't delete the Gradle/cargo build artifacts after signing.
                Useful for iterating during debugging.
  --sign-only   Skip the cargo build and just re-sign the existing unsigned APK.
EOF
      exit 0 ;;
  esac
done

if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "[build-android-apk] Building APK with cargo tauri…"
  cargo tauri android build --apk
fi

UNSIGNED_DIR="$PROJECT_ROOT/src-tauri/gen/android/app/build/outputs/apk/universal/release"
UNSIGNED_APK="$UNSIGNED_DIR/app-universal-release-unsigned.apk"

if [[ ! -f "$UNSIGNED_APK" ]]; then
  echo "[build-android-apk] ERROR: expected unsigned APK at $UNSIGNED_APK" >&2
  exit 1
fi

echo "[build-android-apk] Aligning…"
ALIGNED_APK="$UNSIGNED_DIR/app-universal-release-aligned.apk"
zipalign -f -p 4 "$UNSIGNED_APK" "$ALIGNED_APK"

echo "[build-android-apk] Signing with $KEYSTORE…"
apksigner sign \
  --ks "$KEYSTORE" \
  --ks-key-alias "$KEY_ALIAS" \
  --ks-pass "pass:$STORE_PASS" \
  --key-pass "pass:$KEY_PASS" \
  --out "$OUTPUT_APK" \
  "$ALIGNED_APK"

echo "[build-android-apk] Signed APK: $OUTPUT_APK"
ls -lh "$OUTPUT_APK"

if [[ $KEEP_CACHE -eq 1 ]]; then
  echo "[build-android-apk] --keep-cache: leaving build artifacts in place"
  exit 0
fi

echo "[build-android-apk] Cleaning up build artifacts (~10GB)…"
rm -rf \
  "$PROJECT_ROOT/src-tauri/gen/android/app/build" \
  "$PROJECT_ROOT/src-tauri/gen/android/build" \
  "$PROJECT_ROOT/src-tauri/gen/android/.gradle" \
  "$PROJECT_ROOT/src-tauri/gen/android/app/.cxx" \
  "$PROJECT_ROOT/src-tauri/target/aarch64-linux-android" \
  "$PROJECT_ROOT/src-tauri/target/armv7-linux-androideabi" \
  "$PROJECT_ROOT/src-tauri/target/i686-linux-android" \
  "$PROJECT_ROOT/src-tauri/target/x86_64-linux-android"

echo "[build-android-apk] Done."
