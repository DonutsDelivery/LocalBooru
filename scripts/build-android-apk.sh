#!/usr/bin/env bash
# Build a signed Android APK and clean up the build cache afterward.
#
# Why the cleanup: cargo tauri android build leaves large Gradle artifacts
# under src-tauri/gen/android and Rust outputs under target/*-linux-android.
# Once a verified APK is atomically published at the project root, those
# intermediates are regenerable.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
LOCK_TIMEOUT="${LOCALBOORU_BUILD_LOCK_TIMEOUT:-1800}"
[[ "$LOCK_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "[build-android-apk] ERROR: LOCALBOORU_BUILD_LOCK_TIMEOUT must be a nonnegative number" >&2
  exit 2
}
mkdir -p "$STATE_DIR"
exec 8>"$STATE_DIR/build-cache.lock"
flock -w "$LOCK_TIMEOUT" 8 || {
  echo "[build-android-apk] ERROR: timed out waiting for another build or cleanup" >&2
  exit 1
}

KEYSTORE="${ANDROID_KEYSTORE:-$HOME/.android/debug.keystore}"
KEY_ALIAS="${ANDROID_KEY_ALIAS:-androiddebugkey}"
KEY_PASS="${ANDROID_KEY_PASS:-android}"
STORE_PASS="${ANDROID_STORE_PASS:-android}"
OUTPUT_APK="$PROJECT_ROOT/LocalBooru.apk"

if ! command -v zipalign >/dev/null || ! command -v apksigner >/dev/null; then
  SDK_ROOT="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
  if [[ -n "$SDK_ROOT" && -d "$SDK_ROOT/build-tools" ]]; then
    BUILD_TOOLS_DIR="$(printf '%s\n' "$SDK_ROOT"/build-tools/* | sort -V | tail -n 1)"
    export PATH="$BUILD_TOOLS_DIR:$PATH"
  fi
fi

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
    *)
      echo "[build-android-apk] ERROR: unknown option: $arg" >&2
      exit 2 ;;
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

for tool in zipalign apksigner; do
  command -v "$tool" >/dev/null || {
    echo "[build-android-apk] ERROR: $tool was not found in PATH or Android SDK build-tools" >&2
    exit 1
  }
done

STAGING_DIR="$(mktemp -d "$PROJECT_ROOT/.android-signing.XXXXXX")"
cleanup_staging() {
  rm -rf "$STAGING_DIR"
}
trap cleanup_staging EXIT

ALIGNED_APK="$STAGING_DIR/app-universal-release-aligned.apk"
STAGED_APK="$STAGING_DIR/LocalBooru.apk"

echo "[build-android-apk] Aligning…"
zipalign -f -p 4 "$UNSIGNED_APK" "$ALIGNED_APK"

echo "[build-android-apk] Signing with $KEYSTORE…"
apksigner sign \
  --ks "$KEYSTORE" \
  --ks-key-alias "$KEY_ALIAS" \
  --ks-pass "pass:$STORE_PASS" \
  --key-pass "pass:$KEY_PASS" \
  --v4-signing-enabled false \
  --out "$STAGED_APK" \
  "$ALIGNED_APK"

echo "[build-android-apk] Verifying staged APK…"
apksigner verify --verbose "$STAGED_APK"

mv -f "$STAGED_APK" "$OUTPUT_APK"
rm -f "$OUTPUT_APK.idsig"
echo "[build-android-apk] Published APK: $OUTPUT_APK"
ls -lh "$OUTPUT_APK"

if [[ $KEEP_CACHE -eq 1 ]]; then
  echo "[build-android-apk] --keep-cache: leaving build artifacts in place"
  exit 0
fi

echo "[build-android-apk] Cleaning Android build artifacts…"
"$PROJECT_ROOT/scripts/clean-builds.sh" android --execute --yes

echo "[build-android-apk] Done."
