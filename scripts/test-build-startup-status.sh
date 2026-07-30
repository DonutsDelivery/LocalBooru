#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
inherited_child_pid=""

cleanup() {
  if [[ -n "$inherited_child_pid" ]]; then
    kill "$inherited_child_pid" 2>/dev/null || true
    wait "$inherited_child_pid" 2>/dev/null || true
  fi
  exec 9>&- 2>/dev/null || true
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR/bin" "$TEMP_DIR/home" "$TEMP_DIR/state/localbooru"

cat >"$TEMP_DIR/bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$BUILD_STATUS_TEST_DOCKER_LOG"

if [[ "${1:-}" == image && "${2:-}" == inspect ]]; then
  exit 0
fi

if [[ "${1:-}" == run ]]; then
  for argument in "$@"; do
    if [[ "$argument" == *:/dist ]]; then
      dist="${argument%:/dist}"
      mkdir -p "$dist"
      printf 'test installer\n' >"$dist/LocalBooru-Windows-Setup.exe"
      printf 'test archive\n' >"$dist/LocalBooru-Windows.zip"
      (
        cd "$dist"
        sha256sum LocalBooru-Windows-Setup.exe LocalBooru-Windows.zip >SHA256SUMS-Windows
      )
    fi
  done
  exit 0
fi

exit 0
EOF
chmod +x "$TEMP_DIR/bin/docker"

export HOME="$TEMP_DIR/home"
export XDG_STATE_HOME="$TEMP_DIR/state"
export PATH="$TEMP_DIR/bin:$PATH"
export BUILD_STATUS_TEST_DOCKER_LOG="$TEMP_DIR/docker.log"
export LOCALBOORU_BUILD_LOCK_TIMEOUT=0

# AC: @truthful-container-build-startup ac-locked
printf '%s\n' \
  'pid=4242' \
  'platform=linux' \
  'source=held-source' \
  'started=2026-07-20T08:00:00Z' \
  >"$TEMP_DIR/state/localbooru/build-cache.owner"
exec 9>>"$TEMP_DIR/state/localbooru/build-cache.lock"
flock -n 9
locked_output="$TEMP_DIR/locked.log"
if LOCALBOORU_WINDOWS_BUILD_ROOT="$TEMP_DIR/windows-locked" \
    LOCALBOORU_DIST_WINDOWS_DIR="$TEMP_DIR/windows-locked-dist" \
    "$ROOT/scripts/build-windows-local.sh" >"$locked_output" 2>&1; then
  printf 'locked build unexpectedly succeeded\n' >&2
  exit 1
fi
grep -F 'LOCALBOORU_BUILD_STATUS=LOCKED' "$locked_output" >/dev/null
grep -F 'owner_pid=4242' "$locked_output" >/dev/null
if grep -F 'LOCALBOORU_BUILD_STATUS=STARTED' "$locked_output" >/dev/null; then
  printf 'locked build claimed it started\n' >&2
  exit 1
fi
flock -u 9
exec 9>&-

# AC: @safe-development-startup ac-build-lock-independence
inherited_state="$TEMP_DIR/inherited-state"
mkdir -p "$inherited_state"
INHERITED_STATE="$inherited_state" \
INHERITED_PID_FILE="$TEMP_DIR/inherited-child.pid" \
HELPER="$ROOT/scripts/build-startup-status.sh" \
  bash -c '
    source "$HELPER"
    localbooru_build_acquire_lock "$INHERITED_STATE" linux test-source
    sleep 30 &
    printf "%s\n" "$!" >"$INHERITED_PID_FILE"
  '
inherited_child_pid="$(cat "$TEMP_DIR/inherited-child.pid")"
if ! kill -0 "$inherited_child_pid" 2>/dev/null; then
  printf 'inherited lock test child exited unexpectedly\n' >&2
  exit 1
fi
exec 9>>"$inherited_state/build-cache.lock"
if ! flock -n 9; then
  printf 'surviving child retained completed build ownership\n' >&2
  exit 1
fi
flock -u 9
exec 9>&-
kill "$inherited_child_pid" 2>/dev/null || true
wait "$inherited_child_pid" 2>/dev/null || true
inherited_child_pid=""

# AC: @truthful-container-build-startup ac-1
early_output="$TEMP_DIR/early-failure.log"
if LOCALBOORU_SOURCE_REVISION=does-not-exist \
    LOCALBOORU_WINDOWS_BUILD_ROOT="$TEMP_DIR/windows-early" \
    LOCALBOORU_DIST_WINDOWS_DIR="$TEMP_DIR/windows-early-dist" \
    "$ROOT/scripts/build-windows-local.sh" >"$early_output" 2>&1; then
  printf 'invalid source build unexpectedly succeeded\n' >&2
  exit 1
fi
grep -F 'LOCALBOORU_BUILD_STATUS=FAILED' "$early_output" >/dev/null
if grep -F 'LOCALBOORU_BUILD_STATUS=STARTED' "$early_output" >/dev/null; then
  printf 'preflight failure claimed it started\n' >&2
  exit 1
fi

# AC: @truthful-container-build-startup ac-2
: >"$BUILD_STATUS_TEST_DOCKER_LOG"
windows_output="$TEMP_DIR/windows-started.log"
LOCALBOORU_SOURCE_REVISION=HEAD \
LOCALBOORU_BUILD_JOBS=1 \
LOCALBOORU_WINDOWS_BUILD_ROOT="$TEMP_DIR/windows-build" \
LOCALBOORU_DIST_WINDOWS_DIR="$TEMP_DIR/windows-dist" \
  "$ROOT/scripts/build-windows-local.sh" >"$windows_output" 2>&1
source_commit="$(git -C "$ROOT" rev-parse HEAD)"
grep -F "LOCALBOORU_BUILD_STATUS=STARTED platform=windows source=$source_commit stage=artifacts" \
  "$windows_output" >/dev/null
grep -F "LOCALBOORU_SOURCE_REVISION=$source_commit" "$BUILD_STATUS_TEST_DOCKER_LOG" >/dev/null

: >"$BUILD_STATUS_TEST_DOCKER_LOG"
linux_missing_output="$TEMP_DIR/linux-missing-runtime.log"
if XDG_STATE_HOME="$TEMP_DIR/linux-missing-state" \
    LOCALBOORU_SOURCE_REVISION=HEAD \
    LOCALBOORU_DOCKER_BUILD_ROOT="$TEMP_DIR/linux-missing-build" \
    LOCALBOORU_CCACHE_DIR="$TEMP_DIR/linux-missing-ccache" \
    LOCALBOORU_DIST_LINUX_DIR="$TEMP_DIR/linux-missing-dist" \
    "$ROOT/scripts/build-linux-local.sh" --deb >"$linux_missing_output" 2>&1; then
  printf 'missing native runtime cache unexpectedly started a release\n' >&2
  exit 1
fi
grep -F 'LOCALBOORU_BUILD_STATUS=NEEDS_BOOTSTRAP platform=linux' "$linux_missing_output" >/dev/null
grep -F 'Refusing to take the release' "$linux_missing_output" >/dev/null
if grep -F 'LOCALBOORU_BUILD_STATUS=STARTED' "$linux_missing_output" >/dev/null; then
  printf 'missing native runtime cache claimed it started\n' >&2
  exit 1
fi
[[ ! -s "$BUILD_STATUS_TEST_DOCKER_LOG" ]]
[[ ! -e "$TEMP_DIR/linux-missing-state/localbooru/build-cache.owner" ]]

linux_build="$TEMP_DIR/linux-build"
patch_hash="$(sha256sum "$ROOT/patches/webkitgtk/2.52.3-playbin-video-filter.patch" | cut -d' ' -f1)"
mkdir -p \
  "$linux_build/webkit-build/lib" \
  "$linux_build/webkit-build/bin" \
  "$linux_build/webkitgtk-2.52.3" \
  "$linux_build/vapoursynth-stage"
printf '%s\n' 'localbooru-build-cache-v1' >"$linux_build/.localbooru-build-cache"
: >"$linux_build/webkit-build/.localbooru-config-ubuntu24-gtk3-v2"
touch "$linux_build/webkitgtk-2.52.3/.localbooru-patch-$patch_hash"
touch "$linux_build/vapoursynth-stage/.localbooru-vapoursynth-c05906995662bacd5bddf853d8e68f19286987db"
printf 'webkit runtime\n' >"$linux_build/webkit-build/lib/libwebkit2gtk-4.1.so.0"
printf 'javascript runtime\n' >"$linux_build/webkit-build/lib/libjavascriptcoregtk-4.1.so.0"
printf '#!/usr/bin/env bash\nexit 0\n' >"$linux_build/webkit-build/bin/WebKitWebProcess"
chmod +x "$linux_build/webkit-build/bin/WebKitWebProcess"

linux_output="$TEMP_DIR/linux-started.log"
LOCALBOORU_SOURCE_REVISION=HEAD \
LOCALBOORU_BUILD_JOBS=1 \
LOCALBOORU_DOCKER_BUILD_ROOT="$linux_build" \
LOCALBOORU_CCACHE_DIR="$TEMP_DIR/linux-ccache" \
LOCALBOORU_DIST_LINUX_DIR="$TEMP_DIR/linux-dist" \
  "$ROOT/scripts/build-linux-local.sh" --deb >"$linux_output" 2>&1
grep -F "LOCALBOORU_BUILD_STATUS=STARTED platform=linux source=$source_commit stage=artifacts" \
  "$linux_output" >/dev/null
grep -F "LOCALBOORU_SOURCE_REVISION=$source_commit" "$BUILD_STATUS_TEST_DOCKER_LOG" >/dev/null
grep -F 'LOCALBOORU_ALLOW_NATIVE_RUNTIME_BOOTSTRAP=0' "$BUILD_STATUS_TEST_DOCKER_LOG" >/dev/null

: >"$BUILD_STATUS_TEST_DOCKER_LOG"
linux_bootstrap_output="$TEMP_DIR/linux-bootstrap.log"
LOCALBOORU_SOURCE_REVISION=HEAD \
LOCALBOORU_BUILD_JOBS=1 \
LOCALBOORU_DOCKER_BUILD_ROOT="$TEMP_DIR/linux-bootstrap-build" \
LOCALBOORU_CCACHE_DIR="$TEMP_DIR/linux-bootstrap-ccache" \
LOCALBOORU_DIST_LINUX_DIR="$TEMP_DIR/linux-bootstrap-dist" \
  "$ROOT/scripts/build-linux-local.sh" --bootstrap-native-runtime --deb >"$linux_bootstrap_output" 2>&1
grep -F "LOCALBOORU_BUILD_STATUS=STARTED platform=linux source=$source_commit stage=artifacts" \
  "$linux_bootstrap_output" >/dev/null
grep -F 'LOCALBOORU_ALLOW_NATIVE_RUNTIME_BOOTSTRAP=1' "$BUILD_STATUS_TEST_DOCKER_LOG" >/dev/null

printf 'Build startup status tests passed\n'
