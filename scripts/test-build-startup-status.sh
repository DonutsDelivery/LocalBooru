#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"

cleanup() {
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
linux_output="$TEMP_DIR/linux-started.log"
LOCALBOORU_SOURCE_REVISION=HEAD \
LOCALBOORU_BUILD_JOBS=1 \
LOCALBOORU_DOCKER_BUILD_ROOT="$TEMP_DIR/linux-build" \
LOCALBOORU_CCACHE_DIR="$TEMP_DIR/linux-ccache" \
LOCALBOORU_DIST_LINUX_DIR="$TEMP_DIR/linux-dist" \
  "$ROOT/scripts/build-linux-local.sh" --deb >"$linux_output" 2>&1
grep -F "LOCALBOORU_BUILD_STATUS=STARTED platform=linux source=$source_commit stage=artifacts" \
  "$linux_output" >/dev/null
grep -F "LOCALBOORU_SOURCE_REVISION=$source_commit" "$BUILD_STATUS_TEST_DOCKER_LOG" >/dev/null

printf 'Build startup status tests passed\n'
