#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"

cleanup() {
  exec 9>&- 2>/dev/null || true
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p \
  "$TEMP_DIR/bin" \
  "$TEMP_DIR/home" \
  "$TEMP_DIR/state/localbooru"

cat >"$TEMP_DIR/bin/busctl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

for argument in "$@"; do
  case "$argument" in
    status)
      exit "${BUSCTL_TEST_OWNER:-1}"
      ;;
    call)
      if [[ "${BUSCTL_TEST_FAIL_ONCE:-0}" == "1" && \
          ! -f "$BUSCTL_TEST_FAIL_MARKER" ]]; then
        touch "$BUSCTL_TEST_FAIL_MARKER"
        exit 7
      fi
      printf '%s\n' "$@" >"$BUSCTL_TEST_LOG"
      exit "${BUSCTL_TEST_CALL_STATUS:-0}"
      ;;
  esac
done
exit 2
EOF
chmod +x "$TEMP_DIR/bin/busctl"

export HOME="$TEMP_DIR/home"
export XDG_DATA_HOME="$TEMP_DIR/data"
export XDG_STATE_HOME="$TEMP_DIR/state"
export LOCALBOORU_BUILD_LOCK_TIMEOUT=0
export LOCALBOORU_ENABLE_NATIVE_SVP=0
export BUSCTL_TEST_LOG="$TEMP_DIR/busctl-arguments.log"
export BUSCTL_TEST_FAIL_MARKER="$TEMP_DIR/busctl-failed-once"
export PATH="$TEMP_DIR/bin:$PATH"

exec 9>"$TEMP_DIR/state/localbooru/build-cache.lock"
flock -n 9

# AC: @desktop-launcher-continuity ac-1
# AC: @desktop-launcher-continuity ac-2
# AC: @desktop-launcher-continuity ac-3
BUSCTL_TEST_OWNER=0 BUSCTL_TEST_CALL_STATUS=0 \
  "$ROOT/run.sh" "/tmp/image with spaces.png"
printf '%s\n' \
  --user \
  --quiet \
  call \
  com.localbooru.app.SingleInstance \
  /com/localbooru/app/SingleInstance \
  org.SingleInstance.DBus \
  ExecuteCallback \
  ass \
  2 \
  localbooru \
  "/tmp/image with spaces.png" \
  "$PWD" >"$TEMP_DIR/expected-busctl-arguments.log"
cmp "$TEMP_DIR/expected-busctl-arguments.log" "$BUSCTL_TEST_LOG"

rm -f "$BUSCTL_TEST_LOG" "$BUSCTL_TEST_FAIL_MARKER"
BUSCTL_TEST_OWNER=0 BUSCTL_TEST_FAIL_ONCE=1 \
  "$ROOT/run.sh" "/tmp/retry image.png"
[[ -f "$BUSCTL_TEST_FAIL_MARKER" ]]
grep -Fx "/tmp/retry image.png" "$BUSCTL_TEST_LOG" >/dev/null

if BUSCTL_TEST_OWNER=0 BUSCTL_TEST_CALL_STATUS=7 \
    "$ROOT/run.sh" >/dev/null 2>&1; then
  printf 'launcher accepted a failed activation\n' >&2
  exit 1
fi

flock -u 9
exec 9>&-

mkdir -p \
  "$TEMP_DIR/data/localbooru/local-build" \
  "$TEMP_DIR/data/localbooru/local-build/frontend/dist"
cat >"$TEMP_DIR/data/localbooru/local-build/localbooru" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$LAUNCHER_TEST_LOG"
EOF
chmod +x "$TEMP_DIR/data/localbooru/local-build/localbooru"
printf '<!doctype html>\n' > \
  "$TEMP_DIR/data/localbooru/local-build/frontend/dist/index.html"
export LAUNCHER_TEST_LOG="$TEMP_DIR/installed-arguments.log"

# AC: @desktop-launcher-continuity ac-4
BUSCTL_TEST_OWNER=1 "$ROOT/run.sh" "/tmp/fallback image.png"
printf '%s\n' "/tmp/fallback image.png" >"$TEMP_DIR/expected-installed-arguments.log"
cmp "$TEMP_DIR/expected-installed-arguments.log" "$LAUNCHER_TEST_LOG"

printf 'Launcher behavior tests passed\n'
