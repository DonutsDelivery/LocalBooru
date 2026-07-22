#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
FIRST_PID=""

cleanup() {
  if [[ -n "$FIRST_PID" ]] && kill -0 "$FIRST_PID" 2>/dev/null; then
    touch "$TEMP_DIR/release"
    wait "$FIRST_PID" 2>/dev/null || true
  fi
  exec 9>&- 2>/dev/null || true
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR/bin" "$TEMP_DIR/home" "$TEMP_DIR/state/localbooru" "$TEMP_DIR/target"
cat >"$TEMP_DIR/bin/npm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf 'call\n' >>"$FAKE_NPM_CALLS"
printf '%s\n' "${LOCALBOORU_TASK_QUEUE_WORKERS:-unset}" >>"$FAKE_NPM_WORKERS"
printf '%s\n' "${CARGO_BUILD_JOBS:-unset}" >>"$FAKE_NPM_CARGO_JOBS"
printf '%s\n' "${RUSTC_WRAPPER:-unset}" >>"$FAKE_NPM_RUSTC_WRAPPERS"
printf '%s\n' "$@" >"$FAKE_NPM_ARGUMENTS"
touch "$FAKE_NPM_ENTERED"
if [[ "${FAKE_NPM_BLOCK:-0}" == "1" ]]; then
  while [[ ! -f "$FAKE_NPM_RELEASE" ]]; do
    sleep 0.05
  done
fi
EOF
chmod +x "$TEMP_DIR/bin/npm"
cat >"$TEMP_DIR/bin/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${FAKE_SS_BUSY_PORT:-}" && "$*" == *":$FAKE_SS_BUSY_PORT"* ]]; then
  printf 'LISTEN 0 511 127.0.0.1:%s 0.0.0.0:*\n' "$FAKE_SS_BUSY_PORT"
fi
EOF
chmod +x "$TEMP_DIR/bin/ss"

export HOME="$TEMP_DIR/home"
export XDG_STATE_HOME="$TEMP_DIR/state"
export LOCALBOORU_DEV_TARGET_DIR="$TEMP_DIR/target"
export LOCALBOORU_DEV_LOG="$TEMP_DIR/dev.log"
export LOCALBOORU_WEBKIT_ROOT="$TEMP_DIR/no-webkit"
export FAKE_NPM_CALLS="$TEMP_DIR/calls"
export FAKE_NPM_WORKERS="$TEMP_DIR/workers"
export FAKE_NPM_CARGO_JOBS="$TEMP_DIR/cargo-jobs"
export FAKE_NPM_RUSTC_WRAPPERS="$TEMP_DIR/rustc-wrappers"
export FAKE_NPM_ARGUMENTS="$TEMP_DIR/arguments"
export FAKE_NPM_ENTERED="$TEMP_DIR/entered"
export FAKE_NPM_RELEASE="$TEMP_DIR/release"
export PATH="$TEMP_DIR/bin:$PATH"
unset LOCALBOORU_TASK_QUEUE_WORKERS

# AC: @safe-development-startup ac-single-launch
FAKE_NPM_BLOCK=1 "$ROOT/run-dev.sh" "/tmp/video with spaces.mp4" \
  >"$TEMP_DIR/first-output" 2>&1 &
FIRST_PID=$!
for _ in $(seq 1 100); do
  [[ -f "$FAKE_NPM_ENTERED" ]] && break
  sleep 0.05
done
[[ -f "$FAKE_NPM_ENTERED" ]]

# AC: @safe-development-startup ac-duplicate-launch
if "$ROOT/run-dev.sh" >"$TEMP_DIR/second-output" 2>&1; then
  printf 'duplicate development launch unexpectedly succeeded\n' >&2
  exit 1
fi
grep -F 'A LocalBooru development session is already running.' \
  "$TEMP_DIR/second-output" >/dev/null
[[ "$(wc -l <"$FAKE_NPM_CALLS")" -eq 1 ]]

# AC: @safe-development-startup ac-build-lock-independence
exec 9>"$TEMP_DIR/state/localbooru/build-cache.lock"
flock -n 9

touch "$FAKE_NPM_RELEASE"
wait "$FIRST_PID"
FIRST_PID=""
rm -f "$FAKE_NPM_ENTERED" "$FAKE_NPM_RELEASE"

# AC: @safe-development-startup ac-duplicate-launch
if FAKE_SS_BUSY_PORT=5210 "$ROOT/run-dev.sh" >"$TEMP_DIR/port-output" 2>&1; then
  printf 'development launch ignored a busy Vite port\n' >&2
  exit 1
fi
grep -F 'port 5210 is already in use' "$TEMP_DIR/port-output" >/dev/null
[[ "$(wc -l <"$FAKE_NPM_CALLS")" -eq 1 ]]

# AC: @safe-development-startup ac-single-launch
# AC: @safe-development-startup ac-build-lock-independence
"$ROOT/run-dev.sh" "/tmp/second video.mp4" >/dev/null
[[ "$(wc -l <"$FAKE_NPM_CALLS")" -eq 2 ]]

# AC: @safe-development-startup ac-bounded-workers
mapfile -t worker_values <"$FAKE_NPM_WORKERS"
[[ "${worker_values[0]}" == "1" ]]
[[ "${worker_values[1]}" == "1" ]]
mapfile -t cargo_job_values <"$FAKE_NPM_CARGO_JOBS"
[[ "${cargo_job_values[0]}" == "1" ]]
[[ "${cargo_job_values[1]}" == "1" ]]
mapfile -t rustc_wrappers <"$FAKE_NPM_RUSTC_WRAPPERS"
[[ "${rustc_wrappers[0]}" == "$ROOT/scripts/rustc-host-heavy-build.sh" ]]
[[ "${rustc_wrappers[1]}" == "$ROOT/scripts/rustc-host-heavy-build.sh" ]]

LOCALBOORU_TASK_QUEUE_WORKERS=3 "$ROOT/run-dev.sh" >/dev/null
mapfile -t worker_values <"$FAKE_NPM_WORKERS"
[[ "${worker_values[2]}" == "3" ]]

printf '%s\n' \
  run \
  tauri:dev \
  -- \
  -- \
  -- \
  >"$TEMP_DIR/expected-arguments"
cmp "$TEMP_DIR/expected-arguments" "$FAKE_NPM_ARGUMENTS"

printf 'Development launcher safety tests passed\n'
