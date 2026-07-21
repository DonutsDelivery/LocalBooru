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
touch "$FAKE_NPM_ENTERED"
if [[ "${FAKE_NPM_BLOCK:-0}" == "1" ]]; then
  while [[ ! -f "$FAKE_NPM_RELEASE" ]]; do
    sleep 0.05
  done
fi
EOF
chmod +x "$TEMP_DIR/bin/npm"

export HOME="$TEMP_DIR/home"
export XDG_STATE_HOME="$TEMP_DIR/state"
export LOCALBOORU_DEV_TARGET_DIR="$TEMP_DIR/target"
export LOCALBOORU_WEBKIT_ROOT="$TEMP_DIR/no-webkit"
export FAKE_NPM_CALLS="$TEMP_DIR/calls"
export FAKE_NPM_ENTERED="$TEMP_DIR/entered"
export FAKE_NPM_RELEASE="$TEMP_DIR/release"
export PATH="$TEMP_DIR/bin:$PATH"

# AC: @safe-development-startup ac-single-launch
FAKE_NPM_BLOCK=1 "$ROOT/run-dev.sh" >"$TEMP_DIR/first-output" 2>&1 &
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

# AC: @safe-development-startup ac-single-launch
touch "$FAKE_NPM_RELEASE"
wait "$FIRST_PID"
FIRST_PID=""
rm -f "$FAKE_NPM_ENTERED" "$FAKE_NPM_RELEASE"

# AC: @safe-development-startup ac-build-lock-independence
"$ROOT/run-dev.sh" >"$TEMP_DIR/third-output" 2>&1
[[ "$(wc -l <"$FAKE_NPM_CALLS")" -eq 2 ]]

printf 'Development launcher lock tests passed\n'
