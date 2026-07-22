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
cat >"$TEMP_DIR/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${CARGO_BUILD_JOBS:-unset}" >>"$FAKE_CARGO_JOBS"
printf '%s\n' "$@" >"$FAKE_CARGO_ARGS"
EOF
chmod +x "$TEMP_DIR/bin/cargo"

export HOME="$TEMP_DIR/home"
export XDG_STATE_HOME="$TEMP_DIR/state"
export PATH="$TEMP_DIR/bin:$PATH"
export FAKE_CARGO_JOBS="$TEMP_DIR/jobs"
export FAKE_CARGO_ARGS="$TEMP_DIR/args"
export LOCALBOORU_BUILD_LOCK_TIMEOUT=0

"$ROOT/scripts/run-cargo.sh" test --workspace
[[ "$(<"$FAKE_CARGO_JOBS")" == "2" ]]
printf '%s\n' test --workspace >"$TEMP_DIR/expected-args"
cmp "$TEMP_DIR/expected-args" "$FAKE_CARGO_ARGS"

: >"$FAKE_CARGO_JOBS"
LOCALBOORU_BUILD_JOBS=1 "$ROOT/scripts/run-cargo.sh" check
[[ "$(<"$FAKE_CARGO_JOBS")" == "1" ]]

exec 9>>"$XDG_STATE_HOME/localbooru/build-cache.lock"
flock -n 9
if "$ROOT/scripts/run-cargo.sh" test >"$TEMP_DIR/locked-output" 2>&1; then
  echo "run-cargo unexpectedly bypassed the active build lock" >&2
  exit 1
fi
grep -F 'timed out waiting for another LocalBooru Cargo or release build' \
  "$TEMP_DIR/locked-output" >/dev/null

printf 'Cargo build gate tests passed\n'
