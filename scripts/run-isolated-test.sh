#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/localbooru-isolated.XXXXXX)"
APP_PID=""
HOST_HOME="${HOME:-/tmp}"
HOST_RUSTUP_HOME="${RUSTUP_HOME:-$HOST_HOME/.rustup}"
HOST_CARGO_HOME="${CARGO_HOME:-$HOST_HOME/.cargo}"
HOST_BUILD_GATE="${HOST_HEAVY_BUILD_GATE:-$HOST_HOME/.local/bin/host-heavy-build}"
HOST_BUILD_STATE_DIR="${HOST_HEAVY_BUILD_STATE_DIR:-${XDG_STATE_HOME:-$HOST_HOME/.local/state}/host-heavy-build}"

cleanup() {
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

PORT="${LOCALBOORU_TEST_PORT:-18790}"
READY_TIMEOUT_SECONDS="${LOCALBOORU_TEST_READY_TIMEOUT_SECONDS:-1800}"
if (( $# > 0 )); then
  printf 'the isolated launcher does not accept application file arguments; use synthetic test data through the test profile\n' >&2
  exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  printf 'LOCALBOORU_TEST_PORT must be an unused TCP port from 1024 to 65535\n' >&2
  exit 2
fi
if ! [[ "$READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]{0,4}$ ]] || (( READY_TIMEOUT_SECONDS > 14400 )); then
  printf 'LOCALBOORU_TEST_READY_TIMEOUT_SECONDS must be a positive integer from 1 to 14400\n' >&2
  exit 2
fi
if [[ ! -f "$HOST_BUILD_GATE" || ! -x "$HOST_BUILD_GATE" ]]; then
  printf 'isolated LocalBooru compilation requires an executable host build gate: %s\n' "$HOST_BUILD_GATE" >&2
  printf 'Install the host build gate or set HOST_HEAVY_BUILD_GATE to its executable path before retrying\n' >&2
  exit 75
fi
if ! python3 - "$PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
then
  printf 'isolated test port %s is already in use; choose LOCALBOORU_TEST_PORT\n' "$PORT" >&2
  exit 2
fi

mkdir -p \
  "$TEST_ROOT/home" \
  "$TEST_ROOT/config" \
  "$TEST_ROOT/xdg-data" \
  "$TEST_ROOT/state" \
  "$TEST_ROOT/tmp" \
  "$TEST_ROOT/target"

# Application paths are outside the repository and the user's normal profile.
# Toolchain paths and the host build gate retain their host-wide identity.
# The portable-data override is the application data boundary.
export HOME="$TEST_ROOT/home"
export RUSTUP_HOME="$HOST_RUSTUP_HOME"
export CARGO_HOME="$HOST_CARGO_HOME"
export HOST_HEAVY_BUILD_GATE="$HOST_BUILD_GATE"
export HOST_HEAVY_BUILD_STATE_DIR="$HOST_BUILD_STATE_DIR"
export PATH="$CARGO_HOME/bin:$PATH"
export XDG_CONFIG_HOME="$TEST_ROOT/config"
export XDG_DATA_HOME="$TEST_ROOT/xdg-data"
export XDG_STATE_HOME="$TEST_ROOT/state"
export TMPDIR="$TEST_ROOT/tmp"
export LOCALBOORU_PORTABLE_DATA="$TEST_ROOT/library"
export LOCALBOORU_PORT="$PORT"
export VITE_LOCALBOORU_PORT="$PORT"
export LOCALBOORU_DISABLE_SINGLE_INSTANCE=1
export LOCALBOORU_DEV_TARGET_DIR="$TEST_ROOT/target"
export LOCALBOORU_DEV_LOG="$TEST_ROOT/dev.log"
export LOCALBOORU_ENABLE_NATIVE_SVP=0

printf 'Starting isolated LocalBooru test instance\n'
printf '  data: %s\n' "$LOCALBOORU_PORTABLE_DATA"
printf '  state: %s\n' "$XDG_STATE_HOME"
printf '  port: %s\n' "$LOCALBOORU_PORT"
printf '  temporary root will be removed on exit\n'

"$ROOT/run-dev.sh" "$@" &
APP_PID=$!

READY_DEADLINE=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < READY_DEADLINE )); do
  PROBE_TIMEOUT_SECONDS=$((READY_DEADLINE - SECONDS))
  (( PROBE_TIMEOUT_SECONDS > 0 )) || break
  (( PROBE_TIMEOUT_SECONDS > 2 )) && PROBE_TIMEOUT_SECONDS=2
  if curl --fail --silent \
    --connect-timeout "$PROBE_TIMEOUT_SECONDS" \
    --max-time "$PROBE_TIMEOUT_SECONDS" \
    "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    printf 'Isolated LocalBooru instance is ready on http://127.0.0.1:%s\n' "$PORT"
    wait "$APP_PID"
    exit $?
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    printf 'isolated LocalBooru process exited before readiness\n' >&2
    tail -n 80 "$LOCALBOORU_DEV_LOG" >&2 2>/dev/null || true
    exit 1
  fi
  sleep 0.5
done

printf 'isolated LocalBooru instance did not become ready within %s seconds; increase LOCALBOORU_TEST_READY_TIMEOUT_SECONDS (maximum 14400) if the cold build needs longer\n' "$READY_TIMEOUT_SECONDS" >&2
tail -n 80 "$LOCALBOORU_DEV_LOG" >&2 2>/dev/null || true
exit 1
