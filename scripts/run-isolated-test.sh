#!/usr/bin/env bash
set -euo pipefail
# The background setsid child must inherit this shell's group before creating
# its own session, so its PID is the owned process group used by cleanup.
set +m

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT=""
TEST_ROOT_IDENTITY=""
MAC_CONTAINER_PARENT="${LOCALBOORU_TEST_MACOS_CONTAINER_PARENT:-}"
MAC_CONTAINER_APP="${LOCALBOORU_TEST_MACOS_APP:-}"
APP_PID=""
APP_GROUP=""
HOST_HOME="${HOME:-/tmp}"
HOST_RUSTUP_HOME="${RUSTUP_HOME:-$HOST_HOME/.rustup}"
HOST_CARGO_HOME="${CARGO_HOME:-$HOST_HOME/.cargo}"
HOST_BUILD_GATE="${HOST_HEAVY_BUILD_GATE:-$HOST_HOME/.local/bin/host-heavy-build}"
HOST_BUILD_STATE_DIR="${HOST_HEAVY_BUILD_STATE_DIR:-${XDG_STATE_HOME:-$HOST_HOME/.local/state}/host-heavy-build}"

cleanup() {
  if [[ -n "$APP_GROUP" ]]; then
    # Linux acceptance owns this session/group, including the private bus and
    # its application descendants. Never target another app or user session.
    kill -TERM -- "-$APP_GROUP" 2>/dev/null || true
  elif [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "$APP_PID" ]]; then wait "$APP_PID" 2>/dev/null || true; fi
  if [[ -n "$TEST_ROOT_IDENTITY" ]]; then
    if ! HOME="$HOST_HOME" python3 "$ROOT/scripts/isolated-macos-root.py" cleanup \
      --app "$MAC_CONTAINER_APP" --parent "$MAC_CONTAINER_PARENT" \
      --root "$TEST_ROOT" --identity "$TEST_ROOT_IDENTITY"; then
      printf 'Disposable container cleanup failed; the owned fixture was preserved\n' >&2
      exit 1
    fi
  elif [[ -n "$TEST_ROOT" ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "$MAC_CONTAINER_PARENT" || -n "$MAC_CONTAINER_APP" ]]; then
  if [[ "$(uname -s)" != Darwin || -z "$MAC_CONTAINER_PARENT" || -z "$MAC_CONTAINER_APP" || -z "${LOCALBOORU_TEST_BINARY:-}" ]]; then
    printf 'Container isolation requires macOS, a verified Lite app and container parent, and a prebuilt candidate\n' >&2
    exit 2
  fi
  ROOT_RESULT="$(python3 "$ROOT/scripts/isolated-macos-root.py" create --app "$MAC_CONTAINER_APP" --parent "$MAC_CONTAINER_PARENT")"
  TEST_ROOT="${ROOT_RESULT%$'\n'*}"
  TEST_ROOT_IDENTITY="${ROOT_RESULT##*$'\n'}"
else
  TEST_ROOT="$(mktemp -d /tmp/localbooru-isolated.XXXXXX)"
fi

PORT="${LOCALBOORU_TEST_PORT:-18790}"
READY_TIMEOUT_SECONDS="${LOCALBOORU_TEST_READY_TIMEOUT_SECONDS:-1800}"
TEST_BINARY="${LOCALBOORU_TEST_BINARY:-}"
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
if [[ -n "$TEST_BINARY" ]] && [[ "$TEST_BINARY" != /* || ! -f "$TEST_BINARY" || ! -x "$TEST_BINARY" ]]; then
  printf 'LOCALBOORU_TEST_BINARY must be an absolute path to an executable candidate binary\n' >&2
  exit 2
fi
if [[ -z "$TEST_BINARY" ]] && [[ ! -f "$HOST_BUILD_GATE" || ! -x "$HOST_BUILD_GATE" ]]; then
  printf 'isolated LocalBooru compilation requires an executable host build gate: %s\n' "$HOST_BUILD_GATE" >&2
  printf 'Install the host build gate or set HOST_HEAVY_BUILD_GATE to its executable path before retrying\n' >&2
  exit 75
fi
SESSION_LAUNCH=()
if [[ "$(uname -s)" == "Linux" ]]; then
  DBUS_RUN_SESSION="$(command -v dbus-run-session || true)"
  SETSID="$(command -v setsid || true)"
  if [[ -z "$DBUS_RUN_SESSION" || ! -x "$DBUS_RUN_SESSION" || -z "$SETSID" || ! -x "$SETSID" ]]; then
    printf 'isolated Linux acceptance requires dbus-run-session and setsid; shared desktop sessions are never used\n' >&2
    exit 75
  fi
  SESSION_LAUNCH=("$SETSID" --wait "$DBUS_RUN_SESSION" --)
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
  "$TEST_ROOT/cache" \
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
export XDG_CACHE_HOME="$TEST_ROOT/cache"
export XDG_DATA_HOME="$TEST_ROOT/xdg-data"
export XDG_STATE_HOME="$TEST_ROOT/state"
export TMPDIR="$TEST_ROOT/tmp"
export LOCALBOORU_PORTABLE_DATA="$TEST_ROOT/library"
export LOCALBOORU_PORT="$PORT"
export VITE_LOCALBOORU_PORT="$PORT"
export LOCALBOORU_DISABLE_SINGLE_INSTANCE=1
export LOCALBOORU_DEV_TARGET_DIR="$TEST_ROOT/target"
export CARGO_TARGET_DIR="$TEST_ROOT/target"
export LOCALBOORU_DEV_LOG="$TEST_ROOT/dev.log"
export LOCALBOORU_ENABLE_NATIVE_SVP=0
LOCALBOORU_ISOLATED_INSTANCE_ID="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export LOCALBOORU_ISOLATED_INSTANCE_ID
export VITE_LOCALBOORU_ISOLATED_INSTANCE_ID="$LOCALBOORU_ISOLATED_INSTANCE_ID"

printf 'Starting isolated LocalBooru test instance\n'
printf '  data: %s\n' "$LOCALBOORU_PORTABLE_DATA"
printf '  state: %s\n' "$XDG_STATE_HOME"
printf '  port: %s\n' "$LOCALBOORU_PORT"
printf '  temporary root will be removed on exit\n'

if [[ -n "$TEST_BINARY" ]]; then
  printf '  candidate binary: %s\n' "$TEST_BINARY"
  "${SESSION_LAUNCH[@]}" "$TEST_BINARY" >"$LOCALBOORU_DEV_LOG" 2>&1 &
else
  "${SESSION_LAUNCH[@]}" "$ROOT/run-dev.sh" &
fi
APP_PID=$!
if (( ${#SESSION_LAUNCH[@]} > 0 )); then APP_GROUP="$APP_PID"; fi

READY_DEADLINE=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < READY_DEADLINE )); do
  PROBE_TIMEOUT_SECONDS=$((READY_DEADLINE - SECONDS))
  (( PROBE_TIMEOUT_SECONDS > 0 )) || break
  (( PROBE_TIMEOUT_SECONDS > 2 )) && PROBE_TIMEOUT_SECONDS=2
  if HEALTH_RESPONSE="$(curl --fail --silent \
    --connect-timeout "$PROBE_TIMEOUT_SECONDS" \
    --max-time "$PROBE_TIMEOUT_SECONDS" \
    --max-filesize 4096 \
    "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
    if ! printf '%s' "$HEALTH_RESPONSE" | python3 -c '
import json, os, sys
try:
    response = json.load(sys.stdin)
    valid = (isinstance(response, dict) and response.get("status") == "healthy"
             and response.get("isolated_instance_id") == os.environ["LOCALBOORU_ISOLATED_INSTANCE_ID"])
except (ValueError, TypeError, KeyError):
    valid = False
sys.exit(0 if valid else 1)
'; then
      printf 'isolated readiness response does not belong to this launch; refusing foreign backend\n' >&2
      exit 1
    fi
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
