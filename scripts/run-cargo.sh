#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
LOCK_TIMEOUT="${LOCALBOORU_BUILD_LOCK_TIMEOUT:-1800}"
JOBS="${LOCALBOORU_BUILD_JOBS:-2}"

[[ "$LOCK_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "ERROR: LOCALBOORU_BUILD_LOCK_TIMEOUT must be a nonnegative number" >&2
  exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: LOCALBOORU_BUILD_JOBS must be a positive integer" >&2
  exit 2
}
(($# > 0)) || {
  echo "Usage: scripts/run-cargo.sh <cargo arguments...>" >&2
  exit 2
}

mkdir -p "$STATE_DIR"
exec 8>>"$STATE_DIR/build-cache.lock"
if ! flock -w "$LOCK_TIMEOUT" 8; then
  echo "ERROR: timed out waiting for another LocalBooru Cargo or release build" >&2
  exit 75
fi

export CARGO_BUILD_JOBS="$JOBS"
exec cargo "$@"
