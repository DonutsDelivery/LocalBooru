#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
LOCK_TIMEOUT="${LOCALBOORU_BUILD_LOCK_TIMEOUT:-1800}"
JOBS="${LOCALBOORU_BUILD_JOBS:-1}"

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

LOCK_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/host-heavy-build"
mkdir -p "$LOCK_DIR"
export CARGO_BUILD_JOBS="$JOBS"

# macOS has flock(2), but no flock command in its standard tools.
if [[ "$(uname -s)" == Darwin ]]; then
  exec python3 - "$LOCK_DIR/heavy-build.lock" "$LOCK_TIMEOUT" "$ROOT/scripts/cargo-cache-hygiene.py" "$@" <<'PY'
import fcntl, os, subprocess, sys, time
lock = open(sys.argv[1], 'a')
deadline = time.monotonic() + float(sys.argv[2])
while True:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic() >= deadline:
            sys.exit('ERROR: timed out waiting for the host heavy-build gate')
        time.sleep(0.1)
os.set_inheritable(lock.fileno(), True)
subprocess.run(['python3', sys.argv[3], *sys.argv[4:]], check=True)
result = subprocess.run(['cargo', *sys.argv[4:]])
hygiene = subprocess.run(['python3', sys.argv[3], *sys.argv[4:]])
sys.exit(result.returncode or hygiene.returncode)
PY
fi

exec 8>>"$LOCK_DIR/heavy-build.lock"
if ! flock -w "$LOCK_TIMEOUT" 8; then
  echo "ERROR: timed out waiting for another LocalBooru Cargo or release build" >&2
  exit 75
fi

python3 "$ROOT/scripts/cargo-cache-hygiene.py" "$@"
result=0
cargo "$@" || result=$?
hygiene_result=0
python3 "$ROOT/scripts/cargo-cache-hygiene.py" "$@" || hygiene_result=$?
(( result != 0 )) || result=$hygiene_result
exit "$result"
