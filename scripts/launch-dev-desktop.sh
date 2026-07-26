#!/usr/bin/env bash
# Fast desktop entry for the last successfully built LocalBooru Dev binary.
# Rebuilding is intentionally separate: a desktop click must never wait behind
# the shared compiler lock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VITE_PORT="${LOCALBOORU_DEV_VITE_PORT:-5210}"
DEV_LOG="${LOCALBOORU_DEV_DESKTOP_LOG:-/tmp/localbooru-dev-desktop.log}"

if [[ -n "${LOCALBOORU_DEV_BINARY:-}" ]]; then
    DEV_BINARY="$LOCALBOORU_DEV_BINARY"
elif [[ -x /mnt/storage/Programs/localbooru-target-dev/debug/localbooru ]]; then
    DEV_BINARY="/mnt/storage/Programs/localbooru-target-dev/debug/localbooru"
else
    DEV_BINARY="$ROOT/target/debug/localbooru"
fi

if [[ ! -x "$DEV_BINARY" ]]; then
    echo "LocalBooru Dev binary is missing: $DEV_BINARY" >&2
    echo "Rebuild explicitly with: $ROOT/run-dev.sh" >&2
    exit 1
fi

vite_ready() {
    if command -v ss >/dev/null 2>&1; then
        ss -H -ltn "sport = :$VITE_PORT" 2>/dev/null | grep -q .
    else
        curl --fail --silent --max-time 1 "http://localhost:$VITE_PORT/" >/dev/null 2>&1
    fi
}

if ! vite_ready; then
    echo "Starting LocalBooru Dev frontend on port $VITE_PORT (log: $DEV_LOG)" >&2
    (
        cd "$ROOT/frontend"
        nohup npm run dev -- --port "$VITE_PORT" >>"$DEV_LOG" 2>&1 &
    )

    for _ in {1..40}; do
        vite_ready && break
        sleep 0.25
    done
fi

if ! vite_ready; then
    echo "LocalBooru Dev frontend did not open port $VITE_PORT; see $DEV_LOG" >&2
    exit 1
fi

exec "$DEV_BINARY" "$@"
