#!/usr/bin/env bash
# Fast desktop entry for the last successfully built LocalBooru Dev binary.
# Rebuilding is intentionally separate: a desktop click must never wait behind
# the shared compiler lock.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VITE_PORT="${LOCALBOORU_DEV_VITE_PORT:-5210}"
DEV_LOG="${LOCALBOORU_DEV_DESKTOP_LOG:-/tmp/localbooru-dev-desktop.log}"
DEV_TARGET_DIR="${LOCALBOORU_DEV_TARGET_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/localbooru/builds/dev-target}"

if [[ -n "${LOCALBOORU_DEV_BINARY:-}" ]]; then
    DEV_BINARY="$LOCALBOORU_DEV_BINARY"
else
    DEV_BINARY="$DEV_TARGET_DIR/debug/localbooru"
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

# Desktop-entry launches skip run-dev.sh, so they must select the patched
# WebKit themselves. Without this, VR texImage2D uses system WebKit's
# software upload path and stutters; the Mac WKWebView path is unrelated.
WEBKIT_ROOT="${LOCALBOORU_WEBKIT_ROOT:-/mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched}"
PATCHED_WEBKIT_LIB="$WEBKIT_ROOT/local-build/lib"
PATCHED_WEB_PROCESS="$WEBKIT_ROOT/local-build/bin/WebKitWebProcess"
if [[ "${LOCALBOORU_ENABLE_NATIVE_SVP:-1}" == "1" && -d "$PATCHED_WEBKIT_LIB" && -x "$PATCHED_WEB_PROCESS" ]]; then
    export LOCALBOORU_ENABLE_NATIVE_SVP=1
    export LD_LIBRARY_PATH="$PATCHED_WEBKIT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LOCALBOORU_WEB_PROCESS_PATH="$PATCHED_WEB_PROCESS"
else
    export LOCALBOORU_ENABLE_NATIVE_SVP=0
fi

exec "$DEV_BINARY" "$@"
