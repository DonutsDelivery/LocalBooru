#!/bin/bash
source "$HOME/.cargo/env" 2>/dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${LOCALBOORU_DEV_TARGET_DIR:-}" ]]; then
    export CARGO_TARGET_DIR="$LOCALBOORU_DEV_TARGET_DIR"
elif [[ -d /mnt/storage/Programs && -w /mnt/storage/Programs ]]; then
    export CARGO_TARGET_DIR="/mnt/storage/Programs/localbooru-target-dev"
else
    export CARGO_TARGET_DIR="$ROOT/target"
fi
mkdir -p "$CARGO_TARGET_DIR"
STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
STATE_DIR="$STATE_HOME/localbooru"
mkdir -p "$STATE_DIR"
exec 7>"$STATE_DIR/dev-instance.lock"
if ! flock -n 7; then
    echo "A LocalBooru development session is already running." >&2
    echo "Stop that session before starting another development launch." >&2
    exit 1
fi
cd "$ROOT"
# HW acceleration defaults are applied by the Tauri process when unset.
# The optional patched WebKit runtime still has to be selected before launch.
WEBKIT_ROOT="${LOCALBOORU_WEBKIT_ROOT:-/mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched}"
PATCHED_WEBKIT_LIB="$WEBKIT_ROOT/local-build/lib"
PATCHED_WEB_PROCESS="$WEBKIT_ROOT/local-build/bin/mpv"
if [ "${LOCALBOORU_ENABLE_NATIVE_SVP:-1}" = "1" ] && [ -d "$PATCHED_WEBKIT_LIB" ] && [ -x "$PATCHED_WEB_PROCESS" ]; then
    export LOCALBOORU_ENABLE_NATIVE_SVP=1
    export LD_LIBRARY_PATH="$PATCHED_WEBKIT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LOCALBOORU_WEB_PROCESS_PATH="$PATCHED_WEB_PROCESS"
else
    export LOCALBOORU_ENABLE_NATIVE_SVP=0
fi
exec npm run tauri:dev -- -- -- "$@" >> /tmp/localbooru-dev.log 2>&1
