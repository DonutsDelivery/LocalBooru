#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
APP="$DATA_HOME/localbooru/local-build/localbooru"
FRONTEND="$DATA_HOME/localbooru/local-build/frontend/dist/index.html"
STATE_DIR="$STATE_HOME/localbooru"
LOG="$STATE_DIR/run.log"
SINGLE_INSTANCE_SERVICE="com.localbooru.app.SingleInstance"
SINGLE_INSTANCE_PATH="/com/localbooru/app/SingleInstance"
SINGLE_INSTANCE_INTERFACE="org.SingleInstance.DBus"

mkdir -p "$STATE_DIR"

# Keep desktop-launch logs bounded.
if [[ -f "$LOG" ]] && (( $(stat -c %s "$LOG") > 5242880 )); then
  mv -f "$LOG" "$LOG.1"
fi

if command -v busctl >/dev/null 2>&1; then
  argument_count=$(( $# + 1 ))
  for _ in 1 2; do
    if ! busctl --user status "$SINGLE_INSTANCE_SERVICE" >/dev/null 2>&1; then
      break
    fi
    if busctl --user --quiet call \
        "$SINGLE_INSTANCE_SERVICE" \
        "$SINGLE_INSTANCE_PATH" \
        "$SINGLE_INSTANCE_INTERFACE" \
        ExecuteCallback \
        ass \
        "$argument_count" \
        localbooru \
        "$@" \
        "$PWD" >>"$LOG" 2>&1; then
      exit 0
    fi
  done

  if busctl --user status "$SINGLE_INSTANCE_SERVICE" >/dev/null 2>&1; then
    printf 'ERROR: Failed to activate the running LocalBooru instance\n' >&2
    exit 1
  fi
fi

if [[ ! -x "$APP" || ! -s "$FRONTEND" ]]; then
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "LocalBooru" "Building the local app for first launch…"
  fi

  if ! "$ROOT/scripts/install-local-app.sh" --if-missing >>"$LOG" 2>&1; then
    if command -v notify-send >/dev/null 2>&1; then
      notify-send -u critical "LocalBooru build failed" "See $LOG"
    fi
    exit 1
  fi
fi

cd "$ROOT"

WEBKIT_ROOT="${LOCALBOORU_WEBKIT_ROOT:-/mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched}"
PATCHED_WEBKIT_LIB="$WEBKIT_ROOT/local-build/lib"
PATCHED_WEB_PROCESS="$WEBKIT_ROOT/local-build/bin/mpv"
if [[ "${LOCALBOORU_ENABLE_NATIVE_SVP:-1}" == "1" && -d "$PATCHED_WEBKIT_LIB" && -x "$PATCHED_WEB_PROCESS" ]]; then
  export LOCALBOORU_ENABLE_NATIVE_SVP=1
  export LD_LIBRARY_PATH="$PATCHED_WEBKIT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LOCALBOORU_WEB_PROCESS_PATH="$PATCHED_WEB_PROCESS"
else
  export LOCALBOORU_ENABLE_NATIVE_SVP=0
fi

exec "$APP" "$@" >>"$LOG" 2>&1
