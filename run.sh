#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
APP="$DATA_HOME/localbooru/local-build/localbooru"
FRONTEND="$DATA_HOME/localbooru/local-build/frontend/dist/index.html"
STATE_DIR="$STATE_HOME/localbooru"
LOG="$STATE_DIR/run.log"
DEV_APP_PATHS=(
  "$ROOT/target/debug/localbooru"
  "/mnt/storage/Programs/localbooru-target-dev/debug/localbooru"
)
if [[ -n "${LOCALBOORU_DEV_TARGET_DIR:-}" ]]; then
  DEV_APP_PATHS+=("$LOCALBOORU_DEV_TARGET_DIR/debug/localbooru")
fi

mkdir -p "$STATE_DIR"

stop_development_instance() {
  local proc_link proc_exe pid dev_app
  local -a dev_pids=()

  for proc_link in /proc/[0-9]*/exe; do
    proc_exe="$(readlink -f "$proc_link" 2>/dev/null || true)"
    for dev_app in "${DEV_APP_PATHS[@]}"; do
      if [[ "$proc_exe" == "$dev_app" ]]; then
        pid="${proc_link#/proc/}"
        dev_pids+=("${pid%/exe}")
        break
      fi
    done
  done

  [[ ${#dev_pids[@]} -gt 0 ]] || return 0
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "LocalBooru" "Stopping the development instance before normal launch…"
  fi
  kill -TERM "${dev_pids[@]}" 2>/dev/null || true

  for _ in {1..50}; do
    local running=0
    for pid in "${dev_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && running=1
    done
    [[ $running -eq 0 ]] && return 0
    sleep 0.1
  done

  printf 'ERROR: LocalBooru development instance did not stop gracefully\n' >&2
  return 1
}

stop_development_instance

# Keep desktop-launch logs bounded.
if [[ -f "$LOG" ]] && (( $(stat -c %s "$LOG") > 5242880 )); then
  mv -f "$LOG" "$LOG.1"
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
