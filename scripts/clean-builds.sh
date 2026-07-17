#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
MODE="report"
EXECUTE=0
ASSUME_YES=0

usage() {
  cat <<'EOF'
Usage: scripts/clean-builds.sh [MODE] [--execute] [--yes]

Modes:
  report         Show all managed build/cache paths (default)
  dev            Remove the local Rust debug target
  android        Remove Tauri Android Gradle/CMake and Rust target outputs
  linux-cache    Remove the persistent Linux Docker build root and ccache
  windows-cache  Remove the persistent Windows build root and legacy sccache
  all-caches     Remove every cache listed above

Deletion is a dry run unless --execute is supplied. --yes skips the exact
interactive confirmation and is intended only for deliberate automation.
Release artifacts, installed apps, dependencies, and virtual environments are
never included.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    report|dev|android|linux-cache|windows-cache|all-caches) MODE="$1" ;;
    --execute) EXECUTE=1 ;;
    --yes) ASSUME_YES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option or mode: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

LINUX_BUILD_DEFAULT="$ROOT/build-linux-docker"
LINUX_CCACHE_DEFAULT="$ROOT/.ccache-docker"
WINDOWS_BUILD_DEFAULT="/mnt/storage/Programs/localbooru-build-windows-docker"
WINDOWS_SCCACHE_DEFAULT="/mnt/storage/Programs/localbooru-sccache-windows-docker"
LINUX_BUILD_ROOT="${LOCALBOORU_DOCKER_BUILD_ROOT:-$LINUX_BUILD_DEFAULT}"
LINUX_CCACHE_ROOT="${LOCALBOORU_CCACHE_DIR:-$LINUX_CCACHE_DEFAULT}"
WINDOWS_BUILD_ROOT="${LOCALBOORU_WINDOWS_BUILD_ROOT:-$WINDOWS_BUILD_DEFAULT}"
WINDOWS_SCCACHE_ROOT="${LOCALBOORU_WINDOWS_SCCACHE_ROOT:-$WINDOWS_SCCACHE_DEFAULT}"
DIST_LINUX_ROOT="${LOCALBOORU_DIST_LINUX_DIR:-$ROOT/dist-linux-local}"
DIST_WINDOWS_ROOT="${LOCALBOORU_DIST_WINDOWS_DIR:-$ROOT/dist-windows-local}"
CACHE_MARKER=".localbooru-build-cache"

DEV_PATHS=(
  "$ROOT/target/debug"
  "/mnt/storage/Programs/localbooru-target-dev/debug"
)
ANDROID_PATHS=(
  "$ROOT/src-tauri/gen/android/app/build"
  "$ROOT/src-tauri/gen/android/build"
  "$ROOT/src-tauri/gen/android/.gradle"
  "$ROOT/src-tauri/gen/android/app/.cxx"
  "$ROOT/target/aarch64-linux-android"
  "$ROOT/target/armv7-linux-androideabi"
  "$ROOT/target/i686-linux-android"
  "$ROOT/target/x86_64-linux-android"
  "/mnt/storage/Programs/localbooru-target-dev/aarch64-linux-android"
  "/mnt/storage/Programs/localbooru-target-dev/armv7-linux-androideabi"
  "/mnt/storage/Programs/localbooru-target-dev/i686-linux-android"
  "/mnt/storage/Programs/localbooru-target-dev/x86_64-linux-android"
)
LINUX_PATHS=(
  "$LINUX_BUILD_ROOT"
  "$LINUX_CCACHE_ROOT"
)
WINDOWS_PATHS=(
  "$WINDOWS_BUILD_ROOT"
  "$WINDOWS_SCCACHE_ROOT"
)
ALL_PATHS=(
  "${DEV_PATHS[@]}"
  "${ANDROID_PATHS[@]}"
  "${LINUX_PATHS[@]}"
  "${WINDOWS_PATHS[@]}"
)

case "$MODE" in
  report) PATHS=("${ALL_PATHS[@]}") ;;
  dev) PATHS=("${DEV_PATHS[@]}") ;;
  android) PATHS=("${ANDROID_PATHS[@]}") ;;
  linux-cache) PATHS=("${LINUX_PATHS[@]}") ;;
  windows-cache) PATHS=("${WINDOWS_PATHS[@]}") ;;
  all-caches) PATHS=("${ALL_PATHS[@]}") ;;
esac

canonical() {
  realpath -m -- "$1"
}

has_symlink_component() {
  local current
  current="$(realpath -ms -- "$1")"
  while true; do
    [[ -L "$current" ]] && return 0
    [[ "$current" == "/" ]] && return 1
    current="$(dirname "$current")"
  done
}

ROOT_REAL="$(canonical "$ROOT")"
HOME_REAL="$(canonical "$HOME")"
PROTECTED_PATHS=(
  "$DATA_HOME/localbooru/local-build/localbooru"
  "$ROOT/LocalBooru.apk"
  "$ROOT/updates"
  "$ROOT/dist"
  "$DIST_LINUX_ROOT"
  "$DIST_WINDOWS_ROOT"
)
FIXED_CACHE_PATHS=(
  "${DEV_PATHS[@]}"
  "${ANDROID_PATHS[@]}"
  "$LINUX_BUILD_DEFAULT"
  "$LINUX_CCACHE_DEFAULT"
  "$WINDOWS_BUILD_DEFAULT"
  "$WINDOWS_SCCACHE_DEFAULT"
)

assert_safe() {
  local path="$1"
  local resolved protected protected_resolved fixed fixed_resolved marker_value
  [[ -n "$path" ]] || { echo "ERROR: refusing empty path" >&2; return 1; }
  resolved="$(canonical "$path")"

  case "$resolved" in
    /|"$HOME_REAL"|"$ROOT_REAL")
      echo "ERROR: refusing unsafe path: $resolved" >&2
      return 1
      ;;
  esac

  for protected in "${PROTECTED_PATHS[@]}"; do
    protected_resolved="$(canonical "$protected")"
    if [[ "$resolved" == "$protected_resolved" ||
          "$resolved" == "$protected_resolved"/* ||
          "$protected_resolved" == "$resolved"/* ]]; then
      echo "ERROR: $resolved overlaps protected path $protected_resolved" >&2
      return 1
    fi
  done

  if [[ $EXECUTE -eq 0 ]]; then
    return 0
  fi

  if has_symlink_component "$path"; then
    echo "ERROR: refusing cleanup path with a symlink component: $path" >&2
    return 1
  fi

  for fixed in "${FIXED_CACHE_PATHS[@]}"; do
    fixed_resolved="$(canonical "$fixed")"
    if [[ "$resolved" == "$fixed_resolved" ]]; then
      return 0
    fi
  done

  marker_value=""
  if [[ -f "$resolved/$CACHE_MARKER" ]]; then
    marker_value="$(< "$resolved/$CACHE_MARKER")"
  fi
  if [[ "$marker_value" != "localbooru-build-cache-v1" ]]; then
    echo "ERROR: custom cleanup root lacks a valid $CACHE_MARKER marker: $resolved" >&2
    return 1
  fi
}

if [[ $EXECUTE -eq 1 ]]; then
  STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
  LOCK_FILE="$STATE_DIR/build-cache.lock"
  mkdir -p "$STATE_DIR"
  inherited_lock=""
  if [[ -e "/proc/$$/fd/8" ]]; then
    inherited_lock="$(readlink -f "/proc/$$/fd/8" 2>/dev/null || true)"
  fi
  if [[ "$inherited_lock" == "$(canonical "$LOCK_FILE")" ]] && flock -n 8; then
    :
  else
    exec 8>"$LOCK_FILE"
    flock -n 8 || {
      echo "ERROR: another LocalBooru build or cleanup is active" >&2
      exit 1
    }
  fi
fi

printf 'LocalBooru build cleanup: %s\n' "$MODE"
printf '%-10s %s\n' 'SIZE' 'PATH'
for path in "${PATHS[@]}"; do
  assert_safe "$path"
  resolved="$(canonical "$path")"
  if [[ -e "$resolved" || -L "$resolved" ]]; then
    size="$(du -sh -- "$resolved" 2>/dev/null | cut -f1)"
  else
    size="missing"
  fi
  printf '%-10s %s\n' "$size" "$resolved"
done

if [[ "$MODE" == "report" ]]; then
  exit 0
fi

if [[ $EXECUTE -eq 0 ]]; then
  echo
  echo "Dry run only. Re-run with --execute to delete these paths."
  exit 0
fi

if [[ $ASSUME_YES -eq 0 ]]; then
  if [[ ! -t 0 ]]; then
    echo "ERROR: interactive confirmation required; use --yes only for deliberate automation" >&2
    exit 2
  fi
  printf '\nType DELETE %s to continue: ' "$MODE"
  read -r confirmation
  [[ "$confirmation" == "DELETE $MODE" ]] || {
    echo "Cancelled."
    exit 1
  }
fi

for path in "${PATHS[@]}"; do
  assert_safe "$path"
  resolved="$(canonical "$path")"
  if [[ -e "$resolved" || -L "$resolved" ]]; then
    echo "Removing $resolved"
    rm -rf -- "$resolved"
  fi
done

echo "Cleanup finished."
