#!/usr/bin/env bash

LOCALBOORU_BUILD_STATUS_REPORTED=0
LOCALBOORU_BUILD_LOCK_HELD=0
LOCALBOORU_BUILD_OWNER_FILE=""
LOCALBOORU_BUILD_PLATFORM=""
LOCALBOORU_BUILD_SOURCE=""

localbooru_build_emit_status() {
  local status="$1"
  shift
  printf 'LOCALBOORU_BUILD_STATUS=%s' "$status"
  if (($# > 0)); then
    printf ' %s' "$@"
  fi
  printf '\n'
  LOCALBOORU_BUILD_STATUS_REPORTED=1
}

localbooru_build_owner_value() {
  local key="$1"
  local owner_file="$2"
  [[ -f "$owner_file" ]] || return 1
  grep -m1 "^${key}=" "$owner_file" | cut -d= -f2-
}

localbooru_build_cleanup() {
  local exit_code=$?

  if [[ "$LOCALBOORU_BUILD_LOCK_HELD" == 1 && -n "$LOCALBOORU_BUILD_OWNER_FILE" ]]; then
    local owner_pid=""
    owner_pid="$(localbooru_build_owner_value pid "$LOCALBOORU_BUILD_OWNER_FILE" 2>/dev/null || true)"
    if [[ "$owner_pid" == "$$" ]]; then
      rm -f "$LOCALBOORU_BUILD_OWNER_FILE"
    fi
  fi

  if ((exit_code != 0)) && [[ "$LOCALBOORU_BUILD_STATUS_REPORTED" == 0 ]]; then
    localbooru_build_emit_status FAILED \
      "platform=${LOCALBOORU_BUILD_PLATFORM:-unknown}" \
      "source=${LOCALBOORU_BUILD_SOURCE:-unknown}" \
      "exit_code=$exit_code" >&2
  fi
}

localbooru_build_acquire_lock() {
  local state_dir="$1"
  local platform="$2"
  local requested_source="$3"
  local timeout="${LOCALBOORU_BUILD_LOCK_TIMEOUT:-0}"

  [[ "$timeout" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    printf 'ERROR: LOCALBOORU_BUILD_LOCK_TIMEOUT must be a nonnegative number\n' >&2
    return 2
  }

  LOCALBOORU_BUILD_PLATFORM="$platform"
  LOCALBOORU_BUILD_SOURCE="$requested_source"
  LOCALBOORU_BUILD_OWNER_FILE="$state_dir/build-cache.owner"
  mkdir -p "$state_dir"
  exec 8>>"$state_dir/build-cache.lock"

  local lock_acquired=0
  if [[ "$timeout" == 0 || "$timeout" == 0.0 ]]; then
    flock -n 8 && lock_acquired=1
  else
    flock -w "$timeout" 8 && lock_acquired=1
  fi

  if [[ "$lock_acquired" != 1 ]]; then
    local owner_pid owner_platform owner_source owner_started
    owner_pid="$(localbooru_build_owner_value pid "$LOCALBOORU_BUILD_OWNER_FILE" 2>/dev/null || true)"
    owner_platform="$(localbooru_build_owner_value platform "$LOCALBOORU_BUILD_OWNER_FILE" 2>/dev/null || true)"
    owner_source="$(localbooru_build_owner_value source "$LOCALBOORU_BUILD_OWNER_FILE" 2>/dev/null || true)"
    owner_started="$(localbooru_build_owner_value started "$LOCALBOORU_BUILD_OWNER_FILE" 2>/dev/null || true)"
    localbooru_build_emit_status LOCKED \
      "platform=$platform" \
      "source=$requested_source" \
      "owner_pid=${owner_pid:-unknown}" \
      "owner_platform=${owner_platform:-unknown}" \
      "owner_source=${owner_source:-unknown}" \
      "owner_started=${owner_started:-unknown}" >&2
    return 75
  fi

  LOCALBOORU_BUILD_LOCK_HELD=1
  trap localbooru_build_cleanup EXIT
  localbooru_build_write_owner "$requested_source"
}

localbooru_build_write_owner() {
  local source="$1"
  local temp_file
  LOCALBOORU_BUILD_SOURCE="$source"
  temp_file="${LOCALBOORU_BUILD_OWNER_FILE}.$$"
  {
    printf 'pid=%s\n' "$$"
    printf 'platform=%s\n' "$LOCALBOORU_BUILD_PLATFORM"
    printf 'source=%s\n' "$source"
    printf 'started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'cwd=%s\n' "$PWD"
    printf 'command=%s\n' "$0 $*"
  } >"$temp_file"
  mv -f "$temp_file" "$LOCALBOORU_BUILD_OWNER_FILE"
}

localbooru_build_started() {
  local source="$1"
  local stage="$2"
  [[ "$LOCALBOORU_BUILD_STATUS_REPORTED" == 0 ]] || return 0
  localbooru_build_emit_status STARTED \
    "platform=$LOCALBOORU_BUILD_PLATFORM" \
    "source=$source" \
    "stage=$stage"
}
