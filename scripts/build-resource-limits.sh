#!/usr/bin/env bash
# Shared host preflight and hard limits for release artifact containers.

localbooru_build_resource_limits() {
  local name value
  LOCALBOORU_BUILD_MEMORY_GB="${LOCALBOORU_BUILD_MEMORY_GB:-12}"
  LOCALBOORU_BUILD_SWAP_GB="${LOCALBOORU_BUILD_SWAP_GB:-2}"
  LOCALBOORU_BUILD_CPUS="${LOCALBOORU_BUILD_CPUS:-2}"
  LOCALBOORU_BUILD_MIN_FREE_GB="${LOCALBOORU_BUILD_MIN_FREE_GB:-30}"
  LOCALBOORU_BUILD_ROOT_MIN_FREE_GB="${LOCALBOORU_BUILD_ROOT_MIN_FREE_GB:-50}"
  LOCALBOORU_CARGO_CACHE_LIMIT_GB="${LOCALBOORU_CARGO_CACHE_LIMIT_GB:-20}"
  for name in LOCALBOORU_BUILD_MEMORY_GB LOCALBOORU_BUILD_CPUS LOCALBOORU_BUILD_MIN_FREE_GB LOCALBOORU_BUILD_ROOT_MIN_FREE_GB LOCALBOORU_CARGO_CACHE_LIMIT_GB; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] || {
      echo "ERROR: $name must be a positive integer (at most 99999)" >&2
      return 2
    }
  done
  [[ "$LOCALBOORU_BUILD_SWAP_GB" =~ ^(0|[1-9][0-9]{0,4})$ ]] || {
    echo "ERROR: LOCALBOORU_BUILD_SWAP_GB must be a nonnegative integer (at most 99999)" >&2
    return 2
  }
  [[ "${LOCALBOORU_BUILD_JOBS:-1}" =~ ^[1-9][0-9]{0,4}$ && "${JOBS:-1}" =~ ^[1-9][0-9]{0,4}$ ]] || {
    echo "ERROR: LOCALBOORU_BUILD_JOBS / --jobs must be a positive integer" >&2
    return 2
  }
  # Docker/Podman --memory-swap is RAM + swap, not swap alone.
  LOCALBOORU_CONTAINER_LIMITS=(
    --memory "${LOCALBOORU_BUILD_MEMORY_GB}g"
    --memory-swap "$((LOCALBOORU_BUILD_MEMORY_GB + LOCALBOORU_BUILD_SWAP_GB))g"
    --cpus "$LOCALBOORU_BUILD_CPUS"
  )
}

localbooru_build_check_disk() {
  python3 - "$LOCALBOORU_BUILD_MIN_FREE_GB" "$LOCALBOORU_BUILD_ROOT_MIN_FREE_GB" "$@" <<'PY'
import os
from pathlib import Path
import sys

build_minimum = int(sys.argv[1]) * 1024 ** 3
root_minimum = int(sys.argv[2]) * 1024 ** 3
root_device = Path('/').stat().st_dev
seen = set()
for requested in sys.argv[3:]:
    path = Path(requested).resolve()
    while not path.exists():
        path = path.parent
    device = path.stat().st_dev
    minimum = max(build_minimum, root_minimum) if device == root_device else build_minimum
    if device in seen:
        continue
    seen.add(device)
    volume = os.statvfs(path)
    available = volume.f_bavail * volume.f_frsize
    if available < minimum:
        sys.exit(f'ERROR: {path} has {available / 1024 ** 3:.1f} GiB free; '
                 f'{minimum / 1024 ** 3:.0f} GiB required. Clear obsolete build caches or '
                 'choose a build filesystem with more space; refusing build.')
PY
}
