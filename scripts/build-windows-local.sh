#!/usr/bin/env bash
# Build LocalBooru Windows x64 artifacts locally through MSVC-under-Wine Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
LOCK_TIMEOUT="${LOCALBOORU_BUILD_LOCK_TIMEOUT:-1800}"
[[ "$LOCK_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "ERROR: LOCALBOORU_BUILD_LOCK_TIMEOUT must be a nonnegative number" >&2
  exit 2
}
mkdir -p "$STATE_DIR"
exec 8>"$STATE_DIR/build-cache.lock"
flock -w "$LOCK_TIMEOUT" 8 || {
  echo "ERROR: timed out waiting for another LocalBooru build or cleanup" >&2
  exit 1
}
DOCKERFILE="$ROOT/Dockerfile.windows-release"
REBUILD=0
JOBS="${LOCALBOORU_BUILD_JOBS:-$(nproc)}"

for arg in "$@"; do
  case "$arg" in
    --rebuild-image) REBUILD=1 ;;
    --jobs=*) JOBS="${arg#*=}" ;;
    *) echo "ERROR: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if command -v docker >/dev/null 2>&1; then
  DOCKER=docker
elif command -v podman >/dev/null 2>&1; then
  DOCKER=podman
else
  echo "ERROR: docker or podman is required" >&2
  exit 1
fi

DOCKERFILE_HASH="$(sha256sum "$DOCKERFILE" | cut -c1-12)"
IMAGE="localbooru-windows-release:msvc-wine-$DOCKERFILE_HASH"
BUILD_ROOT="${LOCALBOORU_WINDOWS_BUILD_ROOT:-/mnt/storage/Programs/localbooru-build-windows-docker}"
DIST_PATH="${LOCALBOORU_DIST_WINDOWS_DIR:-$ROOT/dist-windows-local}"
BUILD_LIMIT_GB="${LOCALBOORU_WINDOWS_BUILD_LIMIT_GB:-30}"
CACHE_MARKER=".localbooru-build-cache"
BUILD_ROOT_DEFAULT="/mnt/storage/Programs/localbooru-build-windows-docker"

has_symlink_component() {
  local current
  current="$(realpath -ms -- "$1")"
  while true; do
    [[ -L "$current" ]] && return 0
    [[ "$current" == "/" ]] && return 1
    current="$(dirname "$current")"
  done
}

if has_symlink_component "$BUILD_ROOT"; then
  echo "ERROR: refusing build cache path with a symlink component: $BUILD_ROOT" >&2
  exit 1
fi

BUILD_ROOT="$(realpath -m -- "$BUILD_ROOT")"
BUILD_ROOT_DEFAULT="$(realpath -m -- "$BUILD_ROOT_DEFAULT")"
marker_value=""
if [[ -f "$BUILD_ROOT/$CACHE_MARKER" ]]; then
  marker_value="$(< "$BUILD_ROOT/$CACHE_MARKER")"
fi
if [[ "$BUILD_ROOT" != "$BUILD_ROOT_DEFAULT" && "$marker_value" != "localbooru-build-cache-v1" ]]; then
  first_entry=""
  if [[ -d "$BUILD_ROOT" ]]; then
    first_entry="$(find "$BUILD_ROOT" -mindepth 1 -maxdepth 1 \
      ! -name "$CACHE_MARKER" ! -name '.localbooru-cache-marker.*' -print -quit)"
  fi
  if [[ -n "$first_entry" ]]; then
    echo "ERROR: custom build cache is nonempty and lacks a valid $CACHE_MARKER marker: $BUILD_ROOT" >&2
    exit 1
  fi
  rm -f "$BUILD_ROOT/$CACHE_MARKER" "$BUILD_ROOT"/.localbooru-cache-marker.*
fi
mkdir -p "$BUILD_ROOT"
rm -f "$BUILD_ROOT"/.localbooru-cache-marker.*
if [[ "$marker_value" != "localbooru-build-cache-v1" ]]; then
  marker_temp="$(mktemp "$BUILD_ROOT/.localbooru-cache-marker.XXXXXX")"
  if ! printf '%s\n' 'localbooru-build-cache-v1' > "$marker_temp"; then
    rm -f "$marker_temp"
    echo "ERROR: failed to initialize cache marker for $BUILD_ROOT" >&2
    exit 1
  fi
  mv -f "$marker_temp" "$BUILD_ROOT/$CACHE_MARKER"
fi

if [[ -L "$DIST_PATH" ]]; then
  DIST_PATH="$(readlink -f "$DIST_PATH")"
fi
mkdir -p "$DIST_PATH"
DIST_PATH="$(readlink -f "$DIST_PATH")"

[[ "$BUILD_LIMIT_GB" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: LOCALBOORU_WINDOWS_BUILD_LIMIT_GB must be a positive integer" >&2
  exit 2
}
BUILD_USAGE_KIB="$(du -sk "$BUILD_ROOT" | cut -f1)"
BUILD_LIMIT_KIB=$((BUILD_LIMIT_GB * 1024 * 1024))
if (( BUILD_USAGE_KIB > BUILD_LIMIT_KIB )); then
  echo "ERROR: Windows build cache is $(du -sh "$BUILD_ROOT" | cut -f1), above the ${BUILD_LIMIT_GB}G limit." >&2
  echo "Run: npm run clean:builds -- windows-cache --execute" >&2
  exit 1
fi
printf '==> Windows persistent build cache before build: %s (limit: %sG)\n' \
  "$(du -sh "$BUILD_ROOT" | cut -f1)" "$BUILD_LIMIT_GB"

if [[ "$REBUILD" == 1 ]] || ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Building Windows MSVC/Wine image $IMAGE"
  "$DOCKER" build -t "$IMAGE" - < "$DOCKERFILE"
fi

printf '==> Building Windows x64 artifacts with %s jobs\n' "$JOBS"
"$DOCKER" run --rm --init \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e LOCALBOORU_BUILD_JOBS="$JOBS" \
  -e npm_config_cache=/build/npm-cache \
  -v "$ROOT:/source:ro" \
  -v "$BUILD_ROOT:/build" \
  -v "$DIST_PATH:/dist" \
  -w /build/worktree \
  "$IMAGE" \
  bash /source/scripts/build-windows-docker.sh

(
  cd "$DIST_PATH"
  sha256sum -c SHA256SUMS-Windows
)

printf '==> Windows persistent build cache after build: %s\n' "$(du -sh "$BUILD_ROOT" | cut -f1)"
printf '==> Local Windows artifacts verified in %s:\n' "$DIST_PATH"
find "$DIST_PATH" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' | sort
