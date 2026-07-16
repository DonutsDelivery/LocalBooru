#!/usr/bin/env bash
# Build LocalBooru Windows x64 artifacts locally through MSVC-under-Wine Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
SCCACHE_ROOT="${LOCALBOORU_WINDOWS_SCCACHE_ROOT:-/mnt/storage/Programs/localbooru-sccache-windows-docker}"
DIST_PATH="${LOCALBOORU_DIST_WINDOWS_DIR:-$ROOT/dist-windows-local}"

if [[ -L "$DIST_PATH" ]]; then
  DIST_PATH="$(readlink -f "$DIST_PATH")"
fi
mkdir -p "$BUILD_ROOT" "$SCCACHE_ROOT" "$DIST_PATH"

if [[ "$REBUILD" == 1 ]] || ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Building Windows MSVC/Wine image $IMAGE"
  "$DOCKER" build --progress=plain -t "$IMAGE" -f "$DOCKERFILE" "$ROOT"
fi

printf '==> Building Windows x64 artifacts with %s jobs\n' "$JOBS"
"$DOCKER" run --rm --init \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e LOCALBOORU_BUILD_JOBS="$JOBS" \
  -e npm_config_cache=/build/npm-cache \
  -v "$ROOT:/source:ro" \
  -v "$BUILD_ROOT:/build" \
  -v "$SCCACHE_ROOT:/ccache" \
  -v "$DIST_PATH:/dist" \
  -w /build/worktree \
  "$IMAGE" \
  bash /source/scripts/build-windows-docker.sh

(
  cd "$DIST_PATH"
  sha256sum -c SHA256SUMS-Windows
)

printf '==> Local Windows artifacts verified in %s:\n' "$DIST_PATH"
find "$DIST_PATH" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' | sort
