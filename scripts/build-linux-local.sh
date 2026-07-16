#!/usr/bin/env bash
# Build LocalBooru's Linux release artifacts locally in Docker/Podman.
# This is the authoritative Linux release entry point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE_HASH="$(sha256sum "$ROOT/Dockerfile.linux-release" | cut -c1-12)"
IMAGE="localbooru-linux-release:ubuntu24.04-webkit2.52.3-v1-$DOCKERFILE_HASH"
REBUILD=0
BUNDLES="appimage,deb,rpm"
JOBS="${LOCALBOORU_BUILD_JOBS:-$(nproc)}"

usage() {
  cat <<'EOF'
Usage: scripts/build-linux-local.sh [OPTIONS]

Options:
  --appimage       Build only the AppImage
  --deb            Build only the Debian package
  --rpm            Build only the RPM package
  --rebuild-image  Rebuild the release toolchain image
  --jobs N         Limit parallel compilation (default: host CPU count)
  -h, --help       Show this help

Environment:
  LOCALBOORU_DOCKER_BUILD_ROOT  Persistent build/cache directory
  LOCALBOORU_DIST_LINUX_DIR     Final artifact directory
  LOCALBOORU_BUILD_JOBS         Parallel build limit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --appimage) BUNDLES="appimage"; shift ;;
    --deb) BUNDLES="deb"; shift ;;
    --rpm) BUNDLES="rpm"; shift ;;
    --rebuild-image) REBUILD=1; shift ;;
    --jobs)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: --jobs requires a positive integer" >&2
        exit 2
      }
      JOBS="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if command -v docker >/dev/null 2>&1; then
  CONTAINER=docker
elif command -v podman >/dev/null 2>&1; then
  CONTAINER=podman
else
  echo "ERROR: Docker or Podman is required" >&2
  exit 1
fi

BUILD_ROOT="${LOCALBOORU_DOCKER_BUILD_ROOT:-$ROOT/build-linux-docker}"
DIST_ROOT="${LOCALBOORU_DIST_LINUX_DIR:-$ROOT/dist-linux-local}"
CCACHE_ROOT="${LOCALBOORU_CCACHE_DIR:-$ROOT/.ccache-docker}"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct)}"
for directory in "$BUILD_ROOT" "$DIST_ROOT" "$CCACHE_ROOT"; do
  mkdir -p "$directory"
done
mkdir -p "$BUILD_ROOT/cargo-home" "$BUILD_ROOT/npm-cache"
BUILD_ROOT="$(readlink -f "$BUILD_ROOT")"
DIST_ROOT="$(readlink -f "$DIST_ROOT")"
CCACHE_ROOT="$(readlink -f "$CCACHE_ROOT")"

if [[ "$REBUILD" == 1 ]] || ! "$CONTAINER" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Building release toolchain image $IMAGE"
  "$CONTAINER" build -t "$IMAGE" - < "$ROOT/Dockerfile.linux-release"
fi

echo "==> Building LocalBooru Linux artifacts"
echo "    bundles: $BUNDLES"
echo "    jobs:    $JOBS"
echo "    build:   $BUILD_ROOT"
echo "    output:  $DIST_ROOT"

"$CONTAINER" run --rm \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e CARGO_HOME=/cargo-home \
  -e RUSTUP_HOME=/opt/rustup \
  -e NPM_CONFIG_CACHE=/build/npm-cache \
  -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  -e TZ=UTC \
  -e LC_ALL=C.UTF-8 \
  -e LOCALBOORU_BUILD_JOBS="$JOBS" \
  -e LOCALBOORU_RELEASE_BUNDLES="$BUNDLES" \
  -e CARGO_TARGET_DIR=/build/target \
  -e CCACHE_DIR=/ccache \
  -e CCACHE_MAXSIZE="${LOCALBOORU_CCACHE_SIZE:-30G}" \
  -v "$ROOT:/source:ro" \
  -v "$BUILD_ROOT:/build" \
  -v "$BUILD_ROOT/cargo-home:/cargo-home" \
  -v "$DIST_ROOT:/dist" \
  -v "$CCACHE_ROOT:/ccache" \
  -w /build \
  "$IMAGE" \
  bash /source/scripts/build-linux-docker.sh

echo
echo "==> LocalBooru Linux artifacts"
find "$DIST_ROOT" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' | sort
