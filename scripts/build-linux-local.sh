#!/usr/bin/env bash
# Build LocalBooru's Linux release artifacts locally in Docker/Podman.
# This is the authoritative Linux release entry point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/build-startup-status.sh"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
SOURCE_REVISION="${LOCALBOORU_SOURCE_REVISION:-HEAD}"
DOCKERFILE_HASH="$(sha256sum "$ROOT/Dockerfile.linux-release" | cut -c1-12)"
IMAGE="localbooru-linux-release:ubuntu24.04-webkit2.52.3-v1-$DOCKERFILE_HASH"
REBUILD=0
BOOTSTRAP_NATIVE_RUNTIME=0
BUNDLES="appimage,deb,rpm"
JOBS="${LOCALBOORU_BUILD_JOBS:-2}"

usage() {
  cat <<'EOF'
Usage: scripts/build-linux-local.sh [OPTIONS]

Options:
  --appimage       Build only the AppImage
  --deb            Build only the Debian package
  --rpm            Build only the RPM package
  --rebuild-image  Rebuild the release toolchain image
  --bootstrap-native-runtime
                    Explicitly compile a missing native runtime cache. Normal
                    release runs never start this long-running bootstrap.
  --jobs N         Limit parallel compilation (default: 2)
  -h, --help       Show this help

Environment:
  LOCALBOORU_DOCKER_BUILD_ROOT  Persistent build/cache directory
  LOCALBOORU_DIST_LINUX_DIR     Final artifact directory
  LOCALBOORU_BUILD_JOBS         Parallel build limit
  LOCALBOORU_LINUX_BUILD_LIMIT_GB  Refuse builds above this cache size (default: 30)
  LOCALBOORU_CCACHE_SIZE         ccache size cap (default: 8G)
  LOCALBOORU_NATIVE_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS
                                Bootstrap limit in seconds (default: 900)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --appimage) BUNDLES="appimage"; shift ;;
    --deb) BUNDLES="deb"; shift ;;
    --rpm) BUNDLES="rpm"; shift ;;
    --rebuild-image) REBUILD=1; shift ;;
    --bootstrap-native-runtime) BOOTSTRAP_NATIVE_RUNTIME=1; shift ;;
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
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse --verify "${SOURCE_REVISION}^{commit}")"
GIT_COMMON_DIR="$(realpath "$(git -C "$ROOT" rev-parse --git-common-dir)")"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" show -s --format=%ct "$SOURCE_COMMIT")}"
WEBKIT_PATCH_PATH="patches/webkitgtk/2.52.3-playbin-video-filter.patch"
WEBKIT_PATCH_HASH="$(git -C "$ROOT" show "$SOURCE_COMMIT:$WEBKIT_PATCH_PATH" | sha256sum | cut -d' ' -f1)"
VAPOURSYNTH_COMMIT="c05906995662bacd5bddf853d8e68f19286987db"
CACHE_MARKER=".localbooru-build-cache"

has_symlink_component() {
  local current
  current="$(realpath -ms -- "$1")"
  while true; do
    [[ -L "$current" ]] && return 0
    [[ "$current" == "/" ]] && return 1
    current="$(dirname "$current")"
  done
}

prepare_cache_root() {
  local path="$1"
  local default_path="$2"
  local resolved default_resolved marker_value first_entry marker_temp

  if has_symlink_component "$path"; then
    echo "ERROR: refusing build cache path with a symlink component: $path" >&2
    exit 1
  fi

  resolved="$(realpath -m -- "$path")"
  default_resolved="$(realpath -m -- "$default_path")"
  marker_value=""
  if [[ -f "$resolved/$CACHE_MARKER" ]]; then
    marker_value="$(< "$resolved/$CACHE_MARKER")"
  fi

  if [[ "$resolved" != "$default_resolved" && "$marker_value" != "localbooru-build-cache-v1" ]]; then
    first_entry=""
    if [[ -d "$resolved" ]]; then
      first_entry="$(find "$resolved" -mindepth 1 -maxdepth 1 \
        ! -name "$CACHE_MARKER" ! -name '.localbooru-cache-marker.*' -print -quit)"
    fi
    if [[ -n "$first_entry" ]]; then
      echo "ERROR: custom build cache is nonempty and lacks a valid $CACHE_MARKER marker: $resolved" >&2
      exit 1
    fi
    rm -f "$resolved/$CACHE_MARKER" "$resolved"/.localbooru-cache-marker.*
  fi

  mkdir -p "$resolved"
  rm -f "$resolved"/.localbooru-cache-marker.*
  if [[ "$marker_value" != "localbooru-build-cache-v1" ]]; then
    marker_temp="$(mktemp "$resolved/.localbooru-cache-marker.XXXXXX")"
    if ! printf '%s\n' 'localbooru-build-cache-v1' > "$marker_temp"; then
      rm -f "$marker_temp"
      echo "ERROR: failed to initialize cache marker for $resolved" >&2
      exit 1
    fi
    mv -f "$marker_temp" "$resolved/$CACHE_MARKER"
  fi
  PREPARED_CACHE_ROOT="$resolved"
}

prepare_cache_root "$BUILD_ROOT" "$ROOT/build-linux-docker"
BUILD_ROOT="$PREPARED_CACHE_ROOT"
prepare_cache_root "$CCACHE_ROOT" "$ROOT/.ccache-docker"
CCACHE_ROOT="$PREPARED_CACHE_ROOT"

mkdir -p "$DIST_ROOT"
mkdir -p "$BUILD_ROOT/cargo-home" "$BUILD_ROOT/npm-cache"
DIST_ROOT="$(readlink -f "$DIST_ROOT")"

BUILD_LIMIT_GB="${LOCALBOORU_LINUX_BUILD_LIMIT_GB:-30}"
[[ "$BUILD_LIMIT_GB" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: LOCALBOORU_LINUX_BUILD_LIMIT_GB must be a positive integer" >&2
  exit 2
}
BUILD_USAGE_KIB="$(du -sk "$BUILD_ROOT" | cut -f1)"
BUILD_LIMIT_KIB=$((BUILD_LIMIT_GB * 1024 * 1024))
if (( BUILD_USAGE_KIB > BUILD_LIMIT_KIB )); then
  echo "ERROR: Linux build cache is $(du -sh "$BUILD_ROOT" | cut -f1), above the ${BUILD_LIMIT_GB}G limit." >&2
  echo "Run: npm run clean:builds -- linux-cache --execute" >&2
  exit 1
fi
printf '==> Linux persistent build cache before build: %s (limit: %sG)\n' \
  "$(du -sh "$BUILD_ROOT" | cut -f1)" "$BUILD_LIMIT_GB"

native_runtime_cache_missing=()
if [[ ! -e "$BUILD_ROOT/webkit-build/.localbooru-config-ubuntu24-gtk3-v2" ]]; then
  native_runtime_cache_missing+=("webkit-build/.localbooru-config-ubuntu24-gtk3-v2")
fi
if [[ ! -s "$BUILD_ROOT/webkit-build/lib/libwebkit2gtk-4.1.so.0" ]]; then
  native_runtime_cache_missing+=("webkit-build/lib/libwebkit2gtk-4.1.so.0")
fi
if [[ ! -s "$BUILD_ROOT/webkit-build/lib/libjavascriptcoregtk-4.1.so.0" ]]; then
  native_runtime_cache_missing+=("webkit-build/lib/libjavascriptcoregtk-4.1.so.0")
fi
if [[ ! -x "$BUILD_ROOT/webkit-build/bin/WebKitWebProcess" ]]; then
  native_runtime_cache_missing+=("webkit-build/bin/WebKitWebProcess")
fi
if [[ ! -f "$BUILD_ROOT/webkitgtk-2.52.3/.localbooru-patch-$WEBKIT_PATCH_HASH" ]]; then
  native_runtime_cache_missing+=("webkitgtk-2.52.3/.localbooru-patch-$WEBKIT_PATCH_HASH")
fi
if [[ ! -f "$BUILD_ROOT/vapoursynth-stage/.localbooru-vapoursynth-$VAPOURSYNTH_COMMIT" ]]; then
  native_runtime_cache_missing+=("vapoursynth-stage/.localbooru-vapoursynth-$VAPOURSYNTH_COMMIT")
fi

if ((${#native_runtime_cache_missing[@]})) && [[ "$BOOTSTRAP_NATIVE_RUNTIME" != 1 ]]; then
  localbooru_build_emit_status NEEDS_BOOTSTRAP \
    "platform=linux" \
    "source=$SOURCE_COMMIT" \
    "cache=$BUILD_ROOT" \
    "missing=$(IFS=,; echo "${native_runtime_cache_missing[*]}")" >&2
  cat >&2 <<EOF
ERROR: Linux native runtime cache is incomplete. Refusing to take the release
build token for an implicit WebKitGTK source build.

Bootstrap explicitly (resumable, bounded):
  LOCALBOORU_NATIVE_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS=900 \\
    scripts/build-linux-local.sh --bootstrap-native-runtime --jobs $JOBS
EOF
  exit 75
fi

localbooru_build_acquire_lock "$STATE_DIR" linux "$SOURCE_REVISION"
python3 "$ROOT/scripts/check-release-version.py"
localbooru_build_write_owner "$SOURCE_COMMIT"

if [[ "$REBUILD" == 1 ]] || ! "$CONTAINER" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Building release toolchain image $IMAGE"
  localbooru_build_started "$SOURCE_COMMIT" container-image
  "$CONTAINER" build -t "$IMAGE" - < "$ROOT/Dockerfile.linux-release"
fi

echo "==> Building LocalBooru Linux artifacts"
echo "    source:  $SOURCE_COMMIT"
echo "    bundles: $BUNDLES"
echo "    jobs:    $JOBS"
echo "    build:   $BUILD_ROOT"
echo "    output:  $DIST_ROOT"

localbooru_build_started "$SOURCE_COMMIT" artifacts
"$CONTAINER" run --rm \
  -u "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e CARGO_HOME=/cargo-home \
  -e RUSTUP_HOME=/opt/rustup \
  -e NPM_CONFIG_CACHE=/build/npm-cache \
  -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  -e LOCALBOORU_SOURCE_REVISION="$SOURCE_COMMIT" \
  -e TZ=UTC \
  -e LC_ALL=C.UTF-8 \
  -e LOCALBOORU_BUILD_JOBS="$JOBS" \
  -e LOCALBOORU_RELEASE_BUNDLES="$BUNDLES" \
  -e LOCALBOORU_ALLOW_NATIVE_RUNTIME_BOOTSTRAP="$BOOTSTRAP_NATIVE_RUNTIME" \
  -e LOCALBOORU_NATIVE_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS="${LOCALBOORU_NATIVE_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS:-900}" \
  -e GIT_DIR=/git \
  -e GIT_WORK_TREE=/source \
  -e CARGO_TARGET_DIR=/build/target \
  -e CCACHE_DIR=/ccache \
  -e CCACHE_MAXSIZE="${LOCALBOORU_CCACHE_SIZE:-8G}" \
  -v "$ROOT:/source:ro" \
  -v "$GIT_COMMON_DIR:/git:ro" \
  -v "$BUILD_ROOT:/build" \
  -v "$BUILD_ROOT/cargo-home:/cargo-home" \
  -v "$DIST_ROOT:/dist" \
  -v "$CCACHE_ROOT:/ccache" \
  -w /build \
  "$IMAGE" \
  bash /source/scripts/build-linux-docker.sh

echo
printf '==> Linux persistent build cache after build: %s\n' "$(du -sh "$BUILD_ROOT" | cut -f1)"
echo "==> LocalBooru Linux artifacts"
find "$DIST_ROOT" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' | sort
