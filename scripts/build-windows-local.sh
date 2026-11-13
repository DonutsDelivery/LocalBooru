#!/usr/bin/env bash
# Build LocalBooru Windows x64 artifacts locally through MSVC-under-Wine Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/build-startup-status.sh"
source "$ROOT/scripts/build-resource-limits.sh"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/localbooru"
SOURCE_REVISION="${LOCALBOORU_SOURCE_REVISION:-HEAD}"
localbooru_build_acquire_lock "$STATE_DIR" windows "$SOURCE_REVISION"
python3 "$ROOT/scripts/check-release-version.py"
DOCKERFILE="$ROOT/Dockerfile.windows-release"
REBUILD=0
JOBS="${LOCALBOORU_BUILD_JOBS:-1}"

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
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse --verify "${SOURCE_REVISION}^{commit}")"
GIT_COMMON_DIR="$(realpath "$(git -C "$ROOT" rev-parse --git-common-dir)")"
localbooru_build_write_owner "$SOURCE_COMMIT"
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" show -s --format=%ct "$SOURCE_COMMIT")}"
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

localbooru_build_resource_limits
localbooru_build_check_disk / "$BUILD_ROOT" "$DIST_PATH"

# Windows has no system ffmpeg, so the app ships its own. A GPL build is
# needed for libx264 transcoding; the release tag is permanent.
FFMPEG_ARCHIVE_NAME="ffmpeg-8.1-full_build-shared.zip"
FFMPEG_ARCHIVE_URL="https://github.com/GyanD/codexffmpeg/releases/download/8.1/$FFMPEG_ARCHIVE_NAME"
FFMPEG_ARCHIVE_SHA256="e184e012ca9a3527ed45d61f8e250c50b07b8661d90849820ea754239049efb4"
FFMPEG_ARCHIVE="$BUILD_ROOT/downloads/$FFMPEG_ARCHIVE_NAME"
if [[ ! -f "$FFMPEG_ARCHIVE" ]]; then
  mkdir -p "$BUILD_ROOT/downloads"
  curl -sSfL -o "$FFMPEG_ARCHIVE.part" "$FFMPEG_ARCHIVE_URL"
  mv "$FFMPEG_ARCHIVE.part" "$FFMPEG_ARCHIVE"
fi
echo "$FFMPEG_ARCHIVE_SHA256  $FFMPEG_ARCHIVE" | sha256sum -c -

if [[ "$REBUILD" == 1 ]] || ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Building Windows MSVC/Wine image $IMAGE"
  localbooru_build_started "$SOURCE_COMMIT" container-image
  "$DOCKER" build -t "$IMAGE" - < "$DOCKERFILE"
fi

printf '==> Building Windows x64 artifacts with %s jobs\n' "$JOBS"
printf '    source: %s\n' "$SOURCE_COMMIT"
localbooru_build_started "$SOURCE_COMMIT" artifacts
container_result=0
"$DOCKER" run --rm --init \
  "${LOCALBOORU_CONTAINER_LIMITS[@]}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -e LOCALBOORU_BUILD_JOBS="$JOBS" \
  -e LOCALBOORU_SOURCE_REVISION="$SOURCE_COMMIT" \
  -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
  -e LOCALBOORU_FFMPEG_ARCHIVE="/build/downloads/$FFMPEG_ARCHIVE_NAME" \
  -e LOCALBOORU_FFMPEG_SHA256="$FFMPEG_ARCHIVE_SHA256" \
  -e npm_config_cache=/build/npm-cache \
  -e GIT_DIR=/git \
  -e GIT_WORK_TREE=/source \
  -v "$ROOT:/source:ro" \
  -v "$GIT_COMMON_DIR:/git:ro" \
  -v "$BUILD_ROOT:/build" \
  -v "$DIST_PATH:/dist" \
  -w /build/worktree \
  "$IMAGE" \
  bash /source/scripts/build-windows-docker.sh || container_result=$?

hygiene_result=0
python3 "$ROOT/scripts/cargo-cache-hygiene.py" --target-directory "$BUILD_ROOT/target" || hygiene_result=$?
(( container_result == 0 )) || exit "$container_result"
(( hygiene_result == 0 )) || exit "$hygiene_result"

(
  cd "$DIST_PATH"
  sha256sum -c SHA256SUMS-Windows
)
python3 - "$DIST_PATH" "$SOURCE_COMMIT" "$ROOT" <<'PY'
import hashlib, json, pathlib, subprocess, sys
dist, source, root = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
receipt = json.loads((dist / 'Windows-build-receipt.json').read_text())
recipe = subprocess.check_output(['git', '-C', root, 'show', source + ':scripts/build-windows-docker.sh'])
assert receipt['schema'] == 1 and receipt['source_commit'] == source
assert receipt['target'] == 'x86_64-pc-windows-msvc'
assert receipt['recipe_sha256'] == hashlib.sha256(recipe).hexdigest()
for key, name in [('portable_sha256', 'DonutMediaCenter-Windows.zip'),
                  ('installer_sha256', 'DonutMediaCenter-Windows-Setup.exe')]:
    with (dist / name).open('rb') as stream:
        assert receipt[key] == hashlib.file_digest(stream, 'sha256').hexdigest()
print('Windows source/recipe/artifact receipt verified')
PY

printf '==> Windows persistent build cache after build: %s\n' "$(du -sh "$BUILD_ROOT" | cut -f1)"
printf '==> Local Windows artifacts verified in %s:\n' "$DIST_PATH"
find "$DIST_PATH" -maxdepth 1 -type f -printf '  %f (%s bytes)\n' | sort
