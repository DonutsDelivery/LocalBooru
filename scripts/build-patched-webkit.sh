#!/usr/bin/env bash
set -euo pipefail

# Both the incremental and full compiler paths share the host build token.
if [[ "${_HOST_HEAVY_BUILD_INTERNAL:-0}" != 1 ]]; then
  exec "${HOST_HEAVY_BUILD_HELPER:-$HOME/.local/bin/host-heavy-build}" run \
    --project localbooru-webkit --worktree "$(cd "$(dirname "$0")/.." && pwd)" \
    --wait "${LOCALBOORU_BUILD_LOCK_TIMEOUT:-21600}" -- "$0" "$@"
fi

webkit_root="${LOCALBOORU_WEBKIT_ROOT:-/mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched}"
build_dir="$webkit_root/local-build"
source_dir="$webkit_root/src/webkitgtk-2.52.3"
deps_dir="$webkit_root/user-deps"
cache_dir="${LOCALBOORU_WEBKIT_CCACHE_DIR:-$webkit_root/.ccache}"
jobs="${LOCALBOORU_WEBKIT_JOBS:-2}"
[[ "$jobs" =~ ^[12]$ ]] || { echo "ERROR: WebKit build jobs must be 1 or 2" >&2; exit 2; }
ccache_bin="$(command -v ccache)"

export PATH="$deps_dir/usr/bin:$PATH"
export LD_LIBRARY_PATH="$deps_dir/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RUBYLIB="$deps_dir/usr/lib/ruby/3.4.0/x86_64-linux:$deps_dir/usr/lib/ruby/3.4.0"
export CCACHE_DIR="$cache_dir"
export CCACHE_MAXSIZE="${LOCALBOORU_WEBKIT_CCACHE_SIZE:-30G}"
export NINJA_STATUS='[%f/%t] '

if [[ "${1:-}" == --mpv-isolation-only ]]; then
  [[ $# == 1 && -f "$build_dir/CMakeCache.txt" && -f "$build_dir/build.ninja" ]] || {
    echo "ERROR: incremental isolation upgrade requires the existing 2.52.3 Ninja cache" >&2
    exit 2
  }
  python3 "$(dirname "$0")/prepare-webkit-mpv-isolation.py" \
    --source "$source_dir" --build-cache "$build_dir" --apply
  # No CMake regeneration or whole-source replacement. Ninja rebuilds the two
  # affected unified objects and shared WebKit library. The plan check refuses
  # unrelated dirty dependencies; WebCore objects are linked directly here.
  ninja -C "$build_dir" -j"$jobs" bin/WebKitWebProcess
  install -m 0755 "$build_dir/bin/WebKitWebProcess" "$build_dir/bin/mpv"
  exit 0
fi
[[ $# == 0 ]] || { echo "Usage: $0 [--mpv-isolation-only]" >&2; exit 2; }

mkdir -p "$cache_dir"
"$ccache_bin" --max-size "$CCACHE_MAXSIZE"

# Reuse the existing CMake cache and only add compiler launchers. This does not
# invalidate completed Ninja objects; future recompiles are stored in ccache.
cmake -S "$source_dir" -B "$build_dir" \
  -DRuby_EXECUTABLE="$deps_dir/usr/bin/ruby" \
  -DRuby_VERSION=3.4.8 \
  -DRUBY_EXECUTABLE="$deps_dir/usr/bin/ruby" \
  -DRUBY_VERSION=3.4.8 \
  -DCMAKE_C_COMPILER_LAUNCHER="$ccache_bin" \
  -DCMAKE_CXX_COMPILER_LAUNCHER="$ccache_bin"

cmake --build "$build_dir" --target WebKit WebKitWebProcess -- -j"$jobs"
install -m 0755 "$build_dir/bin/WebKitWebProcess" "$build_dir/bin/mpv"
"$ccache_bin" --show-stats
