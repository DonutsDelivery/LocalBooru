#!/usr/bin/env bash
set -euo pipefail

webkit_root="${LOCALBOORU_WEBKIT_ROOT:-/mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched}"
build_dir="$webkit_root/local-build"
source_dir="$webkit_root/src/webkitgtk-2.52.3"
deps_dir="$webkit_root/user-deps"
cache_dir="${LOCALBOORU_WEBKIT_CCACHE_DIR:-$webkit_root/.ccache}"
jobs="${LOCALBOORU_WEBKIT_JOBS:-4}"
ccache_bin="$(command -v ccache)"

export PATH="$deps_dir/usr/bin:$PATH"
export LD_LIBRARY_PATH="$deps_dir/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RUBYLIB="$deps_dir/usr/lib/ruby/3.4.0/x86_64-linux:$deps_dir/usr/lib/ruby/3.4.0"
export CCACHE_DIR="$cache_dir"
export CCACHE_MAXSIZE="${LOCALBOORU_WEBKIT_CCACHE_SIZE:-30G}"

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
