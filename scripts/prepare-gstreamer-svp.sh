#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${LOCALBOORU_GSTREAMER_SVP_BUILD_DIR:-$root/gstreamer-svp/build}"
install_dir="${LOCALBOORU_GSTREAMER_SVP_DIR:-$HOME/.local/lib/localbooru}"
mkdir -p "$build_dir" "$install_dir"

cflags=(
  -std=c11 -Wall -Wextra -Werror -fPIC
  -ffile-prefix-map="$root"=.
  -fmacro-prefix-map="$root"=.
)
read -r -a gst_flags <<<"$(pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0)"
vsscript_flags=( -lvsscript )
if [[ -n "${LOCALBOORU_VAPOURSYNTH_DIR:-}" ]]; then
  vapoursynth_include_dir="${LOCALBOORU_VAPOURSYNTH_INCLUDE_DIR:-$LOCALBOORU_VAPOURSYNTH_DIR/include}"
  vapoursynth_lib_dir="${LOCALBOORU_VAPOURSYNTH_LIB_DIR:-$LOCALBOORU_VAPOURSYNTH_DIR}"
  vsscript_flags=(
    -I"$vapoursynth_include_dir"
    -L"$vapoursynth_lib_dir"
    '-Wl,-rpath,$ORIGIN/../python-home/lib/python3.12/site-packages/vapoursynth'
    -lvsscript
  )
fi

cc "${cflags[@]}" -shared \
  "$root/gstreamer-svp/src/webkit_video_filter_hook.c" \
  -o "$build_dir/liblocalbooru-webkit-gst-hook.so" \
  "${gst_flags[@]}" -ldl

cc "${cflags[@]}" -shared \
  "$root/gstreamer-svp/src/gstlocalboorupass.c" \
  -o "$build_dir/libgstlocalboorupass.so" \
  "${gst_flags[@]}"

cc "${cflags[@]}" -shared \
  "$root/gstreamer-svp/src/gstlocalbooruvs.c" \
  -o "$build_dir/libgstlocalbooruvs.so" \
  "${gst_flags[@]}" "${vsscript_flags[@]}"

install -m 0755 "$build_dir/liblocalbooru-webkit-gst-hook.so" "$install_dir/"
install -m 0755 "$build_dir/libgstlocalboorupass.so" "$install_dir/"
install -m 0755 "$build_dir/libgstlocalbooruvs.so" "$install_dir/"

printf '%s\n' "$install_dir/liblocalbooru-webkit-gst-hook.so"
printf '%s\n' "$install_dir/libgstlocalboorupass.so"
printf '%s\n' "$install_dir/libgstlocalbooruvs.so"
