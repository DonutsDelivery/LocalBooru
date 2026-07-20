#!/usr/bin/env bash
set -euo pipefail

DIST="${1:?usage: verify-linux-release.sh DIST [bundle-list]}"
BUNDLES="${2:-appimage,deb,rpm}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

has_bundle() {
  [[ ",$BUNDLES," == *",$1,"* ]]
}

require_file() {
  [[ -s "$1" ]] || { echo "ERROR: missing or empty artifact: $1" >&2; exit 1; }
}

verify_runtime_tree() {
  local root="$1"
  local runtime="$root/usr/lib/localbooru/native-svp"
  for path in \
    "$runtime/bin/mpv" \
    "$runtime/lib/libwebkit2gtk-4.1.so.0" \
    "$runtime/lib/libjavascriptcoregtk-4.1.so.0" \
    "$runtime/lib/libjxl.so.0.7" \
    "$runtime/gstreamer/libgstlocalboorupass.so" \
    "$runtime/gstreamer/libgstlocalbooruvs.so" \
    "$runtime/python-home/lib/python3.12/site-packages/vapoursynth/libvsscript.so" \
    "$runtime/licenses/THIRD_PARTY_NOTICES.md"; do
    require_file "$path"
  done
  file "$runtime/bin/mpv" | grep 'ELF 64-bit' >/dev/null
  file "$runtime/gstreamer/libgstlocalbooruvs.so" | grep 'ELF 64-bit' >/dev/null
  readelf -d "$runtime/gstreamer/libgstlocalbooruvs.so" \
    | grep '\$ORIGIN/../python-home/lib/python3.12/site-packages/vapoursynth' >/dev/null
  if find "$runtime" -type f -print0 | xargs -0 grep -IlE '/home/user|/mnt/storage|/home/.*/SVP 4|/build/(worktree|webkit-build)|/source/(scripts|src-tauri|native-video|gstreamer-svp|frontend|release)/' | grep . >/dev/null; then
    echo "ERROR: runtime contains a host or container build path" >&2
    exit 1
  fi
  if find "$runtime" -type f \( -iname '*svpflow*' -o -iname '*SVPManager*' \) | grep . >/dev/null; then
    echo "ERROR: proprietary SVP runtime was bundled" >&2
    exit 1
  fi
}

require_file "$DIST/LocalBooru-Native-Runtime-Sources.tar.xz"
tar -tJf "$DIST/LocalBooru-Native-Runtime-Sources.tar.xz" \
  | grep 'webkitgtk-2.52.3.tar.xz' >/dev/null
tar -tJf "$DIST/LocalBooru-Native-Runtime-Sources.tar.xz" \
  | grep '2.52.3-playbin-video-filter.patch' >/dev/null
tar -tJf "$DIST/LocalBooru-Native-Runtime-Sources.tar.xz" \
  | grep 'vapoursynth-R75.tar.gz' >/dev/null

if has_bundle deb; then
  require_file "$DIST/LocalBooru-Linux.deb"
  dpkg-deb --info "$DIST/LocalBooru-Linux.deb" >/dev/null
  mkdir -p "$WORK/deb"
  dpkg-deb -x "$DIST/LocalBooru-Linux.deb" "$WORK/deb"
  require_file "$WORK/deb/usr/bin/localbooru"
  require_file "$WORK/deb/usr/lib/localbooru/localbooru"
  [[ "$(dpkg-deb -f "$DIST/LocalBooru-Linux.deb" Installed-Size)" -gt 200000 ]]
  verify_runtime_tree "$WORK/deb"
fi

if has_bundle rpm; then
  require_file "$DIST/LocalBooru-Linux.rpm"
  rpm -K "$DIST/LocalBooru-Linux.rpm" 2>&1 | grep -E 'digests OK|NOT OK|NOKEY' >/dev/null
  rpm -qpl "$DIST/LocalBooru-Linux.rpm" | grep '/usr/bin/localbooru' >/dev/null
  rpm -qpl "$DIST/LocalBooru-Linux.rpm" \
    | grep '/usr/lib/localbooru/native-svp/bin/mpv' >/dev/null
  rpm -qpR "$DIST/LocalBooru-Linux.rpm" | grep '^gtk3$' >/dev/null
  rpm -qpR "$DIST/LocalBooru-Linux.rpm" | grep '^webkit2gtk4\.1$' >/dev/null
fi

if has_bundle appimage; then
  require_file "$DIST/LocalBooru-Linux.AppImage"
  file "$DIST/LocalBooru-Linux.AppImage" | grep 'ELF 64-bit' >/dev/null
  test -x "$DIST/LocalBooru-Linux.AppImage"
  (cd "$WORK" && "$DIST/LocalBooru-Linux.AppImage" --appimage-extract >/dev/null)
  test -x "$WORK/squashfs-root/AppRun"
  bash -n "$WORK/squashfs-root/AppRun"
  verify_runtime_tree "$WORK/squashfs-root"

  require_file "$DIST/LocalBooru-Linux.zip"
  unzip -tq "$DIST/LocalBooru-Linux.zip" >/dev/null
  unzip -l "$DIST/LocalBooru-Linux.zip" | grep 'LocalBooru-Linux.AppImage' >/dev/null
fi

# Report, but do not hide, the actual portability floor of the final app.
for binary in \
  "$WORK/deb/usr/lib/localbooru/localbooru" \
  "$WORK/squashfs-root/usr/bin/localbooru"; do
  [[ -f "$binary" ]] || continue
  floor="$(objdump -T "$binary" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1 || true)"
  echo "Verified $(file -b "$binary"); maximum glibc symbol: ${floor:-none}"
done

echo "Linux release artifact verification passed"
