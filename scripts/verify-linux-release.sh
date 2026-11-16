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
  python3 "$SCRIPT_ROOT/scripts/collect-linux-distro-notices.py" "$root" --verify
  for path in \
    "$runtime/bin/mpv" \
    "$runtime/injected-bundle/libwebkit2gtkinjectedbundle.so" \
    "$runtime/bin/WebKitWebProcess" \
    "$runtime/bin/WebKitNetworkProcess" \
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
  file "$runtime/bin/WebKitNetworkProcess" | grep 'ELF 64-bit' >/dev/null
  file "$runtime/injected-bundle/libwebkit2gtkinjectedbundle.so" | grep 'ELF 64-bit' >/dev/null
  if readelf -d "$runtime/injected-bundle/libwebkit2gtkinjectedbundle.so" | grep -E '(RPATH|RUNPATH).*/build/' >/dev/null; then
    echo 'ERROR: injected bundle retains a build-directory runtime path' >&2
    exit 1
  fi
  grep -aF 'LOCALBOORU_WEBKIT_PROCESS_DIR' "$runtime/lib/libwebkit2gtk-4.1.so.0" >/dev/null
  if grep -aF 'WebKitGPUProcess' "$runtime/lib/libwebkit2gtk-4.1.so.0" >/dev/null; then
    require_file "$runtime/bin/WebKitGPUProcess"
    file "$runtime/bin/WebKitGPUProcess" | grep 'ELF 64-bit' >/dev/null
  fi
  grep -aF 'localbooru-svp-snapshots-' "$runtime/lib/libwebkit2gtk-4.1.so.0" >/dev/null
  if [[ -e "$root/AppRun.tauri" ]]; then
    cmp "$root/usr/lib/libwebkit2gtk-4.1.so.0" "$runtime/lib/libwebkit2gtk-4.1.so.0"
    cmp "$root/usr/lib/libjavascriptcoregtk-4.1.so.0" "$runtime/lib/libjavascriptcoregtk-4.1.so.0"
  fi
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

require_file "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz"
require_file "$DIST/DonutMediaCenter-Linux-source.tar.xz"
require_file "$DIST/application-source-receipt.json"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$SCRIPT_ROOT/scripts/export-application-source.py" \
  --verify-receipt "$DIST/application-source-receipt.json" \
  --commit "${LOCALBOORU_SOURCE_REVISION:-}" >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep 'webkitgtk-2.52.3.tar.xz' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep '2.52.3-playbin-video-filter.patch' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep '2.52.3-flatpak-video-bridge.patch' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep '2.52.3-bundled-process-directory.patch' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep 'prepare-webkit-flatpak-bridge.py' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep -Fx './prepare-appimage-desktop.py' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep -Fx './test-prepare-appimage-desktop.py' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep -Fx './stage-appimage-avif.py' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep -Fx './test-stage-appimage-avif.py' >/dev/null
tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" \
  | grep 'vapoursynth-R75.tar.gz' >/dev/null

tar -tJf "$DIST/DonutMediaCenter-Native-Runtime-Sources.tar.xz" | grep "Dockerfile.linux-release-gl" >/dev/null

if has_bundle deb; then
  require_file "$DIST/DonutMediaCenter-Linux.deb"
  dpkg-deb --info "$DIST/DonutMediaCenter-Linux.deb" >/dev/null
  mkdir -p "$WORK/deb"
  dpkg-deb -x "$DIST/DonutMediaCenter-Linux.deb" "$WORK/deb"
  require_file "$WORK/deb/usr/bin/donutmediacenter"
  require_file "$WORK/deb/usr/lib/localbooru/donutmediacenter"
  [[ "$(dpkg-deb -f "$DIST/DonutMediaCenter-Linux.deb" Installed-Size)" -gt 200000 ]]
  verify_runtime_tree "$WORK/deb"
fi

if has_bundle rpm; then
  require_file "$DIST/DonutMediaCenter-Linux.rpm"
  rpm -K "$DIST/DonutMediaCenter-Linux.rpm" 2>&1 | grep -E 'digests OK|NOT OK|NOKEY' >/dev/null
  rpm -qpl "$DIST/DonutMediaCenter-Linux.rpm" | grep '/usr/bin/donutmediacenter' >/dev/null
  rpm -qpl "$DIST/DonutMediaCenter-Linux.rpm" \
    | grep '/usr/lib/localbooru/native-svp/bin/mpv' >/dev/null
  rpm -qpl "$DIST/DonutMediaCenter-Linux.rpm" \
    | grep '/usr/share/doc/localbooru/distro-packages/manifest.json' >/dev/null
  rpm -qpR "$DIST/DonutMediaCenter-Linux.rpm" | grep '^gtk3$' >/dev/null
  rpm -qpR "$DIST/DonutMediaCenter-Linux.rpm" | grep '^webkit2gtk4\.1$' >/dev/null
fi

if has_bundle appimage; then
  require_file "$DIST/DonutMediaCenter-Linux.AppImage"
  file "$DIST/DonutMediaCenter-Linux.AppImage" | grep 'ELF 64-bit' >/dev/null
  test -x "$DIST/DonutMediaCenter-Linux.AppImage"
  (cd "$WORK" && "$DIST/DonutMediaCenter-Linux.AppImage" --appimage-extract >/dev/null)
  test -x "$WORK/squashfs-root/AppRun"
  bash -n "$WORK/squashfs-root/AppRun"
  python3 "$SCRIPT_ROOT/scripts/prepare-appimage-desktop.py" --check "$WORK/squashfs-root/com.localbooru.app.desktop"
  python3 "$SCRIPT_ROOT/scripts/prepare-appimage-desktop.py" --check "$WORK/squashfs-root/usr/share/applications/com.localbooru.app.desktop"
  verify_runtime_tree "$WORK/squashfs-root"
  python3 "$SCRIPT_ROOT/scripts/stage-appimage-avif.py" "$WORK/squashfs-root" --verify
  require_file "$WORK/squashfs-root/usr/lib/gstreamer-1.0/libgstopengl.so"
  require_file "$WORK/squashfs-root/usr/share/doc/gstreamer1.0-gl/copyright"

  require_file "$DIST/DonutMediaCenter-Linux.zip"
  unzip -tq "$DIST/DonutMediaCenter-Linux.zip" >/dev/null
  unzip -l "$DIST/DonutMediaCenter-Linux.zip" | grep 'DonutMediaCenter-Linux.AppImage' >/dev/null
  unzip -l "$DIST/DonutMediaCenter-Linux.zip" | grep 'DonutMediaCenter-Linux-source.tar.xz' >/dev/null
  unzip -l "$DIST/DonutMediaCenter-Linux.zip" | grep 'application-source-receipt.json' >/dev/null
fi

# Report, but do not hide, the actual portability floor of the final app.
for binary in \
  "$WORK/deb/usr/lib/localbooru/donutmediacenter" \
  "$WORK/squashfs-root/usr/bin/donutmediacenter"; do
  [[ -f "$binary" ]] || continue
  floor="$(objdump -T "$binary" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1 || true)"
  echo "Verified $(file -b "$binary"); maximum glibc symbol: ${floor:-none}"
done

echo "Linux release artifact verification passed"
