#!/usr/bin/env bash
# Build a local, non-Flathub Flatpak from an already verified DMC AppImage.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: $0 /absolute/path/to/DonutMediaCenter-Linux.AppImage SHA256 /absolute/output/directory" >&2
  exit 2
fi
image=$1 expected=$2 out=$3
[[ "$image" = /* && "$out" = /* && "$expected" =~ ^[[:xdigit:]]{64}$ ]] || { echo 'Use absolute paths and a SHA-256 digest.' >&2; exit 2; }
[[ -f "$image" && ! -L "$image" && ! -e "$out" ]] || { echo 'Input must be a regular AppImage; output must not exist.' >&2; exit 2; }
for tool in flatpak unsquashfs sha256sum; do command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }; done
actual=$(sha256sum "$image" | cut -d' ' -f1)
[[ "${actual,,}" = "${expected,,}" ]] || { echo 'AppImage SHA-256 does not match.' >&2; exit 1; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
id=online.donutsdelivery.DonutMediaCenter
mkdir -p "$(dirname "$out")"
mkdir "$out"
work=$(mktemp -d "${TMPDIR:-/tmp}/dmc-flatpak-build.XXXXXXXX")
trap 'rm -rf "$work"' EXIT
# AppImage type 2 reports its squashfs offset without launching the app.
offset=$("$image" --appimage-offset)
[[ "$offset" =~ ^[0-9]+$ ]] || { echo 'Invalid AppImage filesystem offset.' >&2; exit 1; }
unsquashfs -no-progress -offset "$offset" -d "$work/appdir" "$image" >/dev/null
[[ -x "$work/appdir/AppRun" && -e "$work/appdir/AppRun.tauri" && -d "$work/appdir/usr/lib/localbooru/native-svp" ]] || { echo 'AppImage lacks the expected DMC runtime.' >&2; exit 1; }
flatpak build-init --arch=x86_64 "$out/build" "$id" org.freedesktop.Sdk org.freedesktop.Platform 25.08
mkdir -p "$out/build/files/lib/donutmediacenter" "$out/build/files/bin" "$out/build/files/share/applications" "$out/build/files/share/icons/hicolor/512x512/apps"
cp -a "$work/appdir/." "$out/build/files/lib/donutmediacenter/"
# linuxdeploy AppRun.wrapped treats quotes in Exec as literal characters.
sed -i 's/^Exec="donutmediacenter"/Exec=donutmediacenter/' "$out/build/files/lib/donutmediacenter/com.localbooru.app.desktop"
cat > "$out/build/files/bin/donutmediacenter" <<'LAUNCHER'
#!/bin/sh
export APPDIR=/app/lib/donutmediacenter
export PATH="$APPDIR/usr/bin:$PATH"
exec "$APPDIR/AppRun" "$@"
LAUNCHER
chmod 755 "$out/build/files/bin/donutmediacenter"
sed -e "s/^Icon=.*/Icon=$id/" "$root/src-tauri/linux/com.localbooru.app.desktop" > "$out/build/files/share/applications/$id.desktop"
install -m 644 "$root/assets/icon.png" "$out/build/files/share/icons/hicolor/512x512/apps/$id.png"
# The app manages arbitrary user-selected libraries, LAN sharing and GPU playback.
flatpak build-finish --command=donutmediacenter --share=network --share=ipc \
  --socket=wayland --socket=fallback-x11 --socket=pulseaudio \
  --device=dri --filesystem=host "$out/build"
flatpak build-export "$out/repo" "$out/build" stable
flatpak build-bundle "$out/repo" "$out/DonutMediaCenter-local.flatpak" "$id" stable
printf '%s  %s\n' "$actual" "$(basename "$image")" > "$out/source-sha256.txt"
printf 'Local bundle: %s\n' "$out/DonutMediaCenter-local.flatpak"
