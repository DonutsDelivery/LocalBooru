#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/dmc-flatpak-test.XXXXXXXX")
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' 'synthetic input' > "$tmp/input.AppImage"
if "$root/scripts/build-flatpak-local.sh" "$tmp/input.AppImage" "$(printf '0%.0s' {1..64})" "$tmp/invalid-output" >"$tmp/log" 2>&1; then
  echo 'Wrong hash unexpectedly accepted' >&2; exit 1
fi
[[ ! -e "$tmp/invalid-output" ]]
mkdir "$tmp/already-exists"
digest=$(sha256sum "$tmp/input.AppImage" | cut -d' ' -f1)
if "$root/scripts/build-flatpak-local.sh" "$tmp/input.AppImage" "$digest" "$tmp/already-exists" >"$tmp/log" 2>&1; then
  echo 'Existing output unexpectedly accepted' >&2; exit 1
fi
[[ -d "$tmp/already-exists" ]]
# Exercise the actual packaging flow with fake archive/Flatpak tools and inspect
# the produced launcher and desktop entry, without accessing any user library.
mkdir "$tmp/bin"
cat > "$tmp/bin/unsquashfs" <<'TOOL'
#!/bin/bash
set -eu
while (($#)); do if [[ "$1" == -d ]]; then dest=$2; shift 2; else shift; fi; done
mkdir -p "$dest/usr/lib/localbooru/native-svp" "$dest/usr/share/icons/Adwaita/cursors" "$dest/usr/share/doc/adwaita-icon-theme"
printf "[Icon Theme]\nName=Adwaita\n" > "$dest/usr/share/icons/Adwaita/index.theme"
printf "synthetic cursor\n" > "$dest/usr/share/icons/Adwaita/cursors/left_ptr"
printf "synthetic license\n" > "$dest/usr/share/doc/adwaita-icon-theme/copyright"
printf '#!/bin/sh\n' > "$dest/AppRun"
chmod +x "$dest/AppRun"
touch "$dest/AppRun.tauri"
printf '[Desktop Entry]\nExec="donutmediacenter" %%U\n' > "$dest/com.localbooru.app.desktop"
TOOL
cat > "$tmp/bin/flatpak" <<'TOOL'
#!/bin/bash
set -eu
if [[ "$1" == build-init ]]; then mkdir -p "$3/files"; fi
if [[ "$1" == build-finish ]]; then printf "%s\n" "$@" > "$FAKE_FINISH"; fi
TOOL
cat > "$tmp/fake.AppImage" <<'TOOL'
#!/bin/sh
printf '123\n'
TOOL
chmod +x "$tmp/bin/unsquashfs" "$tmp/bin/flatpak" "$tmp/fake.AppImage"
digest=$(sha256sum "$tmp/fake.AppImage" | cut -d' ' -f1)
FAKE_FINISH="$tmp/host-args" PATH="$tmp/bin:$PATH" "$root/scripts/build-flatpak-local.sh" "$tmp/fake.AppImage" "$digest" "$tmp/built" > "$tmp/build.log"
FAKE_FINISH="$tmp/selected-args" PATH="$tmp/bin:$PATH" "$root/scripts/build-flatpak-local.sh" "$tmp/fake.AppImage" "$digest" "$tmp/selected" selected-folder > "$tmp/build-selected.log"
if FAKE_FINISH="$tmp/invalid-args" PATH="$tmp/bin:$PATH" "$root/scripts/build-flatpak-local.sh" "$tmp/fake.AppImage" "$digest" "$tmp/no-parent/output" invalid > "$tmp/invalid.log" 2>&1; then
  echo 'Invalid profile unexpectedly accepted' >&2; exit 1
fi
[[ ! -e "$tmp/no-parent" && ! -e "$tmp/invalid-args" ]]
rg -qx -- '--filesystem=host' "$tmp/host-args"
! rg -q -- '^--filesystem=' "$tmp/selected-args"
rg -qx 'host' "$tmp/built/permission-profile.txt"
rg -qx 'selected-folder' "$tmp/selected/permission-profile.txt"
rg -qx 'export GTK_USE_PORTAL=1' "$tmp/selected/build/files/bin/donutmediacenter"
launcher="$tmp/built/build/files/bin/donutmediacenter"
desktop="$tmp/built/build/files/lib/donutmediacenter/com.localbooru.app.desktop"
rg -q '^export APPDIR=/app/lib/donutmediacenter$' "$launcher"
rg -q '^CODECS_LIB="\$PLATFORM_LIB/codecs-extra/lib"$' "$launcher"
rg -q '^export LD_LIBRARY_PATH="\$CODECS_LIB:\$PLATFORM_LIB:\$RUNTIME/lib:' "$launcher"
rg -q '^export GST_PLUGIN_SYSTEM_PATH_1_0="\$CODECS_LIB/gstreamer-1.0:/usr/lib/extensions/gstreamer-1.0:\$PLATFORM_LIB/gstreamer-1.0"$' "$launcher"
rg -q '^export GST_PLUGIN_PATH_1_0="\$RUNTIME/gstreamer"$' "$launcher"
rg -q '^export WEBKIT_INJECTED_BUNDLE_PATH="\$RUNTIME/injected-bundle"$' "$launcher"
rg -q '^exec "\$APPDIR/usr/bin/donutmediacenter" "\$@"$' "$launcher"
! rg -q '^exec "\$APPDIR/AppRun"' "$launcher"
sh -n "$launcher"
test -f "$tmp/built/build/files/share/icons/Adwaita/cursors/left_ptr"
test -f "$tmp/built/build/files/share/doc/adwaita-icon-theme/copyright"
rg -q '^Exec=donutmediacenter %U$' "$desktop"
echo 'Flatpak input and launcher checks passed' 
