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
mkdir -p "$dest/usr/lib/localbooru/native-svp"
printf '#!/bin/sh\n' > "$dest/AppRun"
chmod +x "$dest/AppRun"
touch "$dest/AppRun.tauri"
printf '[Desktop Entry]\nExec="donutmediacenter" %%U\n' > "$dest/com.localbooru.app.desktop"
TOOL
cat > "$tmp/bin/flatpak" <<'TOOL'
#!/bin/bash
set -eu
if [[ "$1" == build-init ]]; then mkdir -p "$3/files"; fi
TOOL
cat > "$tmp/fake.AppImage" <<'TOOL'
#!/bin/sh
printf '123\n'
TOOL
chmod +x "$tmp/bin/unsquashfs" "$tmp/bin/flatpak" "$tmp/fake.AppImage"
digest=$(sha256sum "$tmp/fake.AppImage" | cut -d' ' -f1)
PATH="$tmp/bin:$PATH" "$root/scripts/build-flatpak-local.sh" "$tmp/fake.AppImage" "$digest" "$tmp/built" > "$tmp/build.log"
launcher="$tmp/built/build/files/bin/donutmediacenter"
desktop="$tmp/built/build/files/lib/donutmediacenter/com.localbooru.app.desktop"
rg -q '^export APPDIR=/app/lib/donutmediacenter$' "$launcher"
rg -q '^Exec=donutmediacenter %U$' "$desktop"
echo 'Flatpak input and launcher checks passed' 
