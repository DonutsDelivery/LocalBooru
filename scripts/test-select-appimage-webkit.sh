#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture=$(mktemp -d "${TMPDIR:-/tmp}/dmc runtime selection.XXXXXXXX")
trap 'rm -rf "$fixture"' EXIT
appdir="$fixture/AppDir with spaces"
libdir="$appdir/usr/lib"
runtime="$libdir/localbooru/native-svp/lib"
mkdir -p "$runtime"
touch "$appdir/AppRun"
printf 'system webkit' > "$libdir/libwebkit2gtk-4.1.so.0"
printf 'patched webkit' > "$runtime/libwebkit2gtk-4.1.so.0"
if bash "$repo/scripts/select-appimage-webkit.sh" "$appdir" 2>"$fixture/error"; then
  echo 'Incomplete patched pair unexpectedly accepted' >&2; exit 1
fi
[[ $(cat "$libdir/libwebkit2gtk-4.1.so.0") == 'system webkit' ]]
printf 'patched javascriptcore' > "$runtime/libjavascriptcoregtk-4.1.so.0"
printf 'stale system webkit version' > "$libdir/libwebkit2gtk-4.1.so.0.old"
printf 'unrelated' > "$libdir/libunrelated.so.0"
bash "$repo/scripts/select-appimage-webkit.sh" "$appdir"
for family in libwebkit2gtk-4.1 libjavascriptcoregtk-4.1; do
  [[ -L "$libdir/$family.so.0" ]]
  cmp "$libdir/$family.so.0" "$runtime/$family.so.0"
done
[[ ! -e "$libdir/libwebkit2gtk-4.1.so.0.old" ]]
[[ $(cat "$libdir/libunrelated.so.0") == unrelated ]]
bash "$repo/scripts/select-appimage-webkit.sh" "$appdir"
echo 'AppImage patched runtime selection and incomplete-pair checks passed'
