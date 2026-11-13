#!/usr/bin/env bash
# linuxdeploy prepends AppDir/usr/lib after our launcher sets its runtime path.
# Point its SONAMEs at the patched WebKit pair rather than its system copies.
set -euo pipefail
[[ $# == 1 && "$1" == /* && "$1" != / && -d "$1" && ! -L "$1" ]] || {
  echo 'Usage: select-appimage-webkit.sh /absolute/staged/AppDir' >&2; exit 2;
}
appdir=$1
libdir="$appdir/usr/lib"
runtime="$libdir/localbooru/native-svp/lib"
[[ -e "$appdir/AppRun" && -d "$libdir" && ! -L "$libdir" ]] || {
  echo 'Expected a staged DMC AppImage, refusing to modify libraries.' >&2; exit 1;
}
for family in libwebkit2gtk-4.1 libjavascriptcoregtk-4.1; do
  [[ -s "$runtime/$family.so.0" ]] || { echo "Missing patched runtime: $family" >&2; exit 1; }
done
for family in libwebkit2gtk-4.1 libjavascriptcoregtk-4.1; do
  for old in "$libdir/$family.so"*; do
    [[ -e "$old" || -L "$old" ]] || continue
    rm -- "$old"
  done
  ln -s "localbooru/native-svp/lib/$family.so.0" "$libdir/$family.so.0"
done
