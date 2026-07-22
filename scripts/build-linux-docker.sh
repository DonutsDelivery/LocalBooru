#!/usr/bin/env bash
# Runs inside Dockerfile.linux-release through scripts/build-linux-local.sh.
set -euo pipefail

SOURCE="/source"
BUILD="/build"
ROOT="$BUILD/worktree"
DIST="/dist"
JOBS="${LOCALBOORU_BUILD_JOBS:-2}"
BUNDLES="${LOCALBOORU_RELEASE_BUNDLES:-appimage,deb,rpm}"
WEBKIT_VERSION="2.52.3"
WEBKIT_SHA256="5b3e0d174e63dcc28848b1194e0e7448d5948c3c2427ecd931c2c5be5261aebb"
VAPOURSYNTH_COMMIT="c05906995662bacd5bddf853d8e68f19286987db" # R75
APPIMAGETOOL_SHA256="b90f4a8b18967545fda78a445b27680a1642f1ef9488ced28b65398f2be7add2"

export CCACHE_DIR="${CCACHE_DIR:-/ccache}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-30G}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BUILD/target}"
mkdir -p "$BUILD/downloads" "$DIST" "$CCACHE_DIR"
ccache --max-size "$CCACHE_MAXSIZE" >/dev/null

echo "==> Refreshing isolated source worktree"
SOURCE_REVISION="${LOCALBOORU_SOURCE_REVISION:?LOCALBOORU_SOURCE_REVISION is required}"
git -c safe.directory="$SOURCE" -C "$SOURCE" cat-file -e "${SOURCE_REVISION}^{commit}"
RESOLVED_SOURCE_REVISION="$(git -c safe.directory="$SOURCE" -C "$SOURCE" rev-parse "${SOURCE_REVISION}^{commit}")"
[[ "$RESOLVED_SOURCE_REVISION" == "$SOURCE_REVISION" ]] || {
  echo "ERROR: source revision did not resolve exactly: $SOURCE_REVISION" >&2
  exit 1
}
printf '==> Staging committed source revision %s\n' "$RESOLVED_SOURCE_REVISION"
rm -rf "$ROOT"
mkdir -p "$ROOT"
git -c safe.directory="$SOURCE" -C "$SOURCE" archive --format=tar "$RESOLVED_SOURCE_REVISION" \
  | tar -C "$ROOT" -xf -

rm -f "$DIST/SHA256SUMS" "$DIST/LocalBooru-Native-Runtime-Sources.tar.xz"
[[ ",$BUNDLES," == *",appimage,"* ]] && \
  rm -f "$DIST/LocalBooru-Linux.AppImage" "$DIST/LocalBooru-Linux.zip"
[[ ",$BUNDLES," == *",deb,"* ]] && rm -f "$DIST/LocalBooru-Linux.deb"
[[ ",$BUNDLES," == *",rpm,"* ]] && rm -f "$DIST/LocalBooru-Linux.rpm"

has_bundle() {
  [[ ",$BUNDLES," == *",$1,"* ]]
}

download_checked() {
  local url="$1" output="$2" expected="$3"
  if [[ ! -f "$output" ]] || ! printf '%s  %s\n' "$expected" "$output" | sha256sum -c - >/dev/null 2>&1; then
    rm -f "$output"
    curl -fL --retry 3 --retry-delay 2 -o "$output" "$url"
  fi
  printf '%s  %s\n' "$expected" "$output" | sha256sum -c -
}

prepare_webkit() {
  local tarball="$BUILD/downloads/webkitgtk-$WEBKIT_VERSION.tar.xz"
  local source="$BUILD/webkitgtk-$WEBKIT_VERSION"
  local build_dir="$BUILD/webkit-build"
  local patch_file="$ROOT/patches/webkitgtk/2.52.3-playbin-video-filter.patch"
  local patch_hash
  local configure_stamp="$build_dir/.localbooru-config-ubuntu24-gtk3-v2"
  patch_hash="$(sha256sum "$patch_file" | cut -d' ' -f1)"

  download_checked \
    "https://webkitgtk.org/releases/webkitgtk-$WEBKIT_VERSION.tar.xz" \
    "$tarball" "$WEBKIT_SHA256"

  if [[ ! -f "$source/.localbooru-patch-$patch_hash" ]]; then
    rm -rf "$source" "$build_dir"
    tar -xJf "$tarball" -C "$BUILD"
    patch -d "$source" -p1 < "$patch_file"
    touch "$source/.localbooru-patch-$patch_hash"
  fi

  if [[ ! -f "$configure_stamp" ]]; then
    echo "==> Configuring patched WebKitGTK $WEBKIT_VERSION"
    rm -f "$build_dir"/.localbooru-config-*
    cmake -S "$source" -B "$build_dir" -G Ninja \
      -DPORT=GTK \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DUSE_GTK4=OFF \
      -DENABLE_DOCUMENTATION=OFF \
      -DENABLE_INTROSPECTION=OFF \
      -DENABLE_MINIBROWSER=OFF \
      -DENABLE_API_TESTS=OFF \
      -DENABLE_ENCRYPTED_MEDIA=OFF \
      -DENABLE_SPEECH_SYNTHESIS=OFF \
      -DENABLE_WEB_RTC=OFF \
      -DUSE_LIBBACKTRACE=OFF
    touch "$configure_stamp"
  else
    echo "==> Reusing patched WebKitGTK configuration"
  fi
  rm -f \
    "$build_dir/lib/libjavascriptcoregtk-4.1.so" \
    "$build_dir/lib/libjavascriptcoregtk-4.1.so.0" \
    "$build_dir/lib/libwebkit2gtk-4.1.so" \
    "$build_dir/lib/libwebkit2gtk-4.1.so.0"
  cmake --build "$build_dir" --target WebKit WebKitWebProcess --parallel "$JOBS"
  test -s "$build_dir/lib/libwebkit2gtk-4.1.so.0"
  test -x "$build_dir/bin/WebKitWebProcess"
}

prepare_vapoursynth() {
  local source="$BUILD/vapoursynth-r75"
  local build_dir="$BUILD/vapoursynth-build"
  local stage="$BUILD/vapoursynth-stage"
  local vs_init
  local vs_package

  if [[ ! -d "$source/.git" ]]; then
    rm -rf "$source" "$build_dir" "$stage"
    git clone --filter=blob:none https://github.com/vapoursynth/vapoursynth.git "$source"
  fi
  git -C "$source" fetch --depth 1 origin "$VAPOURSYNTH_COMMIT"
  git -C "$source" checkout --detach "$VAPOURSYNTH_COMMIT"
  [[ "$(git -C "$source" rev-parse HEAD)" == "$VAPOURSYNTH_COMMIT" ]]

  if [[ ! -f "$stage/.localbooru-vapoursynth-$VAPOURSYNTH_COMMIT" ]]; then
    rm -rf "$build_dir" "$stage"
    meson setup "$build_dir" "$source" \
      --prefix=/usr \
      --libdir=lib \
      --buildtype=release
    meson compile -C "$build_dir" -j "$JOBS"
    DESTDIR="$stage" meson install -C "$build_dir"
    touch "$stage/.localbooru-vapoursynth-$VAPOURSYNTH_COMMIT"
  fi

  # R75 installs its C API beside the Python extension. Mirror the conventional
  # include/lib layout in the private build stage for native plugin consumers.
  vs_init="$(find "$stage" -path '*/vapoursynth/__init__.py' -print -quit)"
  test -n "$vs_init"
  vs_package="$(dirname "$vs_init")"
  mkdir -p "$stage/usr/include/vapoursynth" "$stage/usr/lib"
  cp -a "$vs_package/include/." "$stage/usr/include/vapoursynth/"
  cp -a "$vs_package"/libvapoursynth.so* "$stage/usr/lib/"
  cp -a "$vs_package"/libvsscript.so* "$stage/usr/lib/"
}

stage_native_runtime() {
  local runtime="$BUILD/release-runtime/native-svp"
  local webkit="$BUILD/webkit-build"
  local vs_stage="$BUILD/vapoursynth-stage"
  local vs_package

  rm -rf "$runtime"
  mkdir -p "$runtime/bin" "$runtime/lib" "$runtime/gstreamer" \
    "$runtime/python-home/lib/python3.12/site-packages" "$runtime/licenses"

  cp -a "$webkit/lib/libwebkit2gtk-4.1.so.0"* "$runtime/lib/"
  cp -a "$webkit/lib/libjavascriptcoregtk-4.1.so.0"* "$runtime/lib/"
  install -m 0755 "$webkit/bin/WebKitWebProcess" "$runtime/bin/mpv"
  cp -a /usr/lib/x86_64-linux-gnu/libjxl.so.0.7* "$runtime/lib/"
  cp -a /usr/lib/x86_64-linux-gnu/libhwy.so.1* "$runtime/lib/"
  cp -a /usr/lib/x86_64-linux-gnu/libbrotlicommon.so.1* "$runtime/lib/"
  cp -a /usr/lib/x86_64-linux-gnu/libbrotlidec.so.1* "$runtime/lib/"
  cp -a /usr/lib/x86_64-linux-gnu/libbrotlienc.so.1* "$runtime/lib/"
  cp -a /usr/lib/x86_64-linux-gnu/liblcms2.so.2* "$runtime/lib/"

  # Build-tree RPATHs are neither needed nor portable; launchers provide the
  # packaged library search path explicitly.
  for binary in \
    "$runtime/bin/mpv" \
    "$runtime/lib/libwebkit2gtk-4.1.so.0" \
    "$runtime/lib/libjavascriptcoregtk-4.1.so.0"; do
    patchelf --remove-rpath "$binary"
  done

  # Bundle only the Python standard library and shared runtime needed by
  # VapourSynth's embedded script engine; SVPflow itself remains user-supplied.
  cp -a /usr/lib/python3.12 "$runtime/python-home/lib/"
  rm -rf "$runtime/python-home/lib/python3.12/dist-packages" \
         "$runtime/python-home/lib/python3.12/__pycache__" \
         "$runtime/python-home/lib/python3.12/test" \
         "$runtime/python-home/lib/python3.12/ensurepip"
  cp -a /usr/lib/x86_64-linux-gnu/libpython3.12.so* "$runtime/python-home/lib/"

  vs_package="$(find "$vs_stage" -type d -path '*/vapoursynth' \
    \( -path '*/site-packages/*' -o -path '*/dist-packages/*' \) | head -1)"
  [[ -n "$vs_package" ]]
  cp -a "$vs_package" "$runtime/python-home/lib/python3.12/site-packages/"
  cp "$BUILD/vapoursynth-r75/COPYING.LESSER" "$runtime/licenses/VapourSynth-COPYING.LESSER"
  cp "$ROOT/release/linux/THIRD_PARTY_NOTICES.md" "$runtime/licenses/"
  cp "$ROOT/LICENSE" "$runtime/licenses/LocalBooru-LICENSE"
  if [[ -f /usr/share/doc/python3.12/copyright ]]; then
    cp /usr/share/doc/python3.12/copyright "$runtime/licenses/Python-copyright"
  fi
  cp /usr/share/doc/libjxl0.7/copyright "$runtime/licenses/JPEG-XL-copyright"
  cp /usr/share/doc/libhwy1t64/copyright "$runtime/licenses/Highway-copyright"
  cp /usr/share/doc/libbrotli1/copyright "$runtime/licenses/Brotli-copyright"
  cp /usr/share/doc/liblcms2-2/copyright "$runtime/licenses/LCMS2-copyright"

  local vs_lib="$runtime/python-home/lib/python3.12/site-packages/vapoursynth"
  LOCALBOORU_GSTREAMER_SVP_DIR="$runtime/gstreamer" \
  LOCALBOORU_GSTREAMER_SVP_BUILD_DIR="$BUILD/gstreamer-svp-build" \
  LOCALBOORU_VAPOURSYNTH_DIR="$vs_lib" \
  LOCALBOORU_VAPOURSYNTH_INCLUDE_DIR="$vs_stage/usr/include" \
  LOCALBOORU_VAPOURSYNTH_LIB_DIR="$vs_lib" \
    bash "$ROOT/scripts/prepare-gstreamer-svp.sh"

  test -x "$runtime/bin/mpv"
  test -s "$runtime/gstreamer/libgstlocalboorupass.so"
  test -s "$runtime/gstreamer/libgstlocalbooruvs.so"
  test -s "$vs_lib/libvsscript.so"
}

build_tauri_bundles() {
  echo "==> Installing locked JavaScript dependencies"
  (cd "$ROOT" && npm ci)
  (cd "$ROOT/frontend" && npm ci)

  echo "==> Building Tauri bundles: $BUNDLES"
  (cd "$ROOT" && cargo tauri build --ci --bundles "$BUNDLES")
}

install_runtime_tree() {
  local destination="$1"
  mkdir -p "$destination/usr/lib/localbooru"
  rm -rf "$destination/usr/lib/localbooru/native-svp"
  cp -a "$BUILD/release-runtime/native-svp" "$destination/usr/lib/localbooru/"
  mkdir -p "$destination/usr/share/doc/localbooru"
  cp "$ROOT/release/linux/THIRD_PARTY_NOTICES.md" \
    "$destination/usr/share/doc/localbooru/THIRD_PARTY_NOTICES.md"
  cp "$ROOT/LICENSE" "$destination/usr/share/doc/localbooru/LICENSE"
}

package_deb() {
  has_bundle deb || return 0
  local base stage real_binary installed_size
  base="$(find "$CARGO_TARGET_DIR/release/bundle/deb" -maxdepth 1 -name '*.deb' | head -1)"
  [[ -n "$base" ]]
  stage="$BUILD/package-deb"
  rm -rf "$stage"
  dpkg-deb -R "$base" "$stage"
  install_runtime_tree "$stage"

  real_binary="$stage/usr/bin/localbooru"
  test -x "$real_binary"
  install -m 0755 "$real_binary" "$stage/usr/lib/localbooru/localbooru"
  cc -O2 -Wall -Wextra -Werror "$ROOT/release/linux/localbooru-launcher.c" \
    -o "$stage/usr/bin/localbooru"
  installed_size="$(du -sk "$stage/usr" | cut -f1)"
  python3 -c 'import pathlib,sys,re; p=pathlib.Path(sys.argv[1]); s=p.read_text(); p.write_text(re.sub(r"(?m)^Installed-Size:.*$", "Installed-Size: " + sys.argv[2], s))' \
    "$stage/DEBIAN/control" "$installed_size"
  dpkg-deb --root-owner-group --build "$stage" "$DIST/LocalBooru-Linux.deb"
}

package_rpm() {
  has_bundle rpm || return 0
  local version stage
  version="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' \
    "$ROOT/src-tauri/tauri.conf.json")"
  stage="$BUILD/package-rpm"
  rm -rf "$stage"
  mkdir -p "$stage"

  # Reuse the verified Debian payload layout so every Linux format carries the
  # same launcher and native runtime. Metadata remains RPM-specific.
  if [[ -d "$BUILD/package-deb/usr" ]]; then
    cp -a "$BUILD/package-deb/usr" "$stage/"
  else
    local base_rpm extracted real_binary
    base_rpm="$(find "$CARGO_TARGET_DIR/release/bundle/rpm" -maxdepth 1 -name '*.rpm' | head -1)"
    [[ -n "$base_rpm" ]]
    extracted="$BUILD/base-rpm"
    rm -rf "$extracted"; mkdir -p "$extracted"
    (cd "$extracted" && rpm2cpio "$base_rpm" | cpio -idm --quiet)
    cp -a "$extracted/usr" "$stage/"
    install_runtime_tree "$stage"
    real_binary="$stage/usr/bin/localbooru"
    install -m 0755 "$real_binary" "$stage/usr/lib/localbooru/localbooru"
    cc -O2 -Wall -Wextra -Werror "$ROOT/release/linux/localbooru-launcher.c" \
      -o "$stage/usr/bin/localbooru"
  fi

  fpm -s dir -t rpm -C "$stage" \
    -n localbooru -v "$version" --iteration 1 \
    --license MIT --category Graphics \
    --description 'Local image library with automatic tagging' \
    --url 'https://github.com/DonutsDelivery/LocalBooru' \
    --depends gtk3 --depends webkit2gtk4.1 \
    --depends gstreamer1-plugins-base --depends gstreamer1-plugins-good \
    --depends gstreamer1-plugins-bad-free \
    -p "$DIST/LocalBooru-Linux.rpm" .
}

package_appimage() {
  has_bundle appimage || return 0
  local base stage tool tool_status
  base="$(find "$CARGO_TARGET_DIR/release/bundle/appimage" -maxdepth 1 -name '*.AppImage' | head -1)"
  [[ -n "$base" ]]
  stage="$BUILD/AppDir"
  rm -rf "$stage" "$BUILD/squashfs-root"
  (cd "$BUILD" && chmod +x "$base" && "$base" --appimage-extract >/dev/null)
  mv "$BUILD/squashfs-root" "$stage"
  install_runtime_tree "$stage"
  mv "$stage/AppRun" "$stage/AppRun.tauri"
  install -m 0755 "$ROOT/release/linux/AppRun" "$stage/AppRun"

  tool="$BUILD/downloads/appimagetool-x86_64.AppImage"
  download_checked \
    'https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage' \
    "$tool" "$APPIMAGETOOL_SHA256"
  chmod +x "$tool"
  rm -f "$DIST/LocalBooru-Linux.AppImage"
  set +e
  env -u SOURCE_DATE_EPOCH ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 \
    "$tool" "$stage" "$DIST/LocalBooru-Linux.AppImage"
  tool_status=$?
  set -e
  # The 2023 continuous appimagetool wrapper can propagate SIGPIPE (141)
  # after the inner tool has successfully written and finalized the image.
  if [[ "$tool_status" -ne 0 && "$tool_status" -ne 141 ]]; then
    return "$tool_status"
  fi
  test -s "$DIST/LocalBooru-Linux.AppImage"
  chmod +x "$DIST/LocalBooru-Linux.AppImage"
}

package_source_offer() {
  local source_stage="$BUILD/native-runtime-sources"
  rm -rf "$source_stage"
  mkdir -p "$source_stage"
  cp "$BUILD/downloads/webkitgtk-$WEBKIT_VERSION.tar.xz" "$source_stage/"
  cp "$ROOT/patches/webkitgtk/2.52.3-playbin-video-filter.patch" "$source_stage/"
  git -C "$BUILD/vapoursynth-r75" archive --format=tar.gz \
    --output="$source_stage/vapoursynth-R75.tar.gz" "$VAPOURSYNTH_COMMIT"
  cp "$ROOT/Dockerfile.linux-release" "$source_stage/"
  cp "$ROOT/scripts/build-linux-docker.sh" "$source_stage/"
  cp "$ROOT/release/linux/THIRD_PARTY_NOTICES.md" "$source_stage/"
  tar -cJf "$DIST/LocalBooru-Native-Runtime-Sources.tar.xz" \
    -C "$source_stage" .
}

prepare_webkit
prepare_vapoursynth
stage_native_runtime
build_tauri_bundles
package_deb
package_rpm
package_appimage
package_source_offer

if has_bundle appimage; then
  rm -f "$DIST/LocalBooru-Linux.zip"
  (cd "$DIST" && zip -q LocalBooru-Linux.zip \
    LocalBooru-Linux.AppImage LocalBooru-Native-Runtime-Sources.tar.xz)
fi

bash "$ROOT/scripts/verify-linux-release.sh" "$DIST" "$BUNDLES"
(cd "$DIST" && sha256sum LocalBooru-* | sort -k2 > SHA256SUMS)
ccache --show-stats
