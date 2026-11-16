# LocalBooru Linux Native Runtime Notices

DonutMediaCenter itself is distributed under GPL-3.0-only in `LICENSE`.
Retain `NOTICE` and `LICENSES/MIT-legacy.txt` for code previously distributed
under MIT. Branding is addressed separately in `TRADEMARKS.md`.
Third-party runtime components retain the licenses described below.

The staged distribution libraries also carry package copyright notices under
`usr/share/doc/localbooru/distro-packages/`, including referenced common license
texts. Its `manifest.json` records candidate package names, exact binary/source
versions, staged file names and notice hashes. Ambiguous basename owners retain
all candidate notices. This manifest records notice retention; the separate
source correspondence audit establishes binary origins. The application source
includes `scripts/collect-linux-distro-notices.py` to reproduce this step inside
the build image. Existing custom-runtime notices below remain separate.

The Linux release bundles modified WebKitGTK libraries for the
application-selected GStreamer video-filter hook, WebProcess path override,
private video bridge access inside Flatpak's nested WebProcess sandbox,
and discovery of matching bundled WebKit process helpers.
The exact upstream source is WebKitGTK 2.52.3 from:

https://webkitgtk.org/releases/webkitgtk-2.52.3.tar.xz

Upstream SHA-256:

`5b3e0d174e63dcc28848b1194e0e7448d5948c3c2427ecd931c2c5be5261aebb`

The complete LocalBooru patches are included in the source artifact and in the
LocalBooru repository at `patches/webkitgtk/2.52.3-playbin-video-filter.patch`
`patches/webkitgtk/2.52.3-flatpak-video-bridge.patch`, and
`patches/webkitgtk/2.52.3-bundled-process-directory.patch`, applied in that order.
The source artifact also includes `prepare-webkit-flatpak-bridge.py`, which
validates and applies the second patch before rebuilding the affected runtime.
WebKitGTK carries LGPL-2.1-or-later and BSD-family component licenses; consult
the bundled source tree for the license applying to each file.

The Linux release also bundles VapourSynth R75 core and Python bindings from:

https://github.com/vapoursynth/vapoursynth/tree/R75

VapourSynth is distributed under LGPL-2.1; its `COPYING.LESSER` is included in
the source artifact.

Python 3.12 runtime components are redistributed under the Python Software
Foundation License Version 2. The package notice and license text are included as
`licenses/Python-copyright` in the bundled runtime.

The patched WebKit runtime also carries its Ubuntu 24.04 JPEG XL dependency
closure: JPEG XL, Highway, Brotli, and Little CMS 2. Their package copyright
and license notices are included alongside this file in the runtime `licenses`
directory.

The AppImage includes Ubuntu gstreamer1.0-gl; its package copyright and
license notice accompanies the bundled plugin.

The AppImage also retains the build image's AVIF decoder dependency chain,
including libavif, libgav1, libyuv and Abseil. This keeps the replacement
WebKit paired with matching codecs. Package notices are retained in
`distro-packages/`; `avif-dependencies.json` records codec hashes and direct
dependencies. The staging and verification helper is included in the native
source artifact. Host glibc and C/C++ support libraries remain platform
dependencies. The manifest is a byte inventory, not source attestation.

The `libgstlocalboorupass.so` and `libgstlocalbooruvs.so` bridge plugins and
`liblocalbooru-webkit-gst-hook.so` filter hook are DonutMediaCenter components
covered by the repository GPL-3.0-only license. Their source is in
`gstreamer-svp/src/` in the exact application-source archive. Retain the
bundled historical MIT notice for previously MIT-licensed portions.

SVP Manager and SVPflow are **not bundled**. They are detected only from a
legitimate user installation and remain governed by their upstream terms.
GPU vendor drivers are not bundled.
