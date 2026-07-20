# LocalBooru Linux Native Runtime Notices

LocalBooru itself is distributed under the MIT license in `LICENSE`.

The Linux release bundles modified WebKitGTK libraries solely to expose the
application-selected GStreamer video-filter hook and WebProcess path override.
The exact upstream source is WebKitGTK 2.52.3 from:

https://webkitgtk.org/releases/webkitgtk-2.52.3.tar.xz

Upstream SHA-256:

`5b3e0d174e63dcc28848b1194e0e7448d5948c3c2427ecd931c2c5be5261aebb`

The complete LocalBooru patch is included in the source artifact and in the
LocalBooru repository at `patches/webkitgtk/2.52.3-playbin-video-filter.patch`.
WebKitGTK carries LGPL-2.1-or-later and BSD-family component licenses; consult
the bundled source tree for the license applying to each file.

The Linux release also bundles VapourSynth R75 core and Python bindings from:

https://github.com/vapoursynth/vapoursynth/tree/R75

VapourSynth is distributed under LGPL-2.1; its `COPYING.LESSER` is included in
the source artifact.

Python 3.12 runtime components are redistributed under the Python Software
Foundation License Version 2. The license text is included in the runtime
source/notices artifact.

The patched WebKit runtime also carries its Ubuntu 24.04 JPEG XL dependency
closure: JPEG XL, Highway, Brotli, and Little CMS 2. Their package copyright
and license notices are included alongside this file in the runtime `licenses`
directory.

The `libgstlocalboorupass.so` and `libgstlocalbooruvs.so` bridge plugins are
LocalBooru components covered by the repository MIT license.

SVP Manager and SVPflow are **not bundled**. They are detected only from a
legitimate user installation and remain governed by their upstream terms.
GPU vendor drivers are not bundled.
