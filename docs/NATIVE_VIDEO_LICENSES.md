# Native Video Licensing and Provenance

This is an engineering inventory, not legal advice. Reuse decisions must be updated when source headers carry explicit SPDX terms or the copyright owner issues a written relicensing decision.

| Component | Evidence | LocalBooru decision |
|---|---|---|
| LocalBooru application | Root `LICENSE`: MIT | Keep Rust/React/Tauri integration MIT. |
| Newly authored LocalBooru native protocol/coordinator/helper skeleton | Created in this repository without copying Pure Harmony source | Rust protocol/coordinator remain MIT. The separately built helper's final license must match the exact linked FFmpeg configuration; a GPL-enabled FFmpeg build requires GPL-compatible helper distribution and corresponding source. |
| Pure Harmony `video-helper` | `video-helper/CMakeLists.txt:1-8` explicitly calls the helper GPL-2.0-or-later and says its source is published | May be distributed as a separate GPL sidecar if its corresponding source, license and notices are provided. Do not link it into the MIT Tauri binary. Keep IPC/process separation. |
| Pure Harmony plugin/standalone and `plugin/Source/SharedGpuSurface*` | `LICENSE-COMMERCIAL.md` applies to Arbit software; SharedGpuSurface headers have no inspected standalone SPDX grant | **Blocked for direct copying.** Clean-room implement a LocalBooru consumer from documented wire behavior, or obtain an explicit MIT/GPL-compatible relicensing grant from the owner. |
| `SharedGpuSurfaceProtocol.h` | Stored under commercial `plugin/Source`, but included by the GPL helper through `video-helper/CMakeLists.txt:339-342`; no standalone license header found | Licensing is ambiguous. Do not copy until explicitly dual-licensed. Define a fresh LocalBooru protocol and adapt a GPL helper implementation to it. |
| FFmpeg/libav | Current system libraries; Pure Harmony helper declares GPL sidecar specifically because it links FFmpeg | Keep FFmpeg in a separately distributed helper. Record the exact configured FFmpeg license/features in release artifacts and provide required source/notices. Do not link libav into LocalBooru’s MIT Rust binary. |
| `rife_net` | `video-helper/src/rife_net/LICENSE.nihui`; Pure Harmony comments identify MIT upstream | Reuse only after preserving the upstream license and verifying every vendored file’s provenance. Keep optional model licenses separate. |
| ncnn | Pure Harmony build comments identify BSD-3-Clause | Preserve BSD notice and source attribution if enabled. |
| nlohmann/json | MIT | Compatible; include notice in helper third-party manifest. |
| GLFW | zlib/libpng | Compatible; include notice. |
| Vulkan headers/loader | Apache-2.0/MIT-style Khronos terms, depending component | Pin versions and include upstream notices. Do not bundle a GPU vendor driver. |
| SVPflow/SVP Manager/VapourSynth plugins | No redistributable SDK or license grant established in this audit; existing LocalBooru uses user-installed/add-on assets | **Do not bundle or copy SVPflow.** Support only explicit discovery/invocation of a legitimately user-installed runtime until written redistribution/API terms are established. Report its copy mode truthfully. |
| VapourSynth | Open-source core, plugins vary | Core integration alone does not grant rights to proprietary SVP plugins. Audit each shipped plugin separately. |
| Subtitle parser/renderer | Local parser plus libavcodec embedded-subtitle decode; GTK/Pango renders text in the native host | Record libavcodec's exact release configuration. No libass code is currently linked directly by the helper. Fonts remain user/system assets unless separately licensed. |
| Whisper sidecar/models | Existing LocalBooru add-on path | Renderer consumes durable cue files only. Preserve existing model and sidecar notices; do not move Python/model code into the native helper. |

## SVP/VapourSynth native-adapter contract

The existing `addons/svp/app.py` integration is an external-process HLS adapter, not a reusable native-frame API. It discovers `vspipe`, Python `vapoursynth`, and `svpflow1`/`svpflow2` shared libraries from a user installation. Its current pipeline decodes to raw planar `yuv420p`, copies each plane into VapourSynth frames, runs one generated SVPflow graph, emits frames through `vspipe`, re-encodes them, and writes HLS segments. That implementation therefore has multiple CPU copies plus an encode/decode round trip and cannot be described as zero-copy.

The native Lightbox adapter may reuse only the following contract:

- discover a legitimate external VapourSynth/SVPflow installation; never bundle or copy user-installed SVP binaries;
- create one long-lived VapourSynth graph per active native generation;
- exchange bounded in-memory frames with backpressure; no HLS, disk segments, or LocalBooru dependency on MPV;
- flush and rebuild the graph on seek, media replacement, incompatible format changes, or generation change;
- expose source/target FPS, queue depth, scene-change behavior, and every CPU/GPU copy in diagnostics;
- fall back to ordinary native playback when any required runtime, plugin, format, or synchronization capability is unavailable.

Evidence recorded during the 2026-07-12 development audit:

- the installed runtime reports VapourSynth Core R75, API R4.2/R3.6, with `vspipe` at `/usr/bin/vspipe`;
- the inspected external plugin installation exposes `libsvpflow1.so` and `libsvpflow2.so` under `/opt/svp/plugins`;
- VapourSynth upstream ships `COPYING.LESSER` containing LGPL-2.1 terms;
- SVP's public SVPflow manual states that `svpflow1` is distributed under the GNU GPL, but this audit did not establish complete redistribution terms for every SVPflow binary, SVP Manager, platform package, or model/runtime dependency.

Consequently, release builds must treat the whole SVPflow installation as user-supplied and dynamically discovered until a complete per-artifact redistribution audit or written grant exists. LocalBooru must not link or embed MPV as a workaround; MPV may be consulted only as a behavioral reference.


## Required release actions

1. Keep explicit `SPDX-License-Identifier: MIT` headers on every independently authored native-helper source/header/test and its CMake manifest. The linked helper artifact's distribution obligations still follow the exact FFmpeg configuration.
2. If the Pure Harmony GPL helper is reused, ship it as a separate process with complete corresponding source and GPL notices.
3. Obtain an explicit license for `SharedGpuSurfaceProtocol.h` before copying it; otherwise retain the independently designed LocalBooru protocol.
4. Detect, but do not redistribute, SVP unless a written redistributable grant is obtained.
5. Generate a native-helper third-party notice from the exact CMake configuration used for release.
6. Record whether FFmpeg was built GPL, LGPL, or nonfree; reject a `nonfree` release configuration.

`scripts/audit-native-video-release.py` now enforces the machine-checkable portion of this gate. It rejects unresolved dynamic dependencies, any FFmpeg `--enable-nonfree` configuration, a mismatched pinned FFmpeg configuration hash, and release mode without a non-empty notices file or complete dependency-to-SPDX license map. Its JSON manifest records the helper hash, normalized FFmpeg configuration and hash, complete runtime dependency closure with per-file hashes and concluded license expressions, and stable fixture hashes. `distribution_approved` is never true in development mode; release mode requires the pinned configuration hash, notices input, and no `NOASSERTION` dependency. The packaging environment remains responsible for reviewing the license map and generating the corresponding notices.

The 2026-07-13 Arch development build is **not release evidence**. It reports `--enable-gpl --enable-version3` and no `--enable-nonfree`, so the helper built on that host must be treated as GPLv3-compatible. Its very large distro-specific dynamic dependency closure also makes that binary unsuitable for redistribution in Debian/RPM/AppImage artifacts. Release helpers must be built in the target packaging environment from a pinned FFmpeg configuration, and the generated build configuration, dependency closure, notices, source offer, and fixture hashes must accompany each artifact.
