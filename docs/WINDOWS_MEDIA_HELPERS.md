# Source-built Windows media helpers

`scripts/build-windows-media.py` is a prototype for replacing the prebuilt
FFmpeg dependency with an explicit, source-pinned Windows x86-64 build.
It is wired into the local release wrapper but has not passed a real
compiler or Windows runtime run. The current Gyan-based package remains an
inspection artifact with its dependency-source closure unresolved.

The manifest pins the Linux-host LLVM-MinGW UCRT compiler, FFmpeg, x264, WebP,
dav1d, zlib, NVIDIA codec headers, and build-only Meson/NASM. NVIDIA drivers and
the proprietary CUDA toolkit are not redistributed. The container's native
Clang supplies NVPTX compilation; driver availability is a runtime check.
Static codec libraries avoid additional non-system DLLs. LLVM and MinGW runtime
notices accompany the tools.

Required features include CPU H.264/AAC, AV1 software decoding, WebP/PNG/JPEG
thumbnails, HLS, and CUDA decode/scale/NVENC. Missing components, nonfree
configuration, unexpected DLL imports, changed source caches, checksum
mismatches and unsafe archives fail the build.

Run only inside the canonical Windows release container while the host
heavy-build gate is held, with one compiler job. A small derivative
`Dockerfile.windows-media` adds Ninja and pkg-config to the existing Windows
toolchain image; reuse that base rather than rebuild the Microsoft SDK.
The environment marker below is a caller contract, not proof of lock admission.

```sh
DMC_WINDOWS_MEDIA_BUILD_LOCKED=1 python3 scripts/build-windows-media.py \
  --cache /build/windows-media-cache --output /build/new-windows-media-output
```

Use an absolute cache path without whitespace and a fresh absolute output path.
Recognized downloads and verified immutable source/compiler trees are reused.
Each compile attempt gets a fresh private workspace: NASM, codec libraries,
Meson state, FFmpeg objects and staging products are never accepted from a
previous attempt. Finalized helper outputs are reused only after their payload,
source, recipe and notice checks pass. No global compiler installation or user
application profile is used. Outputs include
`bin/ffmpeg.exe`, `bin/ffprobe.exe`, notices, selected sources and a receipt
binding configuration, compiler identity, input/output hashes and PE imports.

```sh
python3 scripts/build-windows-media.py --check-output \
  --output /build/new-windows-media-output
```

Verification binds the payload, input archives and recipe files. It is unsigned
provenance, not independent build attestation. The local wrapper stages helpers, notices, source offer and receipts into
installer and portable payloads, and checks both against the verified output.
The optional MSIX path consumes that portable payload. Application and media
sources also accompany the packages and their checksums. Test H.264/AAC/AV1 decoding, WebP thumbnails, HLS, hardware
fallback and restart on actual Windows. Wine CLI checks are build evidence.

Lightweight fixtures use a disposable directory on owned storage:

```sh
DMC_MEDIA_FIXTURE_ROOT=/absolute/owned/storage/fixtures \
  python3 -B scripts/test-windows-media.py
```
