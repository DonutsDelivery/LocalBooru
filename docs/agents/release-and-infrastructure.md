# Release and Infrastructure Guide

This document is the operational source of truth for LocalBooru release builds.

## Platform ownership

| Platform | Normal build owner | Fallback |
|---|---|---|
| Linux x86_64 | Local Docker/Podman | Manual `Linux release fallback` GitHub workflow |
| Windows | Not yet supported by the Tauri release pipeline | None; the removed workflow built obsolete Electron artifacts |
| macOS | Not yet supported by the Tauri release pipeline | None; the removed workflow built obsolete Electron artifacts |
| iOS | GitHub workflow / Apple toolchain | `.github/workflows/build-ios.yml` |

Do not restore tag-triggered desktop publishing until a current Tauri pipeline has
been implemented and verified for that platform.

## Linux release command

From the repository root:

```bash
npm run release:linux
```

Equivalent direct command:

```bash
./scripts/build-linux-local.sh
```

Target one package format with `--appimage`, `--deb`, or `--rpm`. Rebuild the
toolchain image after changing `Dockerfile.linux-release`:

```bash
./scripts/build-linux-local.sh --rebuild-image
```

The host wrapper owns Docker/Podman invocation. `scripts/build-linux-docker.sh`
is an internal container entry point and must not be run directly on the host.
`scripts/build-tauri-linux.sh` remains only as a compatibility redirect.
The checkout is mounted read-only; the container refreshes an isolated filtered
worktree under the persistent build root before npm, Tauri, or Cargo runs.

## Persistent state

The default persistent directories are:

- `build-linux-docker/` — WebKitGTK, VapourSynth, Cargo, and package staging
- `.ccache-docker/` — C/C++ compiler cache
- `dist-linux-local/` — final verified artifacts

Large build roots can live on another filesystem without changing the source
checkout:

```bash
LOCALBOORU_DOCKER_BUILD_ROOT=/path/to/build-cache \
LOCALBOORU_CCACHE_DIR=/path/to/ccache \
LOCALBOORU_DIST_LINUX_DIR=/path/to/artifacts \
./scripts/build-linux-local.sh
```

`LOCALBOORU_BUILD_JOBS` or `--jobs` limits compilation parallelism. WebKitGTK is
the dominant build cost; do not delete the persistent build root between
releases.

## Linux build contents

The container builds:

1. Node 22 frontend assets from both npm lockfiles.
2. Tauri 2 / Rust release binary from `Cargo.lock`.
3. WebKitGTK 2.52.3 from the pinned upstream tarball and SHA-256.
4. The tracked LocalBooru WebKit patch, including:
   - application-owned GStreamer `video-filter` insertion;
   - `LOCALBOORU_WEB_PROCESS_PATH` selection for the Manager-compatible helper.
5. VapourSynth R75 from a pinned Git commit.
6. LocalBooru's GStreamer pass-through and VapourSynth bridge plugins.
7. AppImage, Debian, RPM, portable ZIP, source-compliance archive, and checksums.

Linux packages bundle only redistributable WebKitGTK, VapourSynth, Python, and
LocalBooru bridge components. They do **not** bundle SVP Manager, SVPflow, GPU
drivers, models, or credentials. A legitimate host installation supplies SVP
Manager/SVPflow at runtime.

The package launcher selects the bundled patched WebKit before the dynamic
loader starts LocalBooru. `LOCALBOORU_ENABLE_NATIVE_SVP=0` remains an explicit
runtime opt-out; ordinary WebKit/GStreamer playback remains available.

## Expected artifacts

`dist-linux-local/` must contain:

- `LocalBooru-Linux.AppImage`
- `LocalBooru-Linux.deb`
- `LocalBooru-Linux.rpm`
- `LocalBooru-Linux.zip`
- `LocalBooru-Native-Runtime-Sources.tar.xz`
- `SHA256SUMS`

The source archive accompanies the modified LGPL runtime and includes the exact
upstream WebKitGTK source tarball, LocalBooru patch, VapourSynth source, and
container build recipe.

## Verification gates

The container calls `scripts/verify-linux-release.sh` before writing checksums.
It verifies:

- package/archive structure and readability;
- the real Tauri executable plus launcher layout;
- patched WebKitGTK and Manager-compatible WebProcess helper presence;
- both LocalBooru GStreamer plugins;
- VapourSynth/Python runtime and plugin RPATH;
- native source-compliance contents;
- absence of SVP Manager/SVPflow and private host paths;
- the final executable's observed glibc symbol floor.

After extraction or installation, perform a host acceptance pass with exactly
one SVP Manager process:

1. Launch LocalBooru from the package, not the source tree.
2. Verify ordinary playback with SVP disabled.
3. Enable SVP and open a representative 4K file from the library.
4. Verify Manager detection, Manager-generated script use, genuine SVP OSD,
   advancing video, and smooth motion.
5. Toggle SVP off and on again.
6. For NVIDIA Optical Flow, explicitly select the NVIDIA rendering device in
   SVP Manager. `Do not change` did not select the GPU in the verified Linux
   environment and caused optical flow to run on the CPU.

Logs prove graph setup; visual motion quality still requires human confirmation.

## GitHub workflow policy

`.github/workflows/build.yml` is manual-only and calls the same Docker wrapper.
It is an emergency fallback, not the normal release path. It uploads artifacts
for inspection but does not create a GitHub release.

Creating tags, pushing commits, signing packages, and publishing a GitHub
release are explicit operator actions after local artifact verification. Never
publish merely because a tag was pushed.
