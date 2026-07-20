# Release and Infrastructure Guide

This document is the operational source of truth for LocalBooru release builds.

## Platform ownership

| Platform | Normal build owner | Fallback |
|---|---|---|
| Linux x86_64 | Local Docker/Podman | Manual `Linux release fallback` GitHub workflow |
| Windows x86_64 | Local MSVC/Wine Docker build | Native `windows-2022` job in the manual fallback workflow |
| macOS universal | Native `macos-14` job in the manual fallback workflow | None; Apple tooling is required |
| iOS | GitHub workflow / Apple toolchain | `.github/workflows/build-ios.yml` |

Desktop workflows upload inspection artifacts only. Tag creation and publication
remain explicit operator actions after every promised platform is verified.

## Windows release command

From the repository root on Linux:

```bash
npm run release:windows
```

Equivalent direct commands:

```bash
./scripts/build-windows-local.sh
./scripts/build-windows-local.sh --rebuild-image
LOCALBOORU_BUILD_JOBS=4 ./scripts/build-windows-local.sh
```

`scripts/build-windows-local.sh` is the host entry point;
`scripts/build-windows-docker.sh` is the internal container entry point. The
image downloads the genuine MSVC toolchain and Windows SDK through the pinned
`mstorsjo/msvc-wine` revision, plus Node 20, Rust, Tauri CLI 2.9.4, LLVM/LLD,
and NSIS. The first image build is intentionally large. Do not redistribute the
downloaded Microsoft toolchain image.

Source staging uses `git archive HEAD`, so the build always uses the exact
committed revision and excludes dirty files, sockets, host `node_modules`, APKs,
and local caches. Commit release changes before invoking the wrapper.

The default persistent directories are:

- `/mnt/storage/Programs/localbooru-build-windows-docker` — Cargo registry,
  target tree, npm state, and disposable worktree;
- `/mnt/storage/Programs/localbooru-sccache-windows-docker` — reserved compiler
  cache directory;
- `dist-windows-local/` — final verified artifacts.

Override them with `LOCALBOORU_WINDOWS_BUILD_ROOT`,
`LOCALBOORU_WINDOWS_SCCACHE_ROOT`, and `LOCALBOORU_DIST_WINDOWS_DIR`.
Ubuntu's sccache 0.7.7 is deliberately not enabled as `RUSTC_WRAPPER`: `cc-rs`
would apply it to MSVC resource preprocessing, where `/showIncludes` detection
fails through Wine. Persistent Cargo target reuse remains active. Rust's large
final Windows link uses native `lld-link` with the genuine MSVC/Windows SDK
libraries; `cl`, `lib`, and resource compilation continue through the MSVC
wrappers.

`dist-windows-local/` must contain:

- `LocalBooru-Windows-Setup.exe` — unsigned NSIS installer;
- `LocalBooru-Windows.zip` — portable x64 executable and license;
- `SHA256SUMS-Windows` — LF-terminated basename-only hashes.

The container verifies ZIP and NSIS integrity, extracts the installer, requires
both standalone and installed `LocalBooru.exe` payloads to be PE32+ x86-64,
rejects private checkout/build paths, and verifies hashes again on the host. The
outer NSIS launcher may correctly be an i386 PE32 stub. Docker/Wine packaging is
not Windows GUI acceptance: before publication, install and run the artifacts
on a real supported Windows system and report Authenticode state accurately.
See `docs/WINDOWS_PACKAGING.md` for the concise command reference.

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
The checkout is mounted read-only; the container resolves the requested commit and
stages it with `git archive`, so uncommitted and untracked host files cannot enter
the release. The resolved source SHA is printed before npm, Tauri, or Cargo runs.

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

`.github/workflows/build.yml` is manual-only. Its Linux job calls the same local
Docker wrapper; Windows uses native `windows-2022`; macOS uses native `macos-14`
to produce an explicitly ad-hoc-signed universal x86_64/arm64 bundle. The macOS
artifacts are not Developer ID signed or notarized, so Gatekeeper may still
require a user override. These jobs upload short-lived inspection artifacts and
never create a GitHub release. The iOS workflow is also manual-only and uploads
an Actions artifact without publishing it.

Creating tags, pushing commits, signing packages, and publishing a GitHub
release are explicit operator actions after local artifact verification. Never
publish merely because a tag was pushed.
