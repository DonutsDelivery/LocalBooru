# Windows packaging

LocalBooru has two equivalent Windows x64 build paths:

- `scripts/build-windows-ci.ps1` runs on native `windows-2022` GitHub Actions.
- `scripts/build-windows-local.sh` runs locally on Linux through the genuine MSVC toolchain and Windows SDK under Wine.

The local path is the normal reproducible Docker entry point:

```bash
./scripts/build-windows-local.sh
./scripts/build-windows-local.sh --rebuild-image
LOCALBOORU_BUILD_JOBS=4 ./scripts/build-windows-local.sh
```

The first image build downloads MSVC, the Windows SDK, Rust, Node.js, and the pinned Tauri CLI, so it is large and slow. Later builds reuse:

- `/mnt/storage/Programs/localbooru-build-windows-docker` for Cargo/npm build state;
- `/mnt/storage/Programs/localbooru-sccache-windows-docker` for `sccache` objects;
- `dist-windows-local/` for final artifacts.

Override these with `LOCALBOORU_WINDOWS_BUILD_ROOT`, `LOCALBOORU_WINDOWS_SCCACHE_ROOT`, and `LOCALBOORU_DIST_WINDOWS_DIR`.

Expected outputs:

- `LocalBooru-Windows.zip` containing the x64 standalone executable and license;
- `LocalBooru-Windows-Setup.exe` NSIS installer;
- `SHA256SUMS-Windows` with LF-terminated basename-only entries.

The verifier checks archive integrity, PE headers, x64 architecture for both the standalone and installed NSIS payload, forbidden build paths, and final hashes. The NSIS launcher itself may be an i386 PE32 stub while installing the x64 PE32+ application. Local/CI artifacts are unsigned unless a separate Authenticode signing stage is configured.
