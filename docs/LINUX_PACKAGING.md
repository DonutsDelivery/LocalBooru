# LocalBooru Linux Packaging

Linux releases are built with the project-owned Docker/Podman toolchain, not
with host libraries and not with the retired Electron GitHub workflow.

Build every Linux format:

```bash
npm run release:linux
```

Build one format:

```bash
npm run tauri:build:linux:appimage
npm run tauri:build:linux:deb
npm run tauri:build:linux:rpm
```

Artifacts are written to `dist-linux-local/` after structural, native-runtime,
license-source, and portability verification. The persistent WebKitGTK build
cache lives in `build-linux-docker/` by default.

The complete operational guide—including external cache paths, artifact names,
SVP runtime ownership, verification gates, and GitHub fallback policy—is:

[`docs/agents/release-and-infrastructure.md`](agents/release-and-infrastructure.md)

Do not add host package-install instructions here. The Dockerfile is the Linux
toolchain source of truth.