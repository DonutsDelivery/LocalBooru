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
## GNOME and Wayland icon identity

The GTK application ID is `com.localbooru.app`, matching the installed
`com.localbooru.app.desktop` and hicolor icon. `StartupWMClass=donutmediacenter`
also matches older binaries and X11 windows. The historical data/package ID is
preserved; this does not create a new library.

The local installer registers this identity automatically, including when an
existing binary skips compilation. To repair an existing local desktop entry
without rebuilding or closing DMC:

```bash
python3 scripts/install-linux-desktop.py --launcher "$PWD/run.sh"
```

Linux packages use the same registrar against their staging directory. It never
updates host caches or favorites when `--data-dir` is supplied. Local repairs
retain old launchers as hidden aliases and migrate an existing GNOME pin to the
canonical desktop entry.
