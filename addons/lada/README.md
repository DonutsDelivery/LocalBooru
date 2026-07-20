# LocalBooru LADA sidecar

This directory is a separately licensed **AGPL-3.0-only** add-on. It is not part of LocalBooru's MIT base application artifact.

The sidecar adapts pinned [LADA](https://github.com/ladaapp/lada) `FrameRestorer` output into LocalBooru's versioned, bounded local protocol. It opens only canonical local paths supplied by LocalBooru, preserves source timestamps, and exposes restored BGR frames through three leased shared-memory buffers.

## User contract

Users install, repair, update, and remove this package through LocalBooru Add-on Manager. They do not configure Python, models, executables, sockets, or a separate manager.

## Commands

```bash
localbooru-lada-sidecar probe --config probe.json
localbooru-lada-sidecar serve --config session.json --socket-fd 3
```

The package builder produces backend and model archives plus `release-manifest.json`:

```bash
./packaging/build-bundles.sh
# Optional: LADA_CUDA_VARIANT=cuda-legacy ./packaging/build-bundles.sh
```

One invocation produces the common runtime, CUDA and Intel XPU layers, model bundle, Corresponding Source, and a single manifest that binds all four installable packages.

Generated release artifacts are not committed or published automatically. Every offered binary must be accompanied by the exact `source.tar.zst`, license notices, hashes, and source link recorded in its manifest.

## License

Copyright 2026 LocalBooru contributors.

This add-on is free software under the GNU Affero General Public License version 3 only. See `LICENSE`. LADA and its first-party weights retain their own copyright and AGPL notices. This is an engineering packaging boundary, not legal advice.
