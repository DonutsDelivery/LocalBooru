# Linux SVP Manager endpoint isolation

The old patched WebProcess relay bound `/tmp/mpvsocket` at application startup
and kept it while idle. It did not steal an already listening socket, but it
prevented a later player from binding SVP's default endpoint. No shared SVP
configuration writes were found in this integration. A VLC launch difference
is a separate player configuration issue; this patch does not change VLC or
Manager settings.

## Discovery and ownership

SVP documents concurrent mpv discovery from `/tmp/mpvSockets/PID` starting with
4.3.191: [SVP mpv guide](https://www.svp-team.com/wiki/SVP:mpv) and
[mpvSockets reference](https://github.com/wis/mpvSockets). This change targets
Linux. Windows and macOS retain their existing endpoints and have not received
new native discovery acceptance testing. Android's direct SVPflow stream does
not use Manager or these endpoints.

The selected video element carries a unique `localbooru-svp-host-…` HTML ID.
WebKit 2.52.3's `HTMLMediaElement::attributeChanged(idAttr)` stores it in `m_id`;
`mediaPlayerElementId()` returns that value, and `MediaPlayer::elementId()`
passes it to GStreamer. Registration happens both when creating the playbin
and when its element ID changes. The actual video player's WebProcess writes
its PID into DMC's private, instance-specific registration directory.

Rust issues a document epoch and accepts monotonic host/phase revisions. It
publishes only the selected registered host. The relay advertises only that
process's PID socket while its lease is active. Manager connections capture
the exact lease and recheck it under the transition lock before commands.
Filter/pause events and delayed frontend remounts also check the same owner.
Late old enables, disables, events and commands cannot affect a successor.

Registration is exclusive; cleanup checks its inode. Relay cleanup removes
only its own PID socket inode. Other mpvSockets entries and `/tmp/mpvsocket`
are never removed. Filter/script/backend paths include the application PID;
registration paths additionally use the existing random snapshot namespace.
The new relay uses `LOCALBOORU_MPV_CONTROL_UPSTREAM_V2`; Rust clears the legacy
key so an unmatched old runtime cannot publish the shared default endpoint.
App and runtime must be deployed together to retain native interpolation.
A missing host registration gets five current-owner attempts over one second
before an explicit availability error. Stale retry receipts cannot affect a
successor; original playback controls become available if registration fails.

## Runtime upgrade and build boundary

`2.52.3-playbin-video-filter.patch` is the durable pristine-source patch.
It applies to the pinned upstream 2.52.3 tar without replacing unrelated
custom changes. `2.52.3-existing-mpv-relay-upgrade.patch` upgrades the existing
external relay and includes exact before/after SHA256 values for two source
files. The preparation helper refuses unknown or partial preimages.

Read-only preimage check:

```bash
python3 scripts/prepare-webkit-mpv-isolation.py \
  --source /mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched/src/webkitgtk-2.52.3
```

The minimal cached target closure, if the cache is otherwise compatible, is:

1. WebCore `UnifiedSource-3c72abbe-58.cpp.o` (GStreamer player).
2. WebKit `UnifiedSource-54928a2b-47.cpp.o` (WebProcess startup/relay).
3. `lib/libwebkit2gtk-4.1.so.0.21.7`, its symlinks and WebProcess link if needed.

WebCore objects link directly into the shared library; there is no additional
WebCore archive. The current library is about 125 MiB; the WebProcess shim is
about 16 KiB. Compile/link duration and peak memory have not been measured for
this upgrade. Two jobs and the host build gate are mandatory.

**Current cache blocker:** the unchanged installed cache's read-only Ninja
plan contains 533 tasks, including bindings and ANGLE/header regeneration.
A reviewed generator-only recovery executed 37 canonical steps with zero
compilation/linking. All 9486 generated files retained their content except
`supplemental_dependency.tmp`; no headers changed. The subsequent full dry-run
still contained 533 tasks. Its dependency provenance/restat behavior must be
resolved before claiming the closure above.
Running two cached compiler recipes and relinking against potentially stale
generated headers is not approved evidence of a compatible runtime.

After review and a compatible cache, the coordinated upgrade command is:

```bash
LOCALBOORU_BUILD_LOCK_TIMEOUT=21600 LOCALBOORU_WEBKIT_JOBS=2 \
  scripts/build-patched-webkit.sh --mpv-isolation-only
```

This wrapper sets the existing dependency environment, acquires the host gate,
checks source hashes, and rejects unrelated Ninja tasks before changing source.
It checks the plan again after patching and then builds the bounded target.
It does not regenerate CMake or replace the full source tree. It currently
refuses the 533-task cache. Stop the DMC instance through its normal Quit action
before replacing its shared runtime, preserve the previous binary/runtime
pair for rollback, and reopen through the canonical launcher only after the
matching app and runtime are installed. Never stop other players or Manager.

## Focused checks and acceptance limits

```bash
NODE_ENV=test node --test frontend/src/utils/svpVideoHost*.test.js
python3 scripts/test-webkit-mpv-isolation.py \
  --source /mnt/storage/Programs/localbooru-webkit2gtk-4.1-patched/src/webkitgtk-2.52.3
source /home/user/.cargo/env
LOCALBOORU_BUILD_LOCK_TIMEOUT=21600 \
  CARGO_TARGET_DIR=/mnt/storage/Projects/extra_storage/localbooru/builds/dev-target \
  scripts/run-cargo.sh test --manifest-path src-tauri/Cargo.toml svp_ -- --nocapture
```

The C++ check copies only the two program source files to disposable temporary
storage, applies the exact upgrade, and executes the production registration,
marker parsing, relay forwarding and inode cleanup helpers. Socket paths in
that compiled fixture point into its temporary directory. It never accesses
the real default socket, Manager configuration or user media. These checks
prove synthetic ownership behavior, not real SVP discovery/interpolation.
Native acceptance remains pending a reviewed compatible runtime deployment.
