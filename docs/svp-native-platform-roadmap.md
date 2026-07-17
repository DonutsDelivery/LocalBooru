# SVP native playback: Linux release scope and cross-platform roadmap

Status: engineering research for the build after the current Linux release.

## Decision for the next build

The next build exposes the Manager-controlled native playback path on **Linux only**. Windows and macOS must not advertise or enter that path. The current frontend gates it with `isLinuxDesktopApp()`, and the Rust Manager bridge is compiled only under `target_os = "linux"`.

This is a release-scope decision, not a claim that VapourSynth or SVPflow are unavailable elsewhere. Both are available on Windows and macOS. LocalBooru's existing sidecar already recognizes:

| Platform | SVPflow modules |
| --- | --- |
| Linux | `libsvpflow1.so`, `libsvpflow2.so` |
| Windows | `svpflow1_vs.dll`, `svpflow2_vs.dll` |
| macOS | `libsvpflow1_vs64.dylib`, `libsvpflow2_vs64.dylib` |

The missing piece is a verified attachment between a cadence-changing VapourSynth graph and the media backend owned by each embedded webview.

## Ownership boundary

LocalBooru is not an interpolation engine. The user's installed SVP Manager owns the graph, cadence, algorithms, presets, and OSD. LocalBooru may expose a Manager-compatible control endpoint, snapshot the exact graph bytes immutably, execute that snapshot, and transport its output to the existing player. It must not generate SVPflow parameters, select an interpolation preset, duplicate frames as a product fallback, or present synthetic interpolation as SVP support.

Fragmented MP4/MSE is only the Windows/macOS attachment transport. A transport fixture such as an identity VapourSynth graph may test muxing, buffering, and seeking, but it is not a product graph or an SVP acceptance signal. Desktop SVP playback requires an active Manager-generated graph; otherwise LocalBooru fails open to one direct source.

## OSD acceptance signal

The red border / red square is **not** the configured SVP Control Panel OSD and is not proof that Manager control or frame interpolation is working. Do not infer graph success from that mark.

The source of truth is the real OSD explicitly enabled in SVP Control Panel settings, followed by visible or measured cadence change. LocalBooru must not draw an imitation watermark.

The installed `libsvpflow2.so` does contain `RemoteControl::hasOSD`, `RemoteControl::getOSD`, and `SmoothFpsCore::drawOSD`, and a traced SVPflow graph attempts to connect to the Manager remote-control socket at `/tmp/com.svpteam.svp`. That mechanism still has to be proven end to end. A configured OSD that does not appear means the control path is not accepted even if a script was attached or the video briefly stopped and resumed.

Before attributing missing OSD or missing visual interpolation to LocalBooru, establish a working system baseline in mpv or VLC. Duplicate Manager processes, stale sockets, temporary-directory mismatches, or broader SVP configuration failures can affect every player. Tests must use one Manager process, one active player-control endpoint, a reachable Manager remote-control socket, and a baseline player that visibly interpolates.

## Linux transition ownership and failure matrix

Exactly one HTML `<video>` pipeline may own audio/video output. Changing SVP state is a transaction:

1. capture time and whether playback was active;
2. pause the current element;
3. remove its `src` and call `load()` to force `emptied` and tear down the old playbin;
4. wait for rapid Manager changes to settle;
5. create one replacement playbin using the final filter state;
6. restore time only after `loadedmetadata`;
7. resume only the replacement pipeline.

Backend script/filter changes are serialized, runtime files are atomically replaced, and only one peer process may own the Manager-compatible socket at a time.

Required failure cases for every native implementation:

| Case | Required result |
| --- | --- |
| duplicate `vf add` with identical script content | no remount |
| duplicate `vf remove` | no remount |
| changed script at the same pathname | one replacement graph |
| add/remove/add burst | old source emptied immediately; only final graph reaches play |
| Manager pause/add/resume transaction | resume is deferred until replacement metadata exists |
| close during transition | source removed, playback disabled, no stale audio |
| seek during graph reset | one final position; no old graph resumes |
| graph creation/evaluation failure | fail open to one unfiltered pipeline, never retain both |
| competing Manager process | rejected while the active controller remains connected |
| EOS during transition | do not restart an ended source as a second audible pipeline |

Runtime evidence must include media lifecycle events (`pause`, `emptied`, `loadedmetadata`, `play`) and WebKit playbin identifiers. A transition fails if an old identifier reaches `play` after its successor becomes active.

## Windows research

### Control plane

Official SVP mpv integration uses `input-ipc-server=mpvpipe` on Windows. A future LocalBooru Manager bridge therefore needs a Windows named-pipe server plus peer/process ownership equivalent to the Linux Unix-socket guard.

### Attachment options

#### A. On-device VapourSynth sidecar -> fragmented MP4 -> MSE (recommended first prototype)

Run the Manager-generated VapourSynth/SVPflow graph locally, encode its output into short fragmented MP4 segments, and append those segments through `MediaSource`/`SourceBuffer` to the existing HTML video element.

Advantages:

- stays inside LocalBooru's existing video element and controls;
- does not embed mpv;
- works with WebView2's supported web-media APIs;
- uses native/on-device VapourSynth and SVPflow;
- can share most processing code with macOS.

Costs:

- decode/process/re-encode rather than a direct decoded-frame filter;
- buffering and A/V timestamp management remain application responsibilities;
- seeking requires a transactional segment reset;
- not a zero-copy path.

This is conceptually related to the existing stream path but should use a bounded, seekable fMP4/MSE protocol rather than preserving the old HLS implementation unchanged.

#### B. WebCodecs frame pipeline

WebCodecs exposes `VideoDecoder`, `VideoFrame`, and per-frame processing, but supplies neither demuxing nor a complete player. Bridging raw frames to native VapourSynth would require custom demux, audio scheduling, seeking, rendering, and synchronization. It would effectively replace HTML video playback ownership, so it is a research path rather than the default.

#### C. Native Media Foundation/DirectComposition renderer

A native renderer could host decoding, VapourSynth, and presentation directly. WebView2 composition APIs can place visuals in a host composition tree, but this creates a second player/renderer and must recreate controls, subtitles, fullscreen, color handling, and synchronization. Do not select it without explicitly revising the single-player ownership requirement.

#### D. Globally registered Media Foundation Transform

Do not rely on registering a system-wide MFT and hoping WebView2 selects it. WebView2 does not expose a supported API for inserting an application-selected transform into its internal media graph. Global registration is invasive and selection is not deterministic.

## macOS research

### Control plane

SVP's documented mpv route on macOS uses `/tmp/mpvsocket`. The Unix control transport can be adapted, but the transport does not provide frame attachment by itself.

### Attachment options

#### A. On-device VapourSynth sidecar -> fragmented MP4 -> MSE (recommended first prototype)

Use the same bounded fMP4/MSE protocol proposed for Windows. WKWebView keeps the existing HTML media element while a native sidecar runs VapourSynth/SVPflow locally. Verify actual codec/MSE behavior on the minimum supported macOS version, including Apple Silicon.

#### B. AVPlayerItem + AVVideoComposition custom compositor

AVFoundation officially supports `AVPlayerItem.videoComposition`, and `AVVideoComposition.customVideoCompositorClass` can select an `AVVideoCompositing` implementation. This is a plausible native VapourSynth attachment only when LocalBooru owns the `AVPlayerItem`.

WKWebView does not expose its internal `AVPlayerItem` for host modification. Using this approach therefore requires a LocalBooru-owned AVPlayer surface and changes playback ownership. It belongs in a separate prototype, not the next build.

#### C. WebCodecs/custom canvas renderer

This has the same demux, audio, seeking, and ownership costs as Windows and should not be the first prototype.

## Implementation order

1. Define one platform-neutral bounded processing-session protocol: open, metadata, start position, seek generation, pause, resume, stop, and terminal error.
2. Remove LocalBooru-generated interpolation from the desktop product path. The browser must not select a graph mode or graph pathname.
3. Extract the existing Linux Manager command, ownership, and immutable-snapshot behavior into a shared core.
4. Add Manager-compatible control transports: Windows `mpvpipe` named pipe and macOS `/tmp/mpvsocket`, while preserving the Linux Unix socket.
5. Execute only the immutable graph snapshot supplied by the active Manager through the bounded fMP4 processor.
6. Attach that output through MSE to the existing HTML video element on Windows and macOS, with transactional generation changes and direct-playback fail-open.
7. Compile and package Windows through the canonical local Docker/MSVC-Wine build.
8. Run native macOS socket tests and universal app/DMG builds in GitHub Actions.
9. Keep real Windows and Apple Silicon playback acceptance open until artifacts can be tested on suitable machines with user-installed SVP.
10. Only reconsider native replacement renderers if MSE cannot meet product requirements.

## Verification boundaries

LocalBooru currently has no owned Windows or macOS playback machines. Verification must distinguish what the available build systems can prove:

- `scripts/build-windows-local.sh` and `scripts/build-windows-docker.sh` can prove Windows compilation, linking, packaging, PE architecture, archive integrity, and absence of bundled proprietary SVP files. Wine/MSVC cross-build success is not WebView2 or real SVP Manager playback acceptance.
- `.github/workflows/build.yml` and `scripts/build-macos-ci.sh` can run native fake-Manager Unix-socket tests, compile universal binaries, inspect arm64/x86_64 slices, and validate app/DMG structure. Unless the runner is Apple Silicon with a real installed SVP setup, this is not Apple Silicon SVP playback acceptance.
- Real Manager discovery, proprietary plugin ABI loading, genuine OSD, presented cadence, dropped frames, sustained MSE behavior, GPU paths, seek quality, and A/V synchronization remain manual native acceptance items.

Build results must be labeled **compile/protocol/package verified**, never **SVP playback verified**, until those native acceptance items are run.

## Sources

- SVP Linux setup and copy-back requirement: <https://www.svp-team.com/wiki/SVP:Linux>
- SVP mpv IPC by platform: <https://www.svp-team.com/wiki/SVP:mpv>
- SVP VLC's player-specific filter/control integration: <https://www.svp-team.com/wiki/SVP:VLC>
- SVP 4 Mac player and installation support: <https://www.svp-team.com/docs/mac/>
- Microsoft WebView2 API overview: <https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/overview-features-apis>
- Microsoft WebView2 network-request interception: <https://learn.microsoft.com/en-us/microsoft-edge/webview2/how-to/webresourcerequested>
- Apple `AVVideoComposition.customVideoCompositorClass`: <https://developer.apple.com/documentation/avfoundation/avvideocomposition/customvideocompositorclass>
- Apple `AVPlayerItem.videoComposition`: <https://developer.apple.com/documentation/avfoundation/avplayeritem/videocomposition>
- Media Source Extensions: <https://developer.mozilla.org/en-US/docs/Web/API/Media_Source_Extensions_API>
- WebCodecs: <https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API>
