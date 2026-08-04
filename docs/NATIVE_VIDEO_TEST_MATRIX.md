# Native Video Test Matrix

## Baseline status (2026-07-13)

- Frontend production build: passing.
- Rust check: passing with five known dead-code warnings for deliberately retained retired optical-flow seams.
- Direct WebKit playback: zero-copy-capable but not guaranteed by diagnostics.
- Legacy SVP desktop delivery remains FFmpeg -> vspipe/SVPflow -> FFmpeg HLS -> Rust proxy -> WebKit decode for non-native targets.
- Native SVP delivery uses one persistent external FFmpeg -> bounded FIFO -> vspipe/SVPflow NVOF -> Y4M -> planar YUV420 SHM -> GTK GLArea pipeline. It creates no media segments, performs YUV conversion on the GPU, and truthfully reports `zero_cpu_copy=false` with `copy_mode=svp_yuv420p_shared_memory_to_gpu`.
- Pure Harmony native viewport evidence: 1920x1080, 63 produced FPS, 59 presented FPS, zero dropped frames on the reported machine.

### Evidence classification for the full-functionality plan

- **Accepted ordinary DMA-BUF baseline:** the managed GTK/EGL path has bounded 15-second, 10-minute, X11/XWayland, recovery, and lifecycle evidence below. Those runs establish the existing ordinary-path baseline; they do not establish the new fit/fill/original geometry or control-parity requirements.
- **Historical SVP regression (resolved):** `/tmp/localbooru-4k-svp-final-runtime.log` and `/tmp/localbooru-4k-svp-surface-debug.log` registered three one-FD/three-plane YUV420 SHM descriptors but remained at `presented_fps=0.0`. The current path keeps the mapped buffers alive without a per-frame CPU clone, reuses per-buffer GL textures, and drives the GLArea from the display frame clock.
- **Accepted current planar-SHM proof:** a controlled muted 30-second 3840×2160 23.976 -> 60 FPS run registered exactly three one-FD/three-plane buffers. It advanced from `accepted=30, draw_completed=28` to `accepted=1639, draw_completed=1638`, held host queue depth at 0–3, reported 0.615–36.237 ms sampled queue latency, and sustained approximately 60 draw completions per second after warm-up. Close discarded and released the one pending lease; no helper, FFmpeg, or vspipe process remained.
- **Accepted canonical-geometry proof:** the shared Rust model covers fit/fill/original, SAR, quarter-turn rotation, UV crop, physical rounding, subtitle/HUD regions, and title-bar-safe insets. Focused geometry, Linux-platform, and protocol tests pass. The same mode is sent from the React Lightbox to the native viewport, while browser playback retains equivalent `object-fit: contain/cover/none` semantics. A real 3840×2160 fit-mode run reached GTK/EGL draw completion without shader/import errors.
- **Accepted ordinary DMA-BUF proof:** the supplied 3840×2160 file reached direct DRM PRIME → DMA-BUF → EGL external-image presentation at approximately 24 source/presented FPS, submitted at least 120 frames, and reported `zero_cpu_copy=true copy_mode=dma_buf_external_oes fallback_reason=none`. A forced invalid DRM node instead reported `zero_cpu_copy=false copy_mode=shared_memory_rgba` with the VA-API error, then completed GTK first-frame presentation and bounded SHM lease recycling. Neither run left an app/helper/FFmpeg/vspipe process.
- **Closed SVP GPU-output feasibility gate:** `native-video/tools/probe_svp_gpu_output.py` loaded the installed plugins, built `SmoothFps_NVOF`, and requested 60 real outputs. They were CPU-addressable `VideoFrame`/`YUV420P8` objects with no CUDA/Vulkan/OpenCL/DMA-BUF/EGLImage handle in frame attributes, properties, or the public R4 header. `docs/SVP_ZERO_COPY_FEASIBILITY.md` records the executable no-go decision; SVP remains truthfully non-zero-copy.

### Bounded runtime template

Runtime validation is explicit, muted, time-bounded, and followed by an owned-process cleanup assertion:

```bash
timeout --signal=TERM --kill-after=5s 45s \
  npm run tauri:dev -- -- -- \
  --mute --svp "/path/to/non-private-fixture.mp4"

pgrep -af '/target/debug/[l]ocalbooru|[l]ocalbooru-native-video|[v]spipe|[f]fmpeg.*fixture|frontend/node_modules/.bin/[v]ite'
```

For deterministic close accounting without relying on frontend automation, the opt-in managed spike can select SVP and close the media before the outer process timeout:

```bash
LOCALBOORU_NATIVE_RUNTIME_SPIKE="/path/to/non-private-fixture.mp4" \
LOCALBOORU_NATIVE_RUNTIME_SPIKE_SVP=1 \
LOCALBOORU_NATIVE_VIEWPORT_SPIKE=1 \
LOCALBOORU_NATIVE_RUNTIME_SPIKE_SECONDS=30 \
timeout --signal=TERM --kill-after=5s 38s ./target/debug/localbooru
```

`npm run build`, `cargo build`, and `scripts/prepare-native-video.sh` are build-only commands. They must not launch LocalBooru or start a GUI playback session.

## Development media

Use one non-private H.264 1080p/60 fixture during implementation. Add HEVC only when a hardware-decode distinction matters. Record fixture hashes and codec metadata, never user-library paths. Reserve VP9, AV1, 24/30 FPS, variable-frame-rate, embedded-track, and 4K coverage for rollout.

## Measurements

During a major phase or hardware-boundary proof, record only the diagnostics relevant to that phase. Do not repeat a passing decode, import, or viewport proof when that path has not changed. Development runtime checks last 30–60 seconds.

### Native SVP adapter evidence

- External discovery: VapourSynth Core R75/API R4.2 and user-supplied `/home/user/SVP 4/plugins/libsvpflow{1,2}.so` detected. The path is runtime evidence only; release code continues to discover a configured user installation rather than bundling it.
- Focused source smoke: the input was recognized as exact `24000/1001`; 120 consecutive complete 3840×2160 YUV420 outputs matched 60 FPS timestamps and had zero equal adjacent sampled-frame hashes. Seek restart returned PTS 1.0. The generated VapourSynth graph now retains the exact source rational instead of approximating it as `23976/1000`.
- Managed helper smoke: `set_interpolation(svp)` -> media open -> playback state -> first frame completed without HLS or disk segments.
- Helper transition smoke: SVP stopped in 1.811 seconds, its FFmpeg/vspipe producers exited, and the helper selected `dma_buf_external_oes`. The full frontend/coordinator preserved-position interaction remains a bounded GUI acceptance item because injecting directly into the helper bypasses Rust's transport registry.
- Repeatable lifecycle probe: `native-video/tools/probe_svp_lifecycle.py` drives ordinary -> SVP -> paused seek -> speed/audio/subtitle changes -> ordinary -> close. Against the supplied 4K file, off -> SVP preserved PTS 0.25025 exactly, paused seek landed on its 1.25025-second target, and SVP -> off moved only 26.042 ms. The selected audio track reopened at the canonical position, subtitle-none and delay commands completed, and no FFmpeg/`vspipe` descendants survived disable or close. The helper exited 0.
- Direct 3840×2160 23.976 -> 60 FPS run: over the final 20 one-second windows, actual SHM lease releases averaged 59.975 FPS, matched production exactly, and reported no additional steady-state drops. Ten startup/catch-up frames were dropped before the steady state and remain visible in cumulative diagnostics.
- Current draw-completion/A/V proof after pacing repair: first-frame readiness now anchors the video wall clock instead of treating external SVP startup time as playback time, and audio remains paused until that first SVP frame. A bounded supplied-4K run advanced from 29 accepted/28 draw-completed frames to 750/750, sustained approximately 60 draw completions per second with queue depth 0–2, and held measured A/V drift between -14 and +35.333 ms after startup. Cumulative drops stayed bounded at five; close released the remaining work and left no helper, FFmpeg, or `vspipe` process.
- The surface-less SDL fallback now accepts planar YUV420 with `SDL_UpdateYUVTexture`; lifecycle probing can no longer crash on the first SVP frame or orphan its external workers. Pause commands emit immediate canonical playback state rather than waiting forever for a frame that a paused session will not produce.
- SVP SHM descriptors now carry source SAR, rotation, matrix, range, and chroma siting. The YUV shader selects BT.601/BT.709/BT.2020 and narrow/full conversion coefficients from that descriptor; a producer test freezes SAR 4:3, 90° rotation, BT.601 full range, and left chroma metadata.
- Process cleanup check: no helper, FFmpeg, or vspipe process remained after close.

### Task 13 sustained-throughput investigation (2026-07-14)

- Repeated bounded supplied-4K runs observed short windows at 58.8–60.6 actual GTK draw completions per second and later windows at 25–40 FPS while SVP continued producing approximately 60 FPS. The user subsequently confirmed that a full-settings game client was sharing the GPU throughout this testing. These cadence and drop observations are therefore confounded and do not establish LocalBooru's isolated steady-state throughput in either direction.
- Stage accounting stayed balanced in the completed probes (`accepted == draw_completed` where a final balanced snapshot was available), queues remained bounded, managed close completed, and no helper, FFmpeg, or `vspipe` child survived. Those lifecycle/accounting facts remain valid; no root-cause or throughput conclusion is drawn from the contaminated performance samples.
- A live profile during the active interval observed approximately 101–118% CPU in the Tauri/GTK process, 77–82% in the helper, 94–128% in the source FFmpeg process, and 178–206% in `vspipe`; the NVIDIA GPU reported 33–43% SM use and roughly 1.5–3.1 GB/s PCIe receive traffic. `vspipe` RSS was approximately 2.29–2.33 GiB, FFmpeg approximately 937 MiB, helper approximately 480 MiB, and the app approximately 337 MiB. Because an unrelated full-settings game client was concurrently using the GPU, these samples are contextual only and cannot attribute GPU saturation or establish LocalBooru's isolated performance.
- Two transport experiments were kept out of the default after bounded observations. A software-YUV → VA-API upload → DRM-PRIME/DMA-BUF path remains truthfully copy-backed and is available only behind `LOCALBOORU_SVP_DMABUF_UPLOAD=1`; a pixel-unpack-buffer experiment was reverted because it showed no benefit in those runs. Concurrent gaming invalidates the comparisons as controlled performance A/B evidence, so neither experiment is considered conclusively faster or slower. The supported/default SVP path remains the simpler planar YUV420 SHM → GTK textures pending an idle-machine comparison.
- The managed runtime harness now has an explicit `LOCALBOORU_NATIVE_RUNTIME_SPIKE_FULLSCREEN=1` control so fullscreen probes exercise the intended native state instead of relying on an ignored test variable.
- The first nominal 600-second SVP run was not accepted as a soak. It exercised the development VA-API/DMA-BUF uploader because the runtime `set_interpolation` transition bypassed the new environment gate, then reached EOF after 220 seconds and failed to restart. During the active interval it ended at 2,098 accepted/2,098 draw-completed frames with 9,956 bounded drops; the remaining diagnostics were idle. An unrelated full-settings game client also shared the GPU, so the cadence/drop measurements are confounded. Managed close completed without fatal/panic output or surviving owned processes.
- Both harness defects were fixed at their source. `VideoPlaybackSession` now waits at EOF for a seek instead of terminating its worker, and the `set_interpolation` transition now respects `LOCALBOORU_SVP_DMABUF_UPLOAD` just like initial open. The playback-session regression test seeks after EOF and receives a new first frame. A 22-second deterministic SVP-SHM loop then crossed EOF four times, reported only `svp_yuv420p_shared_memory_to_gpu`, ended at 1,026 accepted/1,026 draw-completed frames with two drops, and managed close left no helper, FFmpeg, or `vspipe` child.
- The corresponding audio worker had the same EOF lifetime defect. A failing regression proved that a post-EOF paused seek left its submitted clock at the old end position; `AudioPlaybackSession` now waits for seek/stop at EOF. All 19 helper tests pass, and a 1080p A/V SVP loop crossed EOF without the previous approximately +200-second clock jump (`av_drift_ms` stayed between -537.333 and +4.0 ms during the bounded run). The remaining negative end-of-loop drift is visible and is not treated as sync acceptance.
- The corrected 600-second loop harness kept video work alive across two EOF transitions and cleaned up all owned children. It reported 7,421 accepted, 7,420 draw-completed at the final diagnostics snapshot, 26,089 drops, 0.0–56.9 presented FPS, and -546.667 to +200,253.333 ms drift. That run predates the audio-worker EOF fix and ran while the GPU was also being used by an unrelated full-settings game client. It is valid lifecycle/failure-discovery evidence only; its cadence, drop, resource, and drift ranges are not controlled performance evidence.
- The ordinary runtime matrix exercised 1080p60, portrait, anamorphic SAR, rotation metadata, BT.601 limited, BT.709 full, and multitrack fixtures. Every run produced a real first GTK frame through `dma_buf_external_oes`, retained `zero_cpu_copy=true`, had `accepted == draw_completed`, reported zero drops, completed managed close, and left no owned process. Fractional scale factors 1.0/1.25/1.5/2.0 are frozen by the canonical geometry tests; physical monitor transfer and visual placement remain human-only boundaries.
- An optimized Rust application plus Release C++ helper also ran the supplied file for 95 seconds on the default SVP-SHM path. It observed 59.6 produced FPS, 20.9 presented FPS in the last diagnostic, a 27.148-FPS average over its last 60 active windows, 3,246/3,245 accepted/draw-completed, and 2,298 drops. The user has since confirmed that a full-settings World of Warcraft client was sharing the GPU during these profiling runs. The release observation therefore does **not** isolate LocalBooru, cannot establish a host-presentation bottleneck, and cannot be used to accept or reject 4K/60 performance. Managed-close and no-surviving-worker facts remain valid.
- **Rollout decision:** native SVP performance is unmeasured under controlled idle-machine conditions. Keep the persisted default `react` until a clean supplied-4K 24→60 run records steady-state cadence, frame accounting, GPU load, A/V drift, and cleanup without unrelated GPU workloads. Users may explicitly choose native modes, and one settings change provides rollback without media/database migration.

### Native helper lifecycle and timing evidence (2026-07-13)

- Native diagnostics report measured audio/video drift from the SDL queued-audio clock rather than a placeholder. A two-audio-track headless run reported finite drift and switched streams at the current media position.
- Seek latency is measured from command receipt to the first decoded/published frame within 250 ms of the requested media position. The two-audio-track fixture reported 65.87 ms; an older queued pre-seek frame is explicitly excluded.
- First-frame latency is measured in Rust from the generation-scoped native open to the successful GTK presentation callback, not helper decode alone. Stale-generation callbacks cannot complete the measurement.
- The managed DMA-BUF consumer now submits a GL completion fence after every external-image draw, polls it asynchronously with a current GTK GL context, and releases the helper lease only after it signals. A bounded 15-second runtime proof submitted more than 720 frames at approximately 60 FPS without the former per-frame `glFinish` or framebuffer readback and left no helper process.
- Color metadata transport was verified with deterministic files: 640×480 BT.601/narrow/left and 1280×720 BT.709/full/center reached the surface descriptor unchanged, mapped to the expected EGL hints, rendered a managed first frame, and left no helper process. Pixel-level visual/reference parity remains a human acceptance item.
- A debug Debian bundle completed successfully. Extraction found `/usr/bin/localbooru-native-video`; the extracted sidecar negotiated protocol `1000` and exited cleanly. Bundle SHA-256: `f6ae49d1212b8118a3e27830747b8494c766476c593f81748ddf665f1224db69`; packaged helper SHA-256: `53c71d66f603da11a60a848b9422c8be2bd4942250b23e8b9edfdce8f7a6d86e`. AppImage bundling reached the linuxdeploy stage but failed in the current Arch host toolchain; this is packaging-environment evidence, not a helper build/discovery failure.
- Share-session HLS manifests now keep segment access under `/{token}/hls/media`; that route validates the live share token and then reuses the existing Range-aware media response. Focused tests freeze known/unknown duration manifests and prevent regression to an unscoped `/api/images/...` segment URL.
- A deterministic 600-second 1920×1080/60 H.264/AAC fixture completed 604 seconds of managed GTK/EGL DMA-BUF playback: final submitted frame 34,680, approximately 58 produced/presented FPS, zero reported drops, stable helper state after initialization (92 FDs, 264,620 KiB RSS, 11 threads at both sampled checkpoints), no fatal/panic/EGL/GL/fence errors, and no helper remaining after shutdown. The runtime-spike harness ceiling was first raised from 60 to 900 seconds so this rollout gate exercised the renderer rather than silently closing at one minute.
- The latest 2026-07-14 final-gate build reran `npm run build`, 11 focused frontend tests, five Rust direct-file tests, 56 Rust native-video tests, all 19 C++ helper tests, and focused `git diff --check`; all passed. Both optimized Rust and Release C++ artifacts also built successfully. Vite retained pre-existing dynamic/static import and large-chunk warnings, Rust retained five known dead-code warnings for retired optical-flow seams, and GCC emitted an existing nlohmann-json false-positive array-bounds warning during the Release helper build.
- `scripts/prepare-native-video.sh` produced the target-suffixed Release helper without launching an app/helper/vspipe/FFmpeg process. The current build-package and staged `x86_64-unknown-linux-gnu` binaries match at SHA-256 `3ade135f076012b81b317ad2aefb51ffd8e96b6c2611f72b7cc59e441ab6d42b`. The refreshed development provenance manifest records 125 dynamic dependencies and FFmpeg configuration SHA-256 `bdbbfab8bac5a2a07f4a6b223337c46b13cbf6b53c503618467ae2fdf6eb9063`. As designed, `distribution_approved=false`: this Arch development dependency closure and unpinned notices/license map are not release provenance.
- The runtime-spike ceiling is now 43,200 seconds and its opt-in loop mode seeks and resumes at EOF. A two-second fixture looped three times during an eight-second bounded proof with zero fatal/GL/EGL/fence errors and no helper remaining; this keeps a future ten-hour release-candidate soak on a continuous decoded-frame workload without manufacturing a multi-gigabyte fixture.
- EGLImage and external-texture ownership now remains attached to each asynchronous completion fence; destruction occurs only after the fence signals with the GTK GL context current. This closes the previous implicit lifetime gap between draw submission and exact helper lease release.
- The same managed DMA-BUF path completed a bounded X11/XWayland run at approximately 60 FPS with zero reported drops/errors and no helper remaining. Wayland and X11 automated runtime paths therefore both have local evidence; placement/focus remain human checks.
- Safe-mode rollback with native enablement and the runtime spike requested created no helper process. Forced helper `SIGKILL` recovery now has real hardware evidence: the replacement helper was launched from its long-lived runtime-worker parent, reopened generation 1 at position 11.383 seconds with autoplay preserved, and resumed managed DMA-BUF presentation at approximately 60 FPS. Killing that replacement produced no third helper; the coordinator honored the one-attempt ceiling and selected terminal Web/HLS fallback. The root cause was Linux `PR_SET_PDEATHSIG` being tied to the spawning thread: the old implementation spawned a replacement from the short-lived notification thread, so the kernel sent `SIGTERM` as soon as that thread returned.
- Five-minute forced-copy lifecycle run: 5,282 open/seek/close cycles, helper FD count fixed at 6 (`min=max=end=6`), RSS 61,652–116,692 KiB with 115,040 KiB at the end, and no fatal/recoverable helper event. This is bounded development evidence, not the release-candidate long soak.
- Canonical renderer handoff state now travels in every generation-scoped native snapshot: item/generation, position/duration, paused, volume/mute, speed, selected audio/subtitle IDs, subtitle delay, SVP configuration, and display mode. Six coordinator tests include full-state preservation plus stale-generation rejection; the helper integration test verifies authoritative control-state events, and eleven focused frontend policy/lifecycle tests verify bounded Web capture and native-to-Web restoration. Web fallback keeps the native surface visible until the replacement `<video>` has decoded `loadeddata`, then releases it through a generation-checked command; native preparation keeps a direct-file Web frame available until GTK reports its matching first frame. Target-specific browser track/SVP state is logged as an explicit fallback instead of being silently claimed as preserved.

## Development gates

1. 1080p/60 for 30–60 seconds with a recorded bounded drop rate on the preferred path.
2. Repeated random seeks without stale frames or A/V drift.
3. Image -> video waits for matching first frame.
4. Video -> image hides native viewport synchronously.
5. Rapid navigation ignores stale generations.
6. Resize/fullscreen/minimize/tray restore without leaked handles.
7. Helper crash recovers once or preserves position into Web fallback.
8. APK/browser/cast HLS behavior remains unchanged.
9. Whisper cue timing survives pause, seek, speed, and subtitle delay.

## Rollout and default-on gates

1. Broad codec, frame-rate, VFR, embedded-track, and 4K matrix.
2. Ten-minute Linux playback and resize/fullscreen lifecycle run.
3. Release-candidate/default-on soak with stable process, file descriptor, and GPU memory counts.
