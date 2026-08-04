# Native Video Architecture

## Purpose

Desktop-local video playback uses a native, crash-isolated renderer with renderer-owned controls. Images remain in the React Lightbox. Android, browser/LAN clients, Chromecast, and DLNA retain HLS.

## Ownership

- Rust `NativePlaybackCoordinator` is the canonical media/presentation state owner.
- The native helper owns decode, A/V timing, optional interpolation, subtitles, HUD composition, and frame production.
- The platform viewport host imports and presents helper surfaces inside the existing Tauri window.
- React owns navigation/library/image UI and chooses presentation through capability policy.

## State transitions

`PreparingVideo` is not visible until a matching generation emits `FirstFrameReady`. Image navigation increments generation and hides the native viewport before exposing the React image. Late helper events are ignored. Native failure preserves position and selects the Web/HLS fallback.

## Data paths

### Desktop local

`FFmpeg decode -> optional native filter -> GPU compositor + HUD -> shared GPU surface -> native child viewport`

### APK/browser/cast

Existing direct/HLS/cast paths remain unchanged.

## Copy modes

- Preferred Linux mode: VA-API DRM PRIME → DMA-BUF → EGLImage/external texture (`zero_cpu_copy=true`, `copy_mode=dma_buf_external_oes`). The accepted path rejects non-`AV_PIX_FMT_DRM_PRIME` frames and never invokes `av_hwframe_transfer_data`, maps a pixel plane, or copies pixel bytes. `av_hwframe_map(..., AV_HWFRAME_MAP_READ)` exports the DRM PRIME descriptor rather than downloading the frame. Before export, the helper calls `vaSyncSurface`, so decode completion is explicit but currently CPU-blocking rather than represented by a native fence FD. The GTK consumer imports the received plane/object FDs directly with `eglCreateImageKHR` and `glEGLImageTargetTexture2DOES`, then inserts a nonblocking GL completion fence after each external-image draw. It retains the EGLImage, external texture, and lease until that fence signals; production does not use `glFinish`, `glReadPixels`, or a CPU framebuffer copy. The protocol carries an optional producer native-fence FD for a future nonblocking decode-export path.
- DMA-BUF descriptors carry decoded color space (`BT.601`, `BT.709`, or `BT.2020`), narrow/full range, and supported chroma siting. The EGL importer maps these to external-image YUV hints; legacy descriptors fall back to resolution-based BT.601/BT.709 selection, narrow range, and centered chroma rather than assuming full-range BT.709.
- Fallback: bounded CPU shared-memory ring with an explicit reason. Startup remains `zero_cpu_copy=false`/`dma_buf_pending_validation` until a real DRM PRIME frame is exported. Hardware/export failure reports `zero_cpu_copy=false`, `copy_mode=shared_memory_rgba`, and the concrete fallback reason; GTK maps the popup transparently for first-frame preroll so an initially hidden GLArea cannot deadlock lease release.
- Native SVP: NVOF acceleration does not expose exportable GPU output through the installed API. Its accepted/default path is planar YUV420 shared memory uploaded into reusable GL textures and is always labelled `zero_cpu_copy=false`, `copy_mode=svp_yuv420p_shared_memory_to_gpu`. A development-only `LOCALBOORU_SVP_DMABUF_UPLOAD=1` experiment copies those software frames into three lease-aligned VA-API surfaces and reuses their DMA-BUF/EGL objects; it remains `zero_cpu_copy=false`, reports `copy_mode=svp_cpu_to_vaapi_dmabuf`, and is not a rollout path because sustained supplied-4K profiling did not improve cadence. Reusable imports are keyed by EGL display/context, helper generation, buffer ID, and DMA-BUF object identity; the cache is cleared under a current context after the last in-flight fence when the viewport is hidden. See `SVP_ZERO_COPY_FEASIBILITY.md` and its executable probe.
- Do not call a GPU blit strict zero-copy; diagnostics distinguish zero CPU readback from GPU copies.

## Security

Rust authorizes/canonicalizes media paths before helper open. The helper has no network listener and accepts only authenticated parent-local IPC with bounded messages.

The control channel is an inherited stdin pipe and the surface channel is an inherited Unix `SOCK_SEQPACKET`; neither creates a filesystem socket or listening endpoint. Packets are capped at 64 KiB and five ancillary FDs. Surface descriptors validate dimensions, plane/object mappings, offsets, strides, and backing-file size before import. Generated subtitle registration is generation-scoped and restricted to a regular sidecar in the canonical active-media directory.

### Sandbox and resource-limit rollout

The Linux launcher now enforces the baseline process boundary before `exec`:

- `PR_SET_PDEATHSIG=SIGTERM`, with a parent-PID race check, so a helper cannot outlive a crashed/killed LocalBooru parent;
- the helper is spawned by the long-lived native runtime worker, not by a transient notification thread. Linux delivers `PR_SET_PDEATHSIG` when the specific parent thread exits, so this ownership is required for a recovered helper to survive after the old runtime's notification pump returns;
- `PR_SET_NO_NEW_PRIVS=1`;
- `RLIMIT_NOFILE=256` and `RLIMIT_CORE=0`;
- inherited-FD-only control and surface IPC.

A runtime hardware-decode proof confirmed `NoNewPrivs: 1`, `nofile=256/256`, `core=0/0`, managed DMA-BUF presentation at approximately 60 FPS, and no helper after normal parent exit. A separate `SIGKILL` parent-death test confirmed the helper disappeared within the bounded poll window.

Before native playback becomes default-on, validate a syscall filter against the supported hardware matrix rather than applying an untested generic allowlist:

- Linux follow-up: a syscall allowlist covering FFmpeg/VA-API/EGL/GTK, external SVP, system fonts, and audio devices. Address-space/process limits also remain deferred until values are measured against supported 4K and driver allocations; an arbitrary `RLIMIT_AS` can break GPU drivers.
- Windows: restricted token plus Job Object process/memory limits.
- macOS: hardened runtime and a narrow sandbox entitlement set for inherited IPC and explicitly authorized media files.

Sandbox failures must be reported as a one-shot native recovery failure and then use Web/HLS fallback; they must never trigger an unbounded restart loop. The syscall-filter profile remains a release gate because it must be validated across hardware decode, external SVP, system fonts, and audio devices.
