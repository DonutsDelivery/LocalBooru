# SVP zero-copy feasibility

**Decision:** No — the installed SVPflow/VapourSynth API does not expose an
exportable GPU output frame. LocalBooru must retain the planar YUV420 shared-memory
path and must report native SVP as `zero_cpu_copy=false` with
`copy_mode=svp_yuv420p_shared_memory_to_gpu`.

This conclusion is specific to the installed Linux SVPflow/VapourSynth API. NVOF
accelerates motion analysis/interpolation internally, but that does not make the
returned VapourSynth frame externally importable GPU memory.

## Reproducible probe

Run:

```sh
native-video/tools/probe_svp_gpu_output.py \
  --frames 60 \
  --json-output /tmp/localbooru-svp-gpu-output-probe.json
```

The probe deliberately does **not** call `VideoFrame.get_read_ptr()`. It loads the
real plugins, creates a real `SmoothFps_NVOF` graph, requests 60 output frames,
and inspects:

- the registered plugin signatures;
- output node format and cadence;
- output frame type, public methods, and frame properties;
- the installed `VapourSynth4.h` frame API;
- dynamic dependencies and exported SVPflow symbols.

A probe failure exits non-zero. A successful probe writes the complete evidence
as JSON.

## Evidence captured on 2026-07-13

Installed components:

- VapourSynth core R75, API R4.2;
- `/home/user/SVP 4/plugins/libsvpflow1.so`;
- `/home/user/SVP 4/plugins/libsvpflow2.so`.

Registered NVOF signature:

```text
SmoothFps_NVOF(
  clip:vnode;
  opt:data;
  vec_src:vnode:optional;
  src:vnode:optional;
  fps:float:optional
) -> any
```

The executable graph was:

```text
YUV420P8 640x512 24000/1001
→ quarter-resolution 160x128 vector source
→ svp2.SmoothFps_NVOF(rate=60/1, gpuid=0, algo=23)
→ YUV420P8 640x512 60/1
```

All 60 requested outputs were ordinary `VideoFrame` objects. Their only public
frame-access methods/fields were:

```text
close, closed, copy, format, get_read_ptr, get_stride, get_write_ptr,
height, props, readchunks, readonly, width
```

Observed frame properties were limited to:

```text
_DurationDen, _DurationNum, _PTS
```

There was no CUDA, Vulkan, OpenCL, DMA-BUF, EGLImage, device pointer, or external
handle attribute or frame property. The installed public `VapourSynth4.h` likewise
contained no GPU-frame export API matching those mechanisms.

The plugin binary does contain `GPURenderer` and `SmoothFps_NVOF` implementation
symbols. This proves an internal accelerated implementation exists. It does not
provide ownership, synchronization, format, modifier, or lifetime semantics for
an external GPU allocation. Neither plugin has a direct dynamic dependency that
constitutes a public GPU-frame export ABI; both expose the standard VapourSynth
plugin boundary.

## Why the existing transport cannot be zero-copy

The current native SVP path necessarily crosses CPU-visible planes:

1. FFmpeg writes planar YUV420 source frames to a bounded FIFO.
2. The generated VapourSynth source callback copies those bytes into writable
   VapourSynth planes using `get_write_ptr` and `ctypes.memmove`.
3. SVPflow performs NVOF-assisted interpolation internally.
4. `vspipe --y4m` serializes output planes as Y4M bytes.
5. `SvpFrameSource` reads those bytes into a CPU `DecodedVideoFrame`.
6. `ShmSurfaceProducer` copies the frame into the bounded shared-memory pool.
7. GTK uploads those planar surfaces to reusable GL textures.

Avoiding steps 4–7 would require an API returning an externally importable GPU
allocation plus synchronization and lifetime ownership. The installed API returns
only a normal `VideoFrame`; reading its pixels uses CPU plane access.

## Gate result

No versioned CUDA/Vulkan/DMA-BUF adapter will be implemented because there is no
exportable handle contract to adapt. The accepted design is therefore:

```text
FFmpeg → bounded FIFO → VapourSynth/SVPflow NVOF → Y4M
→ bounded planar YUV420 SHM → reusable GL textures → GTK draw completion
```

Required product behavior:

- never describe native SVP as end-to-end zero-copy;
- retain `zero_cpu_copy=false` diagnostics;
- keep queueing and SHM leases bounded;
- count presentation only after GL draw/fence completion;
- revisit this decision only if a documented SVPflow/VapourSynth API later
  exposes an external GPU frame with explicit ownership and synchronization.
