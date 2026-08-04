# Spike 001: WebKitGTK `playbin` video-filter insertion

## Question

Given LocalBooru's existing React `<video>` player, can a GStreamer video filter be inserted below it so the original WebKit player and controls remain in use?

## Evidence

- Local WebKitGTK is `2.52.3`; GStreamer is `1.28.3`.
- WebKit's `MediaPlayerPrivateGStreamer::createGSTPlayBin()` creates a `playbin`/`playbin3`, sets its sinks, and already assigns an `audio-filter` for pitch preservation.
- Both installed `playbin` and `playbin3` expose a writable `video-filter` property of type `GstElement`.
- No `vapoursynth` GStreamer element is installed, so an SVP implementation needs a custom plugin.
- `lbprobe.c` is a disposable in-place `GstVideoFilter`. It proves plugin discovery, raw-video negotiation, and frame delivery.

## Executed checks

Build and inspect the disposable plugin:

```sh
mkdir -p build
cc -std=c11 -Wall -Wextra -Werror -fPIC -shared lbprobe.c \
  -o build/libgstlbprobe.so \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0)
GST_PLUGIN_PATH="$PWD/build" gst-inspect-1.0 lbprobe
```

Direct filter-chain result:

```text
LB_FILTER_NEGOTIATED 640x360 A444_16LE -> A444_16LE
LB_FILTER_FRAME count=1 pts=0:00:00.000000000
LB_FILTER_FRAME count=30 pts=0:00:00.966666666
LB_FILTER_FRAME count=60 pts=0:00:01.966666666
```

Real `playbin` result using `/tmp/localbooru-native-plan-fixture.mp4`:

```sh
GST_PLUGIN_PATH="$PWD/build" gst-launch-1.0 -q playbin \
  uri=file:///tmp/localbooru-native-plan-fixture.mp4 \
  video-filter=lbprobe video-sink=fakesink audio-sink=fakesink
```

```text
LB_FILTER_NEGOTIATED 1920x1080 NV12 -> NV12
LB_FILTER_FRAME count=1 pts=0:00:00.000000000
LB_FILTER_FRAME count=30 pts=0:00:01.209541666
LB_FILTER_FRAME count=60 pts=0:00:02.460791666
LB_FILTER_FRAME count=90 pts=0:00:03.712041666
```

`webkit-video-filter.patch` shows the minimal downstream WebKit insertion point. It is illustrative and was not applied to the system WebKit package.

## Verdict: PARTIAL

### Validated

- GStreamer `playbin` can run a custom video filter while retaining the player that owns the pipeline.
- WebKitGTK uses that same `playbin` property and has a narrow source-level insertion point.
- A downstream WebKit patch can therefore keep LocalBooru's original React/WebKit player and route decoded frames through a custom filter without HLS.

### Still required

- Implement the actual VapourSynth bridge as a `GstVideoFilter`/custom element. Unlike the in-place probe, interpolation changes frame count and timestamps, so it may need a buffered `GstBaseTransform` or aggregator-style design rather than `transform_frame_ip`.
- Integrate SVP Manager's control plane. The installed VLC integration separates `libvapoursynth_plugin.so` from `libsvpcontrol_plugin.so` and exposes script/rate/delay/error functions, but those binaries use VLC's ABI and cannot simply be loaded into GStreamer.
- Build and run a patched WebKitGTK package to prove the HTML `<video>` path specifically.
- Verify tracks, seeking, pause, speed, subtitles, A/V delay, EOS looping, and fallback.
- Measure clean-machine performance. Official SVP MPV guidance requires copy-back decoding for VapourSynth, so SVP-on should not be described as end-to-end zero-copy.

### Recommendation

Pause replacement-player work. Build the smallest WebKitGTK package patch plus a buffered VapourSynth GStreamer filter, then prove the original LocalBooru player reaches Manager-confirmed active playback and responds to the SVP Control Panel. Do not promote this route until that control-plane proof exists.
