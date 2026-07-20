# GStreamer Video Player on Wayland - Technical Findings

This document captures the technical findings from the GStreamer video player prototype for Tauri on Wayland Linux.

## Summary

**GStreamer works on Wayland with hardware acceleration.**  The challenge is integrating it with Tauri's WebView for overlay functionality.

## Environment Tested

- Session: Wayland
- GStreamer: 1.26.10
- Hardware: NVIDIA GPU with NVDEC support
- Tauri: 2.x with webkit2gtk

## Available Components

### Video Sinks (all available)
- `waylandsink` - Direct Wayland surface rendering (best performance)
- `gtksink` - GTK3 widget-based rendering
- `gtkglsink` - GTK3 with OpenGL
- `glimagesink` - OpenGL image sink
- `autovideosink` - Automatic selection

### Hardware Decoders
- `nvh264dec` - NVIDIA H.264 (available, working)
- `nvh265dec` - NVIDIA H.265 (available, working)
- `vaapih264dec` - VA-API H.264 (not available on this system)
- `avdec_h264` - Software fallback (available)

## Architecture Options for Tauri + GStreamer

### Option 1: Separate Video Window (Implemented)

**How it works:**
- GStreamer creates its own Wayland surface via `waylandsink`
- Video plays in a separate window
- WebView remains in Tauri window

**Pros:**
- Simple to implement
- Full hardware acceleration
- Works reliably on Wayland

**Cons:**
- Video window is separate (cannot overlay UI)
- Window management complexity
- Not suitable for lightbox-style overlay

### Option 2: GTK Widget Embedding (Complex)

**How it works:**
- Use `gtksink` which creates a GtkWidget
- Attempt to embed this widget in Tauri's GTK hierarchy
- Use GTK Overlay container for layering

**Challenges:**
- Tauri doesn't expose GTK widget tree easily
- Version conflicts (Tauri uses glib 0.18, gstreamer uses 0.20)
- Would require patching Tauri or using unsafe FFI

**Status:** Not implemented due to version conflicts

### Option 3: Wayland Subsurface Protocol (Advanced)

**How it works:**
- Get wl_surface handle from Tauri window
- Create wl_subsurface for video
- Position subsurface behind/in front of WebView

**Challenges:**
- Requires direct Wayland protocol access
- WebView transparency support varies
- Complex coordinate synchronization

**Status:** Would require significant additional work

### Option 4: Frame Extraction (appsink)

**How it works:**
- Use `appsink` to extract decoded frames
- Send frames to WebView via base64 or canvas
- Render in JavaScript

**Pros:**
- Full control over rendering
- Easy overlay with CSS
- Works with WebView transparency

**Cons:**
- CPU overhead for frame transfer
- Latency added
- Higher memory usage

**Status:** Not implemented but viable for overlays

## Code Structure

```
src-tauri/src/video/
├── mod.rs          # Module exports and documentation
├── player.rs       # GstVideoPlayer - high-level player API
├── overlay.rs      # VideoOverlay - pipeline management
└── commands.rs     # Tauri commands for frontend
```

## Usage from Frontend

```javascript
// Initialize the video player
await invoke('video_init');

// Get system info
const info = await invoke('video_get_system_info');
console.log(info);

// Play a video (opens separate window on Wayland)
await invoke('video_play', { uri: '/path/to/video.mp4' });

// Control playback
await invoke('video_pause');
await invoke('video_resume');
await invoke('video_stop');

// Seek (in seconds)
await invoke('video_seek', { positionSecs: 30.5 });

// Volume (0.0 to 1.0)
await invoke('video_set_volume', { volume: 0.8 });

// Get position/duration
const pos = await invoke('video_get_position');
const dur = await invoke('video_get_duration');

// Cleanup
await invoke('video_cleanup');
```

## Test Binaries

```bash
# Test GStreamer setup
cargo run --bin test_gstreamer

# Test video playback
cargo run --bin test_video_playback -- /path/to/video.mp4
```

## Recommended Approach for LocalBooru

For a media application like LocalBooru, I recommend:

1. **For full-screen/dedicated video view:**
   - Use the current implementation with `waylandsink`
   - Video plays in its own window
   - Full hardware acceleration

2. **For lightbox overlay with UI controls:**
   - Use `<video>` tag in WebView with HTTP streaming
   - Backend transcodes/serves video via HTTP
   - UI controls render in WebView
   - Works cross-platform

3. **Hybrid approach:**
   - Quick preview: WebView video tag
   - Full playback: Native GStreamer window
   - Let user toggle between modes

## Known Limitations

1. **Window overlay not possible on Wayland** - Unlike X11, Wayland doesn't allow window reparenting or XEmbed-style embedding. Each surface is independent.

2. **GTK version conflicts** - Tauri's gtk crate (0.18) uses different glib version than gstreamer-rs (0.20), causing type mismatches.

3. **WebView transparency** - webkit2gtk may not support transparent backgrounds reliably on all compositors.

## Future Improvements

1. Implement `appsink` frame extraction for WebView overlay
2. Investigate wlr-layer-shell for compositor overlay surfaces
3. Consider GTK4/Tauri v3 which might have better integration
4. Add subtitle support via GStreamer's subtitle pipeline

## References

- [GStreamer Wayland Embedding Discussion](https://discourse.gstreamer.org/t/embedding-gstreamer-into-wayland-window/3661)
- [gstreamer-rs documentation](https://gstreamer.pages.freedesktop.org/gstreamer-rs/)
- [Tauri window management](https://tauri.app/v1/api/js/window/)
- [Wayland Protocol Documentation](https://wayland.freedesktop.org/docs/html/)
