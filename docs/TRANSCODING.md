# LocalBooru Transcoding Pipeline

This document describes the GStreamer-based transcoding pipeline implemented for the Tauri migration.

## Overview

The transcoding pipeline converts incompatible video formats (AV1, VP9, HEVC) to browser-compatible H.264 using GStreamer. It supports hardware acceleration (NVENC) and outputs HLS streams for efficient playback.

## Architecture

```
Source Video -> uridecodebin -> videoconvert -> videoscale -> encoder -> h264parse -> mpegtsmux -> hlssink2
                            -> audioconvert -> audioresample -> aacenc  ->
```

### Components

1. **Source Decoding** (`uridecodebin`)
   - Automatically detects and decodes any supported format
   - Handles container parsing and codec selection
   - Supports hardware-accelerated decoding (NVDEC, VA-API)

2. **Video Processing**
   - `videoconvert`: Converts between color formats
   - `videoscale`: Scales to target resolution based on quality preset
   - `capsfilter`: Enforces output dimensions

3. **Video Encoding**
   - Primary: `nvh264enc` (NVIDIA NVENC) - ~10x faster than software
   - Secondary: `vaapih264enc` (Intel/AMD VA-API)
   - Fallback: `x264enc` (Software x264)

4. **Audio Encoding**
   - `avenc_aac` or `faac` or `voaacenc` (AAC output)
   - Automatic resampling and format conversion

5. **HLS Output** (`hlssink2`)
   - Generates M3U8 playlist and TS segments
   - Configurable segment duration (default 4s)
   - Progressive playback support

## Files

### Core Module: `src-tauri/src/video/transcode.rs`

Contains:
- `TranscodeQuality` - Quality presets (Low/Medium/High/Original)
- `HardwareEncoder` - Encoder detection and selection
- `TranscodeConfig` - Pipeline configuration
- `TranscodePipeline` - Main transcoding pipeline
- `TranscodeManager` - Multi-stream management and cache control
- `probe_video_codec()` - Video codec detection
- `needs_transcoding()` - Codec compatibility check

### Tauri Commands: `src-tauri/src/video/transcode_commands.rs`

Frontend-accessible commands:
- `transcode_get_capabilities` - Query hardware encoder availability
- `transcode_check_needed` - Check if video needs transcoding
- `transcode_start` - Start transcoding session
- `transcode_get_progress` - Get encoding progress
- `transcode_is_ready` - Check if HLS playlist is ready
- `transcode_get_playlist` - Get playlist path
- `transcode_stop` - Stop transcoding
- `transcode_stop_all` - Stop all active sessions
- `transcode_cleanup_cache` - Clean up cached segments
- `transcode_set_cache_limit` - Set maximum cache size

## Quality Presets

| Preset   | Resolution | Video Bitrate | Audio Bitrate |
|----------|------------|---------------|---------------|
| Low      | 854x480    | 1 Mbps        | 96 kbps       |
| Medium   | 1280x720   | 2.5 Mbps      | 128 kbps      |
| High     | 1920x1080  | 5 Mbps        | 192 kbps      |
| Original | Source     | 8 Mbps        | 256 kbps      |

## Hardware Encoder Detection

The system automatically detects available hardware encoders at startup:

1. **NVENC (NVIDIA)** - Checked via `nvh264enc` element availability
2. **VA-API (Intel/AMD)** - Checked via `vaapih264enc` element
3. **Software Fallback** - Always uses `x264enc` with ultrafast preset

## Codecs Requiring Transcoding

The following codecs are automatically detected as needing transcoding:

- AV1 (limited browser support)
- VP9 (context-dependent support)
- HEVC/H.265 (poor browser support)
- MPEG-4 Part 2
- MPEG-2
- WMV, VC-1
- Theora
- ProRes, DNxHD, FFV1

## Cache Management

The `TranscodeManager` includes automatic cache management:

- Default limit: 5 GB
- Oldest segments are deleted first when over limit
- Empty stream directories are automatically cleaned up
- Manual cleanup via `transcode_cleanup_cache` command

## Usage Example (Frontend)

```javascript
// Check if transcoding is needed
const needsTranscode = await invoke('transcode_check_needed', {
  videoPath: '/path/to/video.mkv'
});

if (needsTranscode) {
  // Start transcoding
  const { streamId, playlistPath, encoder } = await invoke('transcode_start', {
    request: {
      source_path: '/path/to/video.mkv',
      quality: 'medium',
      start_position: 0.0
    }
  });

  // Wait for playlist to be ready
  while (!(await invoke('transcode_is_ready', { streamId }))) {
    await new Promise(r => setTimeout(r, 100));
  }

  // Play the HLS stream
  videoPlayer.src = playlistPath;
}
```

## GStreamer Dependencies

Required GStreamer plugins:
- `gst-plugins-base` - Core elements (videoconvert, audioresample)
- `gst-plugins-good` - Standard elements (hlssink2)
- `gst-plugins-bad` - Extended elements (nvh264enc, vaapih264enc, hlssink2)
- `gst-plugins-ugly` - x264enc encoder
- `gst-libav` - AAC encoder (avenc_aac)

On Arch Linux:
```bash
pacman -S gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

## Testing

A test binary is provided:

```bash
cargo run --bin test_transcode -- /path/to/video.mp4 [--transcode]
```

This will:
1. Detect hardware encoders
2. Check GStreamer element availability
3. Analyze video codec (if path provided)
4. Optionally transcode a few segments (with --transcode flag)

## Known Limitations

1. **VFR videos** - Currently no special handling for variable frame rate
2. **Subtitles** - Not included in transcoded output
3. **Multi-audio** - Only first audio track is transcoded
4. **Seeking** - Start position seeking not yet implemented

## Future Improvements

- [ ] Add VAAPI encoder setup with proper surface handling
- [ ] Implement start position seeking
- [ ] Add subtitle pass-through or burn-in option
- [ ] Support multiple audio track selection
- [ ] Add adaptive bitrate (ABR) multi-quality output
- [ ] Implement segment caching based on video hash
