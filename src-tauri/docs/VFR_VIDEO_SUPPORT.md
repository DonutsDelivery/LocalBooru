# VFR (Variable Frame Rate) Video Support

This document describes the VFR video handling implementation in LocalBooru's Tauri video player.

## Overview

Variable Frame Rate (VFR) videos are common from:
- Phone recordings (most mobile cameras use VFR)
- Screen recordings (frame rate varies with screen content)
- Game capture software
- Some webcams

HTML5 video elements struggle with VFR content because browsers often:
- Assume constant frame rate
- Interpolate timestamps incorrectly
- Cause audio desync over time
- Have seeking artifacts

GStreamer handles VFR natively through proper PTS (Presentation Timestamp) handling.

## Architecture

### VfrVideoPlayer

The `VfrVideoPlayer` struct in `src/video/vfr.rs` provides:

1. **Proper timestamp preservation**: Uses GStreamer's clock-based synchronization
2. **Multiple seek modes**: Keyframe, Accurate, and SnapToFrame
3. **Frame stepping**: Step forward one frame at a time
4. **VFR detection**: Analyze videos to detect VFR characteristics
5. **Audio/Video sync**: Maintained through GStreamer's shared clock mechanism

### Key Components

```
Pipeline Architecture:
  playbin3 (or playbin)
    |
    +-- decodebin3 (preserves PTS)
    |     |
    |     +-- video decoder (hw or sw)
    |     +-- audio decoder
    |
    +-- video sink (waylandsink/glimagesink)
    |     - sync=true (respects timestamps)
    |
    +-- audio sink (pulsesink)
          - sync=true
          - low latency config
```

## Seek Modes

### SeekMode::Keyframe
- Fast seeking to nearest keyframe
- May not land on exact requested position
- Best for quick scrubbing

### SeekMode::Accurate
- Decodes from keyframe to exact position
- Slower but precise
- Default mode for VFR content

### SeekMode::SnapToFrame
- Snaps to nearest actual frame boundary
- Best for VFR content where frame timestamps vary
- Uses GStreamer's SNAP_NEAREST flag

## VFR Detection

The `analyze_video_for_vfr()` function examines:

1. **Container framerate**: VFR videos often report:
   - 0/1 fps (unknown)
   - 1000/1 fps (VFR marker)
   - Very high values

2. **Frame duration variance**: (TODO)
   - Sample actual frame PTSs
   - Calculate min/max frame durations
   - Large variance indicates VFR

## API Usage

### From Rust

```rust
use localbooru_lib::video::{VfrVideoPlayer, SeekMode, analyze_video_for_vfr};

// Analyze video first
let info = analyze_video_for_vfr("/path/to/video.mp4")?;
if info.is_vfr {
    println!("VFR video detected!");
}

// Create player
let player = VfrVideoPlayer::new()?;

// Play video
player.play_uri("/path/to/video.mp4")?;

// Seek with VFR-aware mode
player.seek_with_mode(10_000_000_000, SeekMode::SnapToFrame)?;

// Step frame by frame
player.step_frame()?;
```

### From Frontend (Tauri Commands)

```javascript
// Analyze video
const info = await invoke('video_vfr_analyze', { uri: '/path/to/video.mp4' });

// Initialize player
await invoke('video_vfr_init');

// Play
await invoke('video_vfr_play', { uri: '/path/to/video.mp4' });

// Seek with specific mode
await invoke('video_vfr_seek_with_mode', {
  positionSecs: 10.0,
  mode: 'snap_to_frame'  // or 'accurate' or 'keyframe'
});

// Step forward one frame
await invoke('video_vfr_step_frame');

// Get stream info
const streamInfo = await invoke('video_vfr_get_stream_info');
```

## Tauri Commands

| Command | Description |
|---------|-------------|
| `video_vfr_analyze` | Analyze video for VFR characteristics |
| `video_vfr_init` | Initialize VFR player |
| `video_vfr_play` | Start playback |
| `video_vfr_pause` | Pause playback |
| `video_vfr_resume` | Resume playback |
| `video_vfr_stop` | Stop playback |
| `video_vfr_seek` | Seek with default mode |
| `video_vfr_seek_with_mode` | Seek with specific mode |
| `video_vfr_step_frame` | Step forward one frame |
| `video_vfr_get_position` | Get current position (seconds) |
| `video_vfr_get_duration` | Get duration (seconds) |
| `video_vfr_get_state` | Get player state |
| `video_vfr_get_stream_info` | Get detailed stream info |
| `video_vfr_get_frame_info` | Get frame rate info |
| `video_vfr_set_seek_mode` | Set default seek mode |
| `video_vfr_get_seek_mode` | Get current seek mode |
| `video_vfr_set_rate` | Set playback rate |
| `video_vfr_set_volume` | Set volume (0.0-1.0) |
| `video_vfr_get_volume` | Get volume |
| `video_vfr_set_muted` | Set mute state |
| `video_vfr_is_muted` | Check if muted |
| `video_vfr_cleanup` | Cleanup player |

## GStreamer Configuration

### Hardware Acceleration

The player automatically detects and uses available decoders:
1. **NVIDIA NVDEC**: `nvh264dec`, `nvh265dec`
2. **VA-API**: `vaapih264dec`, `vaapih265dec`
3. **Software**: `avdec_h264` (ffmpeg)

### Video Sinks

On Wayland (Linux):
1. `waylandsink` - Best performance, direct compositor integration
2. `glimagesink` - OpenGL-based fallback
3. `autovideosink` - Auto-selection

### Audio Configuration

PulseAudio sink with low latency:
- Buffer time: 20ms
- Latency time: 10ms
- Sync enabled

## Testing

### Test Binary

```bash
cd src-tauri
cargo run --bin test_vfr_video -- /path/to/video.mp4
```

Commands:
- `p` - Pause/Resume
- `s` - Seek to 10s (accurate)
- `k` - Seek to 10s (keyframe)
- `f` - Seek to 10s (snap to frame)
- `.` - Step one frame
- `i` - Show stream info
- `q` - Quit

### Creating Test VFR Videos

Use ffmpeg to create a VFR test video:

```bash
# Create VFR video from constant frame rate source
ffmpeg -i input.mp4 -vf "setpts=PTS*random(0)" -vsync vfr output_vfr.mp4
```

Or record from a phone camera (most are VFR by default).

## Known Limitations

1. **VFR Detection**: Currently relies on container metadata. More accurate detection would sample actual frame timestamps.

2. **Timeline Scrubbing**: Fast scrubbing may have slight delays due to accurate seek mode. Consider using keyframe mode for scrubbing and accurate mode for final position.

3. **Frame Stepping**: Only forward stepping is implemented. Reverse stepping would require different GStreamer approach.

## Future Improvements

1. Better VFR detection by sampling actual frame PTSs
2. Reverse frame stepping
3. Timeline thumbnail generation with proper VFR handling
4. Adaptive seek mode selection based on scrubbing speed
