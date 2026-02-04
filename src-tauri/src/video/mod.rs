//! GStreamer Video Player Prototype for Tauri
//!
//! This module provides GStreamer-based video playback that works on Wayland Linux.
//! The approach uses a separate GTK window with GStreamer video rendering that can
//! be positioned alongside or layered with the Tauri WebView window.
//!
//! # Architecture Notes
//!
//! On Wayland, there are several challenges for video playback in Tauri:
//!
//! 1. **No Window Embedding**: Wayland doesn't support embedding windows from different
//!    processes or even different surfaces in the same compositor tree easily. This means
//!    we can't just create a GStreamer video surface and embed it in the WebView.
//!
//! 2. **Compositor-Level Overlays**: The only way to achieve true video overlay on Wayland
//!    is through compositor-specific protocols (like layer-shell) or by using GTK's
//!    native widget hierarchy.
//!
//! 3. **Our Approach**: Since Tauri uses webkit2gtk (GTK3), we can:
//!    - Use `gtksink` which creates a GtkWidget for video rendering
//!    - Attempt to overlay this widget in the GTK widget hierarchy
//!    - Or use a separate positioned window that follows the main window
//!
//! # Hardware Acceleration
//!
//! The pipeline automatically uses hardware decoding when available:
//! - NVIDIA: nvh264dec, nvh265dec (via NVDEC)
//! - VA-API: vaapih264dec, vaapih265dec
//! - Fallback: Software decoding (avdec_h264, etc.)
//!
//! # VFR (Variable Frame Rate) Support
//!
//! The `vfr` module provides specialized handling for VFR videos:
//! - Proper PTS (Presentation Timestamp) preservation
//! - Accurate seeking with frame-snap capability
//! - A/V sync maintained through clock-based synchronization
//! - VFR detection and analysis tools
//!
//! VFR support is the primary reason for using GStreamer over HTML5 video,
//! as browser video elements often struggle with variable frame rate content.
//!
//! # Transcoding
//!
//! The transcode module provides on-the-fly transcoding for incompatible formats:
//! - Converts AV1, VP9, HEVC to H.264 for browser compatibility
//! - Uses hardware encoding (NVENC) when available
//! - Outputs HLS segments for streaming playback
//! - Includes cache management for disk space control
//!
//! # Frame Interpolation
//!
//! The interpolation module provides smooth motion interpolation:
//! - SVP (SmoothVideo Project) integration for professional-grade interpolation
//! - RIFE-NCNN fallback for neural network based interpolation
//! - FFmpeg minterpolate as a universal fallback
//! - Configurable presets and target FPS

mod player;
mod overlay;
mod commands;
pub mod vfr;
pub mod transcode;
pub mod transcode_commands;
pub mod interpolation;
pub mod interpolation_commands;
pub mod events;

pub use player::{GstVideoPlayer, PlayerState, HardwareDecoder};
pub use overlay::{VideoOverlay, test_gstreamer_setup};
pub use commands::*;
pub use vfr::{VfrVideoPlayer, VfrPlayerState, SeekMode, FrameRateInfo, StreamInfo, analyze_video_for_vfr};
pub use transcode::{TranscodeQuality, HardwareEncoder, TranscodeProgress, TranscodeState};
pub use transcode_commands::*;
pub use interpolation::{InterpolationConfig, InterpolationPreset, InterpolationBackend, InterpolatedPlayer, InterpolationState, BackendCapabilities, detect_available_backends, recommend_backend};
pub use interpolation_commands::*;
pub use events::{VideoEvent, VideoEventStreamer, VideoEventState, video_events_start, video_events_stop, video_events_is_active};

// Re-export VFR command types
pub use commands::{VfrVideoPlayerState, FrameRateInfoDto, StreamInfoDto, SeekModeDto};
