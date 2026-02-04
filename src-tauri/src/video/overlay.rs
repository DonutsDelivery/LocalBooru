//! Video Overlay Implementation for Wayland
//!
//! This module provides video overlay capabilities that work on Wayland.
//!
//! # Wayland Video Playback Challenges
//!
//! On Wayland, video overlay is more complex than X11 due to:
//! 1. No XEmbed or window reparenting
//! 2. Strict surface isolation between clients
//! 3. No direct window handle passing between processes
//!
//! # Available Approaches
//!
//! 1. **waylandsink**: Direct Wayland subsurface rendering
//!    - Best performance, hardware overlay
//!    - Requires passing wl_surface handle
//!
//! 2. **autovideosink**: Let GStreamer pick the best sink
//!    - Works but creates separate window
//!
//! 3. **appsink + manual rendering**: Extract frames and render in WebView
//!    - Works but has latency overhead
//!    - Good for overlays/transparency
//!
//! This implementation provides a pipeline-based approach that works
//! with any available video sink.

use gstreamer as gst;
use gstreamer::prelude::*;

/// Test if GStreamer video playback works on the current system
///
/// This function creates a test pipeline and checks if all required
/// elements are available.
pub fn test_gstreamer_setup() -> Result<String, String> {
    gst::init().map_err(|e| format!("GStreamer init failed: {}", e))?;

    let mut report = String::new();
    report.push_str("GStreamer Video Setup Report\n");
    report.push_str("============================\n\n");

    // Check GStreamer version
    let (major, minor, micro, nano) = gst::version();
    report.push_str(&format!("GStreamer version: {}.{}.{}.{}\n", major, minor, micro, nano));

    // Check video sinks
    let sinks = [
        ("waylandsink", "Wayland Direct Sink (best for Wayland)"),
        ("gtksink", "GTK Widget Sink"),
        ("gtkglsink", "GTK GL Sink"),
        ("glimagesink", "OpenGL Image Sink"),
        ("autovideosink", "Auto Video Sink"),
    ];

    report.push_str("\nVideo Sinks:\n");
    for (name, desc) in sinks {
        let available = gst::ElementFactory::find(name).is_some();
        report.push_str(&format!("  {} [{}]: {}\n",
            if available { "[OK]" } else { "[--]" },
            name,
            desc
        ));
    }

    // Check decoders
    let decoders = [
        ("nvh264dec", "NVIDIA H.264 (NVDEC)"),
        ("nvh265dec", "NVIDIA H.265 (NVDEC)"),
        ("nvav1dec", "NVIDIA AV1 (NVDEC)"),
        ("vaapih264dec", "VA-API H.264"),
        ("vaapih265dec", "VA-API H.265"),
        ("avdec_h264", "FFmpeg H.264 (Software)"),
        ("avdec_h265", "FFmpeg H.265 (Software)"),
    ];

    report.push_str("\nVideo Decoders:\n");
    for (name, desc) in decoders {
        let available = gst::ElementFactory::find(name).is_some();
        report.push_str(&format!("  {} [{}]: {}\n",
            if available { "[OK]" } else { "[--]" },
            name,
            desc
        ));
    }

    // Check Wayland-specific support
    report.push_str("\nWayland Support:\n");
    if std::env::var("XDG_SESSION_TYPE").map(|v| v == "wayland").unwrap_or(false) {
        report.push_str("  [OK] Running on Wayland session\n");
    } else {
        report.push_str("  [--] Not running on Wayland (may be X11 or unknown)\n");
    }

    // Check for hardware-accelerated upload elements
    let hw_upload = [
        ("glupload", "OpenGL upload"),
        ("vapostproc", "VA-API post processor"),
        ("nvvidconv", "NVIDIA video converter"),
    ];

    report.push_str("\nHardware Upload/Conversion:\n");
    for (name, desc) in hw_upload {
        let available = gst::ElementFactory::find(name).is_some();
        report.push_str(&format!("  {} [{}]: {}\n",
            if available { "[OK]" } else { "[--]" },
            name,
            desc
        ));
    }

    // Recommended pipeline based on available elements
    report.push_str("\nRecommended Pipeline:\n");
    if gst::ElementFactory::find("nvh264dec").is_some() {
        report.push_str("  NVIDIA: playbin with nvdec + waylandsink\n");
        report.push_str("  Pipeline: nvh264dec -> glupload -> gldownload -> waylandsink\n");
    } else if gst::ElementFactory::find("vaapih264dec").is_some() {
        report.push_str("  VA-API: playbin with vaapi + waylandsink\n");
    } else {
        report.push_str("  Software: playbin with software decode + autovideosink\n");
    }

    Ok(report)
}

/// Configuration for video overlay
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// Video position X
    pub x: i32,
    /// Video position Y
    pub y: i32,
    /// Video width
    pub width: i32,
    /// Video height
    pub height: i32,
    /// Keep aspect ratio
    pub keep_aspect: bool,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 640,
            height: 480,
            keep_aspect: true,
        }
    }
}

/// Video overlay that uses GStreamer pipelines
///
/// This creates a GStreamer pipeline using playbin which automatically
/// handles format negotiation and decoder selection.
pub struct VideoOverlay {
    pipeline: gst::Element,
    config: OverlayConfig,
}

impl VideoOverlay {
    /// Create a new video overlay using playbin
    pub fn new() -> Result<Self, String> {
        // Initialize GStreamer
        gst::init().map_err(|e| format!("Failed to init GStreamer: {}", e))?;

        // Create playbin which handles everything automatically
        let playbin = gst::ElementFactory::make("playbin")
            .name("playbin")
            .build()
            .map_err(|e| format!("Failed to create playbin: {}", e))?;

        // Try to set up the best video sink for Wayland
        if let Some(video_sink) = Self::create_best_video_sink() {
            playbin.set_property("video-sink", &video_sink);
            log::info!("Video sink configured");
        } else {
            log::info!("Using default video sink (autovideosink)");
        }

        Ok(Self {
            pipeline: playbin,
            config: OverlayConfig::default(),
        })
    }

    /// Create the best available video sink for the current platform
    fn create_best_video_sink() -> Option<gst::Element> {
        // On Wayland, prefer waylandsink for best performance
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();

        if session_type == "wayland" {
            // Try waylandsink first
            if let Ok(sink) = gst::ElementFactory::make("waylandsink").build() {
                log::info!("Using waylandsink for Wayland");
                return Some(sink);
            }

            // Fall back to glimagesink with Wayland
            if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
                log::info!("Using glimagesink for Wayland");
                return Some(sink);
            }
        }

        // Fall back to autovideosink
        if let Ok(sink) = gst::ElementFactory::make("autovideosink").build() {
            log::info!("Using autovideosink");
            return Some(sink);
        }

        None
    }

    /// Play a video file
    pub fn play(&self, uri: &str) -> Result<(), String> {
        let full_uri = if uri.starts_with("file://") || uri.starts_with("http") {
            uri.to_string()
        } else {
            format!("file://{}", uri)
        };

        self.pipeline.set_property("uri", &full_uri);

        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("Failed to start playback: {}", e))?;

        log::info!("Playing video: {}", full_uri);
        Ok(())
    }

    /// Pause playback
    pub fn pause(&self) -> Result<(), String> {
        self.pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| format!("Failed to pause: {}", e))?;
        Ok(())
    }

    /// Stop playback
    pub fn stop(&self) -> Result<(), String> {
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| format!("Failed to stop: {}", e))?;
        Ok(())
    }

    /// Set video position and size
    pub fn set_geometry(&mut self, x: i32, y: i32, width: i32, height: i32) {
        self.config.x = x;
        self.config.y = y;
        self.config.width = width;
        self.config.height = height;
        // Note: Position/size control depends on the video sink type
        // waylandsink supports this via wl_subsurface positioning
    }

    /// Check if using hardware acceleration
    pub fn is_hardware_accelerated(&self) -> bool {
        // Check available decoders
        let registry = gst::Registry::get();
        registry.find_feature("nvh264dec", gst::ElementFactory::static_type()).is_some()
            || registry.find_feature("vaapih264dec", gst::ElementFactory::static_type()).is_some()
    }

    /// Get the underlying pipeline for advanced configuration
    pub fn pipeline(&self) -> &gst::Element {
        &self.pipeline
    }
}

impl Drop for VideoOverlay {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Get information about current display server
pub fn get_display_info() -> String {
    let mut info = String::new();

    // Check session type
    if let Ok(session) = std::env::var("XDG_SESSION_TYPE") {
        info.push_str(&format!("Session type: {}\n", session));
    }

    // Check Wayland display
    if let Ok(display) = std::env::var("WAYLAND_DISPLAY") {
        info.push_str(&format!("Wayland display: {}\n", display));
    }

    // Check X11 display (XWayland)
    if let Ok(display) = std::env::var("DISPLAY") {
        info.push_str(&format!("X11 display: {} (may be XWayland)\n", display));
    }

    info
}
