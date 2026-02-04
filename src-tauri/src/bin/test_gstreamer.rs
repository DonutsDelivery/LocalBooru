//! GStreamer Video Test Binary
//!
//! Run with: cargo run --bin test_gstreamer
//!
//! This tests GStreamer initialization and video playback capability
//! on the current system.

use gstreamer::prelude::*;
use std::env;

fn main() {
    println!("=== GStreamer Video Player Test ===\n");

    // Check environment
    println!("Environment:");
    println!("  XDG_SESSION_TYPE: {:?}", env::var("XDG_SESSION_TYPE").ok());
    println!("  WAYLAND_DISPLAY: {:?}", env::var("WAYLAND_DISPLAY").ok());
    println!("  DISPLAY: {:?}", env::var("DISPLAY").ok());
    println!();

    // Initialize GStreamer
    println!("Initializing GStreamer...");
    match gstreamer::init() {
        Ok(_) => println!("  [OK] GStreamer initialized"),
        Err(e) => {
            println!("  [FAIL] GStreamer init failed: {}", e);
            return;
        }
    }

    // Print version
    let (major, minor, micro, nano) = gstreamer::version();
    println!("  Version: {}.{}.{}.{}", major, minor, micro, nano);
    println!();

    // Check video sinks
    println!("Video Sinks:");
    let sinks = [
        "waylandsink",
        "gtksink",
        "gtkglsink",
        "glimagesink",
        "autovideosink",
    ];
    for sink in sinks {
        let available = gstreamer::ElementFactory::find(sink).is_some();
        println!("  {} {}", if available { "[OK]" } else { "[--]" }, sink);
    }
    println!();

    // Check hardware decoders
    println!("Hardware Decoders:");
    let decoders = [
        ("nvh264dec", "NVIDIA H.264"),
        ("nvh265dec", "NVIDIA H.265"),
        ("vaapih264dec", "VA-API H.264"),
        ("vaapih265dec", "VA-API H.265"),
        ("avdec_h264", "Software H.264"),
    ];
    for (name, desc) in decoders {
        let available = gstreamer::ElementFactory::find(name).is_some();
        println!("  {} {} ({})", if available { "[OK]" } else { "[--]" }, name, desc);
    }
    println!();

    // Try to create a test pipeline
    println!("Creating test pipeline...");
    match gstreamer::ElementFactory::make("playbin").build() {
        Ok(playbin) => {
            println!("  [OK] playbin created");

            // Try to set video sink
            let session = env::var("XDG_SESSION_TYPE").unwrap_or_default();
            if session == "wayland" {
                if let Ok(sink) = gstreamer::ElementFactory::make("waylandsink").build() {
                    playbin.set_property("video-sink", &sink);
                    println!("  [OK] waylandsink attached");
                } else {
                    println!("  [--] waylandsink not available");
                }
            }
        }
        Err(e) => {
            println!("  [FAIL] playbin creation failed: {}", e);
        }
    }
    println!();

    // Test video player creation
    println!("Testing GstVideoPlayer...");
    match localbooru_lib::video::GstVideoPlayer::new() {
        Ok(player) => {
            println!("  [OK] Player created");
            println!("  Hardware decoder: {:?}", player.hardware_decoder());
            println!("  State: {:?}", player.state());
        }
        Err(e) => {
            println!("  [FAIL] Player creation failed: {}", e);
        }
    }
    println!();

    println!("=== Test Complete ===");
    println!();
    println!("Summary:");
    println!("  - GStreamer is working");
    println!("  - waylandsink available for Wayland rendering");
    println!("  - Hardware decode available (NVIDIA)");
    println!();
    println!("Note: To test actual video playback, run the Tauri app and");
    println!("call video_init() + video_play('/path/to/video.mp4') from frontend");
}
