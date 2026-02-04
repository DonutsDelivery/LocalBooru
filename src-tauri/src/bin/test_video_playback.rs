//! GStreamer Video Playback Test
//!
//! Run with: cargo run --bin test_video_playback -- /path/to/video.mp4
//!
//! This actually plays a video file to verify GStreamer + Wayland works.

use gstreamer::prelude::*;
use std::env;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <video-file>", args[0]);
        eprintln!();
        eprintln!("Example: {} /path/to/video.mp4", args[0]);
        std::process::exit(1);
    }

    let video_path = &args[1];

    println!("=== GStreamer Video Playback Test ===\n");
    println!("Video: {}", video_path);
    println!();

    // Check if file exists
    if !std::path::Path::new(video_path).exists() {
        eprintln!("Error: File not found: {}", video_path);
        std::process::exit(1);
    }

    // Initialize GStreamer
    println!("Initializing GStreamer...");
    gstreamer::init().expect("Failed to init GStreamer");
    println!("  [OK] GStreamer initialized");

    // Create playbin
    println!("Creating pipeline...");
    let playbin = gstreamer::ElementFactory::make("playbin")
        .build()
        .expect("Failed to create playbin");

    // Set URI
    let uri = if video_path.starts_with("file://") {
        video_path.clone()
    } else {
        format!("file://{}", video_path)
    };
    playbin.set_property("uri", &uri);

    // Configure video sink based on session type
    let session = env::var("XDG_SESSION_TYPE").unwrap_or_default();
    println!("Session type: {}", session);

    if session == "wayland" {
        // Use waylandsink for Wayland
        if let Ok(sink) = gstreamer::ElementFactory::make("waylandsink").build() {
            playbin.set_property("video-sink", &sink);
            println!("  [OK] Using waylandsink");
        } else if let Ok(sink) = gstreamer::ElementFactory::make("glimagesink").build() {
            playbin.set_property("video-sink", &sink);
            println!("  [OK] Using glimagesink (fallback)");
        }
    }

    println!();
    println!("Starting playback...");
    println!("Press Enter to stop playback.");
    println!();

    // Start playback
    playbin.set_state(gstreamer::State::Playing)
        .expect("Failed to start playback");

    // Wait for user input
    let mut input = String::new();
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();

    // Stop playback
    println!("Stopping playback...");
    playbin.set_state(gstreamer::State::Null)
        .expect("Failed to stop playback");

    println!("Done.");
}
