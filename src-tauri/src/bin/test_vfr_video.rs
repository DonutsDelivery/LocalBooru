//! VFR Video Playback Test
//!
//! Run with: cargo run --bin test_vfr_video -- /path/to/video.mp4
//!
//! This tests VFR-aware video playback and seeking.

use std::env;
use std::io::{self, BufRead};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <video-file>", args[0]);
        eprintln!();
        eprintln!("Tests VFR (Variable Frame Rate) video handling.");
        eprintln!();
        eprintln!("Commands during playback:");
        eprintln!("  p     - Pause/Resume");
        eprintln!("  s     - Seek to 10 seconds (accurate)");
        eprintln!("  k     - Seek to 10 seconds (keyframe only)");
        eprintln!("  f     - Seek to 10 seconds (snap to frame)");
        eprintln!("  .     - Step forward one frame");
        eprintln!("  i     - Show stream info");
        eprintln!("  q     - Quit");
        std::process::exit(1);
    }

    let video_path = &args[1];

    println!("=== VFR Video Playback Test ===\n");
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

    // Analyze video for VFR
    println!("\nAnalyzing video for VFR characteristics...");
    match localbooru_lib::video::analyze_video_for_vfr(video_path) {
        Ok(info) => {
            println!("  VFR detected: {}", info.is_vfr);
            println!("  Average FPS: {:.2}", info.average_fps);
            if let Some(fps) = info.container_fps {
                println!("  Container FPS: {:.2}", fps);
            }
            if let Some(min) = info.min_frame_duration_ns {
                println!("  Min frame duration: {:.2} ms", min as f64 / 1_000_000.0);
            }
            if let Some(max) = info.max_frame_duration_ns {
                println!("  Max frame duration: {:.2} ms", max as f64 / 1_000_000.0);
            }
        }
        Err(e) => {
            println!("  [WARN] Could not analyze: {}", e);
        }
    }

    // Create VFR player
    println!("\nCreating VFR video player...");
    let player = match localbooru_lib::video::VfrVideoPlayer::new() {
        Ok(p) => {
            println!("  [OK] VFR player created");
            p
        }
        Err(e) => {
            eprintln!("  [FAIL] Failed to create player: {}", e);
            std::process::exit(1);
        }
    };

    // Start playback
    println!("\nStarting playback...");
    if let Err(e) = player.play_uri(video_path) {
        eprintln!("  [FAIL] Failed to start playback: {}", e);
        std::process::exit(1);
    }
    println!("  [OK] Playback started");

    println!("\n=== Controls ===");
    println!("p=pause/resume, s=seek(accurate), k=seek(keyframe), f=seek(snap)");
    println!(".=step frame, i=info, q=quit");
    println!("================\n");

    // Interactive control loop using a simpler blocking approach
    let mut paused = false;

    // Simple blocking input loop
    let stdin = io::stdin();
    let stdin_lock = stdin.lock();

    println!("Enter commands (press Enter after each):");

    for line in stdin_lock.lines() {
        // Display position before processing command
        if let (Some(pos), Some(dur)) = (player.position(), player.duration()) {
            let pos_secs = pos as f64 / 1_000_000_000.0;
            let dur_secs = dur as f64 / 1_000_000_000.0;
            println!("Position: {:.2}/{:.2}s [{:?}]", pos_secs, dur_secs, player.state());
        }

        let input = match line {
            Ok(s) => s,
            Err(_) => break,
        };

        match input.trim() {
            "p" => {
                if paused {
                    println!("Resuming...");
                    player.resume().ok();
                    paused = false;
                } else {
                    println!("Pausing...");
                    player.pause().ok();
                    paused = true;
                }
            }
            "s" => {
                println!("Seeking to 10s (accurate mode)...");
                player.seek_with_mode(10_000_000_000, localbooru_lib::video::SeekMode::Accurate).ok();
            }
            "k" => {
                println!("Seeking to 10s (keyframe mode)...");
                player.seek_with_mode(10_000_000_000, localbooru_lib::video::SeekMode::Keyframe).ok();
            }
            "f" => {
                println!("Seeking to 10s (snap to frame mode)...");
                player.seek_with_mode(10_000_000_000, localbooru_lib::video::SeekMode::SnapToFrame).ok();
            }
            "." => {
                println!("Stepping forward one frame...");
                player.step_frame().ok();
            }
            "i" => {
                println!("--- Stream Info ---");
                if let Ok(info) = player.query_stream_info() {
                    println!("Duration: {:?}s", info.duration_ns.map(|n| n as f64 / 1e9));
                    println!("Position: {:?}s", info.position_ns.map(|n| n as f64 / 1e9));
                    println!("Seekable: {}", info.seekable);
                    println!("Buffering: {:?}%", info.buffering_percent);
                }
                let frame_info = player.frame_rate_info();
                println!("VFR: {}", frame_info.is_vfr);
                println!("Avg FPS: {:.2}", frame_info.average_fps);
                println!("-------------------");
            }
            "q" => {
                println!("Quitting...");
                break;
            }
            "" => {
                // Empty input, just continue
            }
            other => {
                println!("Unknown command: '{}'. Use p/s/k/f/./i/q", other);
            }
        }

        // Check for end of stream or error
        match player.state() {
            localbooru_lib::video::VfrPlayerState::EndOfStream => {
                println!("\nEnd of stream reached.");
                break;
            }
            localbooru_lib::video::VfrPlayerState::Error(e) => {
                eprintln!("\nError: {}", e);
                break;
            }
            _ => {}
        }
    }

    // Stop playback
    println!("Stopping playback...");
    player.stop().ok();

    println!("Done.");
}
