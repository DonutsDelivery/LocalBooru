//! Test binary for GStreamer transcoding pipeline
//!
//! Run with: cargo run --bin test_transcode -- /path/to/video.mp4

use localbooru_lib::video::transcode::{
    HardwareEncoder, TranscodeConfig, TranscodeManager, TranscodePipeline, TranscodeQuality,
    needs_transcoding, probe_video_codec,
};
use std::path::PathBuf;
use std::time::Duration;

fn main() {
    // Simple logging to stdout (env_logger not needed for test)

    println!("=== GStreamer Transcoding Test ===\n");

    // Test hardware encoder detection
    println!("1. Hardware Encoder Detection:");
    let encoder = HardwareEncoder::detect();
    println!("   Detected encoder: {:?}\n", encoder);

    // Test GStreamer setup
    println!("2. GStreamer Setup:");
    if let Err(e) = gstreamer::init() {
        eprintln!("   ERROR: Failed to initialize GStreamer: {}", e);
        return;
    }
    let (major, minor, micro, nano) = gstreamer::version();
    println!(
        "   GStreamer version: {}.{}.{}.{}\n",
        major, minor, micro, nano
    );

    // Check required elements
    println!("3. Required Elements Check:");
    let required_elements = [
        ("uridecodebin", "URI decoder"),
        ("videoconvert", "Video converter"),
        ("videoscale", "Video scaler"),
        ("x264enc", "x264 encoder (software)"),
        ("nvh264enc", "NVENC encoder (hardware)"),
        ("avenc_aac", "AAC audio encoder"),
        ("mpegtsmux", "MPEG-TS muxer"),
        ("hlssink2", "HLS sink"),
    ];

    for (name, desc) in required_elements {
        let available = gstreamer::ElementFactory::find(name).is_some();
        println!(
            "   {} [{}]: {}",
            if available { "[OK]" } else { "[--]" },
            name,
            desc
        );
    }
    println!();

    // Test codec detection if video path provided
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let video_path = PathBuf::from(&args[1]);

        if video_path.exists() {
            println!("4. Video Analysis:");
            println!("   File: {}", video_path.display());

            // Probe video codec
            match probe_video_codec(&video_path) {
                Ok(codec) => {
                    println!("   Codec: {}", codec);
                    println!(
                        "   Needs transcoding: {}",
                        needs_transcoding(&codec)
                    );
                }
                Err(e) => {
                    println!("   Could not probe codec: {}", e);
                }
            }
            println!();

            // Test transcoding if requested
            if args.len() > 2 && args[2] == "--transcode" {
                println!("5. Transcoding Test:");
                let output_dir = std::env::temp_dir().join("localbooru_transcode_test");

                let config = TranscodeConfig {
                    source_path: video_path,
                    output_dir: output_dir.clone(),
                    quality: TranscodeQuality::Medium,
                    start_position: 0.0,
                    preferred_encoder: None,
                    segment_duration: 4,
                    force_cfr: false,
                };

                match TranscodePipeline::new(config) {
                    Ok(mut pipeline) => {
                        println!("   Stream ID: {}", pipeline.stream_id());
                        println!("   Output: {}", pipeline.hls_dir().display());

                        match pipeline.start() {
                            Ok(()) => {
                                println!("   Pipeline started successfully!");

                                // Monitor progress for 10 seconds
                                for i in 0..20 {
                                    std::thread::sleep(Duration::from_millis(500));
                                    let progress = pipeline.get_progress();
                                    println!(
                                        "   Progress: {:.1}% | Segments: {} | Speed: {:.1}x | State: {:?}",
                                        progress.progress_percent,
                                        progress.segments_ready,
                                        progress.encoding_speed,
                                        progress.state
                                    );

                                    // Check if playlist is ready
                                    if pipeline.is_playlist_ready() && i == 0 {
                                        println!("   Playlist ready at: {}", pipeline.playlist_path().display());
                                    }

                                    // Stop after some segments
                                    if progress.segments_ready >= 3 {
                                        println!("   Stopping after 3 segments...");
                                        break;
                                    }
                                }

                                let _ = pipeline.stop();
                                println!("   Pipeline stopped");

                                // Cleanup
                                let _ = pipeline.cleanup();
                                println!("   Cleaned up output directory");
                            }
                            Err(e) => {
                                println!("   ERROR starting pipeline: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("   ERROR creating pipeline: {}", e);
                    }
                }
            }
        } else {
            println!("   ERROR: File not found: {}", video_path.display());
        }
    } else {
        println!("4. Video Analysis: (skipped - no video path provided)");
        println!("   Usage: test_transcode /path/to/video.mp4 [--transcode]");
    }

    println!("\n=== Test Complete ===");
}
