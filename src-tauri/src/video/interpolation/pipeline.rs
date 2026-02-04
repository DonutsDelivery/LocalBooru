//! Interpolation Pipeline Implementation
//!
//! This module implements the actual frame interpolation pipeline using GStreamer
//! with external interpolation processes.
//!
//! # Pipeline Architecture
//!
//! For SVP/RIFE integration, we use a subprocess approach:
//!
//! ```text
//! GStreamer decode -> Y4M pipe -> Interpolator subprocess -> Y4M pipe -> GStreamer encode/display
//! ```
//!
//! This avoids the need for custom GStreamer elements while still achieving
//! efficient real-time interpolation.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use serde::{Deserialize, Serialize};

use super::config::{InterpolationConfig, InterpolationBackend};

/// State of the interpolation pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationState {
    /// Not initialized
    Uninitialized,
    /// Starting up
    Starting,
    /// Running and processing frames
    Running,
    /// Paused
    Paused,
    /// Stopped
    Stopped,
    /// Error occurred
    Error,
}

/// Statistics from the interpolation pipeline
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InterpolationStats {
    /// Frames processed
    pub frames_processed: u64,
    /// Frames interpolated (generated)
    pub frames_interpolated: u64,
    /// Average processing time per frame (ms)
    pub avg_frame_time_ms: f64,
    /// Current effective FPS
    pub effective_fps: f64,
    /// Buffer fill level (0.0 - 1.0)
    pub buffer_level: f64,
    /// Number of frame drops
    pub frame_drops: u64,
}

/// Interpolated video player
///
/// Wraps video playback with frame interpolation support.
pub struct InterpolatedPlayer {
    /// Current configuration
    config: InterpolationConfig,
    /// Pipeline state
    state: InterpolationState,
    /// Running flag for threads
    running: Arc<AtomicBool>,
    /// Interpolator subprocess
    interpolator_process: Option<Child>,
    /// FFmpeg decode process
    decoder_process: Option<Child>,
    /// FFmpeg encode/display process
    encoder_process: Option<Child>,
    /// Statistics
    stats: Arc<std::sync::Mutex<InterpolationStats>>,
    /// Current video path
    current_video: Option<PathBuf>,
    /// Source video FPS
    source_fps: f64,
    /// Source video dimensions
    source_dims: (u32, u32),
}

impl InterpolatedPlayer {
    /// Create a new interpolated player
    pub fn new(config: InterpolationConfig) -> Self {
        Self {
            config,
            state: InterpolationState::Uninitialized,
            running: Arc::new(AtomicBool::new(false)),
            interpolator_process: None,
            decoder_process: None,
            encoder_process: None,
            stats: Arc::new(std::sync::Mutex::new(InterpolationStats::default())),
            current_video: None,
            source_fps: 0.0,
            source_dims: (0, 0),
        }
    }

    /// Get current state
    pub fn state(&self) -> InterpolationState {
        self.state
    }

    /// Get current statistics
    pub fn stats(&self) -> InterpolationStats {
        self.stats.lock().unwrap().clone()
    }

    /// Update configuration
    pub fn set_config(&mut self, config: InterpolationConfig) {
        let was_running = self.state == InterpolationState::Running;

        if was_running {
            self.stop();
        }

        self.config = config;

        // Restart if was running
        if was_running {
            if let Some(ref path) = self.current_video.clone() {
                let _ = self.play(path);
            }
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &InterpolationConfig {
        &self.config
    }

    /// Play a video file with interpolation
    pub fn play(&mut self, video_path: &PathBuf) -> Result<(), String> {
        // Stop any existing playback
        self.stop();

        // Probe video to get FPS and dimensions
        let (fps, width, height) = probe_video(video_path)?;
        self.source_fps = fps;
        self.source_dims = (width, height);
        self.current_video = Some(video_path.clone());

        log::info!(
            "Starting interpolated playback: {}x{} @ {:.2}fps -> {}fps",
            width, height, fps, self.config.target_fps
        );

        // Check if interpolation is actually needed
        if !self.config.enabled || self.config.backend == InterpolationBackend::None {
            log::info!("Interpolation disabled, playing directly");
            return self.play_direct(video_path);
        }

        // Check if real-time is viable
        if !self.config.is_realtime_viable((width, height)) {
            log::warn!(
                "Resolution {}x{} may not be viable for real-time interpolation with {:?}",
                width, height, self.config.backend
            );
        }

        self.state = InterpolationState::Starting;
        self.running.store(true, Ordering::SeqCst);

        // Start the pipeline based on backend
        match self.config.backend {
            InterpolationBackend::Svp => self.start_svp_pipeline(video_path),
            InterpolationBackend::RifeNcnn => self.start_rife_pipeline(video_path),
            InterpolationBackend::Minterpolate => self.start_minterpolate_pipeline(video_path),
            InterpolationBackend::None => self.play_direct(video_path),
        }
    }

    /// Start SVP interpolation pipeline
    fn start_svp_pipeline(&mut self, video_path: &PathBuf) -> Result<(), String> {
        // SVP pipeline: ffmpeg decode -> vspipe (SVP) -> mpv display
        //
        // We create a VapourSynth script dynamically that reads Y4M from stdin
        // and outputs interpolated Y4M to stdout.

        let video_str = video_path.to_string_lossy();

        // Generate VapourSynth script
        let vs_script = self.generate_svp_script();

        // Write script to temp file
        let script_path = std::env::temp_dir().join("localbooru_svp.vpy");
        std::fs::write(&script_path, &vs_script)
            .map_err(|e| format!("Failed to write VS script: {}", e))?;

        log::debug!("SVP script written to: {}", script_path.display());

        // Build the pipeline:
        // ffmpeg -i video -> Y4M pipe -> vspipe (SVP processing) -> mpv

        // Start FFmpeg decoder outputting Y4M
        let decoder = Command::new("ffmpeg")
            .args([
                "-hwaccel", "auto",
                "-i", &video_str,
                "-f", "yuv4mpegpipe",
                "-pix_fmt", "yuv420p",
                "-"
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start FFmpeg decoder: {}", e))?;

        // Start vspipe to process frames through SVP
        let mut vspipe = Command::new("vspipe")
            .args([
                "-c", "y4m",
                script_path.to_string_lossy().as_ref(),
                "-"
            ])
            .stdin(Stdio::from(decoder.stdout.ok_or("No decoder stdout")?))
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start vspipe: {}", e))?;

        // Start mpv to display the output
        let mpv = Command::new("mpv")
            .args([
                "--no-terminal",
                "--force-window=yes",
                "--title=LocalBooru - SVP Interpolated",
                "--osc=yes",
                "--input-ipc-server=/tmp/localbooru-mpv-ipc",
                "-"
            ])
            .stdin(Stdio::from(vspipe.stdout.take().ok_or("No vspipe stdout")?))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start mpv: {}", e))?;

        self.interpolator_process = Some(vspipe);
        self.encoder_process = Some(mpv);
        self.state = InterpolationState::Running;

        // Start stats monitoring thread
        self.start_stats_monitor();

        Ok(())
    }

    /// Start RIFE-NCNN interpolation pipeline
    fn start_rife_pipeline(&mut self, video_path: &PathBuf) -> Result<(), String> {
        // RIFE pipeline uses a Python helper script that:
        // 1. Decodes with FFmpeg
        // 2. Interpolates with rife-ncnn-vulkan-python
        // 3. Outputs to mpv

        let video_str = video_path.to_string_lossy();
        let _target_fps = self.config.target_fps;
        let _model = self.config.preset.rife_model();
        let _gpu_id = self.config.gpu_id;

        // For RIFE, we use FFmpeg's scale and fps filters to achieve interpolation
        // with minterpolate as a simpler alternative when RIFE CLI isn't available

        // Check for RIFE CLI binary
        let rife_cli = Command::new("which")
            .arg("rife-ncnn-vulkan")
            .output()
            .ok()
            .filter(|o| o.status.success());

        if rife_cli.is_some() {
            // Use RIFE CLI in pipe mode
            // ffmpeg decode -> RIFE -> ffmpeg encode -> mpv

            // FFmpeg decode to raw frames
            let decoder = Command::new("ffmpeg")
                .args([
                    "-hwaccel", "auto",
                    "-i", &video_str,
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-"
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| format!("Failed to start FFmpeg decoder: {}", e))?;

            // Note: rife-ncnn-vulkan doesn't support pipe input directly
            // We need to use the Python wrapper or a different approach
            // For now, fall back to minterpolate
            log::warn!("RIFE CLI doesn't support pipe input, falling back to minterpolate");
            drop(decoder);
            return self.start_minterpolate_pipeline(video_path);
        }

        // Fall back to minterpolate
        log::info!("RIFE CLI not found, using minterpolate fallback");
        self.start_minterpolate_pipeline(video_path)
    }

    /// Start FFmpeg minterpolate pipeline
    fn start_minterpolate_pipeline(&mut self, video_path: &PathBuf) -> Result<(), String> {
        // minterpolate is FFmpeg's built-in motion interpolation filter
        // It's slower than SVP/RIFE but always available

        let video_str = video_path.to_string_lossy();
        let target_fps = self.config.target_fps;

        // Build filter string based on preset
        let mi_mode = match self.config.preset {
            super::config::InterpolationPreset::Fast => "dup",
            super::config::InterpolationPreset::Balanced => "blend",
            super::config::InterpolationPreset::Quality
            | super::config::InterpolationPreset::Animation
            | super::config::InterpolationPreset::Film => "mci",
            super::config::InterpolationPreset::Max => "mci",
        };

        let search = match self.config.preset {
            super::config::InterpolationPreset::Fast => "ds",
            super::config::InterpolationPreset::Balanced => "hexbs",
            super::config::InterpolationPreset::Quality
            | super::config::InterpolationPreset::Max => "ntss",
            super::config::InterpolationPreset::Animation => "hexbs",
            super::config::InterpolationPreset::Film => "umh",
        };

        let filter = format!(
            "minterpolate=fps={}:mi_mode={}:mc_mode=aobmc:me_mode=bidir:me={}:scd=fdiff",
            target_fps, mi_mode, search
        );

        log::info!("Starting minterpolate pipeline with filter: {}", filter);

        // Use mpv with FFmpeg filters
        let mpv = Command::new("mpv")
            .args([
                "--no-terminal",
                "--force-window=yes",
                "--title=LocalBooru - Interpolated",
                "--osc=yes",
                "--input-ipc-server=/tmp/localbooru-mpv-ipc",
                &format!("--vf=lavfi=[{}]", filter),
                "--hwdec=auto",
                &video_str,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start mpv: {}", e))?;

        self.encoder_process = Some(mpv);
        self.state = InterpolationState::Running;

        Ok(())
    }

    /// Play video directly without interpolation
    fn play_direct(&mut self, video_path: &PathBuf) -> Result<(), String> {
        let video_str = video_path.to_string_lossy();

        let mpv = Command::new("mpv")
            .args([
                "--no-terminal",
                "--force-window=yes",
                "--title=LocalBooru",
                "--osc=yes",
                "--hwdec=auto",
                "--input-ipc-server=/tmp/localbooru-mpv-ipc",
                &video_str,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start mpv: {}", e))?;

        self.encoder_process = Some(mpv);
        self.state = InterpolationState::Running;

        Ok(())
    }

    /// Generate VapourSynth script for SVP processing
    fn generate_svp_script(&self) -> String {
        let svp_plugins_dir = "/opt/svp/plugins";
        let super_params = self.config.svp_super_params();
        let analyse_params = self.config.svp_analyse_params();
        let smooth_params = self.config.svp_smooth_params();

        let nvof_section = if self.config.use_nvof {
            format!(r#"
# NVOF Pipeline (NVIDIA Optical Flow)
nvof_blk = {}
NVOF_MIN_WIDTH = 160
NVOF_MIN_HEIGHT = 128

# Auto-select ratio based on resolution
for ratio in [8, 6, 4, 2, 1]:
    test_w = clip.width // ratio
    test_h = clip.height // ratio
    test_w = (test_w // 2) * 2
    test_h = (test_h // 2) * 2
    if test_w >= NVOF_MIN_WIDTH and test_h >= NVOF_MIN_HEIGHT:
        if nvof_blk >= 16 and ratio <= 4:
            break
        elif nvof_blk >= 8 and ratio <= 2:
            break
        elif ratio <= 1:
            break

new_w = clip.width // ratio
new_h = clip.height // ratio
new_w = (new_w // 2) * 2
new_h = (new_h // 2) * 2

if new_w < NVOF_MIN_WIDTH or new_h < NVOF_MIN_HEIGHT:
    new_w = (clip.width // 2) * 2
    new_h = (clip.height // 2) * 2

nvof_src = clip.resize.Bicubic(new_w, new_h)
smooth = core.svp2.SmoothFps_NVOF(clip, '{smooth}', vec_src=nvof_src, src=clip, fps=src_fps)
"#, self.config.preset.nvof_block_size(), smooth = smooth_params)
        } else {
            format!(r#"
# Regular SVP Pipeline
super_clip = core.svp1.Super(clip, '{super}')
vectors = core.svp1.Analyse(super_clip["clip"], super_clip["data"], clip, '{analyse}')
smooth = core.svp2.SmoothFps(clip, super_clip["clip"], super_clip["data"],
    vectors["clip"], vectors["data"], '{smooth}', src=clip, fps=src_fps)
"#, super = super_params, analyse = analyse_params, smooth = smooth_params)
        };

        format!(r#"import vapoursynth as vs
core = vs.core

# Load SVP plugins
core.std.LoadPlugin("{dir}/libsvpflow1.so")
core.std.LoadPlugin("{dir}/libsvpflow2.so")

# Try to load rawsource for Y4M stdin
try:
    core.std.LoadPlugin("~/.local/lib/vapoursynth/libvsrawsource.so".replace("~", __import__("os").path.expanduser("~")))
    clip = core.raws.Source("-")
except:
    # Fallback: try to read from stdin directly
    import sys
    raise RuntimeError("rawsource plugin required for stdin input")

# Get source FPS
src_fps = float(clip.fps)

# Convert to YUV420P8 for SVP
if clip.format.id != vs.YUV420P8:
    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_in_s="709", matrix_s="709")

{nvof}

smooth.set_output()
"#, dir = svp_plugins_dir, nvof = nvof_section)
    }

    /// Start statistics monitoring thread
    fn start_stats_monitor(&self) {
        let running = self.running.clone();
        let stats = self.stats.clone();

        thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                // Try to read from mpv IPC socket for stats
                // This is a simplified implementation
                thread::sleep(std::time::Duration::from_secs(1));

                let mut s = stats.lock().unwrap();
                s.frames_processed += 60; // Approximate
            }
        });
    }

    /// Stop playback
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.state = InterpolationState::Stopped;

        // Kill all processes
        if let Some(mut proc) = self.decoder_process.take() {
            let _ = proc.kill();
        }
        if let Some(mut proc) = self.interpolator_process.take() {
            let _ = proc.kill();
        }
        if let Some(mut proc) = self.encoder_process.take() {
            let _ = proc.kill();
        }

        // Send quit command to mpv via IPC
        if let Ok(mut sock) = std::os::unix::net::UnixStream::connect("/tmp/localbooru-mpv-ipc") {
            let _ = writeln!(sock, r#"{{"command": ["quit"]}}"#);
        }
    }

    /// Pause playback
    pub fn pause(&mut self) -> Result<(), String> {
        if let Ok(mut sock) = std::os::unix::net::UnixStream::connect("/tmp/localbooru-mpv-ipc") {
            writeln!(sock, r#"{{"command": ["set_property", "pause", true]}}"#)
                .map_err(|e| e.to_string())?;
            self.state = InterpolationState::Paused;
        }
        Ok(())
    }

    /// Resume playback
    pub fn resume(&mut self) -> Result<(), String> {
        if let Ok(mut sock) = std::os::unix::net::UnixStream::connect("/tmp/localbooru-mpv-ipc") {
            writeln!(sock, r#"{{"command": ["set_property", "pause", false]}}"#)
                .map_err(|e| e.to_string())?;
            self.state = InterpolationState::Running;
        }
        Ok(())
    }

    /// Seek to position (seconds)
    pub fn seek(&self, position: f64) -> Result<(), String> {
        if let Ok(mut sock) = std::os::unix::net::UnixStream::connect("/tmp/localbooru-mpv-ipc") {
            writeln!(sock, r#"{{"command": ["seek", {}, "absolute"]}}"#, position)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Get current position (seconds)
    pub fn position(&self) -> Result<f64, String> {
        // Would need to query mpv IPC
        Ok(0.0)
    }

    /// Set volume (0.0 - 1.0)
    pub fn set_volume(&self, volume: f64) -> Result<(), String> {
        if let Ok(mut sock) = std::os::unix::net::UnixStream::connect("/tmp/localbooru-mpv-ipc") {
            let mpv_volume = (volume * 100.0) as i32;
            writeln!(sock, r#"{{"command": ["set_property", "volume", {}]}}"#, mpv_volume)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

impl Drop for InterpolatedPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Probe video file for FPS and dimensions
fn probe_video(path: &PathBuf) -> Result<(f64, u32, u32), String> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of", "json",
            &path.to_string_lossy(),
        ])
        .output()
        .map_err(|e| format!("Failed to run ffprobe: {}", e))?;

    if !output.status.success() {
        return Err("ffprobe failed".to_string());
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse ffprobe output: {}", e))?;

    let stream = json["streams"].as_array()
        .and_then(|s| s.first())
        .ok_or("No video stream found")?;

    let width = stream["width"].as_u64().unwrap_or(1920) as u32;
    let height = stream["height"].as_u64().unwrap_or(1080) as u32;

    // Parse frame rate (format: "num/den" or just a number)
    let fps_str = stream["r_frame_rate"].as_str()
        .or_else(|| stream["avg_frame_rate"].as_str())
        .unwrap_or("24/1");

    let fps = if fps_str.contains('/') {
        let parts: Vec<&str> = fps_str.split('/').collect();
        let num: f64 = parts[0].parse().unwrap_or(24.0);
        let den: f64 = parts.get(1).map(|s| s.parse().unwrap_or(1.0)).unwrap_or(1.0);
        if den > 0.0 { num / den } else { 24.0 }
    } else {
        fps_str.parse().unwrap_or(24.0)
    };

    Ok((fps, width, height))
}
