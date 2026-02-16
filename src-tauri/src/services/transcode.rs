//! HLS Transcoding Service
//!
//! Spawns FFmpeg processes to transcode video files into HLS segments.
//! Supports hardware-accelerated encoding (NVENC) with automatic fallback
//! to software encoding (libx264).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;

use dashmap::DashMap;
use serde::Deserialize;
use tokio::process::{Child, Command};

/// Cached hardware encoder availability (detected once).
static HW_ENCODERS: OnceLock<HwEncoders> = OnceLock::new();

struct HwEncoders {
    nvenc: bool,
}

fn detect_hw_encoders() -> &'static HwEncoders {
    HW_ENCODERS.get_or_init(|| {
        let mut hw = HwEncoders { nvenc: false };
        if let Ok(output) = std::process::Command::new("ffmpeg")
            .args(["-encoders"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                hw.nvenc = stdout.contains("h264_nvenc");
                log::info!(
                    "[Transcode] Hardware encoders: NVENC={}",
                    hw.nvenc
                );
            }
        }
        hw
    })
}

/// Video information detected via ffprobe.
struct VideoInfo {
    width: u32,
    height: u32,
    duration: f64,
    avg_fps: f64,
    has_audio: bool,
}

/// Detect video info using ffprobe.
async fn detect_video_info(path: &str) -> VideoInfo {
    let mut info = VideoInfo {
        width: 1920,
        height: 1080,
        duration: 0.0,
        avg_fps: 30.0,
        has_audio: true,
    };

    // Get video stream info
    if let Ok(output) = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,avg_frame_rate",
            "-of", "csv=p=0",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
    {
        if output.status.success() {
            let line = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = line.trim().split(',').collect();
            if parts.len() >= 3 {
                info.width = parts[0].parse().unwrap_or(1920);
                info.height = parts[1].parse().unwrap_or(1080);
                info.duration = parts[2].parse().unwrap_or(0.0);
                if parts.len() >= 4 {
                    if let Some((num, den)) = parts[3].split_once('/') {
                        let n: f64 = num.parse().unwrap_or(0.0);
                        let d: f64 = den.parse().unwrap_or(1.0);
                        if d > 0.0 {
                            info.avg_fps = n / d;
                        }
                    }
                }
            }
        }
    }

    // If stream-level duration is missing (MKV etc.), query format-level
    if info.duration <= 0.0 {
        if let Ok(output) = Command::new("ffprobe")
            .args([
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
            ])
            .arg(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
        {
            if output.status.success() {
                let dur_str = String::from_utf8_lossy(&output.stdout);
                info.duration = dur_str.trim().parse().unwrap_or(0.0);
            }
        }
    }

    // Check for audio stream
    if let Ok(output) = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
    {
        if output.status.success() {
            info.has_audio = !String::from_utf8_lossy(&output.stdout).trim().is_empty();
        }
    }

    log::info!(
        "[Transcode] Detected: {}x{}, {:.1}s, {:.2}fps, audio={}",
        info.width, info.height, info.duration, info.avg_fps, info.has_audio
    );

    info
}

/// A single active transcode stream with its FFmpeg process and HLS output directory.
pub struct TranscodeStream {
    pub stream_id: String,
    pub hls_dir: PathBuf,
    temp_dir: PathBuf,
    process: Option<Child>,
    pub playlist_ready: bool,
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub start_position: f64,
}

impl TranscodeStream {
    fn stop(&mut self) {
        if let Some(ref mut child) = self.process {
            if let Some(pid) = child.id() {
                crate::addons::sidecar::kill_process(pid);
            }
            let _ = child.start_kill();
        }
        self.process = None;

        // Clean up temp directory
        if self.temp_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.temp_dir);
        }
    }
}

impl Drop for TranscodeStream {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Quality preset for transcoding.
#[derive(Debug, Deserialize, Default)]
pub struct QualityPreset {
    pub resolution: Option<String>, // e.g., "720p", "1080p"
    pub bitrate: Option<String>,    // e.g., "4M", "1536K"
}

impl QualityPreset {
    /// Parse resolution string into (width, height) scaling parameters.
    fn target_resolution(&self) -> Option<(u32, u32)> {
        match self.resolution.as_deref() {
            Some("480p") => Some((854, 480)),
            Some("720p") => Some((1280, 720)),
            Some("1080p") => Some((1920, 1080)),
            Some("1440p") => Some((2560, 1440)),
            Some("4k") | Some("2160p") => Some((3840, 2160)),
            _ => None,
        }
    }
}

/// Manages active transcoding streams.
pub struct TranscodeManager {
    streams: DashMap<String, TranscodeStream>,
}

impl TranscodeManager {
    pub fn new() -> Self {
        // Eagerly detect hw encoders at startup
        detect_hw_encoders();
        Self {
            streams: DashMap::new(),
        }
    }

    /// Start a new transcode stream. Returns stream info on success.
    pub async fn start_stream(
        &self,
        file_path: &str,
        start_position: f64,
        quality: &QualityPreset,
        force_cfr: bool,
    ) -> Result<TranscodeStreamInfo, String> {
        // Stop any existing streams first
        self.stop_all();

        let stream_id = uuid::Uuid::new_v4().to_string();

        // Detect video info
        let video_info = detect_video_info(file_path).await;

        // Create temp directory for HLS segments
        let temp_dir = std::env::temp_dir().join(format!("transcode_{}", &stream_id[..8]));
        let hls_dir = temp_dir.join("hls");
        std::fs::create_dir_all(&hls_dir)
            .map_err(|e| format!("Failed to create temp dir: {}", e))?;

        // Build FFmpeg command
        let cmd = build_ffmpeg_command(
            file_path,
            &hls_dir,
            start_position,
            quality,
            force_cfr,
            &video_info,
        );

        log::info!("[Transcode {}] Starting FFmpeg: {}", stream_id, cmd.join(" "));

        // Spawn FFmpeg
        let child = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn FFmpeg: {}", e))?;

        let mut stream = TranscodeStream {
            stream_id: stream_id.clone(),
            hls_dir: hls_dir.clone(),
            temp_dir,
            process: Some(child),
            playlist_ready: false,
            duration: video_info.duration,
            width: video_info.width,
            height: video_info.height,
            start_position,
        };

        // Wait for playlist + first segment
        let playlist_path = hls_dir.join("playlist.m3u8");
        let segment_path = hls_dir.join("segment_0.ts");

        for attempt in 0..200 {
            // 20 seconds max (200 * 100ms)
            if playlist_path.exists() && segment_path.exists() {
                if let Ok(meta) = std::fs::metadata(&segment_path) {
                    if meta.len() > 1000 {
                        stream.playlist_ready = true;
                        log::info!(
                            "[Transcode {}] Ready after {:.1}s",
                            stream_id,
                            attempt as f64 * 0.1
                        );
                        break;
                    }
                }
            }

            // Check if FFmpeg already died
            if let Some(ref mut proc) = stream.process {
                if let Ok(Some(status)) = proc.try_wait() {
                    if !status.success() {
                        let _ = stream.process.take();
                        return Err("FFmpeg exited early".into());
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        if !stream.playlist_ready {
            stream.stop();
            return Err("Timeout waiting for HLS playlist".into());
        }

        let info = TranscodeStreamInfo {
            stream_id: stream_id.clone(),
            stream_url: format!(
                "/api/settings/transcode/stream/{}/playlist.m3u8",
                stream_id
            ),
            duration: video_info.duration,
            start_position,
            source_resolution: Resolution {
                width: video_info.width,
                height: video_info.height,
            },
        };

        self.streams.insert(stream_id, stream);
        Ok(info)
    }

    /// Get the HLS directory for a stream.
    pub fn get_stream_hls_dir(&self, stream_id: &str) -> Option<PathBuf> {
        self.streams.get(stream_id).map(|s| s.hls_dir.clone())
    }

    /// Stop all active streams.
    pub fn stop_all(&self) {
        for mut entry in self.streams.iter_mut() {
            entry.value_mut().stop();
        }
        self.streams.clear();
    }
}

/// Information returned when a transcode stream starts successfully.
#[derive(serde::Serialize)]
pub struct TranscodeStreamInfo {
    pub stream_id: String,
    pub stream_url: String,
    pub duration: f64,
    pub start_position: f64,
    pub source_resolution: Resolution,
}

#[derive(serde::Serialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

/// Build the FFmpeg command line for HLS transcoding.
fn build_ffmpeg_command(
    file_path: &str,
    hls_dir: &Path,
    start_position: f64,
    quality: &QualityPreset,
    force_cfr: bool,
    video_info: &VideoInfo,
) -> Vec<String> {
    let mut cmd: Vec<String> = vec!["ffmpeg".into(), "-y".into()];

    // Hybrid seeking: input seek (fast) + output seek (accurate)
    let mut effective_start = start_position;
    if video_info.duration > 0.0 && effective_start >= video_info.duration {
        effective_start = (video_info.duration - 1.0).max(0.0);
    }

    let (input_seek, output_seek) = if effective_start > 2.0 {
        (effective_start - 2.0, 2.0)
    } else {
        (0.0, effective_start)
    };

    if input_seek > 0.0 {
        cmd.extend(["-ss".into(), format!("{:.3}", input_seek)]);
    }

    cmd.extend(["-i".into(), file_path.into()]);

    if output_seek > 0.0 {
        cmd.extend(["-ss".into(), format!("{:.3}", output_seek)]);
    }

    // Video filter chain
    let mut vf_filters = Vec::new();

    if let Some((width, _height)) = quality.target_resolution() {
        vf_filters.push(format!("scale={}:-2:flags=lanczos", width));
    }

    // Pad to multiple of 2 and ensure yuv420p
    vf_filters.push("pad=ceil(iw/2)*2:ceil(ih/2)*2".into());
    vf_filters.push("format=yuv420p".into());

    if !vf_filters.is_empty() {
        cmd.extend(["-vf".into(), vf_filters.join(",")]);
    }

    // VFR to CFR conversion
    if force_cfr && video_info.avg_fps > 0.0 {
        cmd.extend(["-r".into(), format!("{}", video_info.avg_fps)]);
        cmd.extend(["-vsync".into(), "cfr".into()]);
    }

    // Force keyframes every 2 seconds
    cmd.extend(["-force_key_frames".into(), "expr:gte(t,n_forced*2)".into()]);

    let gop_size = if video_info.avg_fps > 0.0 {
        (video_info.avg_fps * 2.0) as u32
    } else {
        60
    };

    let hw = detect_hw_encoders();
    if hw.nvenc {
        cmd.extend([
            "-c:v".into(), "h264_nvenc".into(),
            "-preset".into(), "p4".into(),
            "-g".into(), gop_size.to_string(),
            "-keyint_min".into(), gop_size.to_string(),
        ]);
    } else {
        cmd.extend([
            "-c:v".into(), "libx264".into(),
            "-preset".into(), "ultrafast".into(),
            "-tune".into(), "zerolatency".into(),
            "-g".into(), gop_size.to_string(),
            "-keyint_min".into(), gop_size.to_string(),
        ]);
    }

    // Bitrate
    if let Some(ref bitrate) = quality.bitrate {
        cmd.extend(["-b:v".into(), bitrate.clone()]);
    } else {
        cmd.extend(["-crf".into(), "23".into()]);
    }

    // Audio
    if video_info.has_audio {
        cmd.extend([
            "-c:a".into(), "aac".into(),
            "-ar".into(), "48000".into(),
            "-ac".into(), "2".into(),
            "-b:a".into(), "192k".into(),
        ]);
    } else {
        cmd.push("-an".into());
    }

    // HLS output
    cmd.extend([
        "-f".into(), "hls".into(),
        "-hls_time".into(), "2".into(),
        "-hls_list_size".into(), "0".into(),
        "-hls_flags".into(), "append_list".into(),
        "-hls_segment_filename".into(),
        hls_dir.join("segment_%d.ts").to_string_lossy().into(),
        hls_dir.join("playlist.m3u8").to_string_lossy().into(),
    ]);

    cmd
}
