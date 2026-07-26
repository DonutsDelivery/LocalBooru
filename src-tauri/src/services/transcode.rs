//! HLS Transcoding Service
//!
//! Spawns FFmpeg processes to transcode video files into HLS segments.
//! Supports hardware-accelerated encoding (NVENC) with automatic fallback
//! to software encoding (libx264).
//! Audio peak safety attenuates overly loud sources via pre-scan volumedetect.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use dashmap::DashMap;
use regex::Regex;
use serde::Deserialize;
use tokio::process::{Child, Command};
use tokio::sync::{watch, RwLock as TokioRwLock};

/// Cached hardware capability detection (done once at startup).
static HW_CAPS: OnceLock<HwCaps> = OnceLock::new();

struct HwCaps {
    nvenc: bool,
    cuda_hwaccel: bool,
    scale_cuda: bool,
}

#[cfg(target_os = "windows")]
fn run_ffmpeg_probe(args: &[&str]) -> std::io::Result<std::process::Output> {
    use std::os::windows::process::CommandExt;
    let mut command = std::process::Command::new("ffmpeg");
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(0x08000000); // CREATE_NO_WINDOW
    command.output()
}

#[cfg(not(target_os = "windows"))]
fn run_ffmpeg_probe(args: &[&str]) -> std::io::Result<std::process::Output> {
    std::process::Command::new("ffmpeg")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
}

impl HwCaps {
    /// Full GPU pipeline available: CUDA decode → GPU scale → NVENC encode
    fn full_gpu(&self) -> bool {
        self.nvenc && self.cuda_hwaccel && self.scale_cuda
    }
}

fn detect_hw_caps() -> &'static HwCaps {
    HW_CAPS.get_or_init(|| {
        let mut hw = HwCaps {
            nvenc: false,
            cuda_hwaccel: false,
            scale_cuda: false,
        };

        // Check encoders (NVENC)
        if let Ok(output) = run_ffmpeg_probe(&["-encoders"]) {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                hw.nvenc = stdout.contains("h264_nvenc");
            }
        }

        // Check hwaccels (cuda)
        if let Ok(output) = run_ffmpeg_probe(&["-hwaccels"]) {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                hw.cuda_hwaccel = stdout.contains("cuda");
            }
        }

        // Check filters (scale_cuda)
        if let Ok(output) = run_ffmpeg_probe(&["-filters"]) {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                hw.scale_cuda = stdout.contains("scale_cuda");
            }
        }

        log::info!(
            "[Transcode] Hardware caps: NVENC={}, CUDA hwaccel={}, scale_cuda={}, full_gpu={}",
            hw.nvenc,
            hw.cuda_hwaccel,
            hw.scale_cuda,
            hw.full_gpu()
        );
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
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration,avg_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
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
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
            ])
            .arg(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
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
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
    {
        if output.status.success() {
            info.has_audio = !String::from_utf8_lossy(&output.stdout).trim().is_empty();
        }
    }

    log::info!(
        "[Transcode] Detected: {}x{}, {:.1}s, {:.2}fps, audio={}",
        info.width,
        info.height,
        info.duration,
        info.avg_fps,
        info.has_audio
    );

    info
}

/// Detect attenuation needed to keep the loudest peak at or below -2 dB.
/// Uses ffmpeg volumedetect on the first 15 seconds for a quick estimate.
/// Returns a non-positive dB value. Quiet sources are never amplified.
pub async fn detect_audio_gain(path: &str) -> Option<f64> {
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            path,
            "-vn", // no video, audio only
            "-af",
            "volumedetect",
            "-t",
            "15", // analyze first 15 seconds
            "-f",
            "null",
            "-",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let re = Regex::new(r"max_volume:\s*([-\d.]+)\s*dB").ok()?;
    let max_volume: f64 = re.captures(&stderr)?.get(1)?.as_str().parse().ok()?;

    // Gain needed to bring peak to -2 dB, but never amplify quiet sources.
    let gain_db = (-2.0 - max_volume).min(0.0);

    // Clamp: don't attenuate more than -12 dB.
    Some(gain_db.clamp(-12.0, 0.0))
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
    transition_epoch: AtomicU64,
    transition_lock: Mutex<()>,
    lifecycle: TokioRwLock<()>,
    shutdown_signal: watch::Sender<bool>,
    shutting_down: AtomicBool,
}

impl TranscodeManager {
    pub fn new() -> Self {
        // Eagerly detect hw encoders at startup
        detect_hw_caps();
        Self {
            streams: DashMap::new(),
            transition_epoch: AtomicU64::new(0),
            transition_lock: Mutex::new(()),
            lifecycle: TokioRwLock::new(()),
            shutdown_signal: watch::channel(false).0,
            shutting_down: AtomicBool::new(false),
        }
    }

    fn claim_transition(&self) -> u64 {
        self.transition_epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn owns_transition(&self, epoch: u64) -> bool {
        !self.shutting_down.load(Ordering::SeqCst)
            && self.transition_epoch.load(Ordering::SeqCst) == epoch
    }

    fn transition_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.transition_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn claim_transition_and_stop(&self) -> Result<u64, String> {
        let _guard = self.transition_guard();
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err("Application shutdown is in progress".into());
        }
        let epoch = self.claim_transition();
        self.stop_registered_streams();
        Ok(epoch)
    }

    fn publish_stream_if_owned(
        &self,
        epoch: u64,
        stream_id: String,
        stream: TranscodeStream,
    ) -> Result<(), TranscodeStream> {
        let _guard = self.transition_guard();
        if self.shutting_down.load(Ordering::SeqCst) || !self.owns_transition(epoch) {
            return Err(stream);
        }
        self.streams.insert(stream_id, stream);
        Ok(())
    }

    fn stop_registered_streams(&self) {
        for mut entry in self.streams.iter_mut() {
            entry.value_mut().stop();
        }
        self.streams.clear();
    }

    /// Start a new transcode stream with optional frame interpolation.
    ///
    /// When `target_fps` is `Some(fps)`, FFmpeg's minterpolate filter is applied
    /// to smoothly increase the frame rate to the given target.
    pub async fn start_stream(
        &self,
        file_path: &str,
        start_position: f64,
        quality: &QualityPreset,
        force_cfr: bool,
    ) -> Result<TranscodeStreamInfo, String> {
        self.start_stream_inner(file_path, start_position, quality, force_cfr, None)
            .await
    }

    /// Start a transcode stream with minterpolate frame interpolation.
    pub async fn start_interpolated_stream(
        &self,
        file_path: &str,
        start_position: f64,
        quality: &QualityPreset,
        target_fps: u32,
    ) -> Result<TranscodeStreamInfo, String> {
        self.start_stream_inner(file_path, start_position, quality, true, Some(target_fps))
            .await
    }

    async fn start_stream_inner(
        &self,
        file_path: &str,
        start_position: f64,
        quality: &QualityPreset,
        force_cfr: bool,
        target_fps: Option<u32>,
    ) -> Result<TranscodeStreamInfo, String> {
        let _lifecycle = self.lifecycle.read().await;
        let mut shutdown = self.shutdown_signal.subscribe();
        let transition_epoch = self.claim_transition_and_stop()?;

        let stream_id = uuid::Uuid::new_v4().to_string();

        // Detect video info and audio gain
        let video_info = tokio::select! {
            info = detect_video_info(file_path) => info,
            _ = shutdown.changed() => {
                return Err("Application shutdown is in progress".into());
            }
        };
        if !self.owns_transition(transition_epoch) {
            return Err("Transcode start was superseded".into());
        }
        let audio_gain_db = if video_info.has_audio {
            tokio::select! {
                gain = detect_audio_gain(file_path) => gain,
                _ = shutdown.changed() => {
                    return Err("Application shutdown is in progress".into());
                }
            }
        } else {
            None
        };
        if !self.owns_transition(transition_epoch) {
            return Err("Transcode start was superseded".into());
        }

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
            target_fps,
            audio_gain_db,
        );

        log::info!(
            "[Transcode {}] Starting FFmpeg: {}",
            stream_id,
            cmd.join(" ")
        );

        // Spawn FFmpeg — on Linux, set PR_SET_PDEATHSIG so the child is killed
        // automatically when the parent process exits (even via SIGKILL / process::exit).
        // On Windows, hide the console window to prevent cmd.exe popup flashes.
        let mut ffmpeg_cmd = Command::new(&cmd[0]);
        ffmpeg_cmd
            .args(&cmd[1..])
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        #[cfg(target_os = "linux")]
        unsafe {
            ffmpeg_cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                Ok(())
            });
        }

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            ffmpeg_cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let child = ffmpeg_cmd
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
        if !self.owns_transition(transition_epoch) {
            stream.stop();
            return Err("Transcode start was superseded".into());
        }

        // Wait for playlist + first segment
        let playlist_path = hls_dir.join("playlist.m3u8");
        let segment_path = hls_dir.join("segment_0.ts");

        for attempt in 0..200 {
            if !self.owns_transition(transition_epoch) {
                stream.stop();
                return Err("Transcode start was superseded".into());
            }
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
            stream_url: format!("/api/settings/transcode/stream/{}/playlist.m3u8", stream_id),
            duration: video_info.duration,
            start_position,
            source_resolution: Resolution {
                width: video_info.width,
                height: video_info.height,
            },
        };

        match self.publish_stream_if_owned(transition_epoch, stream_id, stream) {
            Ok(()) => Ok(info),
            Err(mut stream) => {
                stream.stop();
                Err("Transcode start was superseded".into())
            }
        }
    }

    /// Get the HLS directory for a stream.
    pub fn get_stream_hls_dir(&self, stream_id: &str) -> Option<PathBuf> {
        self.streams.get(stream_id).map(|s| s.hls_dir.clone())
    }

    /// Stop a single active stream.
    pub fn stop_stream(&self, stream_id: &str) -> bool {
        if let Some((_, mut stream)) = self.streams.remove(stream_id) {
            stream.stop();
            true
        } else {
            false
        }
    }

    /// Stop all active streams and invalidate starts that are still preparing.
    pub fn stop_all(&self) {
        let _guard = self.transition_guard();
        self.claim_transition();
        self.stop_registered_streams();
    }

    /// Permanently reject new streams and stop all active or preparing transcodes.
    pub async fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        self.shutdown_signal.send_replace(true);
        let _lifecycle = self.lifecycle.write().await;
        self.stop_all();
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
///
/// Uses a full GPU pipeline (CUDA decode → scale_cuda → NVENC encode) when
/// hardware acceleration is available, keeping CPU usage near zero.
/// Falls back to CPU filters only when minterpolate (frame interpolation) is
/// requested, since that filter has no GPU equivalent.
fn build_ffmpeg_command(
    file_path: &str,
    hls_dir: &Path,
    start_position: f64,
    quality: &QualityPreset,
    force_cfr: bool,
    video_info: &VideoInfo,
    target_fps: Option<u32>,
    audio_gain_db: Option<f64>,
) -> Vec<String> {
    let hw = detect_hw_caps();

    // minterpolate is CPU-only — if requested, we must use the CPU decode path
    let needs_minterpolate = target_fps
        .map(|fps| (fps as f64) > video_info.avg_fps + 1.0)
        .unwrap_or(false);

    let use_gpu_pipeline = hw.full_gpu() && !needs_minterpolate;

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

    // Hardware-accelerated decoding: decode directly to GPU memory
    if use_gpu_pipeline {
        cmd.extend([
            "-hwaccel".into(),
            "cuda".into(),
            "-hwaccel_output_format".into(),
            "cuda".into(),
        ]);
    }

    cmd.extend(["-i".into(), file_path.into()]);

    if output_seek > 0.0 {
        cmd.extend(["-ss".into(), format!("{:.3}", output_seek)]);
    }

    if use_gpu_pipeline {
        // ── Full GPU filter chain ──
        // Frames stay in GPU memory: decode → scale_cuda → NVENC encode
        let mut vf_filters = Vec::new();

        if let Some((width, _height)) = quality.target_resolution() {
            // scale_cuda: -2 ensures even height, width is already even from our presets
            vf_filters.push(format!("scale_cuda={}:-2", width));
        } else {
            // No resolution change — still ensure even dimensions for HLS
            vf_filters.push("scale_cuda=trunc(iw/2)*2:trunc(ih/2)*2".into());
        }

        if !vf_filters.is_empty() {
            cmd.extend(["-vf".into(), vf_filters.join(",")]);
        }
    } else {
        // ── CPU filter chain (fallback, or when minterpolate is needed) ──
        let mut vf_filters = Vec::new();

        if let Some((width, _height)) = quality.target_resolution() {
            vf_filters.push(format!("scale={}:-2:flags=lanczos", width));
        }

        // Frame interpolation via minterpolate (CPU-only filter)
        if let Some(fps) = target_fps {
            if (fps as f64) > video_info.avg_fps + 1.0 {
                vf_filters.push(format!(
                    "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                    fps
                ));
            }
        }

        // Pad to multiple of 2 and ensure yuv420p
        vf_filters.push("pad=ceil(iw/2)*2:ceil(ih/2)*2".into());
        vf_filters.push("format=yuv420p".into());

        if !vf_filters.is_empty() {
            cmd.extend(["-vf".into(), vf_filters.join(",")]);
        }
    }

    // VFR to CFR conversion (use target fps if interpolating)
    let output_fps = target_fps
        .filter(|&fps| (fps as f64) > video_info.avg_fps + 1.0)
        .map(|fps| fps as f64)
        .unwrap_or(video_info.avg_fps);

    if force_cfr && output_fps > 0.0 {
        cmd.extend(["-r".into(), format!("{}", output_fps)]);
        cmd.extend(["-vsync".into(), "cfr".into()]);
    }

    // Force keyframes every 2 seconds
    cmd.extend(["-force_key_frames".into(), "expr:gte(t,n_forced*2)".into()]);

    let gop_size = if video_info.avg_fps > 0.0 {
        (video_info.avg_fps * 2.0) as u32
    } else {
        60
    };

    // Video encoder
    if hw.nvenc {
        cmd.extend([
            "-c:v".into(),
            "h264_nvenc".into(),
            // p1 = fastest preset (lowest latency, ideal for real-time streaming)
            "-preset".into(),
            "p1".into(),
            "-g".into(),
            gop_size.to_string(),
            "-keyint_min".into(),
            gop_size.to_string(),
        ]);
    } else {
        cmd.extend([
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            "ultrafast".into(),
            "-tune".into(),
            "zerolatency".into(),
            "-g".into(),
            gop_size.to_string(),
            "-keyint_min".into(),
            gop_size.to_string(),
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
            "-c:a".into(),
            "aac".into(),
            "-ar".into(),
            "48000".into(),
            "-ac".into(),
            "2".into(),
            "-b:a".into(),
            "192k".into(),
        ]);

        // Attenuate peaks above -2 dB only. Never boost quiet sources.
        if let Some(gain_db) = audio_gain_db {
            if gain_db < -0.5 {
                cmd.extend(["-af".into(), format!("volume={}dB", gain_db)]);
                log::info!("[Transcode] Audio peak attenuation: {:+.1} dB", gain_db);
            }
        }
    } else {
        cmd.push("-an".into());
    }

    // HLS output
    cmd.extend([
        "-f".into(),
        "hls".into(),
        "-hls_time".into(),
        "2".into(),
        "-hls_list_size".into(),
        "0".into(),
        "-hls_flags".into(),
        "append_list".into(),
        "-hls_segment_filename".into(),
        hls_dir.join("segment_%d.ts").to_string_lossy().into(),
        hls_dir.join("playlist.m3u8").to_string_lossy().into(),
    ]);

    cmd
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn manager() -> TranscodeManager {
        TranscodeManager {
            streams: DashMap::new(),
            transition_epoch: AtomicU64::new(0),
            transition_lock: Mutex::new(()),
            lifecycle: TokioRwLock::new(()),
            shutdown_signal: watch::channel(false).0,
            shutting_down: AtomicBool::new(false),
        }
    }

    #[test]
    // AC: @reliable-stream-transitions ac-final-source-owner
    fn newest_start_owns_the_transcode_transition() {
        let manager = manager();
        let first = manager.claim_transition();
        let second = manager.claim_transition();

        assert!(!manager.owns_transition(first));
        assert!(manager.owns_transition(second));
    }

    #[test]
    // AC: @reliable-stream-transitions ac-stop-superseded-producer
    fn stop_invalidates_a_start_that_is_still_preparing() {
        let manager = manager();
        let preparing = manager.claim_transition();

        manager.stop_all();

        assert!(!manager.owns_transition(preparing));
    }

    // AC: @explicit-exit-process-cleanup ac-1
    // AC: @explicit-exit-process-cleanup ac-2
    #[tokio::test]
    async fn shutdown_rejects_new_transcode_transitions() {
        let manager = manager();

        manager.shutdown().await;

        assert!(manager.claim_transition_and_stop().is_err());
    }

    // AC: @explicit-exit-process-cleanup ac-1
    // AC: @explicit-exit-process-cleanup ac-2
    #[tokio::test]
    async fn shutdown_cancels_in_flight_transcode_before_waiting_for_it() {
        let manager = Arc::new(manager());
        let lifecycle = manager.lifecycle.read().await;
        let mut shutdown_signal = manager.shutdown_signal.subscribe();
        let shutdown_manager = manager.clone();
        let shutdown = tokio::spawn(async move {
            shutdown_manager.shutdown().await;
        });

        tokio::time::timeout(Duration::from_secs(1), shutdown_signal.changed())
            .await
            .expect("shutdown signal timed out")
            .expect("shutdown signal closed");
        assert!(*shutdown_signal.borrow());
        assert!(!shutdown.is_finished());

        drop(lifecycle);
        shutdown.await.expect("shutdown task failed");
    }

    #[test]
    // AC: @reliable-stream-transitions ac-stop-superseded-producer
    fn stopped_start_cannot_publish_after_cleanup() {
        let manager = manager();
        let preparing = manager.claim_transition_and_stop().unwrap();
        manager.stop_all();

        let stream = TranscodeStream {
            stream_id: "stale".into(),
            hls_dir: PathBuf::new(),
            temp_dir: PathBuf::new(),
            process: None,
            playlist_ready: true,
            duration: 1.0,
            width: 1,
            height: 1,
            start_position: 0.0,
        };

        assert!(manager
            .publish_stream_if_owned(preparing, "stale".into(), stream)
            .is_err());
        assert!(manager.streams.is_empty());
    }
}
