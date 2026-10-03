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

fn run_ffmpeg_probe(args: &[&str]) -> bool {
    let mut command = std::process::Command::new(crate::platform_paths::helper("ffmpeg"));
    command
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let Ok(mut child) = command.spawn() else {
        return false;
    };
    for _ in 0..50 {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    false
}

impl HwCaps {
    /// Full GPU pipeline available: CUDA decode → GPU scale → NVENC encode
    fn full_gpu(&self) -> bool {
        self.nvenc && self.cuda_hwaccel && self.scale_cuda
    }
}

fn probe_hw_caps(mut probe: impl FnMut(&[&str]) -> bool) -> HwCaps {
    // Build-feature listings succeed even if the driver/device is unavailable.
    let nvenc = probe(&[
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=256x144:rate=1",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-f",
        "null",
        "-",
    ]);
    let cuda = nvenc
        && probe(&[
            "-hide_banner",
            "-loglevel",
            "error",
            "-init_hw_device",
            "cuda=gpu:0",
            "-filter_hw_device",
            "gpu",
            "-f",
            "lavfi",
            "-i",
            "color=size=256x144:rate=1",
            "-frames:v",
            "1",
            "-vf",
            "format=nv12,hwupload,scale_cuda=256:144",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-f",
            "null",
            "-",
        ]);
    HwCaps {
        nvenc,
        cuda_hwaccel: cuda,
        scale_cuda: cuda,
    }
}

fn detect_hw_caps() -> &'static HwCaps {
    HW_CAPS.get_or_init(|| {
        let hw = probe_hw_caps(run_ffmpeg_probe);
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
    if let Ok(output) = Command::new(crate::platform_paths::helper("ffprobe"))
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
        if let Ok(output) = Command::new(crate::platform_paths::helper("ffprobe"))
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
    if let Ok(output) = Command::new(crate::platform_paths::helper("ffprobe"))
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
    let output = Command::new(crate::platform_paths::helper("ffmpeg"))
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
    pub remux: bool,
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
        let audio_gain_db = if video_info.has_audio && !quality.remux {
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

        // Retry hardware startup once using a fully software pipeline. Codec,
        // decode or driver failures can still occur after the capability probe.
        let hw = detect_hw_caps();
        let software = HwCaps {
            nvenc: false,
            cuda_hwaccel: false,
            scale_cuda: false,
        };
        let mut commands = vec![build_ffmpeg_command(
            file_path,
            &hls_dir,
            start_position,
            quality,
            force_cfr,
            &video_info,
            target_fps,
            audio_gain_db,
            hw,
        )];
        if hw.nvenc && !quality.remux {
            commands.push(build_ffmpeg_command(
                file_path,
                &hls_dir,
                start_position,
                quality,
                force_cfr,
                &video_info,
                target_fps,
                audio_gain_db,
                &software,
            ));
        }
        let stream = TranscodeStream {
            stream_id: stream_id.clone(),
            hls_dir: hls_dir.clone(),
            temp_dir,
            process: None,
            playlist_ready: false,
            duration: video_info.duration,
            width: video_info.width,
            height: video_info.height,
            start_position,
        };
        let stream = start_hls_attempts(stream, &commands, quality.remux, || {
            self.owns_transition(transition_epoch)
        })
        .await?;

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

/// Start HLS with bounded retries, retaining stderr while continuously draining
/// the pipe so FFmpeg cannot stall once its diagnostics fill the pipe buffer.
async fn start_hls_attempts(
    mut stream: TranscodeStream,
    commands: &[Vec<String>],
    remux: bool,
    owns_transition: impl Fn() -> bool,
) -> Result<TranscodeStream, String> {
    use tokio::io::AsyncReadExt;
    let mut failures = Vec::new();
    for (attempt, cmd) in commands.iter().enumerate() {
        if !owns_transition() {
            return Err("Transcode start was superseded".into());
        }
        std::fs::create_dir_all(&stream.hls_dir)
            .map_err(|error| format!("Failed to create temp dir: {error}"))?;
        log::info!(
            "[Transcode {}] Starting FFmpeg: {}",
            stream.stream_id,
            cmd.join(" ")
        );
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
            ffmpeg_cmd.creation_flags(0x08000000);
        }
        let mut child = ffmpeg_cmd
            .spawn()
            .map_err(|error| format!("Failed to spawn FFmpeg: {error}"))?;
        let mut stderr = child.stderr.take().expect("piped FFmpeg stderr");
        let diagnostics = std::sync::Arc::new(Mutex::new(Vec::new()));
        let output = diagnostics.clone();
        let mut reader = tokio::spawn(async move {
            let mut buffer = [0u8; 4096];
            while let Ok(count) = stderr.read(&mut buffer).await {
                if count == 0 {
                    break;
                }
                let mut tail = output.lock().unwrap_or_else(|error| error.into_inner());
                tail.extend_from_slice(&buffer[..count]);
                if tail.len() > 8192 {
                    let excess = tail.len() - 8192;
                    tail.drain(..excess);
                }
            }
        });
        stream.process = Some(child);
        let playlist = stream.hls_dir.join("playlist.m3u8");
        let segment = stream.hls_dir.join(if remux {
            "segment_0.m4s"
        } else {
            "segment_0.ts"
        });
        let mut failure = "Timeout waiting for HLS playlist".to_string();
        for tick in 0..200 {
            if !owns_transition() {
                stream.stop();
                reader.abort();
                return Err("Transcode start was superseded".into());
            }
            if playlist.exists()
                && std::fs::metadata(&segment)
                    .map(|meta| meta.len() > 1000)
                    .unwrap_or(false)
            {
                stream.playlist_ready = true;
                log::info!(
                    "[Transcode {}] Ready after {:.1}s",
                    stream.stream_id,
                    tick as f64 * 0.1
                );
                return Ok(stream);
            }
            if let Some(process) = stream.process.as_mut() {
                if let Ok(Some(status)) = process.try_wait() {
                    // Let the stderr reader capture the final error before reporting it.
                    let _ = tokio::time::timeout(Duration::from_secs(1), &mut reader).await;
                    failure = format!("FFmpeg exited before HLS became ready ({status})");
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let detail = {
            let tail = diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            String::from_utf8_lossy(&tail).trim().to_string()
        };
        if !detail.is_empty() {
            failure.push_str(&format!(": {detail}"));
        }
        failures.push(failure);
        stream.stop();
        reader.abort();
        if attempt + 1 < commands.len() {
            log::warn!(
                "[Transcode {}] Hardware startup failed; retrying software encoding",
                stream.stream_id
            );
        }
    }
    Err(failures.join("\nSoftware retry: "))
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
    hw: &HwCaps,
) -> Vec<String> {
    // Packet-copy remux path (e.g. Apple HEVC passthrough): no re-encode.
    if quality.remux {
        return build_packet_copy_remux_command(file_path, hls_dir, start_position, video_info);
    }

    // minterpolate is CPU-only — if requested, we must use the CPU decode path
    let needs_minterpolate = target_fps
        .map(|fps| (fps as f64) > video_info.avg_fps + 1.0)
        .unwrap_or(false);

    let use_gpu_pipeline = hw.full_gpu() && !needs_minterpolate;

    let mut cmd: Vec<String> = vec![
        crate::platform_paths::helper("ffmpeg")
            .to_string_lossy()
            .into_owned(),
        "-y".into(),
    ];

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
        cmd.extend(["-fps_mode".into(), "cfr".into()]);
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

/// Packet-copy HLS remux for HEVC streams (Apple passthrough): wraps the
/// original HEVC bitstream in fMP4 segments without re-encoding.
fn build_packet_copy_remux_command(
    file_path: &str,
    hls_dir: &Path,
    start_position: f64,
    video_info: &VideoInfo,
) -> Vec<String> {
    let mut cmd = vec![
        crate::platform_paths::helper("ffmpeg")
            .to_string_lossy()
            .into_owned(),
        "-y".into(),
    ];
    let effective_start = if video_info.duration > 0.0 {
        start_position.clamp(0.0, (video_info.duration - 1.0).max(0.0))
    } else {
        start_position.max(0.0)
    };
    if effective_start > 0.0 {
        cmd.extend(["-ss".into(), format!("{effective_start:.3}")]);
    }
    cmd.extend([
        "-i".into(),
        file_path.into(),
        "-map".into(),
        "0:v:0".into(),
        "-map".into(),
        "0:a:0?".into(),
        "-c".into(),
        "copy".into(),
        "-bsf:v".into(),
        "hevc_mp4toannexb".into(),
        "-tag:v".into(),
        "hvc1".into(),
        "-avoid_negative_ts".into(),
        "make_zero".into(),
        "-f".into(),
        "hls".into(),
        "-hls_segment_type".into(),
        "fmp4".into(),
        "-hls_time".into(),
        "2".into(),
        "-hls_list_size".into(),
        "0".into(),
        "-hls_flags".into(),
        "append_list+independent_segments".into(),
        "-hls_fmp4_init_filename".into(),
        "init.mp4".into(),
        "-hls_segment_filename".into(),
        hls_dir.join("segment_%d.m4s").to_string_lossy().into(),
        hls_dir.join("playlist.m3u8").to_string_lossy().into(),
    ]);
    cmd
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn failed_device_probes_choose_software_for_encoded_qualities() {
        let hw = probe_hw_caps(|args| {
            assert!(args.contains(&"color=size=256x144:rate=1"));
            assert!(!args.contains(&"-encoders"));
            false
        });
        assert!(!hw.nvenc);
        assert!(!hw.full_gpu());
        let info = VideoInfo {
            width: 1920,
            height: 1080,
            duration: 5.0,
            avg_fps: 24.0,
            has_audio: false,
        };
        for preset in ["480p", "720p", "1080p", "1440p", "4k"] {
            let quality = QualityPreset {
                resolution: Some(preset.into()),
                ..Default::default()
            };
            let cmd = build_ffmpeg_command(
                "/synthetic/video.mp4",
                Path::new("/synthetic/hls"),
                0.0,
                &quality,
                true,
                &info,
                None,
                None,
                &hw,
            );
            assert!(cmd.iter().any(|arg| arg == "libx264"));
            assert!(cmd.iter().any(|arg| arg == "-fps_mode"));
            assert!(!cmd.iter().any(|arg| arg == "-vsync"));
            assert!(!cmd.iter().any(|arg| arg == "h264_nvenc" || arg == "cuda"));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_hardware_hls_retries_software_with_synthetic_video() {
        let root = std::env::temp_dir().join(format!("dmc-synthetic-hls-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("source.mp4");
        let generated = Command::new(crate::platform_paths::helper("ffmpeg"))
            .args([
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=24",
                "-t",
                "5",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&source)
            .output()
            .await
            .unwrap();
        assert!(
            generated.status.success(),
            "{}",
            String::from_utf8_lossy(&generated.stderr)
        );
        let output = root.join("output");
        let hls_dir = output.join("hls");
        let info = VideoInfo {
            width: 160,
            height: 90,
            duration: 5.0,
            avg_fps: 24.0,
            has_audio: false,
        };
        let software = HwCaps {
            nvenc: false,
            cuda_hwaccel: false,
            scale_cuda: false,
        };
        let quality = QualityPreset {
            resolution: Some("480p".into()),
            bitrate: Some("1536K".into()),
            remux: false,
        };
        let commands = vec![
            vec![
                "sh".into(),
                "-c".into(),
                "printf 'synthetic CUDA device unavailable' >&2; exit 1".into(),
            ],
            build_ffmpeg_command(
                source.to_str().unwrap(),
                &hls_dir,
                0.0,
                &quality,
                true,
                &info,
                None,
                None,
                &software,
            ),
        ];
        let stream = TranscodeStream {
            stream_id: "synthetic".into(),
            hls_dir,
            temp_dir: output.clone(),
            process: None,
            playlist_ready: false,
            duration: 5.0,
            width: 160,
            height: 90,
            start_position: 0.0,
        };
        let stream = start_hls_attempts(stream, &commands, false, || true)
            .await
            .unwrap();
        assert!(stream.playlist_ready);
        assert!(
            stream
                .hls_dir
                .join("segment_0.ts")
                .metadata()
                .unwrap()
                .len()
                > 1000
        );
        assert!(
            std::fs::read_to_string(stream.hls_dir.join("playlist.m3u8"))
                .unwrap()
                .contains("#EXTINF")
        );
        drop(stream);
        assert!(!output.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_hls_start_preserves_stderr_and_cleans_output() {
        let root =
            std::env::temp_dir().join(format!("dmc-synthetic-hls-error-{}", uuid::Uuid::new_v4()));
        let stream = TranscodeStream {
            stream_id: "synthetic".into(),
            hls_dir: root.join("hls"),
            temp_dir: root.clone(),
            process: None,
            playlist_ready: false,
            duration: 5.0,
            width: 160,
            height: 90,
            start_position: 0.0,
        };
        let commands = vec![vec![
            "sh".into(),
            "-c".into(),
            "printf 'synthetic encoder unavailable' >&2; exit 1".into(),
        ]];
        let error = start_hls_attempts(stream, &commands, false, || true)
            .await
            .err()
            .unwrap();
        assert!(error.contains("synthetic encoder unavailable"));
        assert!(!root.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn superseded_hls_attempt_never_retries_or_publishes() {
        // AC: @reliable-stream-transitions ac-stop-superseded-producer
        let root =
            std::env::temp_dir().join(format!("dmc-synthetic-hls-stop-{}", uuid::Uuid::new_v4()));
        let stream = TranscodeStream {
            stream_id: "synthetic".into(),
            hls_dir: root.join("hls"),
            temp_dir: root.clone(),
            process: None,
            playlist_ready: false,
            duration: 5.0,
            width: 160,
            height: 90,
            start_position: 0.0,
        };
        let checks = std::sync::atomic::AtomicUsize::new(0);
        let commands = vec![vec!["sleep".into(), "30".into()], vec!["true".into()]];
        let result = start_hls_attempts(stream, &commands, false, || {
            checks.fetch_add(1, Ordering::SeqCst) == 0
        })
        .await;
        assert_eq!(result.err().unwrap(), "Transcode start was superseded");
        assert_eq!(checks.load(Ordering::SeqCst), 2);
        assert!(!root.exists());
    }

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
