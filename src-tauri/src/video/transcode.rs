//! GStreamer-based Transcoding Pipeline for LocalBooru
//!
//! Provides on-the-fly transcoding for incompatible video formats using GStreamer.
//! Supports hardware encoding (NVENC) with automatic fallback to software encoding.
//!
//! # Architecture
//!
//! ```
//! Source Video -> decodebin -> videoconvert -> encoder -> HLS segments
//!                           -> audioconvert -> aacenc  ->
//! ```
//!
//! # Hardware Acceleration
//!
//! - NVENC: Uses nvh264enc for NVIDIA GPUs (best performance)
//! - Software: Falls back to x264enc if no hardware available
//!
//! # Output Formats
//!
//! - HLS: HTTP Live Streaming with .m3u8 playlist and .ts segments
//! - Direct: Raw H.264 stream for direct playback

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_pbutils as gst_pbutils;
use gst_pbutils::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Transcoding quality preset
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TranscodeQuality {
    /// 480p, low bitrate - for mobile/slow networks
    Low,
    /// 720p, medium bitrate - balanced
    Medium,
    /// 1080p, high bitrate - high quality
    High,
    /// Original resolution, high bitrate - maximum quality
    Original,
}

impl TranscodeQuality {
    /// Get target width for this quality preset
    pub fn width(&self) -> Option<i32> {
        match self {
            TranscodeQuality::Low => Some(854),
            TranscodeQuality::Medium => Some(1280),
            TranscodeQuality::High => Some(1920),
            TranscodeQuality::Original => None, // Keep original
        }
    }

    /// Get target bitrate in bits per second
    pub fn bitrate(&self) -> u32 {
        match self {
            TranscodeQuality::Low => 1_000_000,      // 1 Mbps
            TranscodeQuality::Medium => 2_500_000,   // 2.5 Mbps
            TranscodeQuality::High => 5_000_000,     // 5 Mbps
            TranscodeQuality::Original => 8_000_000, // 8 Mbps
        }
    }

    /// Get audio bitrate in bits per second
    pub fn audio_bitrate(&self) -> u32 {
        match self {
            TranscodeQuality::Low => 96_000,
            TranscodeQuality::Medium => 128_000,
            TranscodeQuality::High => 192_000,
            TranscodeQuality::Original => 256_000,
        }
    }
}

/// Hardware encoder type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HardwareEncoder {
    /// NVIDIA NVENC
    Nvenc,
    /// VA-API (Intel/AMD)
    VaApi,
    /// Software x264
    Software,
}

impl HardwareEncoder {
    /// Detect best available encoder
    pub fn detect() -> Self {
        gst::init().ok();
        let registry = gst::Registry::get();

        // Check NVENC (NVIDIA)
        if registry
            .find_feature("nvh264enc", gst::ElementFactory::static_type())
            .is_some()
        {
            log::info!("NVENC hardware encoder available");
            return HardwareEncoder::Nvenc;
        }

        // Check VA-API (Intel/AMD)
        if registry
            .find_feature("vaapih264enc", gst::ElementFactory::static_type())
            .is_some()
        {
            log::info!("VA-API hardware encoder available");
            return HardwareEncoder::VaApi;
        }

        // Fallback to software
        log::info!("Using software encoder (x264)");
        HardwareEncoder::Software
    }

    /// Get GStreamer encoder element name
    fn encoder_name(&self) -> &'static str {
        match self {
            HardwareEncoder::Nvenc => "nvh264enc",
            HardwareEncoder::VaApi => "vaapih264enc",
            HardwareEncoder::Software => "x264enc",
        }
    }
}

/// Transcoding stream state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TranscodeState {
    /// Initializing pipeline
    Initializing,
    /// Transcoding in progress
    Transcoding,
    /// Paused
    Paused,
    /// Completed
    Completed,
    /// Error occurred
    Error(String),
    /// Stopped by user
    Stopped,
}

/// Progress information for a transcode operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeProgress {
    /// Current state
    pub state: TranscodeState,
    /// Progress percentage (0-100)
    pub progress_percent: f64,
    /// Current position in seconds
    pub position_secs: f64,
    /// Total duration in seconds
    pub duration_secs: f64,
    /// Number of HLS segments ready
    pub segments_ready: u32,
    /// Encoding speed (e.g., 2.5x means encoding 2.5 seconds per second)
    pub encoding_speed: f64,
    /// Estimated time remaining in seconds
    pub eta_secs: Option<f64>,
    /// Hardware encoder being used
    pub encoder: HardwareEncoder,
}

/// Configuration for a transcoding session
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    /// Source video path
    pub source_path: PathBuf,
    /// Output directory for HLS segments
    pub output_dir: PathBuf,
    /// Quality preset
    pub quality: TranscodeQuality,
    /// Start position in seconds (for seek-based start)
    pub start_position: f64,
    /// Preferred hardware encoder (None = auto-detect)
    pub preferred_encoder: Option<HardwareEncoder>,
    /// HLS segment duration in seconds
    pub segment_duration: u32,
    /// Force constant frame rate output
    pub force_cfr: bool,
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            source_path: PathBuf::new(),
            output_dir: PathBuf::new(),
            quality: TranscodeQuality::Medium,
            start_position: 0.0,
            preferred_encoder: None,
            segment_duration: 4,
            force_cfr: false,
        }
    }
}

/// GStreamer-based transcoding pipeline
pub struct TranscodePipeline {
    /// Unique stream ID
    stream_id: String,
    /// Configuration
    config: TranscodeConfig,
    /// GStreamer pipeline
    pipeline: Option<gst::Pipeline>,
    /// Current state
    state: Arc<Mutex<TranscodeState>>,
    /// Hardware encoder being used
    encoder: HardwareEncoder,
    /// Start time for speed calculation
    start_time: Option<Instant>,
    /// Source video duration
    duration_secs: f64,
    /// Output HLS directory
    hls_dir: PathBuf,
}

impl TranscodePipeline {
    /// Create a new transcoding pipeline
    pub fn new(config: TranscodeConfig) -> Result<Self, String> {
        gst::init().map_err(|e| format!("Failed to initialize GStreamer: {}", e))?;

        let stream_id = uuid::Uuid::new_v4().to_string();
        let encoder = config
            .preferred_encoder
            .clone()
            .unwrap_or_else(HardwareEncoder::detect);

        let hls_dir = config.output_dir.join(&stream_id);
        std::fs::create_dir_all(&hls_dir)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;

        Ok(Self {
            stream_id,
            config,
            pipeline: None,
            state: Arc::new(Mutex::new(TranscodeState::Initializing)),
            encoder,
            start_time: None,
            duration_secs: 0.0,
            hls_dir,
        })
    }

    /// Get the stream ID
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// Get the HLS output directory
    pub fn hls_dir(&self) -> &Path {
        &self.hls_dir
    }

    /// Get the HLS playlist path
    pub fn playlist_path(&self) -> PathBuf {
        self.hls_dir.join("playlist.m3u8")
    }

    /// Build and start the transcoding pipeline
    pub fn start(&mut self) -> Result<(), String> {
        let pipeline = self.build_pipeline()?;

        // Set up bus watch for messages
        let bus = pipeline.bus().ok_or("Failed to get pipeline bus")?;
        let state_clone = self.state.clone();
        let stream_id = self.stream_id.clone();

        std::thread::spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                match msg.view() {
                    gst::MessageView::Eos(_) => {
                        log::info!("[Transcode {}] End of stream", stream_id);
                        let mut state = state_clone.lock().unwrap();
                        *state = TranscodeState::Completed;
                        break;
                    }
                    gst::MessageView::Error(err) => {
                        let error_msg = format!(
                            "Pipeline error: {} (debug: {:?})",
                            err.error(),
                            err.debug()
                        );
                        log::error!("[Transcode {}] {}", stream_id, error_msg);
                        let mut state = state_clone.lock().unwrap();
                        *state = TranscodeState::Error(error_msg);
                        break;
                    }
                    gst::MessageView::StateChanged(sc) => {
                        if let Some(src) = sc.src() {
                            if src.name() == "pipeline" {
                                log::debug!(
                                    "[Transcode {}] State: {:?} -> {:?}",
                                    stream_id,
                                    sc.old(),
                                    sc.current()
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
        });

        // Start the pipeline
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("Failed to start pipeline: {}", e))?;

        self.pipeline = Some(pipeline);
        self.start_time = Some(Instant::now());

        let mut state = self.state.lock().unwrap();
        *state = TranscodeState::Transcoding;

        log::info!(
            "[Transcode {}] Started with {} encoder",
            self.stream_id,
            self.encoder.encoder_name()
        );

        Ok(())
    }

    /// Build the GStreamer pipeline
    fn build_pipeline(&mut self) -> Result<gst::Pipeline, String> {
        let pipeline = gst::Pipeline::new();

        // Source file URI
        let source_uri = if self.config.source_path.to_string_lossy().starts_with("file://") {
            self.config.source_path.to_string_lossy().to_string()
        } else {
            format!("file://{}", self.config.source_path.display())
        };

        // Create elements
        let source = gst::ElementFactory::make("uridecodebin")
            .name("source")
            .property("uri", &source_uri)
            .build()
            .map_err(|e| format!("Failed to create source: {}", e))?;

        // Video processing chain
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .name("videoconvert")
            .build()
            .map_err(|e| format!("Failed to create videoconvert: {}", e))?;

        let videoscale = gst::ElementFactory::make("videoscale")
            .name("videoscale")
            .build()
            .map_err(|e| format!("Failed to create videoscale: {}", e))?;

        // Scale filter caps (if quality preset specifies width)
        let scale_filter = if let Some(width) = self.config.quality.width() {
            // Calculate height maintaining aspect ratio (height = -1 means auto)
            gst::ElementFactory::make("capsfilter")
                .name("scalefilter")
                .property(
                    "caps",
                    gst::Caps::builder("video/x-raw")
                        .field("width", width)
                        .build(),
                )
                .build()
                .map_err(|e| format!("Failed to create scale filter: {}", e))?
        } else {
            // Pass through without scaling
            gst::ElementFactory::make("identity")
                .name("scalefilter")
                .build()
                .map_err(|e| format!("Failed to create identity: {}", e))?
        };

        // Video encoder
        let video_encoder = self.create_video_encoder()?;

        // H264 parser (needed for muxer)
        let h264parse = gst::ElementFactory::make("h264parse")
            .name("h264parse")
            .build()
            .map_err(|e| format!("Failed to create h264parse: {}", e))?;

        // Audio processing chain
        let audioconvert = gst::ElementFactory::make("audioconvert")
            .name("audioconvert")
            .build()
            .map_err(|e| format!("Failed to create audioconvert: {}", e))?;

        let audioresample = gst::ElementFactory::make("audioresample")
            .name("audioresample")
            .build()
            .map_err(|e| format!("Failed to create audioresample: {}", e))?;

        // AAC audio encoder
        let audio_encoder = gst::ElementFactory::make("avenc_aac")
            .name("audioenc")
            .property("bitrate", self.config.quality.audio_bitrate() as i32)
            .build()
            .or_else(|_| {
                // Fallback to faac if available
                gst::ElementFactory::make("faac")
                    .name("audioenc")
                    .property("bitrate", self.config.quality.audio_bitrate() as i32 / 1000)
                    .build()
            })
            .or_else(|_| {
                // Last resort: voaacenc
                gst::ElementFactory::make("voaacenc")
                    .name("audioenc")
                    .property("bitrate", self.config.quality.audio_bitrate() as i32)
                    .build()
            })
            .map_err(|e| format!("Failed to create audio encoder: {}", e))?;

        // MPEG-TS muxer
        let mpegtsmux = gst::ElementFactory::make("mpegtsmux")
            .name("muxer")
            .build()
            .map_err(|e| format!("Failed to create mpegtsmux: {}", e))?;

        // HLS sink
        let hlssink = gst::ElementFactory::make("hlssink2")
            .name("hlssink")
            .property("target-duration", self.config.segment_duration)
            .property(
                "playlist-location",
                self.hls_dir.join("playlist.m3u8").to_string_lossy().as_ref(),
            )
            .property(
                "location",
                self.hls_dir
                    .join("segment_%05d.ts")
                    .to_string_lossy()
                    .as_ref(),
            )
            .property("max-files", 0u32) // Keep all segments
            .build()
            .map_err(|e| format!("Failed to create hlssink: {}", e))?;

        // Add all elements to pipeline
        pipeline
            .add_many([
                &source,
                &videoconvert,
                &videoscale,
                &scale_filter,
                &video_encoder,
                &h264parse,
                &audioconvert,
                &audioresample,
                &audio_encoder,
                &mpegtsmux,
                &hlssink,
            ])
            .map_err(|e| format!("Failed to add elements to pipeline: {}", e))?;

        // Link video chain: videoconvert -> videoscale -> scalefilter -> encoder -> h264parse -> muxer
        gst::Element::link_many([
            &videoconvert,
            &videoscale,
            &scale_filter,
            &video_encoder,
            &h264parse,
        ])
        .map_err(|e| format!("Failed to link video chain: {}", e))?;

        // Link audio chain: audioconvert -> audioresample -> encoder -> muxer
        gst::Element::link_many([&audioconvert, &audioresample, &audio_encoder])
            .map_err(|e| format!("Failed to link audio chain: {}", e))?;

        // Link h264parse to muxer video pad
        let muxer_video_pad = mpegtsmux
            .request_pad_simple("sink_%d")
            .ok_or("Failed to get muxer video pad")?;
        let h264parse_src = h264parse
            .static_pad("src")
            .ok_or("Failed to get h264parse src pad")?;
        h264parse_src
            .link(&muxer_video_pad)
            .map_err(|e| format!("Failed to link h264parse to muxer: {}", e))?;

        // Link audio encoder to muxer audio pad
        let muxer_audio_pad = mpegtsmux
            .request_pad_simple("sink_%d")
            .ok_or("Failed to get muxer audio pad")?;
        let audio_enc_src = audio_encoder
            .static_pad("src")
            .ok_or("Failed to get audio encoder src pad")?;
        audio_enc_src
            .link(&muxer_audio_pad)
            .map_err(|e| format!("Failed to link audio encoder to muxer: {}", e))?;

        // Link muxer to HLS sink
        gst::Element::link(&mpegtsmux, &hlssink)
            .map_err(|e| format!("Failed to link muxer to hlssink: {}", e))?;

        // Handle dynamic pad linking from uridecodebin
        let videoconvert_weak = videoconvert.downgrade();
        let audioconvert_weak = audioconvert.downgrade();

        source.connect_pad_added(move |_src, src_pad| {
            let pad_name = src_pad.name();
            log::debug!("New pad added: {}", pad_name);

            let caps = src_pad.current_caps();
            let caps_str = caps
                .as_ref()
                .map(|c| c.to_string())
                .unwrap_or_default();

            if caps_str.starts_with("video/") {
                if let Some(videoconvert) = videoconvert_weak.upgrade() {
                    let sink_pad = videoconvert.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            log::error!("Failed to link video pad: {:?}", e);
                        } else {
                            log::info!("Linked video pad");
                        }
                    }
                }
            } else if caps_str.starts_with("audio/") {
                if let Some(audioconvert) = audioconvert_weak.upgrade() {
                    let sink_pad = audioconvert.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            log::error!("Failed to link audio pad: {:?}", e);
                        } else {
                            log::info!("Linked audio pad");
                        }
                    }
                }
            }
        });

        Ok(pipeline)
    }

    /// Create the appropriate video encoder based on hardware availability
    fn create_video_encoder(&self) -> Result<gst::Element, String> {
        let bitrate = self.config.quality.bitrate();

        match &self.encoder {
            HardwareEncoder::Nvenc => {
                gst::ElementFactory::make("nvh264enc")
                    .name("videoenc")
                    .property("bitrate", bitrate / 1000) // NVENC uses kbps
                    .property("preset", 1u32) // 1 = low latency (p1)
                    .property("rc-mode", 2u32) // 2 = CBR
                    .build()
                    .map_err(|e| format!("Failed to create nvh264enc: {}", e))
            }
            HardwareEncoder::VaApi => gst::ElementFactory::make("vaapih264enc")
                .name("videoenc")
                .property("bitrate", bitrate / 1000)
                .property("rate-control", 2u32) // CBR
                .build()
                .map_err(|e| format!("Failed to create vaapih264enc: {}", e)),
            HardwareEncoder::Software => gst::ElementFactory::make("x264enc")
                .name("videoenc")
                .property("bitrate", bitrate / 1000) // x264 uses kbps
                .property("speed-preset", 1u32) // ultrafast
                .property("tune", 0x00000004u32) // zerolatency
                .property("key-int-max", 60u32) // Keyframe every 2 seconds at 30fps
                .build()
                .map_err(|e| format!("Failed to create x264enc: {}", e)),
        }
    }

    /// Get current progress
    pub fn get_progress(&self) -> TranscodeProgress {
        let state = self.state.lock().unwrap().clone();

        let (position_secs, encoding_speed) = if let Some(ref pipeline) = self.pipeline {
            let position = pipeline
                .query_position::<gst::ClockTime>()
                .map(|p| p.seconds() as f64)
                .unwrap_or(0.0);

            let speed = if let Some(start) = self.start_time {
                let elapsed = start.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    position / elapsed
                } else {
                    0.0
                }
            } else {
                0.0
            };

            (position, speed)
        } else {
            (0.0, 0.0)
        };

        let progress_percent = if self.duration_secs > 0.0 {
            (position_secs / self.duration_secs * 100.0).min(100.0)
        } else {
            0.0
        };

        let eta_secs = if encoding_speed > 0.0 && self.duration_secs > 0.0 {
            Some((self.duration_secs - position_secs) / encoding_speed)
        } else {
            None
        };

        let segments_ready = self.count_segments();

        TranscodeProgress {
            state,
            progress_percent,
            position_secs,
            duration_secs: self.duration_secs,
            segments_ready,
            encoding_speed,
            eta_secs,
            encoder: self.encoder.clone(),
        }
    }

    /// Count HLS segments in output directory
    fn count_segments(&self) -> u32 {
        std::fs::read_dir(&self.hls_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "ts")
                            .unwrap_or(false)
                    })
                    .count() as u32
            })
            .unwrap_or(0)
    }

    /// Check if playlist is ready for playback
    pub fn is_playlist_ready(&self) -> bool {
        let playlist = self.playlist_path();
        if !playlist.exists() {
            return false;
        }

        // Check if at least one segment exists
        self.count_segments() > 0
    }

    /// Pause transcoding
    pub fn pause(&mut self) -> Result<(), String> {
        if let Some(ref pipeline) = self.pipeline {
            pipeline
                .set_state(gst::State::Paused)
                .map_err(|e| format!("Failed to pause: {}", e))?;

            let mut state = self.state.lock().unwrap();
            *state = TranscodeState::Paused;
        }
        Ok(())
    }

    /// Resume transcoding
    pub fn resume(&mut self) -> Result<(), String> {
        if let Some(ref pipeline) = self.pipeline {
            pipeline
                .set_state(gst::State::Playing)
                .map_err(|e| format!("Failed to resume: {}", e))?;

            let mut state = self.state.lock().unwrap();
            *state = TranscodeState::Transcoding;
        }
        Ok(())
    }

    /// Stop transcoding
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(ref pipeline) = self.pipeline {
            pipeline
                .set_state(gst::State::Null)
                .map_err(|e| format!("Failed to stop: {}", e))?;
        }

        let mut state = self.state.lock().unwrap();
        *state = TranscodeState::Stopped;

        log::info!("[Transcode {}] Stopped", self.stream_id);
        Ok(())
    }

    /// Cleanup output files
    pub fn cleanup(&self) -> Result<(), String> {
        if self.hls_dir.exists() {
            std::fs::remove_dir_all(&self.hls_dir)
                .map_err(|e| format!("Failed to cleanup: {}", e))?;
        }
        Ok(())
    }
}

impl Drop for TranscodePipeline {
    fn drop(&mut self) {
        if let Some(ref pipeline) = self.pipeline {
            let _ = pipeline.set_state(gst::State::Null);
        }
    }
}

/// Manager for multiple concurrent transcoding streams
pub struct TranscodeManager {
    /// Active streams by ID
    streams: HashMap<String, TranscodePipeline>,
    /// Cache directory for transcoded segments
    cache_dir: PathBuf,
    /// Maximum cache size in bytes
    max_cache_bytes: u64,
    /// Hardware encoder detected once at startup
    hw_encoder: HardwareEncoder,
}

impl TranscodeManager {
    /// Create a new transcode manager
    pub fn new(cache_dir: PathBuf) -> Result<Self, String> {
        gst::init().map_err(|e| format!("Failed to initialize GStreamer: {}", e))?;

        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;

        let hw_encoder = HardwareEncoder::detect();

        Ok(Self {
            streams: HashMap::new(),
            cache_dir,
            max_cache_bytes: 5 * 1024 * 1024 * 1024, // 5 GB default
            hw_encoder,
        })
    }

    /// Get the detected hardware encoder
    pub fn hardware_encoder(&self) -> &HardwareEncoder {
        &self.hw_encoder
    }

    /// Start a new transcoding session
    pub fn start_transcode(
        &mut self,
        source_path: PathBuf,
        quality: TranscodeQuality,
    ) -> Result<String, String> {
        let config = TranscodeConfig {
            source_path,
            output_dir: self.cache_dir.clone(),
            quality,
            preferred_encoder: Some(self.hw_encoder.clone()),
            ..Default::default()
        };

        let mut pipeline = TranscodePipeline::new(config)?;
        let stream_id = pipeline.stream_id().to_string();

        pipeline.start()?;

        self.streams.insert(stream_id.clone(), pipeline);

        Ok(stream_id)
    }

    /// Get progress for a stream
    pub fn get_progress(&self, stream_id: &str) -> Option<TranscodeProgress> {
        self.streams.get(stream_id).map(|p| p.get_progress())
    }

    /// Get HLS playlist path for a stream
    pub fn get_playlist_path(&self, stream_id: &str) -> Option<PathBuf> {
        self.streams.get(stream_id).map(|p| p.playlist_path())
    }

    /// Check if stream is ready for playback
    pub fn is_stream_ready(&self, stream_id: &str) -> bool {
        self.streams
            .get(stream_id)
            .map(|p| p.is_playlist_ready())
            .unwrap_or(false)
    }

    /// Stop a transcoding session
    pub fn stop_transcode(&mut self, stream_id: &str) -> Result<(), String> {
        if let Some(mut pipeline) = self.streams.remove(stream_id) {
            pipeline.stop()?;
        }
        Ok(())
    }

    /// Stop all transcoding sessions
    pub fn stop_all(&mut self) {
        for (_, mut pipeline) in self.streams.drain() {
            let _ = pipeline.stop();
        }
    }

    /// Cleanup cache to stay under limit
    pub fn cleanup_cache(&self) -> Result<u64, String> {
        let mut total_size = 0u64;
        let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();

        // Collect all segment files with their sizes and modification times
        for entry in std::fs::read_dir(&self.cache_dir)
            .map_err(|e| format!("Failed to read cache dir: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();

            if path.is_dir() {
                // This is a stream directory
                for file in std::fs::read_dir(&path)
                    .map_err(|e| format!("Failed to read stream dir: {}", e))?
                {
                    let file = file.map_err(|e| format!("Failed to read file: {}", e))?;
                    let file_path = file.path();
                    if let Ok(metadata) = file_path.metadata() {
                        let size = metadata.len();
                        let modified = metadata.modified().unwrap_or(std::time::UNIX_EPOCH);
                        total_size += size;
                        files.push((file_path, size, modified));
                    }
                }
            }
        }

        // If under limit, nothing to do
        if total_size <= self.max_cache_bytes {
            return Ok(0);
        }

        // Sort by modification time (oldest first)
        files.sort_by_key(|(_, _, time)| *time);

        // Delete oldest files until under limit
        let mut freed = 0u64;
        for (path, size, _) in files {
            if total_size <= self.max_cache_bytes {
                break;
            }

            if std::fs::remove_file(&path).is_ok() {
                total_size -= size;
                freed += size;
                log::debug!("Deleted cached segment: {}", path.display());
            }
        }

        // Clean up empty directories
        for entry in std::fs::read_dir(&self.cache_dir).into_iter().flatten() {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_dir() {
                    if std::fs::read_dir(&path)
                        .map(|mut d| d.next().is_none())
                        .unwrap_or(false)
                    {
                        let _ = std::fs::remove_dir(&path);
                    }
                }
            }
        }

        log::info!("Cache cleanup freed {} bytes", freed);
        Ok(freed)
    }

    /// Set maximum cache size
    pub fn set_max_cache_bytes(&mut self, bytes: u64) {
        self.max_cache_bytes = bytes;
    }
}

impl Drop for TranscodeManager {
    fn drop(&mut self) {
        self.stop_all();
    }
}

/// Check if a video codec requires transcoding for browser playback
pub fn needs_transcoding(codec: &str) -> bool {
    let codec_lower = codec.to_lowercase();

    // Codecs that typically need transcoding for browser playback
    let transcode_codecs = [
        "av1",    // AV1 - limited browser support
        "vp9",    // VP9 - limited in some contexts
        "hevc",   // H.265/HEVC - poor browser support
        "h265",   // H.265/HEVC
        "mpeg4",  // MPEG-4 Part 2
        "mpeg2",  // MPEG-2
        "wmv",    // Windows Media Video
        "vc1",    // VC-1
        "theora", // Theora
        "prores", // ProRes
        "dnxhd",  // DNxHD
        "ffv1",   // FFV1
    ];

    for tc in transcode_codecs {
        if codec_lower.contains(tc) {
            return true;
        }
    }

    false
}

/// Probe video file to get codec information
pub fn probe_video_codec(path: &Path) -> Result<String, String> {
    gst::init().map_err(|e| format!("Failed to init GStreamer: {}", e))?;

    let uri = format!("file://{}", path.display());

    let discoverer = gst_pbutils::Discoverer::new(gst::ClockTime::from_seconds(10))
        .map_err(|e| format!("Failed to create discoverer: {}", e))?;

    let info = discoverer
        .discover_uri(&uri)
        .map_err(|e| format!("Failed to discover video: {}", e))?;

    for stream in info.video_streams() {
        // Use DiscovererStreamInfoExt::caps() method
        if let Some(caps) = DiscovererStreamInfoExt::caps(&stream) {
            if let Some(structure) = caps.structure(0) {
                return Ok(structure.name().as_str().to_string());
            }
        }
    }

    Err("No video stream found".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_encoder_detection() {
        let encoder = HardwareEncoder::detect();
        // Should detect something
        assert!(matches!(
            encoder,
            HardwareEncoder::Nvenc | HardwareEncoder::VaApi | HardwareEncoder::Software
        ));
    }

    #[test]
    fn test_quality_presets() {
        assert_eq!(TranscodeQuality::Low.width(), Some(854));
        assert_eq!(TranscodeQuality::Medium.width(), Some(1280));
        assert_eq!(TranscodeQuality::High.width(), Some(1920));
        assert_eq!(TranscodeQuality::Original.width(), None);
    }

    #[test]
    fn test_needs_transcoding() {
        assert!(needs_transcoding("av1"));
        assert!(needs_transcoding("hevc"));
        assert!(needs_transcoding("h265"));
        assert!(!needs_transcoding("h264"));
        assert!(!needs_transcoding("avc"));
    }
}
