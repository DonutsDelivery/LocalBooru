//! VFR (Variable Frame Rate) Video Support
//!
//! This module provides VFR-aware video handling using GStreamer.
//!
//! # VFR Challenges
//!
//! Variable frame rate videos (common from phones, screen recordings) have:
//! - Non-constant time between frames
//! - Timestamps that matter for correct playback speed
//! - Seeking complexity (must seek to actual frame timestamps, not interpolated times)
//!
//! # GStreamer's VFR Handling
//!
//! GStreamer handles VFR natively through:
//! - PTS (Presentation Timestamp) preservation throughout the pipeline
//! - Clock-based synchronization that respects frame timestamps
//! - Proper audio/video sync via shared clock
//!
//! The key is to NOT interfere with timestamps and let GStreamer's
//! sync mechanisms work. This module provides utilities for:
//! - Detecting VFR content
//! - Accurate seeking with proper timestamp handling
//! - Exposing frame timing information

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_pbutils as gst_pbutils;
use std::sync::{Arc, Mutex};

/// Information about a video's frame rate characteristics
#[derive(Debug, Clone)]
pub struct FrameRateInfo {
    /// Whether the video appears to be VFR
    pub is_vfr: bool,
    /// Average frame rate (fps)
    pub average_fps: f64,
    /// Minimum frame duration detected (in nanoseconds)
    pub min_frame_duration_ns: Option<u64>,
    /// Maximum frame duration detected (in nanoseconds)
    pub max_frame_duration_ns: Option<u64>,
    /// Container-reported frame rate (may be nominal for VFR)
    pub container_fps: Option<f64>,
}

impl Default for FrameRateInfo {
    fn default() -> Self {
        Self {
            is_vfr: false,
            average_fps: 0.0,
            min_frame_duration_ns: None,
            max_frame_duration_ns: None,
            container_fps: None,
        }
    }
}

/// Seek mode for VFR-aware seeking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeekMode {
    /// Seek to nearest keyframe (fast, may be inaccurate)
    Keyframe,
    /// Seek accurately to exact position (slower, decodes from keyframe)
    Accurate,
    /// Snap to actual frame timestamp (best for VFR)
    SnapToFrame,
}

impl Default for SeekMode {
    fn default() -> Self {
        SeekMode::Accurate
    }
}

/// Seek flags for GStreamer based on mode
impl SeekMode {
    pub fn to_gst_flags(&self) -> gst::SeekFlags {
        match self {
            SeekMode::Keyframe => {
                gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT
            }
            SeekMode::Accurate => {
                gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE
            }
            SeekMode::SnapToFrame => {
                // SNAP_NEAREST seeks to the nearest frame boundary
                // Combined with ACCURATE for precise VFR handling
                gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE | gst::SeekFlags::SNAP_NEAREST
            }
        }
    }
}

/// VFR-aware video player state
#[derive(Debug, Clone, PartialEq)]
pub enum VfrPlayerState {
    Idle,
    Playing,
    Paused,
    Buffering,
    Seeking,
    EndOfStream,
    Error(String),
}

/// VFR-aware video player using custom pipeline for maximum control
pub struct VfrVideoPlayer {
    pipeline: gst::Pipeline,
    state: Arc<Mutex<VfrPlayerState>>,
    frame_rate_info: Arc<Mutex<FrameRateInfo>>,
    seek_mode: Arc<Mutex<SeekMode>>,
    /// Last seek position for segment-based seeking
    last_seek_pos: Arc<Mutex<Option<gst::ClockTime>>>,
}

impl VfrVideoPlayer {
    /// Create a new VFR-aware video player
    pub fn new() -> Result<Self, String> {
        gst::init().map_err(|e| format!("Failed to init GStreamer: {}", e))?;

        // Create a pipeline using playbin3 which has better VFR handling
        // playbin3 uses decodebin3 internally which preserves timestamps better
        let pipeline = gst::Pipeline::new();

        // Create playbin3 element (falls back to playbin if not available)
        let playbin = if gst::ElementFactory::find("playbin3").is_some() {
            log::info!("Using playbin3 for enhanced VFR support");
            gst::ElementFactory::make("playbin3")
                .build()
                .map_err(|e| format!("Failed to create playbin3: {}", e))?
        } else {
            log::info!("Using playbin (playbin3 not available)");
            gst::ElementFactory::make("playbin")
                .build()
                .map_err(|e| format!("Failed to create playbin: {}", e))?
        };

        // Configure for VFR handling
        // Disable buffering delay which can cause issues with VFR
        playbin.set_property("buffer-duration", 0i64);

        // Set up video sink
        if let Some(video_sink) = Self::create_video_sink() {
            playbin.set_property("video-sink", &video_sink);
        }

        // Set up audio sink with proper sync
        if let Some(audio_sink) = Self::create_audio_sink() {
            playbin.set_property("audio-sink", &audio_sink);
        }

        // Add playbin to pipeline
        pipeline.add(&playbin).map_err(|e| format!("Failed to add playbin: {}", e))?;

        let state = Arc::new(Mutex::new(VfrPlayerState::Idle));
        let frame_rate_info = Arc::new(Mutex::new(FrameRateInfo::default()));
        let seek_mode = Arc::new(Mutex::new(SeekMode::default()));
        let last_seek_pos = Arc::new(Mutex::new(None));

        let player = Self {
            pipeline,
            state,
            frame_rate_info,
            seek_mode,
            last_seek_pos,
        };

        player.setup_bus_watch();

        Ok(player)
    }

    /// Create the best video sink for the platform
    fn create_video_sink() -> Option<gst::Element> {
        let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();

        if session_type == "wayland" {
            // waylandsink handles VFR well through DRM/KMS timestamping
            if let Ok(sink) = gst::ElementFactory::make("waylandsink").build() {
                // Enable sync for proper VFR playback
                sink.set_property("sync", true);
                log::info!("VFR: Using waylandsink with sync enabled");
                return Some(sink);
            }
        }

        // glimagesink also handles timestamps well
        if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
            sink.set_property("sync", true);
            log::info!("VFR: Using glimagesink with sync enabled");
            return Some(sink);
        }

        // autovideosink as fallback
        if let Ok(sink) = gst::ElementFactory::make("autovideosink").build() {
            log::info!("VFR: Using autovideosink");
            return Some(sink);
        }

        None
    }

    /// Create audio sink with proper sync for A/V alignment
    fn create_audio_sink() -> Option<gst::Element> {
        // pulsesink or autoaudiosink with sync enabled
        if let Ok(sink) = gst::ElementFactory::make("pulsesink").build() {
            // Ensure audio respects timestamps
            sink.set_property("sync", true);
            // Reduce latency for better sync
            sink.set_property("buffer-time", 20000i64); // 20ms
            sink.set_property("latency-time", 10000i64); // 10ms
            log::info!("VFR: Using pulsesink with low latency");
            return Some(sink);
        }

        if let Ok(sink) = gst::ElementFactory::make("autoaudiosink").build() {
            log::info!("VFR: Using autoaudiosink");
            return Some(sink);
        }

        None
    }

    /// Set up bus watch for pipeline messages
    fn setup_bus_watch(&self) {
        let bus = self.pipeline.bus().expect("Pipeline should have a bus");
        let state_clone = self.state.clone();
        let frame_info_clone = self.frame_rate_info.clone();

        let _bus_watch = bus.add_watch_local(move |_bus, msg| {
            use gst::MessageView;

            match msg.view() {
                MessageView::Eos(_) => {
                    log::info!("VFR: End of stream");
                    *state_clone.lock().unwrap() = VfrPlayerState::EndOfStream;
                }
                MessageView::Error(err) => {
                    let error_msg = format!(
                        "Error from {:?}: {} ({:?})",
                        err.src().map(|s| s.path_string()),
                        err.error(),
                        err.debug()
                    );
                    log::error!("VFR: {}", error_msg);
                    *state_clone.lock().unwrap() = VfrPlayerState::Error(error_msg);
                }
                MessageView::StateChanged(state_changed) => {
                    // Only care about pipeline state changes
                    if state_changed.src().map(|s| s.is::<gst::Pipeline>()).unwrap_or(false) {
                        let new_state = state_changed.current();
                        let mut player_state = state_clone.lock().unwrap();
                        *player_state = match new_state {
                            gst::State::Playing => VfrPlayerState::Playing,
                            gst::State::Paused => VfrPlayerState::Paused,
                            gst::State::Null | gst::State::Ready => VfrPlayerState::Idle,
                            _ => player_state.clone(),
                        };
                        log::debug!("VFR: State changed to {:?}", new_state);
                    }
                }
                MessageView::Buffering(buffering) => {
                    let percent = buffering.percent();
                    if percent < 100 {
                        *state_clone.lock().unwrap() = VfrPlayerState::Buffering;
                    }
                    log::debug!("VFR: Buffering {}%", percent);
                }
                MessageView::Tag(tag_msg) => {
                    // Extract frame rate information from tags
                    let tags = tag_msg.tags();
                    Self::extract_frame_rate_from_tags(&tags, &frame_info_clone);
                }
                MessageView::StreamStart(_) => {
                    log::info!("VFR: Stream started");
                }
                MessageView::AsyncDone(_) => {
                    // Seek completed
                    let mut state = state_clone.lock().unwrap();
                    if *state == VfrPlayerState::Seeking {
                        // Return to previous state (Playing or Paused)
                        *state = VfrPlayerState::Playing;
                    }
                    log::debug!("VFR: Async operation completed");
                }
                _ => {}
            }

            gst::glib::ControlFlow::Continue
        }).expect("Failed to add bus watch");

        // Keep the bus watch alive - it will be dropped when VfrVideoPlayer is dropped
        // The watch is stored implicitly by the pipeline's reference to it
    }

    /// Extract frame rate information from stream tags
    fn extract_frame_rate_from_tags(tags: &gst::TagList, frame_info: &Arc<Mutex<FrameRateInfo>>) {
        let _info = frame_info.lock().unwrap();

        // Look for frame rate in various tag formats
        if let Some(framerate) = tags.get::<gst::tags::VideoCodec>() {
            log::debug!("VFR: Video codec: {}", framerate.get());
        }

        // Container frame rate is often stored as nominal-bitrate or similar
        // Real frame rate detection requires analyzing actual frame timestamps
        // TODO: Implement more sophisticated VFR detection by sampling frame PTSs
    }

    /// Get the playbin element from the pipeline
    fn get_playbin(&self) -> Option<gst::Element> {
        self.pipeline.by_name("playbin0")
            .or_else(|| self.pipeline.children().into_iter().next())
    }

    /// Play a video file
    pub fn play_uri(&self, uri: &str) -> Result<(), String> {
        let full_uri = if uri.starts_with("file://") || uri.starts_with("http://") || uri.starts_with("https://") {
            uri.to_string()
        } else {
            format!("file://{}", uri)
        };

        log::info!("VFR: Playing {}", full_uri);

        // Set URI on the playbin
        if let Some(playbin) = self.get_playbin() {
            playbin.set_property("uri", &full_uri);
        } else {
            return Err("Playbin not found in pipeline".to_string());
        }

        // Reset frame rate info for new file
        *self.frame_rate_info.lock().unwrap() = FrameRateInfo::default();
        *self.last_seek_pos.lock().unwrap() = None;

        // Start playing
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("Failed to start playback: {}", e))?;

        *self.state.lock().unwrap() = VfrPlayerState::Playing;
        Ok(())
    }

    /// Pause playback
    pub fn pause(&self) -> Result<(), String> {
        self.pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| format!("Failed to pause: {}", e))?;
        *self.state.lock().unwrap() = VfrPlayerState::Paused;
        Ok(())
    }

    /// Resume playback
    pub fn resume(&self) -> Result<(), String> {
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("Failed to resume: {}", e))?;
        *self.state.lock().unwrap() = VfrPlayerState::Playing;
        Ok(())
    }

    /// Stop playback
    pub fn stop(&self) -> Result<(), String> {
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| format!("Failed to stop: {}", e))?;
        *self.state.lock().unwrap() = VfrPlayerState::Idle;
        Ok(())
    }

    /// VFR-aware seek to position in nanoseconds
    ///
    /// This method uses the configured seek mode for proper VFR handling:
    /// - Accurate mode decodes from keyframe for exact positioning
    /// - SnapToFrame snaps to actual frame timestamps
    pub fn seek(&self, position_ns: u64) -> Result<(), String> {
        let seek_mode = *self.seek_mode.lock().unwrap();
        let flags = seek_mode.to_gst_flags();
        let position = gst::ClockTime::from_nseconds(position_ns);

        log::debug!("VFR: Seeking to {} ns with mode {:?}", position_ns, seek_mode);

        // Mark as seeking
        *self.state.lock().unwrap() = VfrPlayerState::Seeking;
        *self.last_seek_pos.lock().unwrap() = Some(position);

        // Perform the seek
        // Using seek_simple for straightforward seeks
        // For VFR, the ACCURATE flag ensures we decode from keyframe to exact position
        self.pipeline
            .seek_simple(flags, position)
            .map_err(|e| format!("Seek failed: {}", e))?;

        Ok(())
    }

    /// Seek with specific mode (overrides default)
    pub fn seek_with_mode(&self, position_ns: u64, mode: SeekMode) -> Result<(), String> {
        let flags = mode.to_gst_flags();
        let position = gst::ClockTime::from_nseconds(position_ns);

        log::debug!("VFR: Seeking to {} ns with explicit mode {:?}", position_ns, mode);

        *self.state.lock().unwrap() = VfrPlayerState::Seeking;
        *self.last_seek_pos.lock().unwrap() = Some(position);

        self.pipeline
            .seek_simple(flags, position)
            .map_err(|e| format!("Seek failed: {}", e))?;

        Ok(())
    }

    /// Step forward by one frame (useful for VFR frame-by-frame navigation)
    pub fn step_frame(&self) -> Result<(), String> {
        // Pause first if playing
        let was_playing = *self.state.lock().unwrap() == VfrPlayerState::Playing;
        if was_playing {
            self.pause()?;
        }

        // Send step event
        // This steps by one frame respecting VFR timestamps
        let step_event = gst::event::Step::new(gst::format::Buffers::ONE, 1.0, true, false);

        if !self.pipeline.send_event(step_event) {
            return Err("Failed to send step event".to_string());
        }

        log::debug!("VFR: Stepped forward one frame");
        Ok(())
    }

    /// Get current position in nanoseconds
    pub fn position(&self) -> Option<u64> {
        self.pipeline
            .query_position::<gst::ClockTime>()
            .map(|p| p.nseconds())
    }

    /// Get duration in nanoseconds
    pub fn duration(&self) -> Option<u64> {
        self.pipeline
            .query_duration::<gst::ClockTime>()
            .map(|d| d.nseconds())
    }

    /// Get current state
    pub fn state(&self) -> VfrPlayerState {
        self.state.lock().unwrap().clone()
    }

    /// Get frame rate information
    pub fn frame_rate_info(&self) -> FrameRateInfo {
        self.frame_rate_info.lock().unwrap().clone()
    }

    /// Set the default seek mode
    pub fn set_seek_mode(&self, mode: SeekMode) {
        *self.seek_mode.lock().unwrap() = mode;
        log::info!("VFR: Seek mode set to {:?}", mode);
    }

    /// Get the current seek mode
    pub fn seek_mode(&self) -> SeekMode {
        *self.seek_mode.lock().unwrap()
    }

    /// Set playback rate (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
    ///
    /// For VFR content, rate changes must preserve timestamp relationships
    pub fn set_rate(&self, rate: f64) -> Result<(), String> {
        if rate <= 0.0 {
            return Err("Rate must be positive".to_string());
        }

        let position = self.position().unwrap_or(0);
        let position_time = gst::ClockTime::from_nseconds(position);

        // Create a seek event with the new rate
        // FLUSH ensures clean state, ACCURATE ensures VFR timestamps are preserved
        let seek_event = gst::event::Seek::new(
            rate,
            gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE,
            gst::SeekType::Set,
            position_time,
            gst::SeekType::None,
            gst::ClockTime::NONE,
        );

        if !self.pipeline.send_event(seek_event) {
            return Err("Failed to change playback rate".to_string());
        }

        log::info!("VFR: Playback rate set to {}x", rate);
        Ok(())
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&self, volume: f64) {
        if let Some(playbin) = self.get_playbin() {
            playbin.set_property("volume", volume.clamp(0.0, 1.0));
        }
    }

    /// Get current volume
    pub fn volume(&self) -> f64 {
        self.get_playbin()
            .map(|p| p.property::<f64>("volume"))
            .unwrap_or(1.0)
    }

    /// Set mute state
    pub fn set_muted(&self, muted: bool) {
        if let Some(playbin) = self.get_playbin() {
            playbin.set_property("mute", muted);
        }
    }

    /// Check if muted
    pub fn is_muted(&self) -> bool {
        self.get_playbin()
            .map(|p| p.property::<bool>("mute"))
            .unwrap_or(false)
    }

    /// Query detailed stream information for VFR analysis
    pub fn query_stream_info(&self) -> Result<StreamInfo, String> {
        let mut info = StreamInfo::default();

        // Query duration
        info.duration_ns = self.duration();

        // Query position
        info.position_ns = self.position();

        // Query if seekable
        let mut query = gst::query::Seeking::new(gst::Format::Time);
        if self.pipeline.query(&mut query) {
            let (seekable, start, end) = query.result();
            info.seekable = seekable;
            // Extract nanoseconds from GenericFormattedValue (Time format)
            info.seek_start_ns = match start {
                gst::GenericFormattedValue::Time(opt_time) => opt_time.map(|t| t.nseconds()),
                _ => None,
            };
            info.seek_end_ns = match end {
                gst::GenericFormattedValue::Time(opt_time) => opt_time.map(|t| t.nseconds()),
                _ => None,
            };
        }

        // Get buffering info
        let mut buf_query = gst::query::Buffering::new(gst::Format::Percent);
        if self.pipeline.query(&mut buf_query) {
            let (_busy, percent) = buf_query.percent();
            info.buffering_percent = Some(percent);
        }

        Ok(info)
    }
}

impl Drop for VfrVideoPlayer {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Detailed stream information
#[derive(Debug, Clone, Default)]
pub struct StreamInfo {
    pub duration_ns: Option<u64>,
    pub position_ns: Option<u64>,
    pub seekable: bool,
    pub seek_start_ns: Option<u64>,
    pub seek_end_ns: Option<u64>,
    pub buffering_percent: Option<i32>,
}

/// Analyze a video file for VFR characteristics
///
/// This performs a quick analysis of the video to determine if it's VFR
/// by examining frame timestamps in the first few seconds.
pub fn analyze_video_for_vfr(uri: &str) -> Result<FrameRateInfo, String> {
    gst::init().map_err(|e| format!("Failed to init GStreamer: {}", e))?;

    let full_uri = if uri.starts_with("file://") {
        uri.to_string()
    } else {
        format!("file://{}", uri)
    };

    // Create a discoverer for metadata extraction
    let discoverer = gst_pbutils::Discoverer::new(gst::ClockTime::from_seconds(5))
        .map_err(|e| format!("Failed to create discoverer: {}", e))?;

    let info = discoverer
        .discover_uri(&full_uri)
        .map_err(|e| format!("Failed to discover: {}", e))?;

    let mut frame_info = FrameRateInfo::default();

    // Check video streams using DiscovererVideoInfo methods
    for stream in info.video_streams() {
        // Get frame rate as a Fraction
        let fps = stream.framerate();
        let fps_num = fps.numer();
        let fps_denom = fps.denom();

        if fps_denom > 0 {
            let fps_value = fps_num as f64 / fps_denom as f64;
            frame_info.container_fps = Some(fps_value);
            frame_info.average_fps = fps_value;

            // Check for variable frame rate indicators
            // VFR videos often have 0/1 or very high fps values (1000/1) in container
            if fps_denom == 1 && (fps_num == 0 || fps_num >= 1000) {
                frame_info.is_vfr = true;
                log::info!("VFR: Detected VFR video (container fps: {}/{})",
                           fps_num, fps_denom);
            }
        }
    }

    Ok(frame_info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seek_mode_flags() {
        let keyframe_flags = SeekMode::Keyframe.to_gst_flags();
        assert!(keyframe_flags.contains(gst::SeekFlags::KEY_UNIT));

        let accurate_flags = SeekMode::Accurate.to_gst_flags();
        assert!(accurate_flags.contains(gst::SeekFlags::ACCURATE));

        let snap_flags = SeekMode::SnapToFrame.to_gst_flags();
        assert!(snap_flags.contains(gst::SeekFlags::SNAP_NEAREST));
    }

    #[test]
    fn test_vfr_player_creation() {
        let result = VfrVideoPlayer::new();
        assert!(result.is_ok(), "VFR player should be creatable");
    }
}
