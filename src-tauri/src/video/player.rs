//! GStreamer Video Player Implementation
//!
//! Provides hardware-accelerated video playback using GStreamer with automatic
//! hardware decoder selection (NVIDIA NVDEC, VA-API, or software fallback).

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_player as gst_player;
use std::sync::{Arc, Mutex};

/// Video player state
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerState {
    Stopped,
    Playing,
    Paused,
    Buffering,
    Error(String),
}

/// Hardware decoder type detected
#[derive(Debug, Clone, PartialEq)]
pub enum HardwareDecoder {
    Nvidia,
    VaApi,
    Software,
    Unknown,
}

/// GStreamer-based video player
pub struct GstVideoPlayer {
    player: gst_player::Player,
    state: Arc<Mutex<PlayerState>>,
    hw_decoder: HardwareDecoder,
}

impl GstVideoPlayer {
    /// Create a new video player instance
    ///
    /// This initializes GStreamer if not already done and creates a player
    /// with hardware acceleration support.
    pub fn new() -> Result<Self, String> {
        // Initialize GStreamer
        gst::init().map_err(|e| format!("Failed to init GStreamer: {}", e))?;

        // Detect available hardware decoders
        let hw_decoder = Self::detect_hw_decoder();
        log::info!("Hardware decoder detected: {:?}", hw_decoder);

        // Create player with video renderer
        // Using default video renderer which will pick appropriate sink
        let player = gst_player::Player::new(
            None::<gst_player::PlayerVideoRenderer>,
            None::<gst_player::PlayerSignalDispatcher>,
        );

        let state = Arc::new(Mutex::new(PlayerState::Stopped));

        // Connect to player signals
        {
            let state_clone = state.clone();
            player.connect_state_changed(move |_player, new_state| {
                let mut state = state_clone.lock().unwrap();
                *state = match new_state {
                    gst_player::PlayerState::Stopped => PlayerState::Stopped,
                    gst_player::PlayerState::Playing => PlayerState::Playing,
                    gst_player::PlayerState::Paused => PlayerState::Paused,
                    gst_player::PlayerState::Buffering => PlayerState::Buffering,
                    _ => PlayerState::Stopped, // Handle any unknown states
                };
                log::debug!("Player state changed: {:?}", *state);
            });
        }

        {
            let state_clone = state.clone();
            player.connect_error(move |_player, error| {
                let mut state = state_clone.lock().unwrap();
                *state = PlayerState::Error(error.to_string());
                log::error!("Player error: {}", error);
            });
        }

        player.connect_warning(|_player, warning| {
            log::warn!("Player warning: {}", warning);
        });

        player.connect_end_of_stream(|_player| {
            log::info!("End of stream reached");
        });

        Ok(Self {
            player,
            state,
            hw_decoder,
        })
    }

    /// Detect available hardware decoders
    fn detect_hw_decoder() -> HardwareDecoder {
        let registry = gst::Registry::get();

        // Check for NVIDIA decoder
        if registry.find_feature("nvh264dec", gst::ElementFactory::static_type()).is_some() {
            log::info!("NVIDIA hardware decoder available");
            return HardwareDecoder::Nvidia;
        }

        // Check for VA-API decoder
        if registry.find_feature("vaapih264dec", gst::ElementFactory::static_type()).is_some() {
            log::info!("VA-API hardware decoder available");
            return HardwareDecoder::VaApi;
        }

        // Check for software decoder
        if registry.find_feature("avdec_h264", gst::ElementFactory::static_type()).is_some() {
            log::info!("Software decoder (ffmpeg) available");
            return HardwareDecoder::Software;
        }

        HardwareDecoder::Unknown
    }

    /// Play a video file
    pub fn play_uri(&self, uri: &str) -> Result<(), String> {
        // Ensure URI has proper scheme
        let full_uri = if uri.starts_with("file://") || uri.starts_with("http://") || uri.starts_with("https://") {
            uri.to_string()
        } else {
            format!("file://{}", uri)
        };

        log::info!("Playing: {}", full_uri);
        self.player.set_uri(Some(&full_uri));
        self.player.play();
        Ok(())
    }

    /// Pause playback
    pub fn pause(&self) {
        self.player.pause();
    }

    /// Resume playback
    pub fn resume(&self) {
        self.player.play();
    }

    /// Stop playback
    pub fn stop(&self) {
        self.player.stop();
    }

    /// Seek to position in nanoseconds
    pub fn seek(&self, position_ns: u64) {
        self.player.seek(gst::ClockTime::from_nseconds(position_ns));
    }

    /// Get current position in nanoseconds
    pub fn position(&self) -> Option<u64> {
        self.player.position().map(|p| p.nseconds())
    }

    /// Get duration in nanoseconds
    pub fn duration(&self) -> Option<u64> {
        self.player.duration().map(|d| d.nseconds())
    }

    /// Get current state
    pub fn state(&self) -> PlayerState {
        self.state.lock().unwrap().clone()
    }

    /// Get detected hardware decoder
    pub fn hardware_decoder(&self) -> HardwareDecoder {
        self.hw_decoder.clone()
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&self, volume: f64) {
        self.player.set_volume(volume.clamp(0.0, 1.0));
    }

    /// Get current volume
    pub fn volume(&self) -> f64 {
        self.player.volume()
    }

    /// Set mute state
    pub fn set_muted(&self, muted: bool) {
        self.player.set_mute(muted);
    }

    /// Check if muted
    pub fn is_muted(&self) -> bool {
        self.player.is_muted()
    }

    /// Set playback rate (1.0 = normal, 2.0 = 2x speed)
    pub fn set_rate(&self, rate: f64) {
        self.player.set_rate(rate);
    }
}

impl Drop for GstVideoPlayer {
    fn drop(&mut self) {
        self.player.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_creation() {
        // This test requires GStreamer to be installed
        let result = GstVideoPlayer::new();
        assert!(result.is_ok(), "Player should be creatable");
    }

    #[test]
    fn test_hw_detection() {
        let _ = gst::init();
        let hw = GstVideoPlayer::detect_hw_decoder();
        // Should detect something (at least software)
        assert_ne!(hw, HardwareDecoder::Unknown);
    }
}
