//! Video Event Streaming for Tauri
//!
//! This module provides real-time event streaming from the GStreamer video player
//! to the frontend via Tauri events.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, Manager, Emitter};
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};

/// Video playback event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoEvent {
    /// Position update (time in seconds)
    Position {
        position_secs: f64,
        duration_secs: f64,
    },
    /// State changed
    StateChanged {
        state: String,
    },
    /// Buffering progress
    Buffering {
        percent: i32,
    },
    /// End of stream reached
    EndOfStream,
    /// Error occurred
    Error {
        message: String,
    },
    /// Seek completed
    SeekCompleted {
        position_secs: f64,
    },
    /// Volume changed
    VolumeChanged {
        volume: f64,
        muted: bool,
    },
    /// Playback rate changed
    RateChanged {
        rate: f64,
    },
    /// Frame info available (VFR detection)
    FrameInfo {
        is_vfr: bool,
        average_fps: f64,
        container_fps: Option<f64>,
    },
}

/// State for the video event streamer
pub struct VideoEventStreamer {
    /// Whether streaming is active
    is_streaming: bool,
    /// App handle for emitting events
    app_handle: Option<AppHandle>,
}

impl VideoEventStreamer {
    pub fn new() -> Self {
        Self {
            is_streaming: false,
            app_handle: None,
        }
    }

    /// Set the app handle for emitting events
    pub fn set_app_handle(&mut self, handle: AppHandle) {
        self.app_handle = Some(handle);
    }

    /// Emit a video event to the frontend
    pub fn emit(&self, event: VideoEvent) {
        if let Some(ref handle) = self.app_handle {
            if let Err(e) = handle.emit("video_event", event) {
                log::warn!("Failed to emit video event: {}", e);
            }
        }
    }

    /// Emit a position update
    pub fn emit_position(&self, position_secs: f64, duration_secs: f64) {
        self.emit(VideoEvent::Position {
            position_secs,
            duration_secs,
        });
    }

    /// Emit a state change
    pub fn emit_state(&self, state: &str) {
        self.emit(VideoEvent::StateChanged {
            state: state.to_string(),
        });
    }

    /// Emit buffering progress
    pub fn emit_buffering(&self, percent: i32) {
        self.emit(VideoEvent::Buffering { percent });
    }

    /// Emit end of stream
    pub fn emit_eos(&self) {
        self.emit(VideoEvent::EndOfStream);
    }

    /// Emit error
    pub fn emit_error(&self, message: &str) {
        self.emit(VideoEvent::Error {
            message: message.to_string(),
        });
    }

    /// Emit seek completed
    pub fn emit_seek_completed(&self, position_secs: f64) {
        self.emit(VideoEvent::SeekCompleted { position_secs });
    }

    /// Emit volume change
    pub fn emit_volume(&self, volume: f64, muted: bool) {
        self.emit(VideoEvent::VolumeChanged { volume, muted });
    }

    /// Emit rate change
    pub fn emit_rate(&self, rate: f64) {
        self.emit(VideoEvent::RateChanged { rate });
    }

    /// Emit frame info (VFR detection result)
    pub fn emit_frame_info(&self, is_vfr: bool, average_fps: f64, container_fps: Option<f64>) {
        self.emit(VideoEvent::FrameInfo {
            is_vfr,
            average_fps,
            container_fps,
        });
    }
}

impl Default for VideoEventStreamer {
    fn default() -> Self {
        Self::new()
    }
}

/// State wrapper for the event streamer
pub struct VideoEventState(pub Arc<Mutex<VideoEventStreamer>>);

/// Start video event streaming (position polling)
///
/// This starts a background task that polls the video player for position updates
/// and emits them as events to the frontend.
#[tauri::command]
pub async fn video_events_start(
    state: tauri::State<'_, VideoEventState>,
    vfr_state: tauri::State<'_, super::commands::VfrVideoPlayerState>,
    app_handle: AppHandle,
    interval_ms: Option<u64>,
) -> Result<(), String> {
    let mut streamer = state.0.lock().await;

    if streamer.is_streaming {
        return Ok(()); // Already streaming
    }

    streamer.set_app_handle(app_handle.clone());
    streamer.is_streaming = true;

    let interval_duration = Duration::from_millis(interval_ms.unwrap_or(100));
    let vfr_player = vfr_state.0.clone();
    let event_state = state.0.clone();

    // Spawn background task for polling
    tokio::spawn(async move {
        let mut ticker = interval(interval_duration);

        loop {
            ticker.tick().await;

            let streamer = event_state.lock().await;
            if !streamer.is_streaming {
                break;
            }

            let player_guard = vfr_player.lock().await;
            if let Some(ref player) = *player_guard {
                // Get position and duration
                let position = player.position().unwrap_or(0) as f64 / 1_000_000_000.0;
                let duration = player.duration().unwrap_or(0) as f64 / 1_000_000_000.0;

                // Emit position update
                streamer.emit_position(position, duration);

                // Check state
                let state = player.state();
                let state_str = format!("{:?}", state);
                streamer.emit_state(&state_str);
            }

            drop(streamer);
        }

        log::info!("Video event streaming stopped");
    });

    Ok(())
}

/// Stop video event streaming
#[tauri::command]
pub async fn video_events_stop(
    state: tauri::State<'_, VideoEventState>,
) -> Result<(), String> {
    let mut streamer = state.0.lock().await;
    streamer.is_streaming = false;
    Ok(())
}

/// Check if event streaming is active
#[tauri::command]
pub async fn video_events_is_active(
    state: tauri::State<'_, VideoEventState>,
) -> Result<bool, String> {
    let streamer = state.0.lock().await;
    Ok(streamer.is_streaming)
}
