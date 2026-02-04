//! Tauri commands for video playback
//!
//! These commands expose GStreamer video functionality to the frontend.
//!
//! # VFR Support
//!
//! The VFR-related commands (`video_vfr_*`) provide enhanced support for
//! variable frame rate videos, which is the primary reason for using
//! GStreamer over HTML5 video.

use super::{GstVideoPlayer, VideoOverlay, overlay::test_gstreamer_setup};
use super::vfr::{VfrVideoPlayer, VfrPlayerState, SeekMode, FrameRateInfo, StreamInfo, analyze_video_for_vfr};
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

/// State wrapper for the video player
pub struct VideoPlayerState(pub Arc<Mutex<Option<GstVideoPlayer>>>);

/// State wrapper for the video overlay
pub struct VideoOverlayState(pub Arc<Mutex<Option<VideoOverlay>>>);

/// State wrapper for VFR-aware video player
pub struct VfrVideoPlayerState(pub Arc<Mutex<Option<VfrVideoPlayer>>>);

/// Get GStreamer system information
#[tauri::command]
pub fn video_get_system_info() -> Result<String, String> {
    test_gstreamer_setup()
}

/// Initialize the video player
#[tauri::command]
pub async fn video_init(
    state: State<'_, VideoPlayerState>,
) -> Result<String, String> {
    let mut player_guard = state.0.lock().await;

    if player_guard.is_some() {
        return Ok("Player already initialized".to_string());
    }

    let player = GstVideoPlayer::new()?;
    let hw_decoder = format!("{:?}", player.hardware_decoder());
    *player_guard = Some(player);

    Ok(format!("Player initialized with {} decoder", hw_decoder))
}

/// Play a video file
#[tauri::command]
pub async fn video_play(
    state: State<'_, VideoPlayerState>,
    uri: String,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.play_uri(&uri)
}

/// Pause video playback
#[tauri::command]
pub async fn video_pause(
    state: State<'_, VideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.pause();
    Ok(())
}

/// Resume video playback
#[tauri::command]
pub async fn video_resume(
    state: State<'_, VideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.resume();
    Ok(())
}

/// Stop video playback
#[tauri::command]
pub async fn video_stop(
    state: State<'_, VideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.stop();
    Ok(())
}

/// Seek to position (in seconds)
#[tauri::command]
pub async fn video_seek(
    state: State<'_, VideoPlayerState>,
    position_secs: f64,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    let position_ns = (position_secs * 1_000_000_000.0) as u64;
    player.seek(position_ns);
    Ok(())
}

/// Get current playback position (in seconds)
#[tauri::command]
pub async fn video_get_position(
    state: State<'_, VideoPlayerState>,
) -> Result<f64, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    Ok(player.position().map(|ns| ns as f64 / 1_000_000_000.0).unwrap_or(0.0))
}

/// Get video duration (in seconds)
#[tauri::command]
pub async fn video_get_duration(
    state: State<'_, VideoPlayerState>,
) -> Result<f64, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    Ok(player.duration().map(|ns| ns as f64 / 1_000_000_000.0).unwrap_or(0.0))
}

/// Set volume (0.0 to 1.0)
#[tauri::command]
pub async fn video_set_volume(
    state: State<'_, VideoPlayerState>,
    volume: f64,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.set_volume(volume);
    Ok(())
}

/// Get current volume
#[tauri::command]
pub async fn video_get_volume(
    state: State<'_, VideoPlayerState>,
) -> Result<f64, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    Ok(player.volume())
}

/// Set mute state
#[tauri::command]
pub async fn video_set_muted(
    state: State<'_, VideoPlayerState>,
    muted: bool,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.set_muted(muted);
    Ok(())
}

/// Get player state
#[tauri::command]
pub async fn video_get_state(
    state: State<'_, VideoPlayerState>,
) -> Result<String, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    Ok(format!("{:?}", player.state()))
}

/// Cleanup video player
#[tauri::command]
pub async fn video_cleanup(
    state: State<'_, VideoPlayerState>,
) -> Result<(), String> {
    let mut player_guard = state.0.lock().await;
    *player_guard = None;
    Ok(())
}

// =============================================================================
// VFR (Variable Frame Rate) Video Commands
// =============================================================================

/// Serializable frame rate info for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameRateInfoDto {
    pub is_vfr: bool,
    pub average_fps: f64,
    pub min_frame_duration_ms: Option<f64>,
    pub max_frame_duration_ms: Option<f64>,
    pub container_fps: Option<f64>,
}

impl From<FrameRateInfo> for FrameRateInfoDto {
    fn from(info: FrameRateInfo) -> Self {
        Self {
            is_vfr: info.is_vfr,
            average_fps: info.average_fps,
            min_frame_duration_ms: info.min_frame_duration_ns.map(|ns| ns as f64 / 1_000_000.0),
            max_frame_duration_ms: info.max_frame_duration_ns.map(|ns| ns as f64 / 1_000_000.0),
            container_fps: info.container_fps,
        }
    }
}

/// Serializable stream info for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfoDto {
    pub duration_secs: Option<f64>,
    pub position_secs: Option<f64>,
    pub seekable: bool,
    pub seek_start_secs: Option<f64>,
    pub seek_end_secs: Option<f64>,
    pub buffering_percent: Option<i32>,
}

impl From<StreamInfo> for StreamInfoDto {
    fn from(info: StreamInfo) -> Self {
        Self {
            duration_secs: info.duration_ns.map(|ns| ns as f64 / 1_000_000_000.0),
            position_secs: info.position_ns.map(|ns| ns as f64 / 1_000_000_000.0),
            seekable: info.seekable,
            seek_start_secs: info.seek_start_ns.map(|ns| ns as f64 / 1_000_000_000.0),
            seek_end_secs: info.seek_end_ns.map(|ns| ns as f64 / 1_000_000_000.0),
            buffering_percent: info.buffering_percent,
        }
    }
}

/// Seek mode enum for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeekModeDto {
    Keyframe,
    Accurate,
    SnapToFrame,
}

impl From<SeekModeDto> for SeekMode {
    fn from(mode: SeekModeDto) -> Self {
        match mode {
            SeekModeDto::Keyframe => SeekMode::Keyframe,
            SeekModeDto::Accurate => SeekMode::Accurate,
            SeekModeDto::SnapToFrame => SeekMode::SnapToFrame,
        }
    }
}

impl From<SeekMode> for SeekModeDto {
    fn from(mode: SeekMode) -> Self {
        match mode {
            SeekMode::Keyframe => SeekModeDto::Keyframe,
            SeekMode::Accurate => SeekModeDto::Accurate,
            SeekMode::SnapToFrame => SeekModeDto::SnapToFrame,
        }
    }
}

/// Analyze a video file for VFR characteristics
///
/// Returns information about whether the video is VFR and its frame rate properties.
/// This is useful for deciding how to configure playback.
#[tauri::command]
pub async fn video_vfr_analyze(uri: String) -> Result<FrameRateInfoDto, String> {
    // Run analysis in blocking task since it may take a moment
    let result = tokio::task::spawn_blocking(move || {
        analyze_video_for_vfr(&uri)
    })
    .await
    .map_err(|e| format!("Task join error: {}", e))?;

    result.map(|info| info.into())
}

/// Initialize the VFR-aware video player
#[tauri::command]
pub async fn video_vfr_init(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<String, String> {
    let mut player_guard = state.0.lock().await;

    if player_guard.is_some() {
        return Ok("VFR player already initialized".to_string());
    }

    let player = VfrVideoPlayer::new()?;
    *player_guard = Some(player);

    Ok("VFR player initialized".to_string())
}

/// Play a video with VFR-aware handling
#[tauri::command]
pub async fn video_vfr_play(
    state: State<'_, VfrVideoPlayerState>,
    uri: String,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.play_uri(&uri)
}

/// Pause VFR video playback
#[tauri::command]
pub async fn video_vfr_pause(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.pause()
}

/// Resume VFR video playback
#[tauri::command]
pub async fn video_vfr_resume(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.resume()
}

/// Stop VFR video playback
#[tauri::command]
pub async fn video_vfr_stop(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.stop()
}

/// Seek with VFR-aware timestamp handling
///
/// Uses the current seek mode for proper VFR handling.
#[tauri::command]
pub async fn video_vfr_seek(
    state: State<'_, VfrVideoPlayerState>,
    position_secs: f64,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    let position_ns = (position_secs * 1_000_000_000.0) as u64;
    player.seek(position_ns)
}

/// Seek with specific mode
///
/// Allows frontend to choose between:
/// - "keyframe": Fast but may not be exact
/// - "accurate": Exact position, slower
/// - "snap_to_frame": Snaps to nearest actual frame (best for VFR)
#[tauri::command]
pub async fn video_vfr_seek_with_mode(
    state: State<'_, VfrVideoPlayerState>,
    position_secs: f64,
    mode: SeekModeDto,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    let position_ns = (position_secs * 1_000_000_000.0) as u64;
    player.seek_with_mode(position_ns, mode.into())
}

/// Step forward one frame
///
/// Useful for frame-by-frame navigation in VFR content.
#[tauri::command]
pub async fn video_vfr_step_frame(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.step_frame()
}

/// Get current playback position in seconds
#[tauri::command]
pub async fn video_vfr_get_position(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<f64, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(player.position().map(|ns| ns as f64 / 1_000_000_000.0).unwrap_or(0.0))
}

/// Get video duration in seconds
#[tauri::command]
pub async fn video_vfr_get_duration(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<f64, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(player.duration().map(|ns| ns as f64 / 1_000_000_000.0).unwrap_or(0.0))
}

/// Get current player state
#[tauri::command]
pub async fn video_vfr_get_state(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<String, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(format!("{:?}", player.state()))
}

/// Get detailed stream information
#[tauri::command]
pub async fn video_vfr_get_stream_info(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<StreamInfoDto, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.query_stream_info().map(|info| info.into())
}

/// Get frame rate information
#[tauri::command]
pub async fn video_vfr_get_frame_info(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<FrameRateInfoDto, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(player.frame_rate_info().into())
}

/// Set the default seek mode
#[tauri::command]
pub async fn video_vfr_set_seek_mode(
    state: State<'_, VfrVideoPlayerState>,
    mode: SeekModeDto,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.set_seek_mode(mode.into());
    Ok(())
}

/// Get current seek mode
#[tauri::command]
pub async fn video_vfr_get_seek_mode(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<SeekModeDto, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(player.seek_mode().into())
}

/// Set playback rate (e.g., 0.5 for half speed, 2.0 for double speed)
#[tauri::command]
pub async fn video_vfr_set_rate(
    state: State<'_, VfrVideoPlayerState>,
    rate: f64,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.set_rate(rate)
}

/// Set volume (0.0 to 1.0)
#[tauri::command]
pub async fn video_vfr_set_volume(
    state: State<'_, VfrVideoPlayerState>,
    volume: f64,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.set_volume(volume);
    Ok(())
}

/// Get current volume
#[tauri::command]
pub async fn video_vfr_get_volume(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<f64, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(player.volume())
}

/// Set mute state
#[tauri::command]
pub async fn video_vfr_set_muted(
    state: State<'_, VfrVideoPlayerState>,
    muted: bool,
) -> Result<(), String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    player.set_muted(muted);
    Ok(())
}

/// Check if muted
#[tauri::command]
pub async fn video_vfr_is_muted(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<bool, String> {
    let player_guard = state.0.lock().await;
    let player = player_guard.as_ref().ok_or("VFR player not initialized")?;
    Ok(player.is_muted())
}

/// Cleanup VFR video player
#[tauri::command]
pub async fn video_vfr_cleanup(
    state: State<'_, VfrVideoPlayerState>,
) -> Result<(), String> {
    let mut player_guard = state.0.lock().await;
    *player_guard = None;
    Ok(())
}
