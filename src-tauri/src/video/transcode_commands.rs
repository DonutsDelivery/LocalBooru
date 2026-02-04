//! Tauri commands for transcoding control
//!
//! These commands expose the GStreamer transcoding pipeline to the frontend.

use super::transcode::{
    HardwareEncoder, TranscodeManager, TranscodeProgress, TranscodeQuality,
    needs_transcoding, probe_video_codec,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

/// State wrapper for the transcode manager
pub struct TranscodeManagerState(pub Arc<Mutex<Option<TranscodeManager>>>);

/// Information about transcoding capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeCapabilities {
    /// Hardware encoder available
    pub hardware_encoder: String,
    /// Whether NVENC is available
    pub nvenc_available: bool,
    /// Whether VA-API is available
    pub vaapi_available: bool,
    /// Available quality presets
    pub quality_presets: Vec<String>,
}

/// Request to start transcoding
#[derive(Debug, Clone, Deserialize)]
pub struct TranscodeRequest {
    /// Source video path
    pub source_path: String,
    /// Quality preset (low, medium, high, original)
    pub quality: String,
    /// Start position in seconds (optional)
    pub start_position: Option<f64>,
}

/// Response from starting a transcode
#[derive(Debug, Clone, Serialize)]
pub struct TranscodeStartResponse {
    /// Stream ID for tracking
    pub stream_id: String,
    /// Path to HLS playlist
    pub playlist_path: String,
    /// Encoder being used
    pub encoder: String,
}

/// Get transcoding capabilities
#[tauri::command]
pub async fn transcode_get_capabilities(
    state: State<'_, TranscodeManagerState>,
) -> Result<TranscodeCapabilities, String> {
    let mut guard = state.0.lock().await;

    // Initialize manager if needed
    if guard.is_none() {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("localbooru")
            .join("transcode");

        *guard = Some(TranscodeManager::new(cache_dir)?);
    }

    let manager = guard.as_ref().unwrap();
    let hw_encoder = manager.hardware_encoder();

    Ok(TranscodeCapabilities {
        hardware_encoder: format!("{:?}", hw_encoder),
        nvenc_available: matches!(hw_encoder, HardwareEncoder::Nvenc),
        vaapi_available: matches!(hw_encoder, HardwareEncoder::VaApi),
        quality_presets: vec![
            "low".to_string(),
            "medium".to_string(),
            "high".to_string(),
            "original".to_string(),
        ],
    })
}

/// Check if a video needs transcoding for browser playback
#[tauri::command]
pub async fn transcode_check_needed(video_path: String) -> Result<bool, String> {
    let path = PathBuf::from(&video_path);

    if !path.exists() {
        return Err(format!("File not found: {}", video_path));
    }

    match probe_video_codec(&path) {
        Ok(codec) => {
            let needs = needs_transcoding(&codec);
            log::info!(
                "Video {} codec: {}, needs transcoding: {}",
                video_path,
                codec,
                needs
            );
            Ok(needs)
        }
        Err(e) => {
            log::warn!("Failed to probe video codec: {}", e);
            // If we can't probe, assume it might need transcoding
            Ok(true)
        }
    }
}

/// Start transcoding a video
#[tauri::command]
pub async fn transcode_start(
    state: State<'_, TranscodeManagerState>,
    request: TranscodeRequest,
) -> Result<TranscodeStartResponse, String> {
    let mut guard = state.0.lock().await;

    // Initialize manager if needed
    if guard.is_none() {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("localbooru")
            .join("transcode");

        *guard = Some(TranscodeManager::new(cache_dir)?);
    }

    let manager = guard.as_mut().unwrap();

    let quality = match request.quality.to_lowercase().as_str() {
        "low" => TranscodeQuality::Low,
        "medium" => TranscodeQuality::Medium,
        "high" => TranscodeQuality::High,
        "original" => TranscodeQuality::Original,
        _ => TranscodeQuality::Medium,
    };

    let source_path = PathBuf::from(&request.source_path);

    if !source_path.exists() {
        return Err(format!("Source file not found: {}", request.source_path));
    }

    let stream_id = manager.start_transcode(source_path, quality)?;

    let playlist_path = manager
        .get_playlist_path(&stream_id)
        .ok_or("Failed to get playlist path")?
        .to_string_lossy()
        .to_string();

    let encoder = format!("{:?}", manager.hardware_encoder());

    Ok(TranscodeStartResponse {
        stream_id,
        playlist_path,
        encoder,
    })
}

/// Get transcoding progress
#[tauri::command]
pub async fn transcode_get_progress(
    state: State<'_, TranscodeManagerState>,
    stream_id: String,
) -> Result<TranscodeProgress, String> {
    let guard = state.0.lock().await;

    let manager = guard.as_ref().ok_or("Transcode manager not initialized")?;

    manager
        .get_progress(&stream_id)
        .ok_or_else(|| format!("Stream not found: {}", stream_id))
}

/// Check if stream is ready for playback
#[tauri::command]
pub async fn transcode_is_ready(
    state: State<'_, TranscodeManagerState>,
    stream_id: String,
) -> Result<bool, String> {
    let guard = state.0.lock().await;

    let manager = guard.as_ref().ok_or("Transcode manager not initialized")?;

    Ok(manager.is_stream_ready(&stream_id))
}

/// Get HLS playlist path for a stream
#[tauri::command]
pub async fn transcode_get_playlist(
    state: State<'_, TranscodeManagerState>,
    stream_id: String,
) -> Result<String, String> {
    let guard = state.0.lock().await;

    let manager = guard.as_ref().ok_or("Transcode manager not initialized")?;

    manager
        .get_playlist_path(&stream_id)
        .map(|p| p.to_string_lossy().to_string())
        .ok_or_else(|| format!("Stream not found: {}", stream_id))
}

/// Stop a transcoding session
#[tauri::command]
pub async fn transcode_stop(
    state: State<'_, TranscodeManagerState>,
    stream_id: String,
) -> Result<(), String> {
    let mut guard = state.0.lock().await;

    let manager = guard.as_mut().ok_or("Transcode manager not initialized")?;

    manager.stop_transcode(&stream_id)
}

/// Stop all transcoding sessions
#[tauri::command]
pub async fn transcode_stop_all(state: State<'_, TranscodeManagerState>) -> Result<(), String> {
    let mut guard = state.0.lock().await;

    if let Some(manager) = guard.as_mut() {
        manager.stop_all();
    }

    Ok(())
}

/// Cleanup transcode cache
#[tauri::command]
pub async fn transcode_cleanup_cache(
    state: State<'_, TranscodeManagerState>,
) -> Result<u64, String> {
    let guard = state.0.lock().await;

    let manager = guard.as_ref().ok_or("Transcode manager not initialized")?;

    manager.cleanup_cache()
}

/// Set maximum cache size in bytes
#[tauri::command]
pub async fn transcode_set_cache_limit(
    state: State<'_, TranscodeManagerState>,
    max_bytes: u64,
) -> Result<(), String> {
    let mut guard = state.0.lock().await;

    if let Some(manager) = guard.as_mut() {
        manager.set_max_cache_bytes(max_bytes);
    }

    Ok(())
}
