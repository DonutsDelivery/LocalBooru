//! Tauri commands for frame interpolation
//!
//! These commands expose the interpolation pipeline functionality to the frontend,
//! allowing users to enable/disable interpolation, configure settings, and monitor
//! performance.

use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use super::interpolation::{
    InterpolatedPlayer, InterpolationConfig, InterpolationBackend,
    InterpolationPreset, InterpolationState, BackendCapabilities,
    detect_available_backends,
};

/// State wrapper for the interpolated player
pub struct InterpolatedPlayerState(pub Arc<Mutex<Option<InterpolatedPlayer>>>);

/// State wrapper for interpolation configuration (persisted separately)
pub struct InterpolationConfigState(pub Arc<Mutex<InterpolationConfig>>);

/// DTO for backend capabilities (frontend-friendly)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilitiesDto {
    pub backend: String,
    pub available: bool,
    pub status: String,
    pub path: Option<String>,
    pub version: Option<String>,
    pub gpu_available: bool,
    pub nvof_available: bool,
    pub max_resolution: Option<String>,
    pub performance_tier: u8,
}

impl From<BackendCapabilities> for BackendCapabilitiesDto {
    fn from(caps: BackendCapabilities) -> Self {
        Self {
            backend: caps.backend.as_str().to_string(),
            available: caps.available,
            status: caps.status,
            path: caps.path,
            version: caps.version,
            gpu_available: caps.gpu_available,
            nvof_available: caps.nvof_available,
            max_resolution: caps.max_realtime_resolution.map(|(w, h)| format!("{}x{}", w, h)),
            performance_tier: caps.performance_tier,
        }
    }
}

/// DTO for interpolation configuration (frontend-friendly)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationConfigDto {
    pub enabled: bool,
    pub backend: String,
    pub preset: String,
    pub target_fps: u32,
    pub use_nvof: bool,
    pub gpu_id: i32,
    pub svp_algorithm: u32,
    pub artifact_masking: u32,
    pub scene_sensitivity: u32,
}

impl From<&InterpolationConfig> for InterpolationConfigDto {
    fn from(config: &InterpolationConfig) -> Self {
        Self {
            enabled: config.enabled,
            backend: config.backend.as_str().to_string(),
            preset: config.preset.as_str().to_string(),
            target_fps: config.target_fps,
            use_nvof: config.use_nvof,
            gpu_id: config.gpu_id,
            svp_algorithm: config.svp_algorithm,
            artifact_masking: config.artifact_masking,
            scene_sensitivity: config.scene_sensitivity,
        }
    }
}

impl From<InterpolationConfigDto> for InterpolationConfig {
    fn from(dto: InterpolationConfigDto) -> Self {
        Self {
            enabled: dto.enabled,
            backend: InterpolationBackend::from_str(&dto.backend).unwrap_or_default(),
            preset: match dto.preset.as_str() {
                "fast" => InterpolationPreset::Fast,
                "balanced" => InterpolationPreset::Balanced,
                "quality" => InterpolationPreset::Quality,
                "max" => InterpolationPreset::Max,
                "animation" => InterpolationPreset::Animation,
                "film" => InterpolationPreset::Film,
                _ => InterpolationPreset::Balanced,
            },
            target_fps: dto.target_fps,
            use_nvof: dto.use_nvof,
            gpu_id: dto.gpu_id,
            svp_algorithm: dto.svp_algorithm,
            artifact_masking: dto.artifact_masking,
            scene_sensitivity: dto.scene_sensitivity,
            custom_super: None,
            custom_analyse: None,
            custom_smooth: None,
        }
    }
}

/// DTO for interpolation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationStatsDto {
    pub frames_processed: u64,
    pub frames_interpolated: u64,
    pub avg_frame_time_ms: f64,
    pub effective_fps: f64,
    pub buffer_level: f64,
    pub frame_drops: u64,
    pub state: String,
}

/// DTO for preset info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetInfoDto {
    pub id: String,
    pub name: String,
    pub description: String,
}

// =============================================================================
// Tauri Commands
// =============================================================================

/// Detect available interpolation backends
#[tauri::command]
pub fn interpolation_detect_backends() -> Vec<BackendCapabilitiesDto> {
    detect_available_backends()
        .into_iter()
        .map(BackendCapabilitiesDto::from)
        .collect()
}

/// Get current interpolation configuration
#[tauri::command]
pub async fn interpolation_get_config(
    state: State<'_, InterpolationConfigState>,
) -> Result<InterpolationConfigDto, String> {
    let config = state.0.lock().await;
    Ok(InterpolationConfigDto::from(&*config))
}

/// Update interpolation configuration
#[tauri::command]
pub async fn interpolation_set_config(
    config_state: State<'_, InterpolationConfigState>,
    player_state: State<'_, InterpolatedPlayerState>,
    config: InterpolationConfigDto,
) -> Result<(), String> {
    let new_config: InterpolationConfig = config.into();

    // Update config state
    {
        let mut cfg = config_state.0.lock().await;
        *cfg = new_config.clone();
    }

    // Update player if exists
    {
        let mut player_guard = player_state.0.lock().await;
        if let Some(ref mut player) = *player_guard {
            player.set_config(new_config);
        }
    }

    Ok(())
}

/// Enable or disable interpolation
#[tauri::command]
pub async fn interpolation_set_enabled(
    config_state: State<'_, InterpolationConfigState>,
    player_state: State<'_, InterpolatedPlayerState>,
    enabled: bool,
) -> Result<(), String> {
    // Update config
    {
        let mut config = config_state.0.lock().await;
        config.enabled = enabled;

        // If disabling, set backend to None
        if !enabled {
            config.backend = InterpolationBackend::None;
        } else if config.backend == InterpolationBackend::None {
            // Auto-select best backend when enabling
            let backends = detect_available_backends();
            if let Some(best) = backends.iter().filter(|b| b.available).max_by_key(|b| b.performance_tier) {
                config.backend = best.backend;
            }
        }
    }

    // Update player if exists
    {
        let config = config_state.0.lock().await;
        let mut player_guard = player_state.0.lock().await;
        if let Some(ref mut player) = *player_guard {
            player.set_config(config.clone());
        }
    }

    Ok(())
}

/// Set interpolation backend
#[tauri::command]
pub async fn interpolation_set_backend(
    config_state: State<'_, InterpolationConfigState>,
    player_state: State<'_, InterpolatedPlayerState>,
    backend: String,
) -> Result<(), String> {
    let backend_enum = InterpolationBackend::from_str(&backend)
        .ok_or_else(|| format!("Unknown backend: {}", backend))?;

    // Check if backend is available
    let backends = detect_available_backends();
    let backend_caps = backends.iter().find(|b| b.backend == backend_enum);

    if let Some(caps) = backend_caps {
        if !caps.available {
            return Err(format!("Backend '{}' is not available: {}", backend, caps.status));
        }
    }

    // Update config
    {
        let mut config = config_state.0.lock().await;
        config.backend = backend_enum;
        config.enabled = backend_enum != InterpolationBackend::None;
    }

    // Update player if exists
    {
        let config = config_state.0.lock().await;
        let mut player_guard = player_state.0.lock().await;
        if let Some(ref mut player) = *player_guard {
            player.set_config(config.clone());
        }
    }

    Ok(())
}

/// Set target FPS
#[tauri::command]
pub async fn interpolation_set_target_fps(
    config_state: State<'_, InterpolationConfigState>,
    player_state: State<'_, InterpolatedPlayerState>,
    fps: u32,
) -> Result<(), String> {
    // Validate FPS range
    if fps < 24 || fps > 240 {
        return Err("Target FPS must be between 24 and 240".to_string());
    }

    // Update config
    {
        let mut config = config_state.0.lock().await;
        config.target_fps = fps;
    }

    // Update player
    {
        let config = config_state.0.lock().await;
        let mut player_guard = player_state.0.lock().await;
        if let Some(ref mut player) = *player_guard {
            player.set_config(config.clone());
        }
    }

    Ok(())
}

/// Set quality preset
#[tauri::command]
pub async fn interpolation_set_preset(
    config_state: State<'_, InterpolationConfigState>,
    player_state: State<'_, InterpolatedPlayerState>,
    preset: String,
) -> Result<(), String> {
    let preset_enum = match preset.as_str() {
        "fast" => InterpolationPreset::Fast,
        "balanced" => InterpolationPreset::Balanced,
        "quality" => InterpolationPreset::Quality,
        "max" => InterpolationPreset::Max,
        "animation" => InterpolationPreset::Animation,
        "film" => InterpolationPreset::Film,
        _ => return Err(format!("Unknown preset: {}", preset)),
    };

    // Update config
    {
        let mut config = config_state.0.lock().await;
        config.preset = preset_enum;
    }

    // Update player
    {
        let config = config_state.0.lock().await;
        let mut player_guard = player_state.0.lock().await;
        if let Some(ref mut player) = *player_guard {
            player.set_config(config.clone());
        }
    }

    Ok(())
}

/// Get available presets
#[tauri::command]
pub fn interpolation_get_presets() -> Vec<PresetInfoDto> {
    vec![
        PresetInfoDto {
            id: "fast".to_string(),
            name: "Fast".to_string(),
            description: InterpolationPreset::Fast.description().to_string(),
        },
        PresetInfoDto {
            id: "balanced".to_string(),
            name: "Balanced".to_string(),
            description: InterpolationPreset::Balanced.description().to_string(),
        },
        PresetInfoDto {
            id: "quality".to_string(),
            name: "Quality".to_string(),
            description: InterpolationPreset::Quality.description().to_string(),
        },
        PresetInfoDto {
            id: "max".to_string(),
            name: "Maximum".to_string(),
            description: InterpolationPreset::Max.description().to_string(),
        },
        PresetInfoDto {
            id: "animation".to_string(),
            name: "Animation".to_string(),
            description: InterpolationPreset::Animation.description().to_string(),
        },
        PresetInfoDto {
            id: "film".to_string(),
            name: "Film".to_string(),
            description: InterpolationPreset::Film.description().to_string(),
        },
    ]
}

/// Initialize the interpolated player
#[tauri::command]
pub async fn interpolation_init(
    config_state: State<'_, InterpolationConfigState>,
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<String, String> {
    let mut player_guard = player_state.0.lock().await;

    if player_guard.is_some() {
        return Ok("Interpolated player already initialized".to_string());
    }

    let config = config_state.0.lock().await.clone();
    let player = InterpolatedPlayer::new(config);

    *player_guard = Some(player);
    Ok("Interpolated player initialized".to_string())
}

/// Play a video with interpolation
#[tauri::command]
pub async fn interpolation_play(
    player_state: State<'_, InterpolatedPlayerState>,
    path: String,
) -> Result<(), String> {
    let mut player_guard = player_state.0.lock().await;
    let player = player_guard.as_mut().ok_or("Player not initialized")?;

    let video_path = PathBuf::from(&path);
    if !video_path.exists() {
        return Err(format!("Video file not found: {}", path));
    }

    player.play(&video_path)
}

/// Stop interpolated playback
#[tauri::command]
pub async fn interpolation_stop(
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<(), String> {
    let mut player_guard = player_state.0.lock().await;
    if let Some(ref mut player) = *player_guard {
        player.stop();
    }
    Ok(())
}

/// Pause interpolated playback
#[tauri::command]
pub async fn interpolation_pause(
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<(), String> {
    let mut player_guard = player_state.0.lock().await;
    let player = player_guard.as_mut().ok_or("Player not initialized")?;
    player.pause()
}

/// Resume interpolated playback
#[tauri::command]
pub async fn interpolation_resume(
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<(), String> {
    let mut player_guard = player_state.0.lock().await;
    let player = player_guard.as_mut().ok_or("Player not initialized")?;
    player.resume()
}

/// Seek to position (seconds)
#[tauri::command]
pub async fn interpolation_seek(
    player_state: State<'_, InterpolatedPlayerState>,
    position: f64,
) -> Result<(), String> {
    let player_guard = player_state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.seek(position)
}

/// Set volume (0.0 - 1.0)
#[tauri::command]
pub async fn interpolation_set_volume(
    player_state: State<'_, InterpolatedPlayerState>,
    volume: f64,
) -> Result<(), String> {
    let player_guard = player_state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;
    player.set_volume(volume.clamp(0.0, 1.0))
}

/// Get current player state
#[tauri::command]
pub async fn interpolation_get_state(
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<String, String> {
    let player_guard = player_state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;

    let state = match player.state() {
        InterpolationState::Uninitialized => "uninitialized",
        InterpolationState::Starting => "starting",
        InterpolationState::Running => "running",
        InterpolationState::Paused => "paused",
        InterpolationState::Stopped => "stopped",
        InterpolationState::Error => "error",
    };

    Ok(state.to_string())
}

/// Get interpolation statistics
#[tauri::command]
pub async fn interpolation_get_stats(
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<InterpolationStatsDto, String> {
    let player_guard = player_state.0.lock().await;
    let player = player_guard.as_ref().ok_or("Player not initialized")?;

    let stats = player.stats();
    let state = player.state();

    Ok(InterpolationStatsDto {
        frames_processed: stats.frames_processed,
        frames_interpolated: stats.frames_interpolated,
        avg_frame_time_ms: stats.avg_frame_time_ms,
        effective_fps: stats.effective_fps,
        buffer_level: stats.buffer_level,
        frame_drops: stats.frame_drops,
        state: format!("{:?}", state).to_lowercase(),
    })
}

/// Cleanup interpolated player
#[tauri::command]
pub async fn interpolation_cleanup(
    player_state: State<'_, InterpolatedPlayerState>,
) -> Result<(), String> {
    let mut player_guard = player_state.0.lock().await;
    if let Some(mut player) = player_guard.take() {
        player.stop();
    }
    Ok(())
}

/// Check if interpolation is available on this system
#[tauri::command]
pub fn interpolation_is_available() -> bool {
    detect_available_backends()
        .iter()
        .any(|b| b.available && b.backend != InterpolationBackend::None)
}

/// Get recommended backend for given video dimensions
#[tauri::command]
pub fn interpolation_recommend_backend(width: u32, height: u32, prefer_quality: bool) -> Option<String> {
    super::interpolation::recommend_backend(width, height, prefer_quality)
        .map(|b| b.as_str().to_string())
}
