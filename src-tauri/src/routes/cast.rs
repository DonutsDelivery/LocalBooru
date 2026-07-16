//! Cast/Chromecast subsystem routes.
//!
//! Provides bridge routes that proxy to the cast addon sidecar (port 18006)
//! and manage in-memory cast state. The actual device discovery and media
//! casting are handled by the Python sidecar; this module manages the Rust
//! side of the state and proxies HTTP requests.

use std::collections::HashMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use axum::extract::{Path as AxumPath, Request, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::process::Command;
use tower::ServiceExt;
use tower_http::services::ServeFile;
use uuid::Uuid;

use crate::addons::manager::AddonStatus;
use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::server::utils::get_local_ip;
use crate::services::transcode::QualityPreset;

// ─── Cast state types ────────────────────────────────────────────────────────

/// In-memory state tracking the current cast session.
#[derive(Debug, Clone, Serialize)]
pub struct CastState {
    pub active_device: Option<CastDevice>,
    pub status: String,
    pub current_media: Option<CastMedia>,
}

impl CastState {
    pub fn new() -> Self {
        Self {
            active_device: None,
            status: "idle".to_string(),
            current_media: None,
        }
    }
}

impl Default for CastState {
    fn default() -> Self {
        Self::new()
    }
}

/// A discovered cast-capable device on the local network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastDevice {
    pub id: String,
    pub name: String,
    pub device_type: String,
    pub host: String,
}

/// Information about the media currently being cast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastMedia {
    pub image_id: i64,
    pub media_url: String,
    #[serde(default)]
    pub transcode_stream_id: Option<String>,
    #[serde(default)]
    pub svp_stream_id: Option<String>,
    pub position: f64,
    pub duration: f64,
}

// ─── Request models ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct PlayRequest {
    device_id: String,
    file_path: String,
    image_id: i64,
    #[serde(rename = "directory_id")]
    _directory_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ControlRequest {
    action: String,
    value: Option<Value>,
}

// ─── Constants ───────────────────────────────────────────────────────────────

const CAST_ADDON_ID: &str = "cast";
const CAST_ADDON_PORT: u16 = 18006;
const SVP_ADDON_ID: &str = "svp";
static REGISTERED_MEDIA: LazyLock<Mutex<HashMap<String, PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/devices", get(list_devices))
        .route("/devices/refresh", post(refresh_devices))
        .route("/play", post(play))
        .route("/control", post(control))
        .route("/stop", post(stop))
        .route("/status", get(status))
}

pub fn media_router() -> Router<AppState> {
    Router::new().route("/{media_id}/file/{filename}", get(serve_cast_file))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Check whether the cast addon sidecar is currently running.
fn is_addon_running(state: &AppState) -> bool {
    state.addon_manager().get_addon_status(CAST_ADDON_ID) == AddonStatus::Running
}

/// Build the base URL for the cast addon sidecar.
fn addon_base_url() -> String {
    format!("http://127.0.0.1:{}", CAST_ADDON_PORT)
}

fn guess_content_type(path: &Path) -> String {
    mime_guess::from_path(path)
        .first_or_octet_stream()
        .essence_str()
        .to_string()
}

fn register_media(file_path: &Path) -> String {
    // Full UUID (not truncated): this id is the only thing protecting the
    // cast-media URL, which is exempt from JWT so cast devices can fetch it.
    // An 8-char prefix is only 32 bits — brute-forceable on the LAN during an
    // active session. The full token is unguessable.
    let media_id = Uuid::new_v4().to_string();
    REGISTERED_MEDIA
        .lock()
        .expect("cast media registry poisoned")
        .insert(media_id.clone(), file_path.to_path_buf());
    media_id
}

fn unregister_media_url(media_url: &str) {
    let Some((_, rest)) = media_url.split_once("/api/cast-media/") else {
        return;
    };
    let Some((media_id, _)) = rest.split_once('/') else {
        return;
    };
    REGISTERED_MEDIA
        .lock()
        .expect("cast media registry poisoned")
        .remove(media_id);
}

async fn cleanup_registered_cast_media(state: &AppState, current_media: &CastMedia) {
    unregister_media_url(&current_media.media_url);
    if let Some(stream_id) = &current_media.transcode_stream_id {
        state.transcode_manager().stop_stream(stream_id);
    }
    if current_media.svp_stream_id.is_some() {
        stop_svp_cast_stream(state).await;
    }
}

fn parse_fps(raw: &str) -> Option<f64> {
    if let Some((num, den)) = raw.split_once('/') {
        let num: f64 = num.parse().ok()?;
        let den: f64 = den.parse().ok()?;
        if den > 0.0 {
            Some(num / den)
        } else {
            None
        }
    } else {
        raw.parse().ok()
    }
}

fn config_u32(config: &Value, key: &str) -> Option<u32> {
    config.get(key).and_then(|value| {
        value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
    })
}

fn cast_direct_play_reject(path: &Path, reason: &str) -> bool {
    log::info!(
        "[Cast] Transcoding for Chromecast direct-play compatibility: {} ({})",
        path.display(),
        reason
    );
    false
}

async fn is_chromecast_direct_play_safe(path: &Path) -> bool {
    let output = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=format_name:stream=codec_name,profile,level,pix_fmt,width,height,avg_frame_rate",
            "-select_streams",
            "v:0",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .await
    {
        Ok(output) => output,
        Err(e) => return cast_direct_play_reject(path, &format!("ffprobe failed: {}", e)),
    };

    if !output.status.success() {
        return cast_direct_play_reject(path, "ffprobe returned an error");
    }

    let data: Value = match serde_json::from_slice(&output.stdout) {
        Ok(data) => data,
        Err(e) => {
            return cast_direct_play_reject(path, &format!("ffprobe JSON parse failed: {}", e));
        }
    };

    let format_name = data
        .get("format")
        .and_then(|v| v.get("format_name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let format_parts: Vec<&str> = format_name.split(',').map(str::trim).collect();
    let compatible_container = format_parts
        .iter()
        .any(|part| matches!(*part, "mov" | "mp4" | "m4a" | "3gp" | "3g2" | "mj2"));
    if !compatible_container {
        return cast_direct_play_reject(path, "container is not MP4/MOV");
    }

    let Some(stream) = data
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|streams| streams.first())
    else {
        return cast_direct_play_reject(path, "no video stream");
    };

    let codec = stream
        .get("codec_name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if codec != "h264" {
        return cast_direct_play_reject(path, "video codec is not H.264");
    }

    let profile = stream
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if profile.contains("10") || profile.contains("4:2:2") || profile.contains("4:4:4") {
        return cast_direct_play_reject(path, "H.264 profile is not broadly Chromecast-safe");
    }

    let pix_fmt = stream
        .get("pix_fmt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(pix_fmt.as_str(), "yuv420p" | "yuvj420p" | "nv12") {
        return cast_direct_play_reject(path, "pixel format is not 8-bit 4:2:0");
    }

    let level = stream.get("level").and_then(Value::as_i64).unwrap_or(0);
    if level > 41 {
        return cast_direct_play_reject(path, "H.264 level is above 4.1");
    }

    let width = stream.get("width").and_then(Value::as_u64).unwrap_or(0);
    let height = stream.get("height").and_then(Value::as_u64).unwrap_or(0);
    if width == 0 || height == 0 {
        return cast_direct_play_reject(path, "missing video dimensions");
    }
    if width > 1920 || height > 1080 {
        return cast_direct_play_reject(path, "resolution is above 1080p");
    }

    let fps = stream
        .get("avg_frame_rate")
        .and_then(Value::as_str)
        .and_then(parse_fps)
        .unwrap_or(30.0);
    if height > 720 && fps > 30.5 {
        return cast_direct_play_reject(path, "1080p frame rate is above 30 fps");
    }
    if height <= 720 && fps > 60.5 {
        return cast_direct_play_reject(path, "720p frame rate is above 60 fps");
    }

    let audio_output = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .await
    {
        Ok(output) => output,
        Err(e) => return cast_direct_play_reject(path, &format!("audio ffprobe failed: {}", e)),
    };

    let audio_codec = String::from_utf8_lossy(&audio_output.stdout)
        .trim()
        .to_ascii_lowercase();
    if !audio_codec.is_empty() && !matches!(audio_codec.as_str(), "aac" | "mp3") {
        return cast_direct_play_reject(path, "audio codec is not AAC/MP3");
    }

    true
}

struct SvpCastStream {
    stream_id: String,
    media_url: String,
    duration: Option<f64>,
}

fn extract_svp_stream_url(
    sidecar_stream_url: &str,
    local_ip: &str,
    port: u16,
) -> Option<(String, String)> {
    let (_, stream_path) = sidecar_stream_url.split_once("/svp/stream/")?;
    let (stream_id, filename) = stream_path.split_once('/')?;
    if stream_id.is_empty() || filename.is_empty() {
        return None;
    }

    let stream_id = stream_id.to_string();
    Some((
        stream_id.clone(),
        format!(
            "http://{}:{}/api/settings/svp/stream/{}/{}",
            local_ip, port, stream_id, filename
        ),
    ))
}

async fn try_start_svp_cast_stream(
    state: &AppState,
    file_path: &str,
    local_ip: &str,
    port: u16,
) -> Option<SvpCastStream> {
    if state.addon_manager().get_addon_status(SVP_ADDON_ID) != AddonStatus::Running {
        return None;
    }

    let config = crate::routes::settings::get_config_section(state.data_dir(), "svp");
    if !config
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }

    let Some(base_url) = state.addon_manager().addon_url(SVP_ADDON_ID) else {
        return None;
    };

    let target_fps = config_u32(&config, "target_fps").unwrap_or(60);
    let (target_resolution, target_bitrate) = if target_fps > 30 {
        ("720p", "5M")
    } else {
        ("1080p", "8M")
    };

    let mut body = json!({
        "file_path": file_path,
        "start_position": 0.0,
        "target_resolution": target_resolution,
        "target_bitrate": target_bitrate,
    });

    if let (Some(body_obj), Some(config_obj)) = (body.as_object_mut(), config.as_object()) {
        for key in [
            "target_fps",
            "preset",
            "use_nvof",
            "shader",
            "artifact_masking",
            "frame_interpolation",
            "custom_super",
            "custom_analyse",
            "custom_smooth",
        ] {
            if let Some(value) = config_obj.get(key) {
                body_obj.insert(key.to_string(), value.clone());
            }
        }
    }

    let url = format!("{}/svp/play", base_url.trim_end_matches('/'));
    let response = match state.http_client().post(&url).json(&body).send().await {
        Ok(response) => response,
        Err(e) => {
            log::warn!(
                "[Cast] Failed to reach SVP addon for Chromecast cast: {}",
                e
            );
            return None;
        }
    };

    let status = response.status();
    let result: Value = match response.json().await {
        Ok(result) => result,
        Err(e) => {
            log::warn!("[Cast] Invalid SVP response for Chromecast cast: {}", e);
            return None;
        }
    };

    if !status.is_success() {
        let detail = result
            .get("detail")
            .and_then(Value::as_str)
            .unwrap_or("Unknown SVP error");
        log::warn!(
            "[Cast] SVP stream unavailable for Chromecast cast: {}",
            detail
        );
        return None;
    }

    let sidecar_stream_url = result.get("stream_url").and_then(Value::as_str)?;
    let (stream_id, media_url) = match extract_svp_stream_url(sidecar_stream_url, local_ip, port) {
        Some(parts) => parts,
        None => {
            log::warn!(
                "[Cast] Could not rewrite SVP stream URL for Chromecast cast: {}",
                sidecar_stream_url
            );
            stop_svp_cast_stream(state).await;
            return None;
        }
    };

    log::info!(
        "[Cast] Using SVP HLS stream for Chromecast cast: {}",
        stream_id
    );
    Some(SvpCastStream {
        stream_id,
        media_url,
        duration: result.get("duration").and_then(Value::as_f64),
    })
}

async fn stop_svp_cast_stream(state: &AppState) {
    if state.addon_manager().get_addon_status(SVP_ADDON_ID) != AddonStatus::Running {
        return;
    }

    let Some(base_url) = state.addon_manager().addon_url(SVP_ADDON_ID) else {
        return;
    };

    let url = format!("{}/svp/stop", base_url.trim_end_matches('/'));
    if let Err(e) = state.http_client().post(&url).json(&json!({})).send().await {
        log::warn!("[Cast] Failed to stop SVP cast stream: {}", e);
    }
}

async fn serve_cast_file(
    AxumPath((media_id, _filename)): AxumPath<(String, String)>,
    request: Request,
) -> Result<Response, AppError> {
    let file_path = REGISTERED_MEDIA
        .lock()
        .expect("cast media registry poisoned")
        .get(&media_id)
        .cloned()
        .ok_or_else(|| AppError::NotFound("Media not found".into()))?;

    match crate::services::file_tracker::check_file_availability(file_path.to_str().unwrap_or("")) {
        crate::services::file_tracker::FileStatus::Available => {}
        crate::services::file_tracker::FileStatus::DriveOffline => {
            return Err(AppError::ServiceUnavailable("Drive is offline".into()));
        }
        crate::services::file_tracker::FileStatus::Missing => {
            return Err(AppError::NotFound("File not found on disk".into()));
        }
    }

    let response = ServeFile::new(&file_path)
        .oneshot(request)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to serve cast media: {}", e)))?;

    Ok(response.into_response())
}

/// Send a GET request to the cast addon and return the JSON response body.
async fn addon_get(client: &reqwest::Client, path: &str) -> Result<Value, AppError> {
    let url = format!("{}{}", addon_base_url(), path);
    let resp =
        client.get(&url).send().await.map_err(|e| {
            AppError::ServiceUnavailable(format!("Failed to reach cast addon: {}", e))
        })?;
    let body = resp
        .json::<Value>()
        .await
        .map_err(|e| AppError::Internal(format!("Invalid JSON from cast addon: {}", e)))?;
    Ok(body)
}

/// Send a POST request with a JSON body to the cast addon and return the JSON
/// response body.
async fn addon_post(client: &reqwest::Client, path: &str, body: &Value) -> Result<Value, AppError> {
    let url = format!("{}{}", addon_base_url(), path);
    let resp =
        client.post(&url).json(body).send().await.map_err(|e| {
            AppError::ServiceUnavailable(format!("Failed to reach cast addon: {}", e))
        })?;
    let response_body = resp
        .json::<Value>()
        .await
        .map_err(|e| AppError::Internal(format!("Invalid JSON from cast addon: {}", e)))?;
    Ok(response_body)
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/cast/devices -- List available cast devices.
///
/// Proxies to the cast addon sidecar to discover devices. If the addon is not
/// running, returns an empty device list with an `addon_status` field.
async fn list_devices(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    if !is_addon_running(&state) {
        return Ok(Json(json!({
            "devices": [],
            "addon_status": "not_running"
        })));
    }

    let result = addon_get(state.http_client(), "/devices").await?;
    Ok(Json(result))
}

/// POST /api/cast/devices/refresh -- Trigger device re-discovery.
///
/// Proxies to the cast addon sidecar to refresh the list of available devices.
async fn refresh_devices(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    if !is_addon_running(&state) {
        return Err(AppError::ServiceUnavailable(
            "Cast addon is not running".into(),
        ));
    }

    let result = addon_post(state.http_client(), "/devices/refresh", &json!({})).await?;
    Ok(Json(result))
}

/// POST /api/cast/play -- Start casting media to a device.
///
/// Registers or transcodes the requested file path, constructs a LAN-accessible
/// URL, then proxies to the cast addon to begin playback.
async fn play(
    State(state): State<AppState>,
    Json(body): Json<PlayRequest>,
) -> Result<Json<Value>, AppError> {
    if !is_addon_running(&state) {
        return Err(AppError::ServiceUnavailable(
            "Cast addon is not running".into(),
        ));
    }

    let image_id = body.image_id;
    let file_path = PathBuf::from(&body.file_path);
    match crate::services::file_tracker::check_file_availability(&body.file_path) {
        crate::services::file_tracker::FileStatus::Available => {}
        crate::services::file_tracker::FileStatus::DriveOffline => {
            return Err(AppError::ServiceUnavailable("Drive is offline".into()));
        }
        crate::services::file_tracker::FileStatus::Missing => {
            return Err(AppError::NotFound("File not found on disk".into()));
        }
    }

    // Reject any client-supplied path that is not inside a watched directory,
    // so /api/cast/play can't be used to serve/transcode arbitrary host files.
    crate::server::utils::validate_path_in_watch_dir(&state, &body.file_path)?;

    // Construct a URL that the cast device can access over the local network.
    // Cast devices cannot reach 127.0.0.1, so we need the server's LAN IP.
    let local_ip = get_local_ip().ok_or_else(|| {
        AppError::Internal("Could not determine local IP address for casting".into())
    })?;
    let port = state.port();
    let is_chromecast = body.device_id.starts_with("chromecast-");

    let previous_media = {
        let cast_state = state.cast_state();
        let state_guard = cast_state.read().await;
        state_guard.current_media.clone()
    };
    if let Some(current_media) = previous_media {
        cleanup_registered_cast_media(&state, &current_media).await;
    }

    let svp_stream = if is_chromecast {
        try_start_svp_cast_stream(&state, &body.file_path, &local_ip, port).await
    } else {
        None
    };
    let (media_url, content_type, transcode_stream_id, svp_stream_id, stream_duration) =
        if let Some(svp_stream) = svp_stream {
            (
                svp_stream.media_url,
                "application/x-mpegURL".to_string(),
                None,
                Some(svp_stream.stream_id),
                svp_stream.duration,
            )
        } else if is_chromecast && !is_chromecast_direct_play_safe(&file_path).await {
            let quality = QualityPreset {
                resolution: Some("720p".into()),
                bitrate: Some("5M".into()),
            };
            let info = state
                .transcode_manager()
                .start_stream(&body.file_path, 0.0, &quality, true)
                .await
                .map_err(|e| {
                    AppError::Internal(format!("Failed to transcode for Chromecast: {}", e))
                })?;

            (
                format!("http://{}:{}{}", local_ip, port, info.stream_url),
                "application/x-mpegURL".to_string(),
                Some(info.stream_id),
                None,
                Some(info.duration),
            )
        } else {
            let media_id = register_media(&file_path);
            let ext = file_path
                .extension()
                .and_then(|e| e.to_str())
                .filter(|e| !e.is_empty())
                .map(|e| format!(".{}", e))
                .unwrap_or_else(|| ".mp4".to_string());
            (
                format!(
                    "http://{}:{}/api/cast-media/{}/file/media{}",
                    local_ip, port, media_id, ext
                ),
                guess_content_type(&file_path),
                None,
                None,
                None,
            )
        };
    let title = file_path
        .file_stem()
        .and_then(|name| name.to_str())
        .map(|name| format!("LocalBooru #{} - {}", image_id, name))
        .unwrap_or_else(|| format!("LocalBooru #{}", image_id));

    // Proxy to the cast addon to start playback
    let addon_body = json!({
        "device_id": body.device_id.clone(),
        "media_url": media_url.clone(),
        "content_type": content_type,
        "title": title,
        "image_id": image_id,
    });
    let result = match addon_post(state.http_client(), "/play", &addon_body).await {
        Ok(result) => result,
        Err(e) => {
            unregister_media_url(&media_url);
            if let Some(stream_id) = &transcode_stream_id {
                state.transcode_manager().stop_stream(stream_id);
            }
            if svp_stream_id.is_some() {
                stop_svp_cast_stream(&state).await;
            }
            return Err(e);
        }
    };

    let duration = result
        .get("duration")
        .and_then(|v| v.as_f64())
        .or(stream_duration)
        .unwrap_or(0.0);

    // Update in-memory cast state
    {
        let cast_state = state.cast_state();
        let mut state_guard = cast_state.write().await;
        state_guard.status = "casting".to_string();
        state_guard.active_device = Some(CastDevice {
            id: body.device_id.clone(),
            name: result
                .get("device_name")
                .and_then(|v| v.as_str())
                .unwrap_or(&body.device_id)
                .to_string(),
            device_type: result
                .get("device_type")
                .and_then(|v| v.as_str())
                .unwrap_or("chromecast")
                .to_string(),
            host: result
                .get("device_host")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        });
        state_guard.current_media = Some(CastMedia {
            image_id,
            media_url: media_url.clone(),
            transcode_stream_id,
            svp_stream_id,
            position: 0.0,
            duration,
        });
    }

    Ok(Json(json!({
        "success": true,
        "status": "casting",
        "media_url": media_url,
        "file_path": body.file_path,
        "duration": duration,
    })))
}

/// POST /api/cast/control -- Send a playback control command.
///
/// Accepts actions: "play"/"resume", "pause", "seek", "volume". Proxies to the cast
/// addon and returns the current status.
async fn control(
    State(state): State<AppState>,
    Json(body): Json<ControlRequest>,
) -> Result<Json<Value>, AppError> {
    if !is_addon_running(&state) {
        return Err(AppError::ServiceUnavailable(
            "Cast addon is not running".into(),
        ));
    }

    // Validate action
    let valid_actions = ["play", "resume", "pause", "seek", "volume"];
    if !valid_actions.contains(&body.action.as_str()) {
        return Err(AppError::BadRequest(format!(
            "Invalid action '{}'. Must be one of: {}",
            body.action,
            valid_actions.join(", ")
        )));
    }

    let addon_body = json!({
        "action": body.action,
        "value": body.value,
    });
    let result = addon_post(state.http_client(), "/control", &addon_body).await?;

    // Update in-memory state based on the action
    {
        let cast_state = state.cast_state();
        let mut state_guard = cast_state.write().await;

        match body.action.as_str() {
            "pause" => {
                state_guard.status = "paused".to_string();
            }
            "play" | "resume" => {
                state_guard.status = "casting".to_string();
            }
            "seek" => {
                if let Some(ref mut media) = state_guard.current_media {
                    if let Some(pos) = body.value.as_ref().and_then(|v| v.as_f64()) {
                        media.position = pos;
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Json(result))
}

/// POST /api/cast/stop -- Stop casting and clear state.
///
/// Proxies to the cast addon to stop playback, then resets the in-memory
/// cast state to idle.
async fn stop(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    // Attempt to proxy to the addon even if it might have stopped
    if is_addon_running(&state) {
        // Best-effort: if the addon fails to respond, we still clear local state
        let _ = addon_post(state.http_client(), "/stop", &json!({})).await;
    }

    // Clear in-memory cast state
    let current_media = {
        let cast_state = state.cast_state();
        let mut state_guard = cast_state.write().await;
        let current_media = state_guard.current_media.take();
        state_guard.status = "idle".to_string();
        state_guard.active_device = None;
        current_media
    };
    if let Some(current_media) = current_media {
        cleanup_registered_cast_media(&state, &current_media).await;
    }

    Ok(Json(json!({
        "success": true,
        "status": "idle",
    })))
}

/// GET /api/cast/status -- SSE stream of current cast state.
///
/// Returns the in-memory cast state as an SSE stream, polling every 2 seconds.
/// If the addon is running, also queries it for live playback status and merges
/// the results. The JSON response shape is unchanged -- it is delivered as the
/// `data` field of each SSE event.
async fn status(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));

        loop {
            interval.tick().await;

            let cast_state = state.cast_state();
            let state_guard = cast_state.read().await;
            let local_state = state_guard.clone();
            drop(state_guard);

            let mut response = json!({
                "status": local_state.status,
                "active_device": local_state.active_device,
                "current_media": local_state.current_media,
            });

            // If the addon is running and we're actively casting, query for live status
            if is_addon_running(&state) && local_state.status != "idle" {
                match addon_get(state.http_client(), "/status").await {
                    Ok(live) => {
                        response["live"] = live.clone();

                        let cast_state = state.cast_state();
                        let mut state_guard = cast_state.write().await;
                        let mut media_to_cleanup = None;

                        // Sync position from live status back into our in-memory state
                        if let Some(pos) = live.get("position").and_then(|v| v.as_f64()) {
                            if let Some(ref mut media) = state_guard.current_media {
                                media.position = pos;
                            }
                        }

                        // Sync status from addon (it may have stopped on its own)
                        if let Some(addon_status) = live.get("status").and_then(|v| v.as_str()) {
                            match addon_status {
                                "idle" | "stopped" => {
                                    media_to_cleanup = state_guard.current_media.take();
                                    state_guard.status = "idle".to_string();
                                    state_guard.active_device = None;
                                }
                                "paused" => {
                                    state_guard.status = "paused".to_string();
                                }
                                "playing" | "casting" => {
                                    state_guard.status = "casting".to_string();
                                }
                                _ => {}
                            }
                        }
                        drop(state_guard);

                        if let Some(current_media) = media_to_cleanup {
                            cleanup_registered_cast_media(&state, &current_media).await;
                        }
                    }
                    Err(_) => {
                        // Addon is running but failed to respond -- include a warning
                        response["live_status_error"] = json!("Failed to query live status from addon");
                    }
                }
            }

            response["addon_running"] = json!(is_addon_running(&state));

            yield Ok(Event::default().data(response.to_string()));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
