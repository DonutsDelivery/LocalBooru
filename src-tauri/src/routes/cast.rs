//! Cast/Chromecast subsystem routes.
//!
//! Provides bridge routes that proxy to the cast addon sidecar (port 18006)
//! and manage in-memory cast state. The actual device discovery and media
//! casting are handled by the Python sidecar; this module manages the Rust
//! side of the state and proxies HTTP requests.

use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::addons::manager::AddonStatus;
use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::server::utils::get_local_ip;

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
    pub position: f64,
    pub duration: f64,
}

// ─── Request models ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct PlayRequest {
    device_id: String,
    image_id: i64,
    directory_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ControlRequest {
    action: String,
    value: Option<Value>,
}

// ─── Constants ───────────────────────────────────────────────────────────────

const CAST_ADDON_ID: &str = "cast";
const CAST_ADDON_PORT: u16 = 18006;

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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Check whether the cast addon sidecar is currently running.
fn is_addon_running(state: &AppState) -> bool {
    state.addon_manager().get_addon_status(CAST_ADDON_ID) == AddonStatus::Running
}

/// Build the base URL for the cast addon sidecar.
fn addon_base_url() -> String {
    format!("http://127.0.0.1:{}", CAST_ADDON_PORT)
}

/// Send a GET request to the cast addon and return the JSON response body.
async fn addon_get(client: &reqwest::Client, path: &str) -> Result<Value, AppError> {
    let url = format!("{}{}", addon_base_url(), path);
    let resp = client.get(&url).send().await.map_err(|e| {
        AppError::ServiceUnavailable(format!("Failed to reach cast addon: {}", e))
    })?;
    let body = resp.json::<Value>().await.map_err(|e| {
        AppError::Internal(format!("Invalid JSON from cast addon: {}", e))
    })?;
    Ok(body)
}

/// Send a POST request with a JSON body to the cast addon and return the JSON
/// response body.
async fn addon_post(client: &reqwest::Client, path: &str, body: &Value) -> Result<Value, AppError> {
    let url = format!("{}{}", addon_base_url(), path);
    let resp = client.post(&url).json(body).send().await.map_err(|e| {
        AppError::ServiceUnavailable(format!("Failed to reach cast addon: {}", e))
    })?;
    let response_body = resp.json::<Value>().await.map_err(|e| {
        AppError::Internal(format!("Invalid JSON from cast addon: {}", e))
    })?;
    Ok(response_body)
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/cast/devices -- List available cast devices.
///
/// Proxies to the cast addon sidecar to discover devices. If the addon is not
/// running, returns an empty device list with an `addon_status` field.
async fn list_devices(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
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
async fn refresh_devices(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
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
/// Looks up the media file in the directory database, constructs a URL
/// accessible from the local network (using the server's LAN IP and port),
/// then proxies to the cast addon to begin playback.
async fn play(
    State(state): State<AppState>,
    Json(body): Json<PlayRequest>,
) -> Result<Json<Value>, AppError> {
    if !is_addon_running(&state) {
        return Err(AppError::ServiceUnavailable(
            "Cast addon is not running".into(),
        ));
    }

    // Look up the media file path from the directory database.
    // If directory_id is provided, look up directly; otherwise search all directories.
    let state_clone = state.clone();
    let image_id = body.image_id;
    let directory_id_opt = body.directory_id;

    let (file_path, resolved_directory_id) = tokio::task::spawn_blocking(move || {
        // Determine which directory IDs to search
        let dir_ids_to_search: Vec<i64> = match directory_id_opt {
            Some(dir_id) => {
                if !state_clone.directory_db().db_exists(dir_id) {
                    return Err(AppError::NotFound(format!(
                        "Directory {} not found",
                        dir_id
                    )));
                }
                vec![dir_id]
            }
            None => state_clone.directory_db().get_all_directory_ids(),
        };

        // Search through directories for the image
        for dir_id in &dir_ids_to_search {
            if !state_clone.directory_db().db_exists(*dir_id) {
                continue;
            }
            let dir_pool = match state_clone.directory_db().get_pool(*dir_id) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let dir_conn = match dir_pool.get() {
                Ok(c) => c,
                Err(_) => continue,
            };

            let result: Result<String, _> = dir_conn.query_row(
                "SELECT original_path FROM image_files WHERE image_id = ?1 LIMIT 1",
                params![image_id],
                |row| row.get(0),
            );

            if let Ok(path) = result {
                return Ok((path, *dir_id));
            }
        }

        Err(AppError::NotFound(format!(
            "Image {} not found in any directory",
            image_id
        )))
    })
    .await??;

    // Construct a URL that the cast device can access over the local network.
    // Cast devices cannot reach 127.0.0.1, so we need the server's LAN IP.
    let local_ip = get_local_ip().ok_or_else(|| {
        AppError::Internal("Could not determine local IP address for casting".into())
    })?;
    let port = state.port();
    let media_url = format!(
        "http://{}:{}/api/images/{}/file?directory_id={}",
        local_ip, port, image_id, resolved_directory_id
    );

    // Proxy to the cast addon to start playback
    let addon_body = json!({
        "device_id": body.device_id,
        "media_url": media_url,
        "image_id": image_id,
    });
    let result = addon_post(state.http_client(), "/play", &addon_body).await?;

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
            position: 0.0,
            duration: result
                .get("duration")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        });
    }

    Ok(Json(json!({
        "success": true,
        "status": "casting",
        "media_url": media_url,
        "file_path": file_path,
    })))
}

/// POST /api/cast/control -- Send a playback control command.
///
/// Accepts actions: "play", "pause", "seek", "volume". Proxies to the cast
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
    let valid_actions = ["play", "pause", "seek", "volume"];
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
            "play" => {
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
async fn stop(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // Attempt to proxy to the addon even if it might have stopped
    if is_addon_running(&state) {
        // Best-effort: if the addon fails to respond, we still clear local state
        let _ = addon_post(state.http_client(), "/stop", &json!({})).await;
    }

    // Clear in-memory cast state
    {
        let cast_state = state.cast_state();
        let mut state_guard = cast_state.write().await;
        state_guard.status = "idle".to_string();
        state_guard.active_device = None;
        state_guard.current_media = None;
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

                        // Sync position from live status back into our in-memory state
                        if let Some(pos) = live.get("position").and_then(|v| v.as_f64()) {
                            let cast_state = state.cast_state();
                            let mut state_guard = cast_state.write().await;
                            if let Some(ref mut media) = state_guard.current_media {
                                media.position = pos;
                            }
                            // Sync status from addon (it may have stopped on its own)
                            if let Some(addon_status) = live.get("status").and_then(|v| v.as_str()) {
                                match addon_status {
                                    "idle" | "stopped" => {
                                        state_guard.status = "idle".to_string();
                                        state_guard.active_device = None;
                                        state_guard.current_media = None;
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
