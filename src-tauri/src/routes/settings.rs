use axum::extract::{Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{delete, get};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::server::error::AppError;
use crate::server::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(get_all_settings).post(update_settings))
        .route(
            "/saved-searches",
            get(list_saved_searches).post(create_saved_search),
        )
        .route("/saved-searches/{search_id}", delete(delete_saved_search))
        .route(
            "/video-playback",
            get(get_video_playback).post(update_video_playback),
        )
        .route("/util/dimensions", get(get_dimensions))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn settings_path(data_dir: &Path) -> PathBuf {
    data_dir.join("settings.json")
}

/// Load the raw saved settings from disk. Returns `{}` if the file does not
/// exist or cannot be parsed.
fn load_settings(data_dir: &Path) -> Value {
    let path = settings_path(data_dir);
    if path.exists() {
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| json!({}))
    } else {
        json!({})
    }
}

/// Persist settings to disk as pretty-printed JSON.
fn save_settings_to_file(data_dir: &Path, settings: &Value) -> std::io::Result<()> {
    let path = settings_path(data_dir);
    let json_str = serde_json::to_string_pretty(settings)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json_str)
}

/// Return the full default settings structure.
fn get_defaults() -> Value {
    json!({
        "network": {
            "local_network_enabled": false,
            "public_network_enabled": false,
            "local_port": 8790,
            "public_port": 8791,
            "auth_required_level": "local_network",
            "upnp_enabled": false,
            "allow_settings_local_network": false
        },
        "video_playback": {
            "auto_advance_enabled": false,
            "auto_advance_delay": 5
        },
        "saved_searches": []
    })
}

/// Deep-merge `overlay` on top of `base`. For objects, keys in `overlay`
/// overwrite matching keys in `base`; for all other types the overlay value
/// wins outright.
fn deep_merge(base: &Value, overlay: &Value) -> Value {
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            let mut merged = base_map.clone();
            for (key, overlay_val) in overlay_map {
                let merged_val = if let Some(base_val) = base_map.get(key) {
                    deep_merge(base_val, overlay_val)
                } else {
                    overlay_val.clone()
                };
                merged.insert(key.clone(), merged_val);
            }
            Value::Object(merged)
        }
        // Non-object: overlay wins
        (_, overlay) => overlay.clone(),
    }
}

/// Merge saved settings on top of the defaults so the caller always gets a
/// complete structure.
fn merge_with_defaults(saved: &Value) -> Value {
    deep_merge(&get_defaults(), saved)
}

/// Generate a short hex ID for saved searches (12 hex chars derived from the
/// current timestamp plus a UUID suffix to avoid collisions).
fn generate_search_id() -> String {
    let uuid_val = uuid::Uuid::new_v4();
    let bytes = uuid_val.as_bytes();
    // Take first 6 bytes → 12 hex chars
    bytes[..6]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

// ─── Route handlers ──────────────────────────────────────────────────────────

/// GET / — Return all settings (defaults merged with saved overrides).
async fn get_all_settings(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let saved = load_settings(&data_dir);
        merge_with_defaults(&saved)
    })
    .await?;

    Ok(Json(result))
}

/// POST / — Merge incoming JSON into existing settings and persist.
async fn update_settings(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let existing = load_settings(&data_dir);
        let merged = deep_merge(&existing, &body);
        save_settings_to_file(&data_dir, &merged)?;
        Ok::<_, AppError>(merge_with_defaults(&merged))
    })
    .await??;

    Ok(Json(result))
}

// ─── Saved searches ──────────────────────────────────────────────────────────

/// GET /saved-searches — List all saved searches.
async fn list_saved_searches(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let settings = load_settings(&data_dir);
        let searches = settings
            .get("saved_searches")
            .cloned()
            .unwrap_or_else(|| json!([]));
        searches
    })
    .await?;

    Ok(Json(result))
}

#[derive(Debug, Deserialize)]
struct SavedSearchCreate {
    name: String,
    filters: Value,
}

/// POST /saved-searches — Create a new saved search.
async fn create_saved_search(
    State(state): State<AppState>,
    Json(body): Json<SavedSearchCreate>,
) -> Result<Json<Value>, AppError> {
    if body.name.trim().is_empty() {
        return Err(AppError::BadRequest(
            "Saved search name cannot be empty".into(),
        ));
    }

    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut settings = load_settings(&data_dir);

        let id = generate_search_id();
        let now = chrono::Utc::now().to_rfc3339();

        let new_search = json!({
            "id": id,
            "name": body.name,
            "filters": body.filters,
            "created_at": now
        });

        // Ensure saved_searches is an array, then push the new entry
        let searches = settings
            .as_object_mut()
            .ok_or_else(|| AppError::Internal("Settings file is not a JSON object".into()))?
            .entry("saved_searches")
            .or_insert_with(|| json!([]));

        if let Some(arr) = searches.as_array_mut() {
            arr.push(new_search.clone());
        } else {
            // Recover: replace with a fresh array containing the new entry
            *searches = json!([new_search.clone()]);
        }

        save_settings_to_file(&data_dir, &settings)?;
        Ok::<_, AppError>(new_search)
    })
    .await??;

    Ok(Json(result))
}

/// DELETE /saved-searches/:search_id — Delete a saved search by ID.
async fn delete_saved_search(
    State(state): State<AppState>,
    AxumPath(search_id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut settings = load_settings(&data_dir);

        let removed = if let Some(arr) = settings
            .get_mut("saved_searches")
            .and_then(|v| v.as_array_mut())
        {
            let before_len = arr.len();
            arr.retain(|entry| {
                entry
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|id| id != search_id)
                    .unwrap_or(true)
            });
            arr.len() < before_len
        } else {
            false
        };

        if !removed {
            return Err(AppError::NotFound(format!(
                "Saved search '{}' not found",
                search_id
            )));
        }

        save_settings_to_file(&data_dir, &settings)?;
        Ok::<_, AppError>(json!({ "success": true }))
    })
    .await??;

    Ok(Json(result))
}

// ─── Video playback ──────────────────────────────────────────────────────────

/// GET /video-playback — Get video playback configuration.
async fn get_video_playback(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let settings = load_settings(&data_dir);
        let defaults = get_defaults();

        let default_vp = defaults
            .get("video_playback")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let saved_vp = settings
            .get("video_playback")
            .cloned()
            .unwrap_or_else(|| json!({}));

        deep_merge(&default_vp, &saved_vp)
    })
    .await?;

    Ok(Json(result))
}

/// POST /video-playback — Update video playback configuration.
async fn update_video_playback(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut settings = load_settings(&data_dir);

        let existing_vp = settings
            .get("video_playback")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let merged_vp = deep_merge(&existing_vp, &body);

        settings
            .as_object_mut()
            .ok_or_else(|| AppError::Internal("Settings file is not a JSON object".into()))?
            .insert("video_playback".into(), merged_vp.clone());

        save_settings_to_file(&data_dir, &settings)?;

        // Return the full video_playback section (with defaults filled in)
        let defaults = get_defaults();
        let default_vp = defaults
            .get("video_playback")
            .cloned()
            .unwrap_or_else(|| json!({}));
        Ok::<_, AppError>(deep_merge(&default_vp, &merged_vp))
    })
    .await??;

    Ok(Json(result))
}

// ─── Utility: file dimensions ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DimensionsQuery {
    file_path: String,
}

/// Known video file extensions (lowercase, without leading dot).
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "webm", "avi", "mov", "wmv", "flv", "m4v", "ts", "mpg", "mpeg", "3gp", "ogv",
];

/// GET /util/dimensions — Get the pixel dimensions of an image or video file.
async fn get_dimensions(
    Query(q): Query<DimensionsQuery>,
) -> Result<Json<Value>, AppError> {
    let file_path = q.file_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        let path = Path::new(&file_path);

        if !path.exists() {
            return json!({
                "success": false,
                "error": "File not found"
            });
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
            // Use ffprobe for video files
            match get_video_dimensions(path) {
                Ok((w, h)) => json!({
                    "success": true,
                    "width": w,
                    "height": h
                }),
                Err(e) => json!({
                    "success": false,
                    "error": e
                }),
            }
        } else {
            // Use the `image` crate for image files
            match image::image_dimensions(path) {
                Ok((w, h)) => json!({
                    "success": true,
                    "width": w,
                    "height": h
                }),
                Err(e) => json!({
                    "success": false,
                    "error": format!("Failed to read image dimensions: {}", e)
                }),
            }
        }
    })
    .await?;

    Ok(Json(result))
}

/// Run ffprobe to extract width and height from a video file.
fn get_video_dimensions(path: &Path) -> Result<(u64, u64), String> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("Failed to run ffprobe: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffprobe error: {}", stderr.trim()));
    }

    let parsed: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse ffprobe output: {}", e))?;

    let stream = parsed
        .get("streams")
        .and_then(|s| s.as_array())
        .and_then(|arr| arr.first())
        .ok_or_else(|| "No video stream found".to_string())?;

    let width = stream
        .get("width")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "Missing width in ffprobe output".to_string())?;
    let height = stream
        .get("height")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "Missing height in ffprobe output".to_string())?;

    Ok((width, height))
}
