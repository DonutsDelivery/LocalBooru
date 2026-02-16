use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Json, Response};
use axum::routing::{delete, get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

use crate::addons::manager::AddonStatus;
use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::transcode::QualityPreset;

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
        // ─── Config routes ──────────────────────────────────────────────
        .route(
            "/optical-flow",
            get(get_optical_flow).post(update_optical_flow),
        )
        .route("/svp", get(get_svp).post(update_svp))
        .route("/whisper", get(get_whisper).post(update_whisper))
        .route("/cast", get(get_cast).post(update_cast))
        .route("/age-detection/status", get(get_age_detection_status))
        .route("/age-detection/toggle", post(toggle_age_detection))
        .route("/migration", get(get_migration))
        .route("/video-info", post(get_video_info_endpoint))
        // ─── Transcode streaming ────────────────────────────────────────
        .route("/transcode/play", post(start_transcode_stream))
        .route("/transcode/stop", post(stop_transcode_streams))
        .route(
            "/transcode/stream/{stream_id}/{filename}",
            get(serve_transcode_file),
        )
        // ─── Optical flow streaming (sidecar bridge) ────────────────────
        .route("/optical-flow/play", post(bridge_optical_flow_play))
        .route("/optical-flow/stop", post(bridge_optical_flow_stop))
        .route(
            "/optical-flow/stream/{stream_id}/{filename}",
            get(bridge_optical_flow_stream),
        )
        // ─── SVP streaming (sidecar bridge) ─────────────────────────────
        .route("/svp/play", post(bridge_svp_play))
        .route("/svp/stop", post(bridge_svp_stop))
        .route(
            "/svp/stream/{stream_id}/{filename}",
            get(bridge_svp_stream),
        )
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
        "saved_searches": [],
        "optical_flow": {
            "enabled": false,
            "target_fps": 60,
            "use_gpu": true,
            "quality": "fast"
        },
        "svp": {
            "enabled": false,
            "target_fps": 60,
            "preset": "balanced",
            "use_nvof": true,
            "shader": 23,
            "artifact_masking": 100,
            "frame_interpolation": 2,
            "custom_super": null,
            "custom_analyse": null,
            "custom_smooth": null
        },
        "whisper": {
            "enabled": false,
            "auto_generate": false,
            "model_size": "medium",
            "language": "ja",
            "task": "translate",
            "chunk_duration": 30,
            "beam_size": 8,
            "device": "auto",
            "compute_type": "auto",
            "vad_filter": true,
            "suppress_nst": true,
            "cache_subtitles": true,
            "subtitle_font": "Trebuchet MS",
            "subtitle_font_size": 1.3,
            "subtitle_style": "outline",
            "subtitle_color": "#ffffff",
            "subtitle_outline_color": "#000000",
            "subtitle_bg_opacity": 0.75
        },
        "cast": {
            "enabled": true
        },
        "age_detection": {
            "enabled": false
        }
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

/// Read a single config section from settings, merging with defaults.
fn get_config_section(data_dir: &Path, section: &str) -> Value {
    let settings = load_settings(data_dir);
    let defaults = get_defaults();

    let default_section = defaults
        .get(section)
        .cloned()
        .unwrap_or_else(|| json!({}));
    let saved_section = settings
        .get(section)
        .cloned()
        .unwrap_or_else(|| json!({}));

    deep_merge(&default_section, &saved_section)
}

/// Update a single config section: deep-merge the incoming body with the
/// existing saved section, persist, and return the merged result (with defaults).
fn update_config_section(
    data_dir: &Path,
    section: &str,
    body: &Value,
) -> Result<Value, AppError> {
    let mut settings = load_settings(data_dir);

    let existing = settings
        .get(section)
        .cloned()
        .unwrap_or_else(|| json!({}));
    let merged = deep_merge(&existing, body);

    settings
        .as_object_mut()
        .ok_or_else(|| AppError::Internal("Settings file is not a JSON object".into()))?
        .insert(section.into(), merged.clone());

    save_settings_to_file(data_dir, &settings)?;

    let defaults = get_defaults();
    let default_section = defaults
        .get(section)
        .cloned()
        .unwrap_or_else(|| json!({}));
    Ok(deep_merge(&default_section, &merged))
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
        get_config_section(&data_dir, "video_playback")
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
        update_config_section(&data_dir, "video_playback", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── Optical flow ────────────────────────────────────────────────────────────

/// GET /optical-flow — Get optical flow configuration + backend status stub.
async fn get_optical_flow(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "optical_flow");
        // Stub backend status — real detection requires the addon sidecar
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "backend".into(),
                json!({
                    "cv2_cuda_available": false,
                    "cuda_available": false,
                    "gpu_backend": null,
                    "cv2_available": false,
                    "any_backend_available": false
                }),
            );
        }
        config
    })
    .await?;

    Ok(Json(result))
}

/// POST /optical-flow — Update optical flow configuration.
async fn update_optical_flow(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "optical_flow", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── SVP ─────────────────────────────────────────────────────────────────────

/// GET /svp — Get SVP configuration + status stub.
async fn get_svp(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "svp");
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "status".into(),
                json!({
                    "ready": false,
                    "vapoursynth_available": false,
                    "svp_plugins_available": false,
                    "vspipe_available": false,
                    "source_filter_available": false,
                    "nvenc_available": false,
                    "nvof_ready": false,
                    "bestsource_available": false,
                    "ffms2_available": false,
                    "lsmas_available": false,
                    "missing": []
                }),
            );
            obj.insert(
                "presets".into(),
                json!({
                    "fast": { "name": "Fast", "description": "Low quality, minimal GPU usage" },
                    "balanced": { "name": "Balanced", "description": "Good quality/performance balance" },
                    "quality": { "name": "Quality", "description": "High quality, more GPU usage" },
                    "max": { "name": "Maximum", "description": "Best quality, maximum GPU usage" },
                    "animation": { "name": "Animation", "description": "Optimized for animated content" },
                    "film": { "name": "Film", "description": "Optimized for live-action film" }
                }),
            );
        }
        config
    })
    .await?;

    Ok(Json(result))
}

/// POST /svp — Update SVP configuration.
async fn update_svp(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "svp", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── Whisper ─────────────────────────────────────────────────────────────────

/// GET /whisper — Get whisper subtitle configuration + status stub.
async fn get_whisper(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "whisper");
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "status".into(),
                json!({
                    "faster_whisper_installed": false,
                    "cuda_available": false
                }),
            );
        }
        config
    })
    .await?;

    Ok(Json(result))
}

/// POST /whisper — Update whisper subtitle configuration.
async fn update_whisper(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "whisper", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── Cast ────────────────────────────────────────────────────────────────────

/// GET /cast — Get cast configuration + status stub.
async fn get_cast(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "cast");
        if let Some(obj) = config.as_object_mut() {
            obj.insert("installing".into(), json!(false));
            obj.insert(
                "status".into(),
                json!({
                    "pychromecast_installed": false,
                    "upnp_installed": false,
                    "aiohttp_installed": false
                }),
            );
        }
        config
    })
    .await?;

    Ok(Json(result))
}

/// POST /cast — Update cast configuration.
async fn update_cast(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "cast", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── Age detection ───────────────────────────────────────────────────────────

/// GET /age-detection/status — Check age detection addon status.
async fn get_age_detection_status(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let addon_mgr = state.addon_manager();

    let addon_info = addon_mgr.get_addon("age-detector");
    let (installed, running) = match &addon_info {
        Some(info) => (info.installed, info.status == AddonStatus::Running),
        None => (false, false),
    };

    let config = tokio::task::spawn_blocking(move || {
        get_config_section(&data_dir, "age_detection")
    })
    .await?;

    let enabled = config
        .get("enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(Json(json!({
        "enabled": enabled,
        "installed": installed,
        "running": running,
        "installing": false,
        "dependencies": {}
    })))
}

#[derive(Debug, Deserialize)]
struct AgeDetectionToggle {
    enabled: bool,
}

/// POST /age-detection/toggle — Toggle age detection enabled flag.
async fn toggle_age_detection(
    State(state): State<AppState>,
    Json(body): Json<AgeDetectionToggle>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let enabled = body.enabled;

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "age_detection", &json!({ "enabled": enabled }))
    })
    .await??;

    Ok(Json(json!({
        "success": true,
        "enabled": result.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false)
    })))
}

// ─── Migration ───────────────────────────────────────────────────────────────

/// GET /migration — Get migration status stub.
async fn get_migration() -> Json<Value> {
    Json(json!({
        "current_mode": "system",
        "status": null
    }))
}

// ─── Video info ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct VideoInfoRequest {
    file_path: String,
}

/// POST /video-info — Get video information including VFR detection.
async fn get_video_info_endpoint(
    Json(body): Json<VideoInfoRequest>,
) -> Result<Json<Value>, AppError> {
    let file_path = body.file_path;

    let result = tokio::task::spawn_blocking(move || {
        let path = Path::new(&file_path);

        if !path.exists() {
            return json!({ "success": false, "error": "File not found" });
        }

        // Run ffprobe to get codec, dimensions, duration, pixel format, and frame rates
        let output = match std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,duration,pix_fmt,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
            ])
            .arg(path)
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                return json!({ "success": false, "error": format!("Failed to run ffprobe: {}", e) });
            }
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return json!({ "success": false, "error": format!("ffprobe error: {}", stderr.trim()) });
        }

        let parsed: Value = match serde_json::from_slice(&output.stdout) {
            Ok(v) => v,
            Err(e) => {
                return json!({ "success": false, "error": format!("Failed to parse ffprobe output: {}", e) });
            }
        };

        let stream = match parsed
            .get("streams")
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
        {
            Some(s) => s,
            None => {
                return json!({ "success": false, "error": "No video stream found" });
            }
        };

        // Extract basic info
        let codec = stream.get("codec_name").and_then(|v| v.as_str()).unwrap_or("");
        let width = stream.get("width").and_then(|v| v.as_u64()).unwrap_or(0);
        let height = stream.get("height").and_then(|v| v.as_u64()).unwrap_or(0);
        let duration = stream
            .get("duration")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok());
        let pix_fmt = stream.get("pix_fmt").and_then(|v| v.as_str()).unwrap_or("");

        // VFR detection
        let r_frame_rate = stream
            .get("r_frame_rate")
            .and_then(|v| v.as_str())
            .unwrap_or("0/1");
        let avg_frame_rate = stream
            .get("avg_frame_rate")
            .and_then(|v| v.as_str())
            .unwrap_or("0/1");

        let r_fps = parse_fps(r_frame_rate);
        let avg_fps = parse_fps(avg_frame_rate);

        let is_vfr = detect_vfr(r_fps, avg_fps);

        json!({
            "success": true,
            "codec": codec,
            "width": width,
            "height": height,
            "duration": duration,
            "pix_fmt": pix_fmt,
            "is_vfr": is_vfr,
            "r_frame_rate": r_frame_rate,
            "avg_frame_rate": avg_frame_rate,
            "r_fps": (r_fps * 100.0).round() / 100.0,
            "avg_fps": (avg_fps * 100.0).round() / 100.0
        })
    })
    .await?;

    Ok(Json(result))
}

/// Parse a frame rate string like "30000/1001" or "30" into an f64.
fn parse_fps(rate_str: &str) -> f64 {
    if let Some((num, den)) = rate_str.split_once('/') {
        let n: f64 = num.parse().unwrap_or(0.0);
        let d: f64 = den.parse().unwrap_or(1.0);
        if d != 0.0 {
            n / d
        } else {
            0.0
        }
    } else {
        rate_str.parse().unwrap_or(0.0)
    }
}

/// Detect VFR by comparing r_frame_rate vs avg_frame_rate.
/// Common VFR time bases (container frame rates) include 120, 240, 300, 600, 1000.
fn detect_vfr(r_fps: f64, avg_fps: f64) -> bool {
    if r_fps <= 0.0 || avg_fps <= 0.0 {
        return false;
    }

    const VFR_TIME_BASES: &[u64] = &[60, 90, 120, 180, 240, 300, 360, 480, 600, 1000];

    let ratio = r_fps / avg_fps;
    (ratio > 2.0) || (VFR_TIME_BASES.contains(&(r_fps as u64)) && ratio > 1.5)
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

// ─── Transcode streaming ─────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct TranscodePlayRequest {
    file_path: String,
    #[serde(default)]
    start_position: f64,
    #[serde(default)]
    quality_preset: Option<String>,
    #[serde(default)]
    force_cfr: bool,
}

/// POST /transcode/play — Start an HLS transcode stream.
async fn start_transcode_stream(
    State(state): State<AppState>,
    Json(body): Json<TranscodePlayRequest>,
) -> Result<Json<Value>, AppError> {
    let path = Path::new(&body.file_path);
    if !path.exists() {
        return Err(AppError::NotFound("Video file not found".into()));
    }

    // Parse quality preset
    let quality = match body.quality_preset.as_deref() {
        Some("480p") => QualityPreset {
            resolution: Some("480p".into()),
            bitrate: Some("1536K".into()),
        },
        Some("720p") => QualityPreset {
            resolution: Some("720p".into()),
            bitrate: Some("4M".into()),
        },
        Some("1080p") => QualityPreset {
            resolution: Some("1080p".into()),
            bitrate: Some("8M".into()),
        },
        Some("1440p") => QualityPreset {
            resolution: Some("1440p".into()),
            bitrate: Some("14M".into()),
        },
        Some("4k") | Some("2160p") => QualityPreset {
            resolution: Some("4k".into()),
            bitrate: Some("25M".into()),
        },
        _ => QualityPreset::default(), // Original quality, CRF mode
    };

    let info = state
        .transcode_manager()
        .start_stream(&body.file_path, body.start_position, &quality, body.force_cfr)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to start transcode: {}", e)))?;

    Ok(Json(json!({
        "success": true,
        "stream_id": info.stream_id,
        "stream_url": info.stream_url,
        "duration": info.duration,
        "start_position": info.start_position,
        "source_resolution": {
            "width": info.source_resolution.width,
            "height": info.source_resolution.height,
        },
        "message": "Transcoding started"
    })))
}

/// POST /transcode/stop — Stop all active transcode streams.
async fn stop_transcode_streams(
    State(state): State<AppState>,
) -> Json<Value> {
    state.transcode_manager().stop_all();
    Json(json!({ "success": true, "message": "All transcode streams stopped" }))
}

/// GET /transcode/stream/{stream_id}/{filename} — Serve HLS .m3u8 or .ts files.
async fn serve_transcode_file(
    State(state): State<AppState>,
    AxumPath((stream_id, filename)): AxumPath<(String, String)>,
) -> Result<Response, AppError> {
    let hls_dir = state
        .transcode_manager()
        .get_stream_hls_dir(&stream_id)
        .ok_or_else(|| AppError::NotFound("Stream not found".into()))?;

    let file_path = hls_dir.join(&filename);
    if !file_path.exists() {
        return Err(AppError::NotFound("HLS file not found".into()));
    }

    // Determine content type
    let content_type = if filename.ends_with(".m3u8") {
        "application/vnd.apple.mpegurl"
    } else if filename.ends_with(".ts") {
        "video/mp2t"
    } else {
        "application/octet-stream"
    };

    let data = tokio::fs::read(&file_path).await?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, data.len())
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(Body::from(data))
        .unwrap())
}

// ─── Sidecar bridge helpers ──────────────────────────────────────────────────

/// Forward a POST request to a sidecar addon and return its JSON response.
/// For "play" endpoints, rewrites `stream_url` to route through the Rust proxy.
async fn bridge_post(
    state: &AppState,
    addon_id: &str,
    sidecar_path: &str,
    body: &Value,
    rewrite_stream_url: Option<&str>,
) -> Result<Json<Value>, AppError> {
    let base_url = state
        .addon_manager()
        .addon_url(addon_id)
        .ok_or_else(|| {
            AppError::ServiceUnavailable(format!(
                "Addon '{}' is not running. Install and start it first.",
                addon_id
            ))
        })?;

    let url = format!("{}{}", base_url.trim_end_matches('/'), sidecar_path);

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(body)
        .send()
        .await
        .map_err(|e| {
            AppError::ServiceUnavailable(format!("Failed to reach addon '{}': {}", addon_id, e))
        })?;

    let status = response.status();
    let mut result: Value = response.json().await.map_err(|e| {
        AppError::Internal(format!("Invalid response from addon '{}': {}", addon_id, e))
    })?;

    if !status.is_success() {
        let detail = result
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown sidecar error");
        return Err(AppError::Internal(detail.to_string()));
    }

    // Rewrite stream_url to route through Rust server
    if let Some(prefix) = rewrite_stream_url {
        if let Some(sidecar_url) = result.get("stream_url").and_then(|v| v.as_str()) {
            // Extract stream_id and filename from the sidecar's URL
            // e.g., "/optical-flow/stream/abc123/playlist.m3u8"
            let parts: Vec<&str> = sidecar_url.trim_start_matches('/').split('/').collect();
            if parts.len() >= 3 {
                let stream_id = parts[parts.len() - 2];
                let filename = parts[parts.len() - 1];
                let new_url = format!(
                    "/api/settings/{}/stream/{}/{}",
                    prefix, stream_id, filename
                );
                result["stream_url"] = json!(new_url);
            }
        }
    }

    Ok(Json(result))
}

/// Forward a GET request for HLS files to a sidecar addon, streaming the response.
async fn bridge_stream(
    state: &AppState,
    addon_id: &str,
    sidecar_path: &str,
) -> Result<Response, AppError> {
    let base_url = state
        .addon_manager()
        .addon_url(addon_id)
        .ok_or_else(|| {
            AppError::ServiceUnavailable(format!(
                "Addon '{}' is not running",
                addon_id
            ))
        })?;

    let url = format!("{}{}", base_url.trim_end_matches('/'), sidecar_path);

    let client = reqwest::Client::new();
    let response = client.get(&url).send().await.map_err(|e| {
        AppError::ServiceUnavailable(format!("Failed to reach addon '{}': {}", addon_id, e))
    })?;

    let status = StatusCode::from_u16(response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();

    let body_bytes = response.bytes().await.map_err(|e| {
        AppError::Internal(format!("Failed to read response from addon '{}': {}", addon_id, e))
    })?;

    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, &content_type)
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(Body::from(body_bytes))
        .unwrap())
}

// ─── Optical flow bridge ─────────────────────────────────────────────────────

/// POST /optical-flow/play — Start optical flow interpolated stream via sidecar.
async fn bridge_optical_flow_play(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    bridge_post(
        &state,
        "frame-interpolation",
        "/optical-flow/play",
        &body,
        Some("optical-flow"),
    )
    .await
}

/// POST /optical-flow/stop — Stop optical flow streams via sidecar.
async fn bridge_optical_flow_stop(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    bridge_post(
        &state,
        "frame-interpolation",
        "/optical-flow/stop",
        &json!({}),
        None,
    )
    .await
}

/// GET /optical-flow/stream/{stream_id}/{filename} — Serve HLS files via sidecar.
async fn bridge_optical_flow_stream(
    State(state): State<AppState>,
    AxumPath((stream_id, filename)): AxumPath<(String, String)>,
) -> Result<Response, AppError> {
    bridge_stream(
        &state,
        "frame-interpolation",
        &format!("/optical-flow/stream/{}/{}", stream_id, filename),
    )
    .await
}

// ─── SVP bridge ──────────────────────────────────────────────────────────────

/// POST /svp/play — Start SVP interpolated stream via sidecar.
async fn bridge_svp_play(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    bridge_post(&state, "svp", "/svp/play", &body, Some("svp")).await
}

/// POST /svp/stop — Stop SVP streams via sidecar.
async fn bridge_svp_stop(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    bridge_post(&state, "svp", "/svp/stop", &json!({}), None).await
}

/// GET /svp/stream/{stream_id}/{filename} — Serve SVP HLS files via sidecar.
async fn bridge_svp_stream(
    State(state): State<AppState>,
    AxumPath((stream_id, filename)): AxumPath<(String, String)>,
) -> Result<Response, AppError> {
    bridge_stream(
        &state,
        "svp",
        &format!("/svp/stream/{}/{}", stream_id, filename),
    )
    .await
}
