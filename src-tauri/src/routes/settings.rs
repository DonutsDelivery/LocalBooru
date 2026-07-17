use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, OriginalUri, Path as AxumPath, Query, State};
use axum::http::{header, Method, StatusCode};
use axum::response::{Json, Response};
use axum::routing::{any, delete, get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::addons::manager::AddonStatus;
use crate::server::error::AppError;
use crate::server::middleware::AccessTier;
use crate::server::state::AppState;
use crate::services::transcode::QualityPreset;

static LEGACY_SVP_TRANSITION_EPOCH: AtomicU64 = AtomicU64::new(0);
static LEGACY_SVP_CLIENT_TRANSITION_EPOCH: AtomicU64 = AtomicU64::new(0);
static LEGACY_SVP_TRANSITION_LOCK: Mutex<()> = Mutex::new(());

fn owns_legacy_svp_transition(epoch: u64) -> bool {
    LEGACY_SVP_TRANSITION_EPOCH.load(Ordering::SeqCst) == epoch
}

fn claim_legacy_svp_client_transition(epoch: Option<u64>) -> Option<u64> {
    let _guard = LEGACY_SVP_TRANSITION_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(epoch) = epoch {
        let previous = LEGACY_SVP_CLIENT_TRANSITION_EPOCH.load(Ordering::SeqCst);
        if epoch < previous {
            return None;
        }
        LEGACY_SVP_CLIENT_TRANSITION_EPOCH.store(epoch, Ordering::SeqCst);
    }
    Some(LEGACY_SVP_TRANSITION_EPOCH.fetch_add(1, Ordering::SeqCst) + 1)
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(get_all_settings).post(update_settings))
        .nest("/family-mode", family_mode_router())
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
        .route("/svp", get(get_svp).post(update_svp))
        .route("/whisper", get(get_whisper).post(update_whisper))
        .route("/cast", get(get_cast).post(update_cast))
        .route(
            "/auto-tagger",
            get(get_auto_tagger).post(update_auto_tagger),
        )
        .route("/age-detection/status", get(get_age_detection_status))
        .route("/age-detection/toggle", post(toggle_age_detection))
        .route("/age-detection/install", post(install_age_detection))
        .route("/cast/install", post(install_cast))
        .route("/video-info", post(get_video_info_endpoint))
        .route("/audio-gain", post(get_audio_gain_endpoint))
        // ─── Transcode streaming ────────────────────────────────────────
        .route("/transcode/play", post(start_transcode_stream))
        .route("/transcode/stop", post(stop_transcode_streams))
        .route(
            "/transcode/stream/{stream_id}/{filename}",
            get(serve_transcode_file),
        )
        // Compatibility for existing desktop/mobile clients that still send this
        // cleanup request after the optical-flow settings UI was retired.
        .route("/optical-flow/stop", post(bridge_optical_flow_stop))
        // ─── SVP streaming (sidecar bridge) ─────────────────────────────
        .route("/svp/play", post(bridge_svp_play))
        .route("/svp/stop", post(bridge_svp_stop))
        .route("/svp/stream/{stream_id}/{filename}", get(bridge_svp_stream))
        .route("/svp/sessions", any(bridge_svp_sessions_root))
        .route("/svp/sessions/{*path}", any(bridge_svp_sessions_path))
        // ─── Whisper subtitle streaming (sidecar bridge) ──────────────────
        .route("/whisper/install", post(bridge_whisper_install))
        .route("/whisper/generate", post(bridge_whisper_generate))
        .route("/whisper/stop", post(bridge_whisper_stop))
        .route(
            "/whisper/vtt/{stream_id}/subtitles.vtt",
            get(bridge_whisper_vtt),
        )
        .route("/whisper/events/{stream_id}", get(bridge_whisper_events))
}

// ─── Family mode sub-router ──────────────────────────────────────────────────

fn family_mode_router() -> Router<AppState> {
    Router::new()
        .route("/", get(get_family_mode).post(configure_family_mode))
        .route("/unlock", post(unlock_family_mode))
        .route("/lock", post(lock_family_mode))
}

// ─── Family mode PIN hashing ─────────────────────────────────────────────────

fn hash_pin(pin: &str) -> Result<String, AppError> {
    use argon2::password_hash::{rand_core::OsRng, PasswordHasher, SaltString};
    use argon2::Argon2;
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(pin.as_bytes(), &salt)
        .map_err(|e| AppError::Internal(format!("PIN hashing error: {}", e)))?;
    Ok(hash.to_string())
}

fn verify_pin(pin: &str, stored_hash: &str) -> Result<bool, AppError> {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    use argon2::Argon2;
    let parsed_hash = PasswordHash::new(stored_hash)
        .map_err(|e| AppError::Internal(format!("Invalid stored hash: {}", e)))?;
    Ok(Argon2::default()
        .verify_password(pin.as_bytes(), &parsed_hash)
        .is_ok())
}

// ─── Family-mode PIN brute-force backoff (M3) ────────────────────────────────
// In-memory failed-attempt tracker keyed by client IP. The first
// FAMILY_PIN_FREE_ATTEMPTS wrong PINs are answered immediately; each further
// failure locks that IP out for an exponentially growing window (capped). A
// correct PIN clears the IP's counter. State is process-local (resets on
// restart) — fine here: an attacker can't trigger a restart remotely, and the
// owner who knows the PIN succeeds on the first try and never accrues a lockout.
// Keying by IP means a LAN attacker can't lock the host (localhost) out.
const FAMILY_PIN_FREE_ATTEMPTS: u32 = 5;
const FAMILY_PIN_BASE_LOCKOUT_SECS: u64 = 15;
const FAMILY_PIN_MAX_LOCKOUT_SECS: u64 = 900; // 15 min cap

struct PinAttempts {
    failures: u32,
    locked_until: Option<std::time::Instant>,
}

static FAMILY_PIN_GUARD: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<std::net::IpAddr, PinAttempts>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// If the IP is currently locked out, returns `Err(remaining_seconds)`.
fn family_pin_check_lock(ip: std::net::IpAddr) -> Result<(), u64> {
    let map = FAMILY_PIN_GUARD.lock().unwrap();
    if let Some(entry) = map.get(&ip) {
        if let Some(until) = entry.locked_until {
            let now = std::time::Instant::now();
            if now < until {
                return Err((until - now).as_secs() + 1);
            }
        }
    }
    Ok(())
}

fn family_pin_record_failure(ip: std::net::IpAddr) {
    let mut map = FAMILY_PIN_GUARD.lock().unwrap();
    let entry = map.entry(ip).or_insert(PinAttempts {
        failures: 0,
        locked_until: None,
    });
    entry.failures += 1;
    if entry.failures > FAMILY_PIN_FREE_ATTEMPTS {
        let over = entry.failures - FAMILY_PIN_FREE_ATTEMPTS; // 1, 2, 3, ...
        let secs = FAMILY_PIN_BASE_LOCKOUT_SECS
            .saturating_mul(1u64 << (over - 1).min(6))
            .min(FAMILY_PIN_MAX_LOCKOUT_SECS);
        entry.locked_until = Some(std::time::Instant::now() + std::time::Duration::from_secs(secs));
    }
}

fn family_pin_record_success(ip: std::net::IpAddr) {
    FAMILY_PIN_GUARD.lock().unwrap().remove(&ip);
}

// ─── Family mode request models ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct FamilyModeConfigure {
    enabled: Option<bool>,
    pin: Option<String>,
    auto_lock_on_start: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct FamilyModeUnlock {
    pin: String,
}

// ─── Family mode handlers ────────────────────────────────────────────────────

/// GET /family-mode — Get family mode status.
async fn get_family_mode(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let is_locked = state.is_family_mode_locked();

    let result = tokio::task::spawn_blocking(move || {
        let config = get_config_section(&data_dir, "family_mode");
        let enabled = config
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let has_pin = config
            .get("pin_hash")
            .map(|v| !v.is_null() && v.as_str().map(|s| !s.is_empty()).unwrap_or(false))
            .unwrap_or(false);
        let auto_lock_on_start = config
            .get("auto_lock_on_start")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        json!({
            "enabled": enabled,
            "is_locked": is_locked,
            "has_pin": has_pin,
            "auto_lock_on_start": auto_lock_on_start
        })
    })
    .await?;

    Ok(Json(result))
}

/// POST /family-mode — Configure family mode (localhost only).
async fn configure_family_mode(
    State(state): State<AppState>,
    connect_info: ConnectInfo<SocketAddr>,
    Json(body): Json<FamilyModeConfigure>,
) -> Result<Json<Value>, AppError> {
    // Enforce localhost-only access
    let tier = AccessTier::from_ip(&connect_info.0.ip());
    if tier != AccessTier::Localhost {
        return Err(AppError::Forbidden(
            "Family mode configuration is only accessible from localhost".into(),
        ));
    }

    let data_dir = state.data_dir().to_path_buf();

    // Build the update payload
    let mut update = json!({});
    if let Some(enabled) = body.enabled {
        update["enabled"] = json!(enabled);
    }
    if let Some(auto_lock_on_start) = body.auto_lock_on_start {
        update["auto_lock_on_start"] = json!(auto_lock_on_start);
    }
    if let Some(ref pin) = body.pin {
        if pin.len() < 4 {
            return Err(AppError::BadRequest(
                "PIN must be at least 4 characters".into(),
            ));
        }
        let pin_hash = hash_pin(pin)?;
        update["pin_hash"] = json!(pin_hash);
    }

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "family_mode", &update)
    })
    .await??;

    // If disabling family mode, also unlock it
    if body.enabled == Some(false) {
        state.set_family_mode_locked(false);
        if let Some(events) = state.events() {
            events
                .library
                .broadcast("family_mode", json!({"is_locked": false}));
        }
    }

    Ok(Json(result))
}

/// POST /family-mode/unlock — Unlock family mode by verifying PIN.
async fn unlock_family_mode(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(body): Json<FamilyModeUnlock>,
) -> Result<Json<Value>, AppError> {
    let ip = addr.ip();

    // Brute-force backoff: reject early if this IP is currently locked out.
    if let Err(secs) = family_pin_check_lock(ip) {
        return Err(AppError::TooManyRequests(format!(
            "Too many incorrect PIN attempts. Try again in {} seconds.",
            secs
        )));
    }

    let data_dir = state.data_dir().to_path_buf();
    let pin = body.pin;

    let stored_hash = tokio::task::spawn_blocking(move || {
        let config = get_config_section(&data_dir, "family_mode");
        config
            .get("pin_hash")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    })
    .await?;

    let stored_hash = stored_hash
        .ok_or_else(|| AppError::BadRequest("No PIN has been configured for family mode".into()))?;

    let valid = verify_pin(&pin, &stored_hash)?;
    if !valid {
        family_pin_record_failure(ip);
        return Err(AppError::Forbidden("Invalid PIN".into()));
    }

    family_pin_record_success(ip);
    state.set_family_mode_locked(false);
    if let Some(events) = state.events() {
        events
            .library
            .broadcast("family_mode", json!({"is_locked": false}));
    }

    Ok(Json(json!({ "success": true, "is_locked": false })))
}

/// POST /family-mode/lock — Lock family mode (no PIN needed).
async fn lock_family_mode(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    state.set_family_mode_locked(true);
    if let Some(events) = state.events() {
        events
            .library
            .broadcast("family_mode", json!({"is_locked": true}));
    }

    Ok(Json(json!({ "success": true, "is_locked": true })))
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

/// Persist settings to disk as pretty-printed JSON without credential material.
fn save_settings_to_file(data_dir: &Path, settings: &Value) -> std::io::Result<()> {
    let path = settings_path(data_dir);
    let mut sanitized = settings.clone();
    if let Some(object) = sanitized.as_object_mut() {
        object.remove("jwt_secret");
    }
    let json_str = serde_json::to_string_pretty(&sanitized)
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
            "auto_advance_delay": 5,
            "desktop_player_mode": "react",
            "native_video_force_copy": false,
            "native_video_diagnostics": false
        },
        "saved_searches": [],

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
        "auto_tagger": {
            "model": "vit-v3",
            "general_threshold": 0.35,
            "character_threshold": 0.75,
            "device": "auto"
        },
        "age_detection": {
            "enabled": false
        },
        "family_mode": {
            "enabled": false,
            "pin_hash": null,
            "auto_lock_on_start": true
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
    bytes[..6].iter().map(|b| format!("{:02x}", b)).collect()
}

/// Read a single config section from settings, merging with defaults.
pub(crate) fn get_config_section(data_dir: &Path, section: &str) -> Value {
    let settings = load_settings(data_dir);
    let defaults = get_defaults();

    let default_section = defaults.get(section).cloned().unwrap_or_else(|| json!({}));
    let saved_section = settings.get(section).cloned().unwrap_or_else(|| json!({}));

    deep_merge(&default_section, &saved_section)
}

/// Update a single config section: deep-merge the incoming body with the
/// existing saved section, persist, and return the merged result (with defaults).
fn update_config_section(data_dir: &Path, section: &str, body: &Value) -> Result<Value, AppError> {
    let mut settings = load_settings(data_dir);

    let existing = settings.get(section).cloned().unwrap_or_else(|| json!({}));
    let merged = deep_merge(&existing, body);

    settings
        .as_object_mut()
        .ok_or_else(|| AppError::Internal("Settings file is not a JSON object".into()))?
        .insert(section.into(), merged.clone());

    save_settings_to_file(data_dir, &settings)?;

    let defaults = get_defaults();
    let default_section = defaults.get(section).cloned().unwrap_or_else(|| json!({}));
    Ok(deep_merge(&default_section, &merged))
}

// ─── Route handlers ──────────────────────────────────────────────────────────

/// GET / — Return all settings (defaults merged with saved overrides).
async fn get_all_settings(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let mut result = tokio::task::spawn_blocking(move || {
        let saved = load_settings(&data_dir);
        merge_with_defaults(&saved)
    })
    .await?;

    // Never expose the JWT signing secret through the settings API.
    if let Some(obj) = result.as_object_mut() {
        obj.remove("jwt_secret");
    }

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
        // Never allow the JWT signing secret to be set or overwritten via the settings
        // API; the existing on-disk secret is preserved by leaving it out of the merge.
        let mut body = body;
        if let Some(obj) = body.as_object_mut() {
            obj.remove("jwt_secret");
        }
        let merged = deep_merge(&existing, &body);
        save_settings_to_file(&data_dir, &merged)?;
        let mut out = merge_with_defaults(&merged);
        if let Some(obj) = out.as_object_mut() {
            obj.remove("jwt_secret");
        }
        Ok::<_, AppError>(out)
    })
    .await??;

    Ok(Json(result))
}

// ─── Saved searches ──────────────────────────────────────────────────────────

/// GET /saved-searches — List all saved searches.
async fn list_saved_searches(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let settings = load_settings(&data_dir);
        let searches = settings
            .get("saved_searches")
            .cloned()
            .unwrap_or_else(|| json!([]));
        json!({ "searches": searches })
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
        Ok::<_, AppError>(json!({ "search": new_search }))
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

const DESKTOP_PLAYER_MODES: &[&str] = &["react", "native", "native_svp"];

fn migrate_desktop_player_mode(settings: &mut Value) -> bool {
    let Some(settings_object) = settings.as_object_mut() else {
        return false;
    };
    let legacy_svp_enabled = settings_object
        .get("svp")
        .and_then(|value| value.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let video_playback = settings_object
        .entry("video_playback")
        .or_insert_with(|| json!({}));
    let Some(video_object) = video_playback.as_object_mut() else {
        return false;
    };
    let existing_mode = video_object
        .get("desktop_player_mode")
        .and_then(Value::as_str)
        .filter(|mode| DESKTOP_PLAYER_MODES.contains(mode))
        .map(str::to_owned);
    let legacy_native_enabled = video_object
        .get("native_video_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mode = existing_mode.unwrap_or_else(|| {
        if legacy_native_enabled && legacy_svp_enabled {
            "native_svp".to_string()
        } else if legacy_native_enabled {
            "native".to_string()
        } else {
            "react".to_string()
        }
    });
    let changed = video_object.get("desktop_player_mode") != Some(&json!(mode))
        || video_object.contains_key("native_video_enabled");
    video_object.insert("desktop_player_mode".into(), json!(mode));
    video_object.remove("native_video_enabled");
    changed
}

pub(crate) fn get_desktop_player_mode(data_dir: &Path) -> String {
    let mut settings = load_settings(data_dir);
    if migrate_desktop_player_mode(&mut settings) {
        if let Err(error) = save_settings_to_file(data_dir, &settings) {
            log::warn!("Failed to persist desktop player migration: {error}");
        }
    }
    settings
        .get("video_playback")
        .and_then(|value| value.get("desktop_player_mode"))
        .and_then(Value::as_str)
        .unwrap_or("react")
        .to_string()
}

/// GET /video-playback — Get video playback configuration.
async fn get_video_playback(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        get_desktop_player_mode(&data_dir);
        get_config_section(&data_dir, "video_playback")
    })
    .await?;

    Ok(Json(result))
}

fn validate_video_playback_config(body: &mut Value) -> Result<(), AppError> {
    if let Some(delay) = body.get("auto_advance_delay").and_then(|v| v.as_i64()) {
        body["auto_advance_delay"] = json!(delay.clamp(1, 30));
    }
    if let Some(mode) = body.get("desktop_player_mode") {
        let mode = mode
            .as_str()
            .ok_or_else(|| AppError::BadRequest("desktop_player_mode must be a string".into()))?;
        if !DESKTOP_PLAYER_MODES.contains(&mode) {
            return Err(AppError::BadRequest(format!(
                "desktop_player_mode must be one of: {}",
                DESKTOP_PLAYER_MODES.join(", ")
            )));
        }
    }
    if let Some(legacy_enabled) = body.get("native_video_enabled").cloned() {
        let legacy_enabled = legacy_enabled
            .as_bool()
            .ok_or_else(|| AppError::BadRequest("native_video_enabled must be a boolean".into()))?;
        if body.get("desktop_player_mode").is_none() {
            body["desktop_player_mode"] = json!(if legacy_enabled { "native" } else { "react" });
        }
        body.as_object_mut()
            .ok_or_else(|| AppError::BadRequest("video playback update must be an object".into()))?
            .remove("native_video_enabled");
    }
    for key in ["native_video_force_copy", "native_video_diagnostics"] {
        if body.get(key).is_some_and(|value| !value.is_boolean()) {
            return Err(AppError::BadRequest(format!("{key} must be a boolean")));
        }
    }
    Ok(())
}

/// POST /video-playback — Update video playback configuration with validation.
async fn update_video_playback(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    validate_video_playback_config(&mut body)?;

    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        get_desktop_player_mode(&data_dir);
        update_config_section(&data_dir, "video_playback", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── Optical flow ────────────────────────────────────────────────────────────

/// GET /optical-flow — Get optical flow configuration + backend status.
///
/// Frame interpolation is handled natively via FFmpeg's minterpolate filter,
/// so no addon/sidecar is required.
async fn get_optical_flow(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "optical_flow");
        if let Some(obj) = config.as_object_mut() {
            // FFmpeg minterpolate is always available — no addon needed
            obj.insert(
                "backend".into(),
                json!({
                    "any_backend_available": true,
                    "name": "ffmpeg_minterpolate"
                }),
            );
        }
        config
    })
    .await?;

    Ok(Json(result))
}

/// Valid quality presets for optical flow.
const OPTICAL_FLOW_QUALITIES: &[&str] = &[
    "low",
    "medium",
    "high",
    "svp",
    "gpu_native",
    "realtime",
    "fast",
    "balanced",
    "quality",
];

/// POST /optical-flow — Update optical flow configuration with validation.
async fn update_optical_flow(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    // Validate quality enum
    if let Some(quality) = body.get("quality").and_then(|v| v.as_str()) {
        if !OPTICAL_FLOW_QUALITIES.contains(&quality) {
            return Err(AppError::BadRequest(format!(
                "Invalid quality '{}'. Must be one of: {}",
                quality,
                OPTICAL_FLOW_QUALITIES.join(", ")
            )));
        }
    }

    // Clamp target_fps to 15-120
    if let Some(fps) = body.get("target_fps").and_then(|v| v.as_i64()) {
        body["target_fps"] = json!(fps.clamp(15, 120));
    }

    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        update_config_section(&data_dir, "optical_flow", &body)
    })
    .await??;

    Ok(Json(result))
}

// ─── Auto Tagger ────────────────────────────────────────────────────────────

const AUTO_TAGGER_MODELS: &[&str] = &["vit-v3", "eva02-large-v3", "swinv2-v3"];
const AUTO_TAGGER_DEVICES: &[&str] = &["auto", "cuda", "cpu"];

fn validate_auto_tagger_config(body: &mut Value) -> Result<(), AppError> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| AppError::BadRequest("Auto Tagger settings must be an object".into()))?;

    if let Some(model) = object.get("model").and_then(Value::as_str) {
        if !AUTO_TAGGER_MODELS.contains(&model) {
            return Err(AppError::BadRequest(format!(
                "Invalid Auto Tagger model '{}'. Must be one of: {}",
                model,
                AUTO_TAGGER_MODELS.join(", ")
            )));
        }
    }

    if let Some(device) = object.get("device").and_then(Value::as_str) {
        if !AUTO_TAGGER_DEVICES.contains(&device) {
            return Err(AppError::BadRequest(format!(
                "Invalid Auto Tagger device '{}'. Must be one of: {}",
                device,
                AUTO_TAGGER_DEVICES.join(", ")
            )));
        }
    }

    for field in ["general_threshold", "character_threshold"] {
        if let Some(value) = object.get(field).and_then(Value::as_f64) {
            object.insert(field.into(), json!(value.clamp(0.0, 1.0)));
        } else if object.contains_key(field) {
            return Err(AppError::BadRequest(format!("{} must be a number", field)));
        }
    }

    Ok(())
}

async fn get_auto_tagger(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let config =
        tokio::task::spawn_blocking(move || get_config_section(&data_dir, "auto_tagger")).await?;
    Ok(Json(config))
}

async fn update_auto_tagger(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    validate_auto_tagger_config(&mut body)?;
    let should_restart =
        state.addon_manager().get_addon_status("auto-tagger") == AddonStatus::Running;
    let data_dir = state.data_dir().to_path_buf();
    let config =
        tokio::task::spawn_blocking(move || update_config_section(&data_dir, "auto_tagger", &body))
            .await??;

    if should_restart {
        state
            .addon_manager()
            .stop_addon("auto-tagger")
            .await
            .map_err(AppError::Internal)?;
        state
            .addon_manager()
            .start_addon("auto-tagger")
            .await
            .map_err(AppError::Internal)?;
    }

    Ok(Json(config))
}

// ─── SVP ─────────────────────────────────────────────────────────────────────

/// GET /svp — Get SVP configuration + status from addon.
async fn get_svp(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let addon_mgr = state.addon_manager();

    let addon_info = addon_mgr.get_addon("svp");
    let (installed, running) = match &addon_info {
        Some(info) => (info.installed, info.status == AddonStatus::Running),
        None => (false, false),
    };

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "svp");
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "status".into(),
                json!({
                    "installed": installed,
                    "running": running,
                    "ready": running
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

/// Valid SVP shader values.
const SVP_VALID_SHADERS: &[i64] = &[1, 2, 11, 13, 21, 23];

/// POST /svp — Update SVP configuration with validation.
async fn update_svp(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    // Validate shader enum
    if let Some(shader) = body.get("shader").and_then(|v| v.as_i64()) {
        if !SVP_VALID_SHADERS.contains(&shader) {
            return Err(AppError::BadRequest(format!(
                "Invalid shader {}. Must be one of: {}",
                shader,
                SVP_VALID_SHADERS
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }

    // Clamp target_fps to 15-144
    if let Some(fps) = body.get("target_fps").and_then(|v| v.as_i64()) {
        body["target_fps"] = json!(fps.clamp(15, 144));
    }

    let data_dir = state.data_dir().to_path_buf();

    let result =
        tokio::task::spawn_blocking(move || update_config_section(&data_dir, "svp", &body))
            .await??;

    Ok(Json(result))
}

// ─── Whisper ─────────────────────────────────────────────────────────────────

/// GET /whisper — Get whisper subtitle configuration + status from addon.
async fn get_whisper(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let addon_mgr = state.addon_manager();

    let addon_info = addon_mgr.get_addon(WHISPER_ADDON_ID);
    let (installed, running) = match &addon_info {
        Some(info) => (info.installed, info.status == AddonStatus::Running),
        None => (false, false),
    };

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "whisper");
        if let Some(obj) = config.as_object_mut() {
            obj.insert(
                "status".into(),
                json!({
                    "installed": installed,
                    "running": running,
                    "faster_whisper_installed": installed
                }),
            );
        }
        config
    })
    .await?;

    Ok(Json(result))
}

/// Valid whisper model sizes.
const WHISPER_MODEL_SIZES: &[&str] = &["tiny", "base", "small", "medium", "large-v2", "large-v3"];
/// Valid whisper tasks.
const WHISPER_TASKS: &[&str] = &["transcribe", "translate"];
/// Valid whisper devices.
const WHISPER_DEVICES: &[&str] = &["cpu", "cuda", "auto"];
/// Valid whisper compute types.
const WHISPER_COMPUTE_TYPES: &[&str] = &["float16", "float32", "int8", "int8_float16", "auto"];

/// POST /whisper — Update whisper subtitle configuration with validation.
async fn update_whisper(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    // Validate model_size enum
    if let Some(model_size) = body.get("model_size").and_then(|v| v.as_str()) {
        if !WHISPER_MODEL_SIZES.contains(&model_size) {
            return Err(AppError::BadRequest(format!(
                "Invalid model_size '{}'. Must be one of: {}",
                model_size,
                WHISPER_MODEL_SIZES.join(", ")
            )));
        }
    }

    // Validate task enum
    if let Some(task) = body.get("task").and_then(|v| v.as_str()) {
        if !WHISPER_TASKS.contains(&task) {
            return Err(AppError::BadRequest(format!(
                "Invalid task '{}'. Must be one of: {}",
                task,
                WHISPER_TASKS.join(", ")
            )));
        }
    }

    // Validate device enum
    if let Some(device) = body.get("device").and_then(|v| v.as_str()) {
        if !WHISPER_DEVICES.contains(&device) {
            return Err(AppError::BadRequest(format!(
                "Invalid device '{}'. Must be one of: {}",
                device,
                WHISPER_DEVICES.join(", ")
            )));
        }
    }

    // Validate compute_type enum
    if let Some(compute_type) = body.get("compute_type").and_then(|v| v.as_str()) {
        if !WHISPER_COMPUTE_TYPES.contains(&compute_type) {
            return Err(AppError::BadRequest(format!(
                "Invalid compute_type '{}'. Must be one of: {}",
                compute_type,
                WHISPER_COMPUTE_TYPES.join(", ")
            )));
        }
    }

    // Clamp temperature to 0.0-1.0
    if let Some(temp) = body.get("temperature").and_then(|v| v.as_f64()) {
        body["temperature"] = json!(temp.clamp(0.0, 1.0));
    }

    let data_dir = state.data_dir().to_path_buf();

    let result =
        tokio::task::spawn_blocking(move || update_config_section(&data_dir, "whisper", &body))
            .await??;

    Ok(Json(result))
}

// ─── Cast ────────────────────────────────────────────────────────────────────

/// GET /cast — Get cast configuration + status from addon.
async fn get_cast(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let addon_mgr = state.addon_manager();

    let addon_info = addon_mgr.get_addon("cast");
    let (installed, running) = match &addon_info {
        Some(info) => (info.installed, info.status == AddonStatus::Running),
        None => (false, false),
    };

    let result = tokio::task::spawn_blocking(move || {
        let mut config = get_config_section(&data_dir, "cast");
        if let Some(obj) = config.as_object_mut() {
            obj.insert("installing".into(), json!(false));
            obj.insert("installed".into(), json!(installed));
            obj.insert("running".into(), json!(running));
            obj.insert(
                "status".into(),
                json!({
                    "installed": installed,
                    "running": running
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

    let result =
        tokio::task::spawn_blocking(move || update_config_section(&data_dir, "cast", &body))
            .await??;

    Ok(Json(result))
}

// ─── Age detection ───────────────────────────────────────────────────────────

/// GET /age-detection/status — Check age detection addon status.
async fn get_age_detection_status(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let addon_mgr = state.addon_manager();

    let addon_info = addon_mgr.get_addon("age-detector");
    let (installed, running) = match &addon_info {
        Some(info) => (info.installed, info.status == AddonStatus::Running),
        None => (false, false),
    };

    let config =
        tokio::task::spawn_blocking(move || get_config_section(&data_dir, "age_detection")).await?;

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

// ─── Addon install bridges ───────────────────────────────────────────────

/// POST /age-detection/install — Install the age-detection addon (create venv, install deps).
async fn install_age_detection(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        state_clone
            .addon_manager()
            .install_addon("age-detector")
            .map_err(|e| {
                AppError::Internal(format!("Failed to install age-detection addon: {}", e))
            })
    })
    .await??;

    Ok(Json(json!({
        "status": "installed",
        "addon_id": "age-detector",
        "message": "Age detection addon installed successfully"
    })))
}

/// POST /cast/install — Install the cast addon (create venv, install deps).
async fn install_cast(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        state_clone
            .addon_manager()
            .install_addon("cast")
            .map_err(|e| AppError::Internal(format!("Failed to install cast addon: {}", e)))
    })
    .await??;

    Ok(Json(json!({
        "status": "installed",
        "addon_id": "cast",
        "message": "Cast addon installed successfully"
    })))
}

// ─── Video info ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct VideoInfoRequest {
    file_path: String,
}

/// POST /video-info — Get video information including VFR detection.
async fn get_video_info_endpoint(
    State(state): State<AppState>,
    Json(body): Json<VideoInfoRequest>,
) -> Result<Json<Value>, AppError> {
    let file_path = body.file_path;

    // Reject paths outside the media library before handing them to ffprobe.
    crate::server::utils::validate_path_in_watch_dir(&state, &file_path)?;

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
    State(state): State<AppState>,
    Query(q): Query<DimensionsQuery>,
) -> Result<Json<Value>, AppError> {
    let file_path = q.file_path.clone();

    // Reject paths outside the media library before handing them to ffprobe.
    crate::server::utils::validate_path_in_watch_dir(&state, &file_path)?;

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
            match get_video_info(path) {
                Ok(info) => {
                    let mut result = json!({
                        "success": true,
                        "width": info.width,
                        "height": info.height
                    });
                    if let Some(fps) = info.fps {
                        result["fps"] = json!(fps);
                    }
                    result
                }
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

/// Video dimension and FPS info from ffprobe.
pub(crate) struct VideoInfo {
    pub(crate) width: u64,
    pub(crate) height: u64,
    pub(crate) fps: Option<f64>,
}

/// Run ffprobe to extract width, height, and FPS from a video file.
pub(crate) fn get_video_info(path: &Path) -> Result<VideoInfo, String> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
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

    // Parse FPS from r_frame_rate (e.g., "24000/1001") or avg_frame_rate
    let fps = stream
        .get("r_frame_rate")
        .and_then(|v| v.as_str())
        .or_else(|| stream.get("avg_frame_rate").and_then(|v| v.as_str()))
        .and_then(|s| {
            if let Some((num, den)) = s.split_once('/') {
                let n: f64 = num.parse().ok()?;
                let d: f64 = den.parse().ok()?;
                if d > 0.0 {
                    Some(n / d)
                } else {
                    None
                }
            } else {
                s.parse().ok()
            }
        });

    Ok(VideoInfo { width, height, fps })
}

/// POST /audio-gain — Detect attenuation needed to keep a file's peak at or below -2 dBFS.
async fn get_audio_gain_endpoint(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let file_path = match body.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p.to_string(),
        None => return Json(json!({ "gain_db": serde_json::Value::Null })),
    };
    // Reject paths outside the media library before probing with ffmpeg.
    let in_library = crate::server::utils::validate_path_in_watch_dir(&state, &file_path).is_ok();
    if !in_library {
        return Json(json!({ "gain_db": serde_json::Value::Null }));
    }
    let gain_db = crate::services::transcode::detect_audio_gain(&file_path).await;
    Json(json!({ "gain_db": gain_db }))
}

// ─── Transcode streaming ─────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct TranscodePlayRequest {
    file_path: String,
    #[serde(default)]
    start_position: f64,
    #[serde(default)]
    quality_preset: Option<String>,
    #[serde(default = "default_true")]
    force_cfr: bool,
}

/// Parse a quality preset string into FFmpeg parameters.
fn parse_quality_preset(preset: Option<&str>) -> QualityPreset {
    match preset {
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
        Some("1080p_enhanced") => QualityPreset {
            resolution: Some("1080p".into()),
            bitrate: Some("20M".into()),
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
    }
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
    // Reject paths outside the media library before transcoding.
    crate::server::utils::validate_path_in_watch_dir(&state, &body.file_path)?;

    // Parse quality preset
    let quality = parse_quality_preset(body.quality_preset.as_deref());

    let info = state
        .transcode_manager()
        .start_stream(
            &body.file_path,
            body.start_position,
            &quality,
            body.force_cfr,
        )
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
async fn stop_transcode_streams(State(state): State<AppState>) -> Json<Value> {
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
    let base_url = state.addon_manager().addon_url(addon_id).ok_or_else(|| {
        AppError::ServiceUnavailable(format!(
            "Addon '{}' is not running. Install and start it first.",
            addon_id
        ))
    })?;

    let url = format!("{}{}", base_url.trim_end_matches('/'), sidecar_path);

    let client = state.http_client();
    let response = client.post(&url).json(body).send().await.map_err(|e| {
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
                let new_url = format!("/api/settings/{}/stream/{}/{}", prefix, stream_id, filename);
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
    let base_url = state.addon_manager().addon_url(addon_id).ok_or_else(|| {
        AppError::ServiceUnavailable(format!("Addon '{}' is not running", addon_id))
    })?;

    let url = format!("{}{}", base_url.trim_end_matches('/'), sidecar_path);

    let client = state.http_client();
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
        AppError::Internal(format!(
            "Failed to read response from addon '{}': {}",
            addon_id, e
        ))
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

/// POST /optical-flow/play — Start interpolated stream via FFmpeg minterpolate.
async fn bridge_optical_flow_play(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let file_path = body
        .get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::BadRequest("file_path is required".into()))?;

    let path = Path::new(file_path);
    if !path.exists() {
        return Err(AppError::NotFound("Video file not found".into()));
    }
    // Reject paths outside the media library before interpolating.
    crate::server::utils::validate_path_in_watch_dir(&state, file_path)?;

    let start_position = body
        .get("start_position")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    // Read target_fps from optical_flow config
    let data_dir = state.data_dir().to_path_buf();
    let config =
        tokio::task::spawn_blocking(move || get_config_section(&data_dir, "optical_flow")).await?;
    let target_fps = config
        .get("target_fps")
        .and_then(|v| v.as_u64())
        .unwrap_or(60) as u32;

    // Parse quality preset (if provided)
    let quality = parse_quality_preset(body.get("quality_preset").and_then(|v| v.as_str()));

    let info = state
        .transcode_manager()
        .start_interpolated_stream(file_path, start_position, &quality, target_fps)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to start interpolated stream: {}", e)))?;

    Ok(Json(json!({
        "success": true,
        "stream_id": info.stream_id,
        "stream_url": info.stream_url,
        "duration": info.duration,
        "start_position": info.start_position,
        "source_resolution": {
            "width": info.source_resolution.width,
            "height": info.source_resolution.height,
        }
    })))
}

/// POST /optical-flow/stop — Stop interpolated streams.
async fn bridge_optical_flow_stop(State(state): State<AppState>) -> Json<Value> {
    state.transcode_manager().stop_all();
    Json(json!({ "success": true, "message": "All interpolated streams stopped" }))
}

/// GET /optical-flow/stream/{stream_id}/{filename} — Serve HLS files.
///
/// Reuses the same TranscodeManager, so stream URLs from optical-flow/play
/// also work through the transcode/stream endpoint. This endpoint is kept
/// for backward compatibility with the frontend API paths.
async fn bridge_optical_flow_stream(
    State(state): State<AppState>,
    AxumPath((stream_id, filename)): AxumPath<(String, String)>,
) -> Result<Response, AppError> {
    serve_transcode_file(State(state), AxumPath((stream_id, filename))).await
}

// ─── SVP bridge ──────────────────────────────────────────────────────────────

/// POST /svp/play — Start SVP interpolated stream via sidecar.
async fn bridge_svp_play(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let body_obj = body
        .as_object_mut()
        .ok_or_else(|| AppError::BadRequest("SVP play request must be an object".to_string()))?;
    let client_transition_id = body_obj
        .remove("client_transition_id")
        .map(|value| {
            value.as_u64().filter(|epoch| *epoch > 0).ok_or_else(|| {
                AppError::BadRequest("client_transition_id must be a positive integer".to_string())
            })
        })
        .transpose()?;
    let transition_epoch = claim_legacy_svp_client_transition(client_transition_id)
        .ok_or_else(|| AppError::ServiceUnavailable("SVP start was superseded".to_string()))?;
    body_obj.insert("transition_id".to_string(), json!(transition_epoch));
    if state.addon_manager().get_addon_status("svp") != AddonStatus::Running {
        state
            .addon_manager()
            .start_addon("svp")
            .await
            .map_err(AppError::ServiceUnavailable)?;
    }
    if !owns_legacy_svp_transition(transition_epoch) {
        return Err(AppError::ServiceUnavailable(
            "SVP start was superseded".to_string(),
        ));
    }

    let config = get_config_section(state.data_dir(), "svp");
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
            "target_bitrate",
        ] {
            if !body_obj.contains_key(key) {
                if let Some(value) = config_obj.get(key) {
                    body_obj.insert(key.to_string(), value.clone());
                }
            }
        }
    }

    let result = bridge_post(&state, "svp", "/svp/play", &body, Some("svp")).await?;
    if !owns_legacy_svp_transition(transition_epoch) {
        return Err(AppError::ServiceUnavailable(
            "SVP start was superseded".to_string(),
        ));
    }
    Ok(result)
}

/// POST /svp/stop — Stop SVP streams via sidecar.
async fn bridge_svp_stop(
    State(state): State<AppState>,
    body: Option<Json<Value>>,
) -> Result<Json<Value>, AppError> {
    let client_transition_id = body
        .as_ref()
        .and_then(|Json(value)| value.get("client_transition_id"))
        .map(|value| {
            value.as_u64().filter(|epoch| *epoch > 0).ok_or_else(|| {
                AppError::BadRequest("client_transition_id must be a positive integer".to_string())
            })
        })
        .transpose()?;
    let transition_epoch = claim_legacy_svp_client_transition(client_transition_id)
        .ok_or_else(|| AppError::ServiceUnavailable("SVP stop was superseded".to_string()))?;
    if state.addon_manager().get_addon_status("svp") != AddonStatus::Running {
        return Ok(Json(json!({
            "success": true,
            "message": "SVP addon is not running; nothing to stop"
        })));
    }

    bridge_post(
        &state,
        "svp",
        "/svp/stop",
        &json!({ "transition_id": transition_epoch }),
        None,
    )
    .await
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

fn prepare_svp_session_open(
    mut request: Value,
    graph: Option<crate::svp_manager_snapshot::ManagerGraphSnapshot>,
) -> Result<(String, Bytes), AppError> {
    let object = request
        .as_object_mut()
        .ok_or_else(|| AppError::BadRequest("Session request must be an object".to_string()))?;
    if object.contains_key("graph") {
        return Err(AppError::BadRequest(
            "SVP graph selection is owned by the desktop Manager bridge".to_string(),
        ));
    }
    let file_path = object
        .get("file_path")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::BadRequest("file_path is required".to_string()))?
        .to_owned();
    let graph = graph.ok_or_else(|| {
        AppError::ServiceUnavailable("No trusted SVP Manager graph is available".to_string())
    })?;
    object.insert(
        "graph".to_string(),
        serde_json::to_value(graph)
            .map_err(|error| AppError::Internal(format!("Failed to encode SVP graph: {error}")))?,
    );
    let body = serde_json::to_vec(&request)
        .map(Bytes::from)
        .map_err(|error| {
            AppError::Internal(format!("Failed to encode session request: {error}"))
        })?;
    Ok((file_path, body))
}

async fn bridge_svp_sessions_root(
    State(state): State<AppState>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    body: Bytes,
) -> Result<Response, AppError> {
    if method == Method::POST {
        let request: Value = serde_json::from_slice(&body)
            .map_err(|error| AppError::BadRequest(format!("Invalid session request: {}", error)))?;
        let graph = state.manager_graph_snapshots().current();
        let (file_path, body) = prepare_svp_session_open(request, graph)?;
        crate::server::utils::validate_path_in_watch_dir(&state, &file_path)?;
        return bridge_svp_session_request(&state, method, "/svp/sessions", uri.query(), body)
            .await;
    }
    bridge_svp_session_request(&state, method, "/svp/sessions", uri.query(), body).await
}

async fn bridge_svp_sessions_path(
    State(state): State<AppState>,
    AxumPath(path): AxumPath<String>,
    method: Method,
    OriginalUri(uri): OriginalUri,
    body: Bytes,
) -> Result<Response, AppError> {
    bridge_svp_session_request(
        &state,
        method,
        &format!("/svp/sessions/{}", path),
        uri.query(),
        body,
    )
    .await
}

async fn bridge_svp_session_request(
    state: &AppState,
    method: Method,
    sidecar_path: &str,
    query: Option<&str>,
    body: Bytes,
) -> Result<Response, AppError> {
    if state.addon_manager().get_addon_status("svp") != AddonStatus::Running {
        state
            .addon_manager()
            .start_addon("svp")
            .await
            .map_err(AppError::ServiceUnavailable)?;
    }
    let base_url = state
        .addon_manager()
        .addon_url("svp")
        .ok_or_else(|| AppError::ServiceUnavailable("SVP addon is not running".to_string()))?;
    let mut url = format!("{}{}", base_url.trim_end_matches('/'), sidecar_path);
    if let Some(query) = query {
        url.push('?');
        url.push_str(query);
    }
    let request_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|error| AppError::BadRequest(format!("Invalid request method: {}", error)))?;
    let mut request = state.http_client().request(request_method, &url);
    if !body.is_empty() {
        request = request
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_vec());
    }
    let response = request.send().await.map_err(|error| {
        AppError::ServiceUnavailable(format!("Failed to reach SVP addon: {}", error))
    })?;
    let status = StatusCode::from_u16(response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = response
        .bytes()
        .await
        .map_err(|error| AppError::Internal(format!("Failed to read SVP response: {}", error)))?;
    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .body(Body::from(bytes))
        .unwrap())
}

// ─── Whisper subtitle bridge ─────────────────────────────────────────────────

const WHISPER_ADDON_ID: &str = "whisper-subtitles";

/// Check that the whisper addon is currently running and return an error if not.
fn require_whisper_running(state: &AppState) -> Result<(), AppError> {
    let status = state.addon_manager().get_addon_status(WHISPER_ADDON_ID);
    if status != AddonStatus::Running {
        return Err(AppError::ServiceUnavailable(
            "Whisper addon is not running. Install and start it first.".into(),
        ));
    }
    Ok(())
}

/// POST /whisper/install — Install the whisper-subtitles addon (create venv, pip install faster-whisper).
async fn bridge_whisper_install(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let addon_info = state.addon_manager().get_addon(WHISPER_ADDON_ID);
    let already_installed = addon_info
        .as_ref()
        .map(|info| info.installed)
        .unwrap_or(false);

    if already_installed {
        return Ok(Json(json!({
            "status": "installed",
            "addon_id": WHISPER_ADDON_ID,
            "message": "Whisper addon is already installed"
        })));
    }

    let state_clone = state.clone();

    tokio::task::spawn_blocking(move || {
        state_clone
            .addon_manager()
            .install_addon(WHISPER_ADDON_ID)
            .map_err(|e| AppError::Internal(format!("Failed to install whisper addon: {}", e)))
    })
    .await??;

    Ok(Json(json!({
        "status": "installed",
        "addon_id": WHISPER_ADDON_ID,
        "message": "Whisper addon installed successfully"
    })))
}

#[derive(Debug, Deserialize)]
struct WhisperGenerateRequest {
    file_path: String,
    #[serde(default)]
    image_id: Option<i64>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    start_position: Option<f64>,
}

/// POST /whisper/generate — Start subtitle generation for a video file.
///
/// Generates a unique stream_id, proxies the request to the whisper sidecar,
/// and returns URLs for the VTT file and SSE event stream.
async fn bridge_whisper_generate(
    State(state): State<AppState>,
    Json(body): Json<WhisperGenerateRequest>,
) -> Result<Json<Value>, AppError> {
    require_whisper_running(&state)?;

    let file_path = Path::new(&body.file_path);
    if !file_path.exists() {
        return Err(AppError::NotFound("Video file not found".into()));
    }
    // Reject paths outside the media library before handing to the whisper sidecar.
    crate::server::utils::validate_path_in_watch_dir(&state, &body.file_path)?;

    // Generate a unique stream ID
    let stream_id = uuid::Uuid::new_v4().to_string();

    // Load whisper settings to pass along as generation config
    let data_dir = state.data_dir().to_path_buf();
    let whisper_config =
        tokio::task::spawn_blocking(move || get_config_section(&data_dir, "whisper")).await?;

    // Build the request body for the sidecar
    let mut sidecar_body = json!({
        "file_path": body.file_path,
        "stream_id": stream_id,
        "config": whisper_config,
    });
    if let Some(image_id) = body.image_id {
        sidecar_body["image_id"] = json!(image_id);
    }
    if let Some(ref language) = body.language {
        sidecar_body["language"] = json!(language);
    }
    if let Some(ref task) = body.task {
        sidecar_body["task"] = json!(task);
    }
    if let Some(start_position) = body.start_position {
        sidecar_body["start_position"] = json!(start_position);
    }

    // Proxy to the whisper addon
    let base_url = state
        .addon_manager()
        .addon_url(WHISPER_ADDON_ID)
        .ok_or_else(|| {
            AppError::ServiceUnavailable(
                "Whisper addon is not running. Install and start it first.".into(),
            )
        })?;

    let url = format!("{}/whisper/generate", base_url.trim_end_matches('/'));

    let response = state
        .http_client()
        .post(&url)
        .json(&sidecar_body)
        .send()
        .await
        .map_err(|e| {
            AppError::ServiceUnavailable(format!("Failed to reach whisper addon: {}", e))
        })?;

    let status = response.status();
    let result: Value = response
        .json()
        .await
        .map_err(|e| AppError::Internal(format!("Invalid response from whisper addon: {}", e)))?;

    if !status.is_success() {
        let detail = result
            .get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown whisper addon error");
        return Err(AppError::Internal(detail.to_string()));
    }

    // Build response with URLs routed through the Rust server
    let vtt_url = format!("/api/settings/whisper/vtt/{}/subtitles.vtt", stream_id);
    let events_url = format!("/api/settings/whisper/events/{}", stream_id);

    // Merge the sidecar response with our constructed URLs
    let mut response_json = result;
    response_json["stream_id"] = json!(stream_id);
    response_json["vtt_url"] = json!(vtt_url);
    response_json["events_url"] = json!(events_url);

    Ok(Json(response_json))
}

pub(crate) async fn generate_whisper_for_native(
    state: AppState,
    file_path: String,
    image_id: i64,
    start_position: f64,
) -> Result<Value, AppError> {
    bridge_whisper_generate(
        State(state),
        Json(WhisperGenerateRequest {
            file_path,
            image_id: Some(image_id),
            language: None,
            task: None,
            start_position: Some(start_position),
        }),
    )
    .await
    .map(|Json(value)| value)
}

pub(crate) async fn whisper_status_for_native(
    state: &AppState,
    stream_id: &str,
) -> Result<Value, AppError> {
    require_whisper_running(state)?;
    let base_url = state
        .addon_manager()
        .addon_url(WHISPER_ADDON_ID)
        .ok_or_else(|| AppError::ServiceUnavailable("Whisper addon is not running".into()))?;
    let response = state
        .http_client()
        .get(format!(
            "{}/whisper/status/{}",
            base_url.trim_end_matches('/'),
            stream_id
        ))
        .send()
        .await
        .map_err(|error| {
            AppError::ServiceUnavailable(format!("Failed to query Whisper status: {error}"))
        })?;
    let status = response.status();
    let value: Value = response
        .json()
        .await
        .map_err(|error| AppError::Internal(format!("Invalid Whisper status: {error}")))?;
    if status.is_success() {
        Ok(value)
    } else {
        Err(AppError::Internal(
            value
                .get("detail")
                .and_then(Value::as_str)
                .unwrap_or("Whisper status request failed")
                .to_string(),
        ))
    }
}

/// POST /whisper/stop — Stop active whisper subtitle generation.
async fn bridge_whisper_stop(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    require_whisper_running(&state)?;

    bridge_post(&state, WHISPER_ADDON_ID, "/whisper/stop", &json!({}), None).await
}

/// GET /whisper/vtt/{stream_id}/subtitles.vtt — Proxy the growing VTT file from the sidecar.
///
/// Sets Content-Type to text/vtt so browsers can consume it directly.
async fn bridge_whisper_vtt(
    State(state): State<AppState>,
    AxumPath(stream_id): AxumPath<String>,
) -> Result<Response, AppError> {
    require_whisper_running(&state)?;

    let base_url = state
        .addon_manager()
        .addon_url(WHISPER_ADDON_ID)
        .ok_or_else(|| AppError::ServiceUnavailable("Whisper addon is not running".into()))?;

    let url = format!(
        "{}/whisper/vtt/{}/subtitles.vtt",
        base_url.trim_end_matches('/'),
        stream_id
    );

    let response = state.http_client().get(&url).send().await.map_err(|e| {
        AppError::ServiceUnavailable(format!("Failed to reach whisper addon: {}", e))
    })?;

    let status = StatusCode::from_u16(response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let body_bytes = response.bytes().await.map_err(|e| {
        AppError::Internal(format!(
            "Failed to read VTT response from whisper addon: {}",
            e
        ))
    })?;

    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/vtt; charset=utf-8")
        .header(header::CONTENT_LENGTH, body_bytes.len())
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(Body::from(body_bytes))
        .unwrap())
}

/// GET /whisper/events/{stream_id} — SSE proxy for real-time subtitle cue events.
///
/// Connects to the whisper sidecar's SSE endpoint and forwards each event
/// to the client. Each event contains a new subtitle cue as it's generated.
async fn bridge_whisper_events(
    State(state): State<AppState>,
    AxumPath(stream_id): AxumPath<String>,
) -> Result<Response, AppError> {
    require_whisper_running(&state)?;

    let base_url = state
        .addon_manager()
        .addon_url(WHISPER_ADDON_ID)
        .ok_or_else(|| AppError::ServiceUnavailable("Whisper addon is not running".into()))?;

    let url = format!(
        "{}/whisper/events/{}",
        base_url.trim_end_matches('/'),
        stream_id
    );

    let response = state
        .http_client()
        .get(&url)
        .header("Accept", "text/event-stream")
        .send()
        .await
        .map_err(|e| {
            AppError::ServiceUnavailable(format!(
                "Failed to reach whisper addon SSE endpoint: {}",
                e
            ))
        })?;

    if !response.status().is_success() {
        let status_code = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        return Err(AppError::Internal(format!(
            "Whisper addon SSE endpoint returned {}: {}",
            status_code, body
        )));
    }

    // Stream the SSE response body through to the client
    let byte_stream = response.bytes_stream();

    let body = Body::from_stream(byte_stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
        .header(header::CONNECTION, "keep-alive")
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .body(body)
        .unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    // AC: @credential-storage ac-5
    #[tokio::test]
    async fn settings_api_rejects_and_redacts_signing_credential_material() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-settings-credential-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(
            settings_path(&data_dir),
            r#"{"jwt_secret":"legacy-secret","network":{"enabled":true}}"#,
        )
        .unwrap();
        let state = AppState::new(&data_dir, 0).unwrap();
        std::fs::write(
            settings_path(&data_dir),
            r#"{"jwt_secret":"must-be-redacted","network":{"enabled":true}}"#,
        )
        .unwrap();

        let get_response = router()
            .with_state(state.clone())
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let get_body = axum::body::to_bytes(get_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let get_json: Value = serde_json::from_slice(&get_body).unwrap();
        assert!(get_json.get("jwt_secret").is_none());

        save_settings_to_file(
            &data_dir,
            &json!({"jwt_secret": "must-not-persist", "network": {"enabled": true}}),
        )
        .unwrap();
        assert!(load_settings(&data_dir).get("jwt_secret").is_none());

        let post_response = router()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"jwt_secret":"replacement","network":{"enabled":false}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(post_response.status(), StatusCode::OK);
        let saved = load_settings(&data_dir);
        assert!(saved.get("jwt_secret").is_none());
        assert_eq!(saved["network"]["enabled"], false);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    // AC: @reliable-stream-transitions ac-final-source-owner
    // AC: @reliable-stream-transitions ac-stop-superseded-producer
    fn legacy_svp_transition_epoch_assigns_one_latest_owner() {
        let first = claim_legacy_svp_client_transition(None).unwrap();
        let second = claim_legacy_svp_client_transition(None).unwrap();
        assert!(!owns_legacy_svp_transition(first));
        assert!(owns_legacy_svp_transition(second));

        claim_legacy_svp_client_transition(None).unwrap();
        assert!(!owns_legacy_svp_transition(second));
    }

    #[test]
    // AC: @reliable-stream-transitions ac-stop-superseded-producer
    fn legacy_svp_rejects_a_delayed_client_transition() {
        let base = LEGACY_SVP_CLIENT_TRANSITION_EPOCH.load(Ordering::SeqCst);
        let newest = base + 2;
        assert!(claim_legacy_svp_client_transition(Some(newest)).is_some());
        assert!(claim_legacy_svp_client_transition(Some(newest - 1)).is_none());
        assert!(claim_legacy_svp_client_transition(Some(newest)).is_some());
    }

    #[tokio::test]
    async fn retired_optical_flow_stop_remains_compatible_with_existing_clients() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-optical-flow-stop-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        let state = AppState::new(&data_dir, 0).unwrap();

        let response = router()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/optical-flow/stop")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[test]
    // AC: @svp-platform-routing ac-opaque-session-request
    fn session_open_injects_only_the_backend_owned_manager_graph() {
        let graph = crate::svp_manager_snapshot::ManagerGraphSnapshot {
            kind: "manager_snapshot",
            revision: 7,
            snapshot_path: "/private/graph-7.vpy".to_string(),
            snapshot_sha256: "a".repeat(64),
        };
        let (file_path, body) = prepare_svp_session_open(
            json!({"file_path": "/media/video.mp4", "generation": 1}),
            Some(graph),
        )
        .unwrap();
        let request: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(file_path, "/media/video.mp4");
        assert_eq!(request["graph"]["kind"], "manager_snapshot");
        assert_eq!(request["graph"]["revision"], 7);
        assert_eq!(request["graph"]["snapshot_path"], "/private/graph-7.vpy");
        assert_eq!(request["graph"]["snapshot_sha256"], "a".repeat(64));
    }

    #[test]
    // AC: @svp-platform-routing ac-opaque-session-request
    fn session_open_rejects_client_graphs_and_missing_manager_graphs() {
        let client_graph = prepare_svp_session_open(
            json!({"file_path": "/media/video.mp4", "graph": {"kind": "manager_snapshot"}}),
            None,
        );
        assert!(matches!(client_graph, Err(AppError::BadRequest(_))));

        let unavailable = prepare_svp_session_open(json!({"file_path": "/media/video.mp4"}), None);
        assert!(matches!(unavailable, Err(AppError::ServiceUnavailable(_))));
    }

    #[test]
    fn desktop_player_settings_validate_modes_and_legacy_updates() {
        let defaults = get_defaults();
        let playback = &defaults["video_playback"];
        assert_eq!(playback["desktop_player_mode"], json!("react"));
        assert_eq!(playback["native_video_force_copy"], json!(false));
        assert_eq!(playback["native_video_diagnostics"], json!(false));

        let mut valid = json!({
            "desktop_player_mode": "native_svp",
            "native_video_force_copy": true,
            "native_video_diagnostics": true,
            "auto_advance_delay": 99,
        });
        validate_video_playback_config(&mut valid).unwrap();
        assert_eq!(valid["auto_advance_delay"], json!(30));

        let mut legacy = json!({ "native_video_enabled": true });
        validate_video_playback_config(&mut legacy).unwrap();
        assert_eq!(legacy["desktop_player_mode"], json!("native"));
        assert!(legacy.get("native_video_enabled").is_none());

        let mut invalid_mode = json!({ "desktop_player_mode": "automatic" });
        assert!(validate_video_playback_config(&mut invalid_mode).is_err());
        let mut invalid = json!({ "native_video_enabled": "yes" });
        assert!(validate_video_playback_config(&mut invalid).is_err());
    }

    #[test]
    fn desktop_player_migration_maps_legacy_native_and_svp_booleans() {
        for (native, svp, expected) in [
            (false, false, "react"),
            (false, true, "react"),
            (true, false, "native"),
            (true, true, "native_svp"),
        ] {
            let mut settings = json!({
                "video_playback": { "native_video_enabled": native },
                "svp": { "enabled": svp },
            });
            assert!(migrate_desktop_player_mode(&mut settings));
            assert_eq!(settings["video_playback"]["desktop_player_mode"], expected);
            assert!(settings["video_playback"]
                .get("native_video_enabled")
                .is_none());
            assert!(!migrate_desktop_player_mode(&mut settings));
        }
    }

    #[test]
    fn desktop_player_migration_is_persisted_for_normal_restart() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-desktop-player-migration-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(
            settings_path(&data_dir),
            r#"{"video_playback":{"native_video_enabled":true},"svp":{"enabled":true}}"#,
        )
        .unwrap();

        assert_eq!(get_desktop_player_mode(&data_dir), "native_svp");
        let persisted = load_settings(&data_dir);
        assert_eq!(
            persisted["video_playback"]["desktop_player_mode"],
            "native_svp"
        );
        assert!(persisted["video_playback"]
            .get("native_video_enabled")
            .is_none());
        std::fs::remove_dir_all(data_dir).unwrap();
    }

    // AC: @addon-settings ac-3
    #[test]
    fn auto_tagger_validation_rejects_unknown_model_and_normalizes_thresholds() {
        let mut config = json!({
            "model": "unknown",
            "device": "cuda",
            "general_threshold": 2.0,
            "character_threshold": -1.0,
        });
        assert!(validate_auto_tagger_config(&mut config).is_err());

        config["model"] = json!("vit-v3");
        validate_auto_tagger_config(&mut config).unwrap();
        assert_eq!(config["general_threshold"], json!(1.0));
        assert_eq!(config["character_threshold"], json!(0.0));
    }
}
