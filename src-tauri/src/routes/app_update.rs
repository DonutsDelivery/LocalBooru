use std::path::PathBuf;

use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Json, Response};
use axum::routing::get;
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::fs::File;
use tokio_util::io::ReaderStream;

use crate::server::error::AppError;
use crate::server::state::AppState;

/// Current app version, read from Cargo.toml at compile time.
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/check", get(check_update))
        .route("/download", get(download_update))
}

// ─── Request models ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CheckUpdateQuery {
    #[serde(default = "default_platform")]
    platform: String,
    #[serde(default = "default_version")]
    current_version: String,
}

fn default_platform() -> String {
    "android".into()
}

fn default_version() -> String {
    "0.0.0".into()
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/app-update/check — Check if an update is available.
///
/// Query params:
///   - `platform` (default "android")
///   - `current_version` (default "0.0.0")
///
/// Returns `{ "version": "x.y.z", "apk_available": bool }`
async fn check_update(
    State(state): State<AppState>,
    Query(q): Query<CheckUpdateQuery>,
) -> Result<Json<Value>, AppError> {
    let update_available = q.current_version != APP_VERSION;
    let apk_path = find_apk_path(&state);
    let apk_available = apk_path
        .map(|p| p.exists())
        .unwrap_or(false);

    Ok(Json(json!({
        "version": APP_VERSION,
        "update_available": update_available,
        "apk_available": apk_available && q.platform == "android",
    })))
}

/// GET /api/app-update/download — Download the APK file.
async fn download_update(
    State(state): State<AppState>,
) -> Result<Response, AppError> {
    let apk_path = find_apk_path(&state)
        .ok_or_else(|| AppError::NotFound("No APK file found".into()))?;

    if !apk_path.exists() {
        return Err(AppError::NotFound("APK file not found".into()));
    }

    let file = File::open(&apk_path).await?;
    let metadata = file.metadata().await?;
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/vnd.android.package-archive")
        .header(header::CONTENT_LENGTH, metadata.len())
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"LocalBooru.apk\"",
        )
        .body(body)
        .unwrap())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Look for `LocalBooru.apk` in known locations:
/// 1. `{data_dir}/../updates/LocalBooru.apk`
/// 2. Next to the current executable
fn find_apk_path(state: &AppState) -> Option<PathBuf> {
    // Try data_dir parent's updates/ folder (e.g. ~/.localbooru/../updates/)
    let updates_dir = state.data_dir().parent()?.join("updates");
    let candidate = updates_dir.join("LocalBooru.apk");
    if candidate.exists() {
        return Some(candidate);
    }

    // Try next to the current executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidate = exe_dir.join("LocalBooru.apk");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // Return the preferred path even if it doesn't exist yet,
    // so check_update can report apk_available = false
    Some(updates_dir.join("LocalBooru.apk"))
}
