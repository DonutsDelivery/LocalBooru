//! SVP Web Video routes — yt-dlp integration for downloading and streaming
//! web videos through the SVP frame-interpolation pipeline.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::state::AppState;

// ─── Download tracking ───────────────────────────────────────────────────────

/// Status of a web video download managed by yt-dlp.
#[derive(Debug, Clone)]
pub struct WebDownload {
    pub status: WebDownloadStatus,
    pub progress: f64,
    pub filename: Option<String>,
    pub error: Option<String>,
    pub url: String,
    pub quality: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WebDownloadStatus {
    Downloading,
    Ready,
    Failed,
}

impl WebDownloadStatus {
    fn as_str(&self) -> &'static str {
        match self {
            WebDownloadStatus::Downloading => "downloading",
            WebDownloadStatus::Ready => "ready",
            WebDownloadStatus::Failed => "failed",
        }
    }
}

/// Shared registry of active web video downloads, keyed by download_id (UUID).
pub type WebDownloadRegistry = Arc<DashMap<String, WebDownload>>;

/// Create a new empty download registry.
pub fn create_download_registry() -> WebDownloadRegistry {
    Arc::new(DashMap::new())
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/play", post(svp_web_play))
        .route("/status/{download_id}", get(svp_web_status))
        .route("/drm-check", get(svp_web_drm_check))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Check whether yt-dlp is available in PATH.
fn ytdlp_available() -> bool {
    std::process::Command::new("yt-dlp")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Build the download directory inside data_dir.
fn downloads_dir(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("web_downloads")
}

/// Map yt-dlp quality string to a format selector.
fn quality_to_format(quality: &str) -> String {
    match quality {
        "360p" => "bestvideo[height<=360]+bestaudio/best[height<=360]/best".into(),
        "480p" => "bestvideo[height<=480]+bestaudio/best[height<=480]/best".into(),
        "720p" => "bestvideo[height<=720]+bestaudio/best[height<=720]/best".into(),
        "1080p" => "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best".into(),
        "1440p" => "bestvideo[height<=1440]+bestaudio/best[height<=1440]/best".into(),
        "4k" | "2160p" => "bestvideo[height<=2160]+bestaudio/best[height<=2160]/best".into(),
        "best" | _ => "bestvideo+bestaudio/best".into(),
    }
}

// ─── Route handlers ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct WebPlayRequest {
    url: String,
    #[serde(default = "default_quality")]
    quality: String,
}

fn default_quality() -> String {
    "720p".into()
}

/// POST /svp/web/play — Start downloading a web video via yt-dlp, then stream
/// it through SVP. Returns a download_id for tracking progress.
async fn svp_web_play(
    State(state): State<AppState>,
    Json(body): Json<WebPlayRequest>,
) -> Result<Json<Value>, AppError> {
    // Check yt-dlp availability
    if !ytdlp_available() {
        return Err(AppError::ServiceUnavailable(
            "yt-dlp is not installed or not found in PATH. Install it with: pip install yt-dlp"
                .into(),
        ));
    }

    if body.url.trim().is_empty() {
        return Err(AppError::BadRequest("URL cannot be empty".into()));
    }

    let download_id = uuid::Uuid::new_v4().to_string();
    let data_dir = state.data_dir().to_path_buf();
    let dl_dir = downloads_dir(&data_dir);

    // Ensure downloads directory exists
    tokio::fs::create_dir_all(&dl_dir).await?;

    // Register the download
    let registry = state.web_download_registry();
    registry.insert(
        download_id.clone(),
        WebDownload {
            status: WebDownloadStatus::Downloading,
            progress: 0.0,
            filename: None,
            error: None,
            url: body.url.clone(),
            quality: body.quality.clone(),
        },
    );

    // Spawn the download in the background
    let id = download_id.clone();
    let url = body.url.clone();
    let quality = body.quality.clone();
    let reg = registry.clone();
    let state_clone = state.clone();

    tokio::spawn(async move {
        run_ytdlp_download(id, url, quality, dl_dir, reg, state_clone).await;
    });

    Ok(Json(json!({
        "download_id": download_id,
        "status": "downloading"
    })))
}

/// Background task: run yt-dlp, parse progress, update registry, and trigger
/// SVP playback when the download completes.
async fn run_ytdlp_download(
    download_id: String,
    url: String,
    quality: String,
    dl_dir: PathBuf,
    registry: WebDownloadRegistry,
    state: AppState,
) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::process::Command;

    let format = quality_to_format(&quality);
    let output_template = dl_dir
        .join("%(title)s-%(id)s.%(ext)s")
        .to_string_lossy()
        .to_string();

    // Spawn yt-dlp with progress output
    let child = Command::new("yt-dlp")
        .args([
            "--no-playlist",
            "--newline",
            "--progress",
            "-f",
            &format,
            "--merge-output-format",
            "mp4",
            "-o",
            &output_template,
            &url,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            if let Some(mut entry) = registry.get_mut(&download_id) {
                entry.status = WebDownloadStatus::Failed;
                entry.error = Some(format!("Failed to spawn yt-dlp: {}", e));
            }
            return;
        }
    };

    // Read stdout for progress parsing
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        let reg = registry.clone();
        let id = download_id.clone();

        while let Ok(Some(line)) = lines.next_line().await {
            // yt-dlp progress lines look like:
            // [download]  45.2% of ~100.00MiB at 5.00MiB/s ETA 00:10
            if line.contains("[download]") && line.contains('%') {
                if let Some(pct) = parse_ytdlp_progress(&line) {
                    if let Some(mut entry) = reg.get_mut(&id) {
                        entry.progress = pct;
                    }
                }
            }

            // yt-dlp prints the final merged filename like:
            // [Merger] Merging formats into "file.mp4"
            // or [download] Destination: file.mp4
            if line.contains("[Merger] Merging formats into")
                || (line.contains("[download] Destination:") && !line.contains("has already"))
            {
                if let Some(fname) = extract_filename_from_line(&line) {
                    if let Some(mut entry) = reg.get_mut(&id) {
                        entry.filename = Some(fname);
                    }
                }
            }

            // Already downloaded line
            if line.contains("has already been downloaded") {
                if let Some(fname) = extract_filename_from_line(&line) {
                    if let Some(mut entry) = reg.get_mut(&id) {
                        entry.filename = Some(fname);
                        entry.progress = 100.0;
                    }
                }
            }
        }
    }

    // Wait for process to finish
    let exit_status = child.wait().await;

    match exit_status {
        Ok(status) if status.success() => {
            // Try to find the downloaded file if we don't have it yet
            if let Some(mut entry) = registry.get_mut(&download_id) {
                // If we still don't have a filename, scan the downloads dir
                if entry.filename.is_none() {
                    if let Some(fname) = find_latest_file(&dl_dir).await {
                        entry.filename = Some(fname);
                    }
                }

                entry.progress = 100.0;
                entry.status = WebDownloadStatus::Ready;

                // Try to start SVP playback on the downloaded file
                if let Some(ref filename) = entry.filename {
                    let file_path = filename.clone();
                    let addon_mgr = state.addon_manager();
                    if addon_mgr.addon_url("svp").is_some() {
                        log::info!(
                            "[SVP Web] Download complete, starting SVP stream for: {}",
                            file_path
                        );
                        // Fire-and-forget SVP play request via the sidecar bridge
                        let client = state.http_client();
                        if let Some(base_url) = addon_mgr.addon_url("svp") {
                            let play_url =
                                format!("{}/svp/play", base_url.trim_end_matches('/'));
                            let _ = client
                                .post(&play_url)
                                .json(&json!({ "file_path": file_path }))
                                .send()
                                .await;
                        }
                    }
                }
            }
        }
        Ok(status) => {
            if let Some(mut entry) = registry.get_mut(&download_id) {
                entry.status = WebDownloadStatus::Failed;
                entry.error = Some(format!(
                    "yt-dlp exited with status: {}",
                    status.code().unwrap_or(-1)
                ));
            }
        }
        Err(e) => {
            if let Some(mut entry) = registry.get_mut(&download_id) {
                entry.status = WebDownloadStatus::Failed;
                entry.error = Some(format!("Failed to wait for yt-dlp: {}", e));
            }
        }
    }
}

/// Parse a percentage from a yt-dlp progress line.
fn parse_ytdlp_progress(line: &str) -> Option<f64> {
    // Find the percentage pattern like "45.2%"
    let trimmed = line.trim();
    for part in trimmed.split_whitespace() {
        if part.ends_with('%') {
            let num_str = part.trim_end_matches('%');
            if let Ok(pct) = num_str.parse::<f64>() {
                return Some(pct.clamp(0.0, 100.0));
            }
        }
    }
    None
}

/// Extract a filename from a yt-dlp output line.
fn extract_filename_from_line(line: &str) -> Option<String> {
    // [Merger] Merging formats into "file.mp4"
    if let Some(start) = line.find('"') {
        if let Some(end) = line.rfind('"') {
            if end > start {
                return Some(line[start + 1..end].to_string());
            }
        }
    }
    // [download] Destination: file.mp4
    if let Some(idx) = line.find("Destination:") {
        let rest = line[idx + "Destination:".len()..].trim();
        if !rest.is_empty() {
            return Some(rest.to_string());
        }
    }
    None
}

/// Find the most recently modified file in a directory.
async fn find_latest_file(dir: &PathBuf) -> Option<String> {
    let mut entries = tokio::fs::read_dir(dir).await.ok()?;
    let mut latest: Option<(String, std::time::SystemTime)> = None;

    while let Ok(Some(entry)) = entries.next_entry().await {
        if let Ok(meta) = entry.metadata().await {
            if meta.is_file() {
                if let Ok(modified) = meta.modified() {
                    let path = entry.path().to_string_lossy().to_string();
                    if latest.as_ref().map_or(true, |(_, t)| modified > *t) {
                        latest = Some((path, modified));
                    }
                }
            }
        }
    }

    latest.map(|(path, _)| path)
}

// ─── Status endpoint ─────────────────────────────────────────────────────────

/// GET /svp/web/status/{download_id} — Return download progress.
async fn svp_web_status(
    State(state): State<AppState>,
    AxumPath(download_id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let registry = state.web_download_registry();

    let entry = registry
        .get(&download_id)
        .ok_or_else(|| AppError::NotFound(format!("Download '{}' not found", download_id)))?;

    Ok(Json(json!({
        "download_id": download_id,
        "status": entry.status.as_str(),
        "progress": entry.progress,
        "filename": entry.filename,
        "error": entry.error,
    })))
}

// ─── DRM check endpoint ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DrmCheckQuery {
    url: String,
}

/// GET /svp/web/drm-check?url=... — Check if a URL is DRM-protected or live.
/// Runs `yt-dlp --dump-json` to inspect the URL metadata.
async fn svp_web_drm_check(
    Query(q): Query<DrmCheckQuery>,
) -> Result<Json<Value>, AppError> {
    if !ytdlp_available() {
        return Err(AppError::ServiceUnavailable(
            "yt-dlp is not installed or not found in PATH".into(),
        ));
    }

    if q.url.trim().is_empty() {
        return Err(AppError::BadRequest("URL cannot be empty".into()));
    }

    let url = q.url.clone();

    let result = tokio::process::Command::new("yt-dlp")
        .args(["--dump-json", "--no-playlist", &url])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| AppError::Internal(format!("Failed to run yt-dlp: {}", e)))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        // Check if the error message indicates DRM
        let is_drm = stderr.contains("DRM")
            || stderr.contains("drm")
            || stderr.contains("This video is DRM protected");
        if is_drm {
            return Ok(Json(json!({
                "drm": true,
                "live": false,
                "title": null,
                "error": stderr.trim()
            })));
        }
        return Err(AppError::Internal(format!(
            "yt-dlp failed: {}",
            stderr.trim()
        )));
    }

    let info: Value = serde_json::from_slice(&result.stdout).map_err(|e| {
        AppError::Internal(format!("Failed to parse yt-dlp JSON output: {}", e))
    })?;

    // Check for DRM indicators
    let has_drm = info.get("drm_style").is_some()
        || info
            .get("drm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        || info
            .get("_has_drm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

    // Check for live stream indicators
    let is_live = info
        .get("is_live")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || info
            .get("live_status")
            .and_then(|v| v.as_str())
            .map(|s| s == "is_live" || s == "is_upcoming")
            .unwrap_or(false);

    let title = info
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(Json(json!({
        "drm": has_drm,
        "live": is_live,
        "title": title,
    })))
}
