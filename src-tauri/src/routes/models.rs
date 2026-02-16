//! ML Model management routes — download, track, and query status of ML models
//! used by LocalBooru addons (auto-tagger, age-detector, etc.).

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::state::AppState;

// ─── Model registry types ────────────────────────────────────────────────────

/// Known ML models and their remote URLs / expected files.
pub struct ModelDefinition {
    pub name: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    /// URL to download the model from.
    pub download_url: &'static str,
    /// Expected filename inside `data_dir/models/{name}/`.
    pub filename: &'static str,
}

/// Static registry of known models.
pub const KNOWN_MODELS: &[ModelDefinition] = &[
    ModelDefinition {
        name: "auto-tagger",
        display_name: "WD14 Auto-Tagger",
        description: "WD14 ViT tagger for automatic image tagging (anime/illustration focused)",
        download_url: "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx",
        filename: "model.onnx",
    },
    ModelDefinition {
        name: "age-detector",
        display_name: "Age Detection Model",
        description: "Deep learning model for detecting apparent age in images",
        download_url: "https://huggingface.co/nateraw/age-detection/resolve/main/model.onnx",
        filename: "model.onnx",
    },
];

/// Download status for a model.
#[derive(Debug, Clone)]
pub struct ModelDownloadState {
    pub status: ModelStatus,
    pub progress: f64,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    /// Not downloaded and not currently downloading.
    NotDownloaded,
    /// Currently being downloaded.
    Downloading,
    /// Download complete, model files present.
    Downloaded,
    /// Download failed.
    Failed,
}

impl ModelStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ModelStatus::NotDownloaded => "not_downloaded",
            ModelStatus::Downloading => "downloading",
            ModelStatus::Downloaded => "downloaded",
            ModelStatus::Failed => "failed",
        }
    }
}

/// Shared registry of model download states, keyed by model name.
pub type ModelRegistry = Arc<DashMap<String, ModelDownloadState>>;

/// Create a new empty model registry.
pub fn create_model_registry() -> ModelRegistry {
    Arc::new(DashMap::new())
}

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_models))
        .route("/download", post(start_model_download))
        .route("/{model_name}/progress", get(model_download_progress))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return the directory where a model's files should live.
fn model_dir(data_dir: &std::path::Path, model_name: &str) -> PathBuf {
    data_dir.join("models").join(model_name)
}

/// Check whether a model's expected file exists on disk.
fn model_file_exists(data_dir: &std::path::Path, def: &ModelDefinition) -> bool {
    model_dir(data_dir, def.name).join(def.filename).exists()
}

/// Look up a model definition by name.
fn find_model_def(name: &str) -> Option<&'static ModelDefinition> {
    KNOWN_MODELS.iter().find(|m| m.name == name)
}

// ─── Route handlers ──────────────────────────────────────────────────────────

/// GET /settings/models — Return a list of known ML models with their download
/// status (checks both disk and active download state).
async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let registry = state.model_registry();

    let models: Vec<Value> = KNOWN_MODELS
        .iter()
        .map(|def| {
            let on_disk = model_file_exists(&data_dir, def);
            let dir = model_dir(&data_dir, def.name);

            // Check the in-memory registry for active download state
            let (status, progress, error) = if let Some(entry) = registry.get(def.name) {
                (
                    entry.status.as_str().to_string(),
                    entry.progress,
                    entry.error.clone(),
                )
            } else if on_disk {
                ("downloaded".to_string(), 100.0, None)
            } else {
                ("not_downloaded".to_string(), 0.0, None)
            };

            json!({
                "name": def.name,
                "display_name": def.display_name,
                "description": def.description,
                "status": status,
                "progress": progress,
                "on_disk": on_disk,
                "model_dir": dir.to_string_lossy(),
                "filename": def.filename,
                "error": error,
            })
        })
        .collect();

    Ok(Json(json!({ "models": models })))
}

// ─── Download endpoint ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ModelDownloadRequest {
    model_name: String,
}

/// POST /settings/models/download — Start downloading a model in the background.
/// Tracks progress in the shared model registry.
async fn start_model_download(
    State(state): State<AppState>,
    Json(body): Json<ModelDownloadRequest>,
) -> Result<Json<Value>, AppError> {
    let def = find_model_def(&body.model_name).ok_or_else(|| {
        AppError::BadRequest(format!(
            "Unknown model '{}'. Known models: {}",
            body.model_name,
            KNOWN_MODELS
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;

    let data_dir = state.data_dir().to_path_buf();
    let registry = state.model_registry();

    // Check if already downloaded on disk
    if model_file_exists(&data_dir, def) {
        // Update registry to reflect downloaded state
        registry.insert(
            def.name.to_string(),
            ModelDownloadState {
                status: ModelStatus::Downloaded,
                progress: 100.0,
                bytes_downloaded: 0,
                total_bytes: None,
                error: None,
            },
        );
        return Ok(Json(json!({
            "model_name": def.name,
            "status": "downloaded",
            "message": "Model already downloaded"
        })));
    }

    // Check if already downloading
    if let Some(entry) = registry.get(def.name) {
        if entry.status == ModelStatus::Downloading {
            return Ok(Json(json!({
                "model_name": def.name,
                "status": "downloading",
                "progress": entry.progress,
                "message": "Download already in progress"
            })));
        }
    }

    // Register as downloading
    registry.insert(
        def.name.to_string(),
        ModelDownloadState {
            status: ModelStatus::Downloading,
            progress: 0.0,
            bytes_downloaded: 0,
            total_bytes: None,
            error: None,
        },
    );

    // Spawn background download
    let model_name = def.name.to_string();
    let download_url = def.download_url.to_string();
    let filename = def.filename.to_string();
    let reg = registry.clone();

    tokio::spawn(async move {
        run_model_download(model_name, download_url, filename, data_dir, reg).await;
    });

    Ok(Json(json!({
        "model_name": def.name,
        "status": "downloading",
        "message": "Download started"
    })))
}

/// Background task: download a model file with progress tracking.
/// Uses `response.chunk()` for incremental streaming with progress updates.
async fn run_model_download(
    model_name: String,
    download_url: String,
    filename: String,
    data_dir: PathBuf,
    registry: ModelRegistry,
) {
    use tokio::io::AsyncWriteExt;

    let dir = model_dir(&data_dir, &model_name);

    // Create the model directory
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        if let Some(mut entry) = registry.get_mut(&model_name) {
            entry.status = ModelStatus::Failed;
            entry.error = Some(format!("Failed to create model directory: {}", e));
        }
        return;
    }

    let file_path = dir.join(&filename);
    let temp_path = dir.join(format!("{}.part", &filename));

    // Build HTTP client with generous timeout for large models
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            if let Some(mut entry) = registry.get_mut(&model_name) {
                entry.status = ModelStatus::Failed;
                entry.error = Some(format!("Failed to create HTTP client: {}", e));
            }
            return;
        }
    };

    let mut response = match client.get(&download_url).send().await {
        Ok(r) => r,
        Err(e) => {
            if let Some(mut entry) = registry.get_mut(&model_name) {
                entry.status = ModelStatus::Failed;
                entry.error = Some(format!("Failed to start download: {}", e));
            }
            return;
        }
    };

    if !response.status().is_success() {
        if let Some(mut entry) = registry.get_mut(&model_name) {
            entry.status = ModelStatus::Failed;
            entry.error = Some(format!(
                "HTTP {} from download URL",
                response.status().as_u16()
            ));
        }
        return;
    }

    let total_bytes = response.content_length();

    // Update total size in registry
    if let Some(mut entry) = registry.get_mut(&model_name) {
        entry.total_bytes = total_bytes;
    }

    // Open temp file for writing
    let mut file = match tokio::fs::File::create(&temp_path).await {
        Ok(f) => f,
        Err(e) => {
            if let Some(mut entry) = registry.get_mut(&model_name) {
                entry.status = ModelStatus::Failed;
                entry.error = Some(format!("Failed to create temp file: {}", e));
            }
            return;
        }
    };

    // Stream chunks with progress tracking
    let mut bytes_downloaded: u64 = 0;

    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if let Err(e) = file.write_all(&chunk).await {
                    if let Some(mut entry) = registry.get_mut(&model_name) {
                        entry.status = ModelStatus::Failed;
                        entry.error = Some(format!("Failed to write data: {}", e));
                    }
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return;
                }

                bytes_downloaded += chunk.len() as u64;
                let progress = if let Some(total) = total_bytes {
                    if total > 0 {
                        (bytes_downloaded as f64 / total as f64 * 100.0).min(100.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                if let Some(mut entry) = registry.get_mut(&model_name) {
                    entry.bytes_downloaded = bytes_downloaded;
                    entry.progress = progress;
                }
            }
            Ok(None) => {
                // All chunks received — download complete
                break;
            }
            Err(e) => {
                if let Some(mut entry) = registry.get_mut(&model_name) {
                    entry.status = ModelStatus::Failed;
                    entry.error = Some(format!("Download error: {}", e));
                }
                let _ = tokio::fs::remove_file(&temp_path).await;
                return;
            }
        }
    }

    // Flush and close the file
    if let Err(e) = file.flush().await {
        if let Some(mut entry) = registry.get_mut(&model_name) {
            entry.status = ModelStatus::Failed;
            entry.error = Some(format!("Failed to flush file: {}", e));
        }
        let _ = tokio::fs::remove_file(&temp_path).await;
        return;
    }
    drop(file);

    // Rename temp file to final path
    if let Err(e) = tokio::fs::rename(&temp_path, &file_path).await {
        if let Some(mut entry) = registry.get_mut(&model_name) {
            entry.status = ModelStatus::Failed;
            entry.error = Some(format!("Failed to rename temp file: {}", e));
        }
        let _ = tokio::fs::remove_file(&temp_path).await;
        return;
    }

    // Mark as downloaded
    if let Some(mut entry) = registry.get_mut(&model_name) {
        entry.status = ModelStatus::Downloaded;
        entry.progress = 100.0;
        entry.bytes_downloaded = bytes_downloaded;
    }

    log::info!(
        "[Models] Successfully downloaded model '{}' ({} bytes)",
        model_name,
        bytes_downloaded
    );
}

// ─── Progress endpoint ───────────────────────────────────────────────────────

/// GET /settings/models/{model_name}/progress — Return download progress for a
/// specific model.
async fn model_download_progress(
    State(state): State<AppState>,
    AxumPath(model_name): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    // Verify the model name is known
    let def = find_model_def(&model_name).ok_or_else(|| {
        AppError::BadRequest(format!(
            "Unknown model '{}'. Known models: {}",
            model_name,
            KNOWN_MODELS
                .iter()
                .map(|m| m.name)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })?;

    let data_dir = state.data_dir().to_path_buf();
    let registry = state.model_registry();

    // Check in-memory state first
    if let Some(entry) = registry.get(&model_name) {
        return Ok(Json(json!({
            "model_name": model_name,
            "status": entry.status.as_str(),
            "progress": entry.progress,
            "bytes_downloaded": entry.bytes_downloaded,
            "total_bytes": entry.total_bytes,
            "error": entry.error,
        })));
    }

    // Fall back to checking disk
    let on_disk = model_file_exists(&data_dir, def);
    let status = if on_disk {
        "downloaded"
    } else {
        "not_downloaded"
    };

    Ok(Json(json!({
        "model_name": model_name,
        "status": status,
        "progress": if on_disk { 100.0 } else { 0.0 },
        "bytes_downloaded": 0,
        "total_bytes": null,
        "error": null,
    })))
}
