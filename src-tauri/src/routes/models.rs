//! ML model management routes for add-on model downloads.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use dashmap::DashMap;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;

use crate::server::error::AppError;
use crate::server::state::AppState;

#[derive(Clone, Copy)]
pub struct ModelFileDefinition {
    pub download_url: &'static str,
    pub filename: &'static str,
}

pub struct ModelDefinition {
    pub name: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    /// Relative to `{data_dir}/models`; allows tagger models to share a namespace.
    pub storage_dir: &'static str,
    pub files: &'static [ModelFileDefinition],
}

const VIT_V3_FILES: &[ModelFileDefinition] = &[
    ModelFileDefinition {
        download_url: "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx",
        filename: "model.onnx",
    },
    ModelFileDefinition {
        download_url:
            "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv",
        filename: "selected_tags.csv",
    },
];
const EVA02_LARGE_V3_FILES: &[ModelFileDefinition] = &[
    ModelFileDefinition { download_url: "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx", filename: "model.onnx" },
    ModelFileDefinition { download_url: "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv", filename: "selected_tags.csv" },
];
const SWINV2_V3_FILES: &[ModelFileDefinition] = &[
    ModelFileDefinition {
        download_url:
            "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx",
        filename: "model.onnx",
    },
    ModelFileDefinition {
        download_url:
            "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv",
        filename: "selected_tags.csv",
    },
];
const AGE_DETECTOR_FILES: &[ModelFileDefinition] = &[ModelFileDefinition {
    download_url: "https://huggingface.co/nateraw/age-detection/resolve/main/model.onnx",
    filename: "model.onnx",
}];

pub const KNOWN_MODELS: &[ModelDefinition] = &[
    ModelDefinition {
        name: "vit-v3",
        display_name: "ViT V3",
        description: "WD14 ViT V3 image tagger",
        storage_dir: "tagger/vit-v3",
        files: VIT_V3_FILES,
    },
    ModelDefinition {
        name: "eva02-large-v3",
        display_name: "EVA02 Large V3",
        description: "Higher-quality WD14 EVA02 Large V3 image tagger",
        storage_dir: "tagger/eva02-large-v3",
        files: EVA02_LARGE_V3_FILES,
    },
    ModelDefinition {
        name: "swinv2-v3",
        display_name: "SwinV2 V3",
        description: "WD14 SwinV2 V3 image tagger",
        storage_dir: "tagger/swinv2-v3",
        files: SWINV2_V3_FILES,
    },
    ModelDefinition {
        name: "age-detector",
        display_name: "Age Detection Model",
        description: "Deep learning model for apparent age detection",
        storage_dir: "age-detector",
        files: AGE_DETECTOR_FILES,
    },
];

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
    NotDownloaded,
    Downloading,
    Downloaded,
    Failed,
}
impl ModelStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NotDownloaded => "not_downloaded",
            Self::Downloading => "downloading",
            Self::Downloaded => "downloaded",
            Self::Failed => "failed",
        }
    }
}

pub type ModelRegistry = Arc<DashMap<String, ModelDownloadState>>;
pub fn create_model_registry() -> ModelRegistry {
    Arc::new(DashMap::new())
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_models))
        .route("/download", post(start_model_download))
        .route("/{model_name}/progress", get(model_download_progress))
}

fn model_dir(data_dir: &Path, def: &ModelDefinition) -> PathBuf {
    data_dir.join("models").join(def.storage_dir)
}
fn model_files_exist(data_dir: &Path, def: &ModelDefinition) -> bool {
    let dir = model_dir(data_dir, def);
    def.files
        .iter()
        .all(|file| dir.join(file.filename).is_file())
}
fn find_model_def(name: &str) -> Option<&'static ModelDefinition> {
    KNOWN_MODELS.iter().find(|model| model.name == name)
}
fn status_json(def: &ModelDefinition, data_dir: &Path, registry: &ModelRegistry) -> Value {
    let on_disk = model_files_exist(data_dir, def);
    let state = registry.get(def.name);
    let (status, progress, error) = match state.as_deref() {
        Some(state) => (state.status.as_str(), state.progress, state.error.clone()),
        None if on_disk => ("downloaded", 100.0, None),
        None => ("not_downloaded", 0.0, None),
    };
    json!({
        "name": def.name, "display_name": def.display_name, "description": def.description,
        "status": status, "progress": progress, "on_disk": on_disk,
        "model_dir": model_dir(data_dir, def).to_string_lossy(),
        "files": def.files.iter().map(|file| file.filename).collect::<Vec<_>>(), "error": error,
    })
}

async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let registry = state.model_registry();
    Ok(Json(
        json!({ "models": KNOWN_MODELS.iter().map(|def| status_json(def, &data_dir, &registry)).collect::<Vec<_>>() }),
    ))
}

#[derive(Debug, Deserialize)]
struct ModelDownloadRequest {
    model_name: String,
}

async fn start_model_download(
    State(state): State<AppState>,
    Json(body): Json<ModelDownloadRequest>,
) -> Result<Json<Value>, AppError> {
    let def = find_model_def(&body.model_name)
        .ok_or_else(|| AppError::BadRequest(format!("Unknown model '{}'", body.model_name)))?;
    let data_dir = state.data_dir().to_path_buf();
    let registry = state.model_registry();
    if model_files_exist(&data_dir, def) {
        registry.insert(
            def.name.into(),
            ModelDownloadState {
                status: ModelStatus::Downloaded,
                progress: 100.0,
                bytes_downloaded: 0,
                total_bytes: None,
                error: None,
            },
        );
        return Ok(Json(
            json!({ "model_name": def.name, "status": "downloaded" }),
        ));
    }
    if registry
        .get(def.name)
        .is_some_and(|entry| entry.status == ModelStatus::Downloading)
    {
        return Ok(Json(
            json!({ "model_name": def.name, "status": "downloading" }),
        ));
    }
    registry.insert(
        def.name.into(),
        ModelDownloadState {
            status: ModelStatus::Downloading,
            progress: 0.0,
            bytes_downloaded: 0,
            total_bytes: None,
            error: None,
        },
    );
    let name = def.name.to_string();
    let dir = model_dir(&data_dir, def);
    let files = def.files.to_vec();
    let registry_for_download = registry.clone();
    tokio::spawn(async move {
        run_model_download(name, dir, files, registry_for_download).await;
    });
    Ok(Json(
        json!({ "model_name": def.name, "status": "downloading" }),
    ))
}

async fn fail_download(registry: &ModelRegistry, model_name: &str, error: impl std::fmt::Display) {
    if let Some(mut entry) = registry.get_mut(model_name) {
        entry.status = ModelStatus::Failed;
        entry.error = Some(error.to_string());
    }
}

async fn run_model_download(
    model_name: String,
    dir: PathBuf,
    files: Vec<ModelFileDefinition>,
    registry: ModelRegistry,
) {
    if let Err(error) = tokio::fs::create_dir_all(&dir).await {
        fail_download(&registry, &model_name, error).await;
        return;
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            fail_download(&registry, &model_name, error).await;
            return;
        }
    };
    let file_count = files.len().max(1) as f64;
    for (index, definition) in files.iter().enumerate() {
        let target = dir.join(definition.filename);
        if target.is_file() {
            continue;
        }
        let temporary = dir.join(format!("{}.part", definition.filename));
        let mut response = match client.get(definition.download_url).send().await {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => {
                fail_download(
                    &registry,
                    &model_name,
                    format!(
                        "HTTP {} downloading {}",
                        response.status(),
                        definition.filename
                    ),
                )
                .await;
                return;
            }
            Err(error) => {
                fail_download(&registry, &model_name, error).await;
                return;
            }
        };
        let total = response.content_length();
        let mut downloaded = 0u64;
        let mut output = match tokio::fs::File::create(&temporary).await {
            Ok(file) => file,
            Err(error) => {
                fail_download(&registry, &model_name, error).await;
                return;
            }
        };
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    if let Err(error) = output.write_all(&chunk).await {
                        let _ = tokio::fs::remove_file(&temporary).await;
                        fail_download(&registry, &model_name, error).await;
                        return;
                    }
                    downloaded += chunk.len() as u64;
                    if let Some(mut entry) = registry.get_mut(&model_name) {
                        let file_progress = total
                            .filter(|size| *size > 0)
                            .map(|size| downloaded as f64 / size as f64)
                            .unwrap_or(0.0);
                        entry.progress =
                            ((index as f64 + file_progress) / file_count * 100.0).min(99.0);
                        entry.bytes_downloaded = downloaded;
                        entry.total_bytes = total;
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    let _ = tokio::fs::remove_file(&temporary).await;
                    fail_download(&registry, &model_name, error).await;
                    return;
                }
            }
        }
        if let Err(error) = output.flush().await {
            let _ = tokio::fs::remove_file(&temporary).await;
            fail_download(&registry, &model_name, error).await;
            return;
        }
        drop(output);
        if let Err(error) = tokio::fs::rename(&temporary, &target).await {
            fail_download(&registry, &model_name, error).await;
            return;
        }
    }
    if let Some(mut entry) = registry.get_mut(&model_name) {
        entry.status = ModelStatus::Downloaded;
        entry.progress = 100.0;
        entry.error = None;
    }
}

async fn model_download_progress(
    State(state): State<AppState>,
    AxumPath(model_name): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let def = find_model_def(&model_name)
        .ok_or_else(|| AppError::BadRequest(format!("Unknown model '{}'", model_name)))?;
    let data_dir = state.data_dir().to_path_buf();
    let registry = state.model_registry();
    Ok(Json(status_json(def, &data_dir, &registry)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // AC: @addon-settings ac-3
    #[test]
    fn tagger_models_require_onnx_and_tag_csv() {
        for model in KNOWN_MODELS
            .iter()
            .filter(|model| model.storage_dir.starts_with("tagger/"))
        {
            assert!(model.files.iter().any(|file| file.filename == "model.onnx"));
            assert!(model
                .files
                .iter()
                .any(|file| file.filename == "selected_tags.csv"));
        }
    }
}
