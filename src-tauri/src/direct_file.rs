use axum::{
    extract::{Path as AxumPath, Request, State},
    response::{IntoResponse, Response},
};
use serde::Serialize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri_plugin_dialog::{DialogExt, FilePath};
use tower::ServiceExt;
use tower_http::services::ServeFile;

use crate::server::error::AppError;
use crate::server::state::AppState;

const VIDEO_EXTENSIONS: &[&str] = &[
    "3gp", "avi", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts", "ogv", "ts",
    "webm", "wmv",
];
const IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpeg", "jpg", "png", "tif", "tiff", "webp"];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DirectFileRequest {
    pub id: i64,
    pub filename: String,
    pub original_filename: String,
    pub file_path: PathBuf,
    pub url: String,
    pub direct_file_token: String,
    pub direct_file: bool,
    pub svp: bool,
    pub muted: bool,
    pub video_fps: Option<f64>,
    pub video_width: Option<u64>,
    pub video_height: Option<u64>,
}

#[derive(Default)]
pub struct DirectFileState(Mutex<Option<DirectFileRequest>>);

impl DirectFileState {
    pub fn new(request: Option<DirectFileRequest>) -> Self {
        Self(Mutex::new(request))
    }
}

pub fn direct_file_request(path: impl AsRef<Path>) -> Result<DirectFileRequest, String> {
    let canonical = path
        .as_ref()
        .canonicalize()
        .map_err(|error| format!("cannot open media file: {error}"))?;
    if !canonical.is_file() {
        return Err("selected media path is not a regular file".to_string());
    }
    let extension = canonical
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| "selected file has no supported media extension".to_string())?;
    if !VIDEO_EXTENSIONS.contains(&extension.as_str())
        && !IMAGE_EXTENSIONS.contains(&extension.as_str())
    {
        return Err(format!("unsupported media extension: {extension}"));
    }
    let filename = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "selected media filename is not valid UTF-8".to_string())?
        .to_string();
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    // Database media IDs are positive. Keep direct-file IDs negative and
    // inside JavaScript's exact integer range.
    let id = -((hasher.finish() % i32::MAX as u64) as i64) - 1;
    let video_info = VIDEO_EXTENSIONS
        .contains(&extension.as_str())
        .then(|| crate::routes::settings::get_video_info(&canonical).ok())
        .flatten();
    Ok(DirectFileRequest {
        id,
        original_filename: filename.clone(),
        filename,
        file_path: canonical,
        url: String::new(),
        direct_file_token: String::new(),
        direct_file: true,
        svp: false,
        muted: false,
        video_fps: video_info.as_ref().and_then(|info| info.fps),
        video_width: video_info.as_ref().map(|info| info.width),
        video_height: video_info.as_ref().map(|info| info.height),
    })
}

pub fn direct_file_request_from_args(
    args: impl IntoIterator<Item = PathBuf>,
) -> Option<DirectFileRequest> {
    let args: Vec<PathBuf> = args.into_iter().skip(1).collect();
    let svp = args.iter().any(|argument| argument == "--svp");
    let muted = args.iter().any(|argument| argument == "--mute");
    let path = args
        .iter()
        .find(|argument| !argument.to_string_lossy().starts_with('-'))?;
    let mut request = direct_file_request(path).ok()?;
    request.svp = svp;
    request.muted = muted;
    Some(request)
}

pub fn prepare_direct_media_file(
    path: PathBuf,
    state: tauri::State<'_, AppState>,
) -> Result<DirectFileRequest, String> {
    let mut request = direct_file_request(path)?;
    let token = state.register_direct_file(&request.file_path);
    request.url = format!("/api/direct-files/{token}");
    request.direct_file_token = token;
    Ok(request)
}

#[tauri::command]
pub async fn pick_direct_media_file(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<Option<DirectFileRequest>, String> {
    let selected = app
        .dialog()
        .file()
        .add_filter(
            "Media files",
            &[VIDEO_EXTENSIONS, IMAGE_EXTENSIONS].concat(),
        )
        .blocking_pick_file();
    match selected {
        Some(FilePath::Path(path)) => prepare_direct_media_file(path, state).map(Some),
        Some(FilePath::Url(_)) => Err("selected media is not a local filesystem path".to_string()),
        None => Ok(None),
    }
}

#[tauri::command]
pub fn take_startup_media_file(
    state: tauri::State<'_, DirectFileState>,
    app_state: tauri::State<'_, AppState>,
) -> Result<Option<DirectFileRequest>, String> {
    let mut request = state
        .0
        .lock()
        .map_err(|_| "startup media state lock poisoned".to_string())?
        .take();
    if let Some(request) = &mut request {
        let token = app_state.register_direct_file(&request.file_path);
        request.url = format!("/api/direct-files/{token}");
        request.direct_file_token = token;
        log::info!("[DirectFile] opening startup media {}", request.filename);
    }
    Ok(request)
}

#[tauri::command]
pub fn release_direct_media_file(token: String, state: tauri::State<'_, AppState>) {
    state.revoke_direct_file(&token);
}

/// Serve one explicitly selected non-library file with HTTP Range support.
pub async fn serve_direct_media_file(
    State(state): State<AppState>,
    AxumPath(token): AxumPath<String>,
    request: Request,
) -> Result<Response, AppError> {
    let path = state
        .direct_file_path(&token)
        .ok_or_else(|| AppError::NotFound("Direct media file is no longer available".into()))?;
    let response = ServeFile::new(path)
        .oneshot(request)
        .await
        .map_err(|error| AppError::Internal(format!("Failed to serve direct media: {error}")))?;
    Ok(response.into_response())
}

#[tauri::command]
pub fn report_direct_file_stage(stage: String) {
    log::info!("[DirectFile] frontend {stage}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_single_positional_video_argument() {
        let fixture = std::env::temp_dir().join("localbooru-direct-file-test.mp4");
        std::fs::write(&fixture, b"fixture").unwrap();
        let request = direct_file_request_from_args([PathBuf::from("localbooru"), fixture.clone()])
            .expect("video argument should be accepted");
        assert_eq!(request.file_path, fixture.canonicalize().unwrap());
        assert_eq!(request.filename, "localbooru-direct-file-test.mp4");
        let _ = std::fs::remove_file(fixture);
    }

    #[test]
    fn accepts_a_single_positional_image_argument() {
        let fixture = std::env::temp_dir().join("localbooru-direct-file-test.webp");
        std::fs::write(&fixture, b"fixture").unwrap();
        let request = direct_file_request_from_args([PathBuf::from("localbooru"), fixture.clone()])
            .expect("image argument should be accepted");
        assert_eq!(request.file_path, fixture.canonicalize().unwrap());
        assert_eq!(request.filename, "localbooru-direct-file-test.webp");
        let _ = std::fs::remove_file(fixture);
    }

    #[test]
    fn carries_svp_startup_intent_before_or_after_the_media_path() {
        let fixture = std::env::temp_dir().join("localbooru-direct-file-svp-test.mp4");
        std::fs::write(&fixture, b"fixture").unwrap();
        for args in [
            vec![
                PathBuf::from("localbooru"),
                PathBuf::from("--svp"),
                fixture.clone(),
            ],
            vec![
                PathBuf::from("localbooru"),
                fixture.clone(),
                PathBuf::from("--svp"),
            ],
        ] {
            let request =
                direct_file_request_from_args(args).expect("SVP media argument should be accepted");
            assert!(request.svp);
        }
        let _ = std::fs::remove_file(fixture);
    }

    #[test]
    fn carries_muted_startup_intent_before_or_after_the_media_path() {
        let fixture = std::env::temp_dir().join("localbooru-direct-file-muted-test.mp4");
        std::fs::write(&fixture, b"fixture").unwrap();
        for args in [
            vec![
                PathBuf::from("localbooru"),
                PathBuf::from("--mute"),
                fixture.clone(),
            ],
            vec![
                PathBuf::from("localbooru"),
                fixture.clone(),
                PathBuf::from("--mute"),
            ],
        ] {
            let request = direct_file_request_from_args(args)
                .expect("muted media argument should be accepted");
            assert!(request.muted);
        }
        let _ = std::fs::remove_file(fixture);
    }

    #[test]
    fn ignores_flags_and_rejects_non_media_files() {
        let fixture = std::env::temp_dir().join("localbooru-direct-file-test.txt");
        std::fs::write(&fixture, b"fixture").unwrap();
        assert!(direct_file_request_from_args([
            PathBuf::from("localbooru"),
            PathBuf::from("--verbose"),
            fixture.clone(),
        ])
        .is_none());
        let _ = std::fs::remove_file(fixture);
    }
}
