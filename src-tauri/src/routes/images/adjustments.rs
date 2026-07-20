use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Json, Response};
use image::{DynamicImage, ImageBuffer, Rgb};
use rusqlite::{params, TransactionBehavior};
use serde::Deserialize;
use serde_json::json;
use tokio::fs::File;

use crate::db::library::LibraryContext;
use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::importer;

#[derive(Debug, Clone, Deserialize)]
pub struct ImageAdjustmentRequest {
    #[serde(default)]
    pub brightness: i32,
    #[serde(default)]
    pub contrast: i32,
    #[serde(default)]
    pub gamma: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ApplyAdjustmentRequest {
    #[serde(flatten)]
    pub adjustments: ImageAdjustmentRequest,
    pub expected_file_hash: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageLocatorQuery {
    pub library_id: String,
    pub directory_id: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreviewQuery {
    pub library_id: String,
    pub directory_id: i64,
    pub adjustment_hash: String,
    pub preview_key: String,
    pub source_file_hash: String,
}

struct ResolvedImage {
    library: Arc<LibraryContext>,
    path: PathBuf,
    file_hash: String,
}

struct TempPath {
    path: PathBuf,
    armed: bool,
}

impl TempPath {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// POST /api/images/:image_id/preview-adjust — Generate an immutable preview generation.
pub async fn preview_adjust(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
    Json(adjustments): Json<ImageAdjustmentRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validate_adjustments(&adjustments)?;

    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let resolved =
        tokio::task::spawn_blocking(move || resolve_image(&state_clone, &locator_clone, image_id))
            .await??;
    let canonical_locator = ImageLocatorQuery {
        library_id: resolved.library.uuid.clone(),
        directory_id: locator.directory_id,
    };
    let source_file_hash = resolved.file_hash;
    let adjustment_hash = adjustment_hash(&adjustments);
    let preview_key = format!("{:032x}", unique_suffix());
    let filename = preview_cache_filename(
        &canonical_locator,
        image_id,
        &source_file_hash,
        &adjustment_hash,
        &preview_key,
    );
    let url = preview_url(
        &canonical_locator,
        image_id,
        &source_file_hash,
        &adjustment_hash,
        &preview_key,
    );
    let cache_dir = resolved.library.data_dir.join("preview_cache");
    let source_path = resolved.path;
    let expected_source_file_hash = source_file_hash.clone();
    let brightness = adjustments.brightness;
    let contrast = adjustments.contrast;
    let gamma = adjustments.gamma;

    tokio::task::spawn_blocking(move || {
        ensure_path_hash(&source_path, &expected_source_file_hash)?;
        std::fs::create_dir_all(&cache_dir)?;
        let source_snapshot =
            create_validated_snapshot(&source_path, &cache_dir, &expected_source_file_hash)?;
        let image = image::open(&source_snapshot.path)
            .map_err(|error| AppError::Internal(format!("Failed to open image: {}", error)))?;
        let destination = cache_dir.join(filename);
        let mut temporary = TempPath::new(cache_dir.join(format!(
            ".preview-{}-{}.localbooru-previewing",
            std::process::id(),
            unique_suffix()
        )));
        apply_adjustments_to_image(&image, &adjustments)
            .save_with_format(&temporary.path, image::ImageFormat::WebP)
            .map_err(|error| AppError::Internal(format!("Failed to save preview: {}", error)))?;
        sync_file_contents(&temporary.path)?;
        replace_file(&temporary.path, &destination)
            .map_err(|error| AppError::Internal(format!("Failed to publish preview: {}", error)))?;
        temporary.disarm();
        Ok::<_, AppError>(())
    })
    .await??;

    Ok(Json(json!({
        "preview_url": url,
        "adjustment_hash": adjustment_hash,
        "preview_key": preview_key,
        "source_file_hash": source_file_hash,
        "adjustments": { "brightness": brightness, "contrast": contrast, "gamma": gamma }
    })))
}

/// GET /api/images/:image_id/preview — Serve one exact immutable preview generation.
pub async fn get_preview(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(query): Query<PreviewQuery>,
) -> Result<Response, AppError> {
    validate_preview_query(&query)?;
    let locator = ImageLocatorQuery {
        library_id: query.library_id,
        directory_id: query.directory_id,
    };
    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let resolved =
        tokio::task::spawn_blocking(move || resolve_image(&state_clone, &locator_clone, image_id))
            .await??;
    if resolved.file_hash != query.source_file_hash {
        return Err(AppError::NotFound(
            "Preview source is no longer current".into(),
        ));
    }
    let canonical = ImageLocatorQuery {
        library_id: resolved.library.uuid.clone(),
        directory_id: locator.directory_id,
    };
    let path = resolved
        .library
        .data_dir
        .join("preview_cache")
        .join(preview_cache_filename(
            &canonical,
            image_id,
            &query.source_file_hash,
            &query.adjustment_hash,
            &query.preview_key,
        ));
    if !path.is_file() {
        return Err(AppError::NotFound("No matching preview found".into()));
    }

    let file = File::open(&path).await?;
    let metadata = file.metadata().await?;
    let body = axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(file));
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "image/webp")
        .header("Content-Length", metadata.len())
        .body(body)
        .unwrap())
}

/// DELETE /api/images/:image_id/preview — Discard one exact preview generation.
pub async fn discard_preview(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(query): Query<PreviewQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    validate_preview_query(&query)?;
    let locator = ImageLocatorQuery {
        library_id: query.library_id,
        directory_id: query.directory_id,
    };
    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let resolved =
        tokio::task::spawn_blocking(move || resolve_image(&state_clone, &locator_clone, image_id))
            .await??;
    let canonical = ImageLocatorQuery {
        library_id: resolved.library.uuid.clone(),
        directory_id: locator.directory_id,
    };
    let path = resolved
        .library
        .data_dir
        .join("preview_cache")
        .join(preview_cache_filename(
            &canonical,
            image_id,
            &query.source_file_hash,
            &query.adjustment_hash,
            &query.preview_key,
        ));
    let deleted = match std::fs::remove_file(path) {
        Ok(()) => 1,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error.into()),
    };
    Ok(Json(json!({"deleted": deleted})))
}

/// POST /api/images/:image_id/adjust — Apply adjustments to one exact original file.
pub async fn apply_adjust(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
    Json(request): Json<ApplyAdjustmentRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validate_adjustments(&request.adjustments)?;

    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let result = tokio::task::spawn_blocking(move || {
        let resolved = resolve_image(&state_clone, &locator_clone, image_id)?;
        ensure_expected_hash(&resolved, &request.expected_file_hash)?;
        if request.adjustments.brightness == 0
            && request.adjustments.contrast == 0
            && request.adjustments.gamma == 0
        {
            return Ok(json!({ "adjusted": false, "message": "No adjustments needed" }));
        }
        apply_to_resolved_image(
            resolved,
            &locator_clone,
            image_id,
            &request.adjustments,
            &request.expected_file_hash,
            false,
        )
    })
    .await??;
    Ok(Json(result))
}

fn resolve_image(
    state: &AppState,
    locator: &ImageLocatorQuery,
    image_id: i64,
) -> Result<ResolvedImage, AppError> {
    let library = state.resolve_library(Some(&locator.library_id))?;
    if !library.directory_db.db_exists(locator.directory_id) {
        return Err(AppError::NotFound(format!(
            "Directory {} not found in library '{}'",
            locator.directory_id, library.uuid
        )));
    }
    let pool = library.directory_db.get_pool(locator.directory_id)?;
    let conn = pool.get()?;
    let file_hash = conn
        .query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        )
        .map_err(|error| match error {
            rusqlite::Error::QueryReturnedNoRows => AppError::NotFound(format!(
                "Image {} not found in library '{}' directory {}",
                image_id, library.uuid, locator.directory_id
            )),
            other => other.into(),
        })?;
    let mut statement = conn.prepare(
        "SELECT original_path FROM image_files
         WHERE image_id = ?1 AND file_exists = 1 ORDER BY id",
    )?;
    let paths: Vec<PathBuf> = statement
        .query_map(params![image_id], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        return Err(AppError::NotFound("Image file not found".into()));
    }
    if paths.len() != 1 {
        return Err(AppError::BadRequest(
            "Image adjustment is ambiguous because multiple existing file paths share this image"
                .into(),
        ));
    }
    let path = paths.into_iter().next().unwrap();
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|_| AppError::NotFound("Image file not found on disk".into()))?;
    if metadata.file_type().is_symlink() {
        return Err(AppError::BadRequest(
            "Image adjustments do not follow symbolic links".into(),
        ));
    }
    if !metadata.is_file() {
        return Err(AppError::NotFound("Image file not found on disk".into()));
    }
    drop(statement);
    drop(conn);
    Ok(ResolvedImage {
        library,
        path,
        file_hash,
    })
}

fn ensure_expected_hash(resolved: &ResolvedImage, expected: &str) -> Result<(), AppError> {
    if resolved.file_hash != expected {
        return Err(AppError::BadRequest(
            "Image changed since the adjustment operation started".into(),
        ));
    }
    ensure_path_hash(&resolved.path, expected)
}

fn ensure_path_hash(path: &Path, expected: &str) -> Result<(), AppError> {
    let actual = importer::calculate_quick_hash(&path.to_string_lossy())
        .map_err(|error| AppError::Internal(format!("Hash error: {}", error)))?;
    if actual != expected {
        return Err(AppError::BadRequest(
            "Image changed since the adjustment operation started".into(),
        ));
    }
    Ok(())
}

fn create_validated_snapshot(
    source: &Path,
    snapshot_dir: &Path,
    expected: &str,
) -> Result<TempPath, AppError> {
    let extension = source
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let snapshot = TempPath::new(snapshot_dir.join(format!(
        ".preview-source-{}-{}.{}",
        std::process::id(),
        unique_suffix(),
        extension
    )));
    std::fs::copy(source, &snapshot.path)?;
    ensure_path_hash(&snapshot.path, expected)?;
    Ok(snapshot)
}

fn adjustment_hash(adjustments: &ImageAdjustmentRequest) -> String {
    format!(
        "{:016x}",
        xxhash_rust::xxh64::xxh64(
            format!(
                "{}_{}_{}",
                adjustments.brightness, adjustments.contrast, adjustments.gamma
            )
            .as_bytes(),
            0,
        )
    )
}

fn preview_cache_prefix(locator: &ImageLocatorQuery, image_id: i64) -> String {
    format!(
        "{}_{}_{}_",
        locator.library_id, locator.directory_id, image_id
    )
}

fn preview_cache_filename(
    locator: &ImageLocatorQuery,
    image_id: i64,
    source_file_hash: &str,
    adjustment_hash: &str,
    preview_key: &str,
) -> String {
    format!(
        "{}{}_{}_{}.webp",
        preview_cache_prefix(locator, image_id),
        source_file_hash,
        adjustment_hash,
        preview_key
    )
}

fn preview_url(
    locator: &ImageLocatorQuery,
    image_id: i64,
    source_file_hash: &str,
    adjustment_hash: &str,
    preview_key: &str,
) -> String {
    format!(
        "/api/images/{}/preview?library_id={}&directory_id={}&source_file_hash={}&adjustment_hash={}&preview_key={}",
        image_id,
        encode_query_component(&locator.library_id),
        locator.directory_id,
        encode_query_component(source_file_hash),
        adjustment_hash,
        preview_key
    )
}

pub(crate) fn encode_query_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            use std::fmt::Write;
            let _ = write!(encoded, "%{:02X}", byte);
        }
    }
    encoded
}

fn validate_preview_component(value: &str, name: &str) -> Result<(), AppError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(AppError::BadRequest(format!("Invalid {}", name)));
    }
    Ok(())
}

fn validate_preview_query(query: &PreviewQuery) -> Result<(), AppError> {
    validate_preview_component(&query.adjustment_hash, "adjustment hash")?;
    validate_preview_component(&query.preview_key, "preview key")?;
    validate_preview_component(&query.source_file_hash, "source file hash")
}

fn apply_to_resolved_image(
    resolved: ResolvedImage,
    locator: &ImageLocatorQuery,
    image_id: i64,
    adjustments: &ImageAdjustmentRequest,
    expected_file_hash: &str,
    force_commit_failure: bool,
) -> Result<serde_json::Value, AppError> {
    apply_to_resolved_image_with_hook(
        resolved,
        locator,
        image_id,
        adjustments,
        expected_file_hash,
        force_commit_failure,
        || {},
    )
}

fn apply_to_resolved_image_with_hook(
    resolved: ResolvedImage,
    locator: &ImageLocatorQuery,
    image_id: i64,
    adjustments: &ImageAdjustmentRequest,
    expected_file_hash: &str,
    force_commit_failure: bool,
    before_replace: impl FnOnce(),
) -> Result<serde_json::Value, AppError> {
    ensure_expected_hash(&resolved, expected_file_hash)?;
    let extension = resolved
        .path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_lowercase();
    let format = image_format(&extension)?;
    let original_metadata = std::fs::symlink_metadata(&resolved.path)?;
    let image = image::open(&resolved.path)
        .map_err(|error| AppError::Internal(format!("Failed to open image: {}", error)))?;
    let mut adjusted_temp = TempPath::new(adjustment_temp_path(&resolved.path));
    let adjusted = apply_adjustments_to_image(&image, adjustments);
    let (width, height) = (adjusted.width(), adjusted.height());
    adjusted
        .save_with_format(&adjusted_temp.path, format)
        .map_err(|error| AppError::Internal(format!("Failed to save adjusted image: {}", error)))?;
    std::fs::set_permissions(&adjusted_temp.path, original_metadata.permissions())?;
    sync_file_contents(&adjusted_temp.path)?;

    let temporary_string = adjusted_temp.path.to_string_lossy();
    let new_hash = importer::calculate_quick_hash(&temporary_string)
        .map_err(|error| AppError::Internal(format!("Hash error: {}", error)))?;
    let perceptual_hash = importer::calculate_perceptual_hash(&temporary_string);
    let metadata = std::fs::metadata(&adjusted_temp.path)?;
    let file_size = metadata.len() as i64;
    let file_modified_at = modified_at_rfc3339(&metadata);

    let pool = resolved
        .library
        .directory_db
        .get_pool(locator.directory_id)?;
    let mut connection = pool.get()?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    if transaction
        .query_row(
            "SELECT id FROM images WHERE file_hash = ?1 AND id != ?2",
            params![&new_hash, image_id],
            |row| row.get::<_, i64>(0),
        )
        .is_ok()
    {
        return Err(AppError::BadRequest(
            "Adjusted image duplicates another image in this directory".into(),
        ));
    }
    let canonical_filename = format!("{}.{}", &new_hash[..16.min(new_hash.len())], extension);
    let updated = transaction.execute(
        "UPDATE images
         SET filename = ?1, file_hash = ?2, perceptual_hash = ?3,
             width = ?4, height = ?5, file_size = ?6,
             file_modified_at = ?7, updated_at = datetime('now')
         WHERE id = ?8 AND file_hash = ?9",
        params![
            &canonical_filename,
            &new_hash,
            perceptual_hash,
            width as i32,
            height as i32,
            file_size,
            file_modified_at,
            image_id,
            expected_file_hash
        ],
    )?;
    if updated != 1 {
        return Err(AppError::BadRequest(
            "Image changed while adjustments were being prepared".into(),
        ));
    }

    let thumbnails_dir = resolved.library.thumbnails_dir();
    std::fs::create_dir_all(&thumbnails_dir)?;
    let thumbnail_path = thumbnails_dir.join(format!("{}.webp", &new_hash[..16]));
    let mut thumbnail_temp = if thumbnail_path.exists() {
        None
    } else {
        let temporary = TempPath::new(thumbnails_dir.join(format!(
            ".thumbnail-{}-{}.tmp.webp",
            std::process::id(),
            unique_suffix()
        )));
        if !importer::generate_thumbnail_from_image(
            &adjusted,
            &temporary.path.to_string_lossy(),
            400,
        ) {
            return Err(AppError::Internal(
                "Failed to generate adjusted image thumbnail".into(),
            ));
        }
        Some(temporary)
    };

    let mut backup = TempPath::new(adjustment_backup_path(&resolved.path));
    std::fs::copy(&resolved.path, &backup.path)?;
    std::fs::set_permissions(&backup.path, original_metadata.permissions())?;
    sync_file_contents(&backup.path)?;
    ensure_path_hash(&backup.path, expected_file_hash)?;

    before_replace();
    replace_file_if_hash_matches(&mut adjusted_temp, &resolved.path, expected_file_hash)?;

    let mut created_thumbnail = false;
    if let Some(temporary) = thumbnail_temp.as_mut() {
        if let Err(error) = replace_file(&temporary.path, &thumbnail_path) {
            restore_original_if_hash_matches(&mut backup, &resolved.path, &new_hash)?;
            return Err(AppError::Internal(format!(
                "Failed to publish adjusted thumbnail: {}",
                error
            )));
        }
        temporary.disarm();
        created_thumbnail = true;
    }

    if ensure_path_hash(&resolved.path, &new_hash).is_err() {
        if created_thumbnail && !thumbnail_hash_in_use(&resolved.library, &new_hash, None, None)? {
            let _ = std::fs::remove_file(&thumbnail_path);
        }
        return Err(AppError::BadRequest(
            "Image changed before adjusted metadata could be committed".into(),
        ));
    }

    let commit_result = if force_commit_failure {
        drop(transaction);
        Err(rusqlite::Error::ExecuteReturnedResults)
    } else {
        transaction.commit()
    };
    if let Err(error) = commit_result {
        let restore_result =
            restore_original_if_hash_matches(&mut backup, &resolved.path, &new_hash);
        if created_thumbnail && !thumbnail_hash_in_use(&resolved.library, &new_hash, None, None)? {
            let _ = std::fs::remove_file(&thumbnail_path);
        }
        restore_result?;
        return Err(AppError::Internal(format!(
            "Failed to commit adjusted metadata: {}",
            error
        )));
    }
    if std::fs::remove_file(&backup.path).is_ok() {
        backup.disarm();
    }

    if resolved.file_hash != new_hash
        && !thumbnail_hash_in_use(
            &resolved.library,
            &resolved.file_hash,
            Some(locator.directory_id),
            Some(image_id),
        )?
    {
        let old_thumbnail = thumbnails_dir.join(format!(
            "{}.webp",
            &resolved.file_hash[..16.min(resolved.file_hash.len())]
        ));
        let _ = std::fs::remove_file(old_thumbnail);
    }

    let library_id = encode_query_component(&resolved.library.uuid);
    Ok(json!({
        "adjusted": true,
        "brightness": adjustments.brightness,
        "contrast": adjustments.contrast,
        "gamma": adjustments.gamma,
        "file_hash": new_hash,
        "filename": canonical_filename,
        "file_size": file_size,
        "width": width,
        "height": height,
        "file_modified_at": file_modified_at,
        "url": format!("/api/images/{}/file?directory_id={}&library_id={}&file_hash={}", image_id, locator.directory_id, library_id, new_hash),
        "thumbnail_url": format!("/api/images/{}/thumbnail?directory_id={}&library_id={}&file_hash={}", image_id, locator.directory_id, library_id, new_hash)
    }))
}

fn restore_original_if_hash_matches(
    backup: &mut TempPath,
    destination: &Path,
    installed_hash: &str,
) -> Result<(), AppError> {
    replace_file_if_hash_matches(backup, destination, installed_hash)
}

pub(crate) fn thumbnail_hash_in_use(
    library: &LibraryContext,
    hash: &str,
    except_directory: Option<i64>,
    except_image: Option<i64>,
) -> Result<bool, AppError> {
    for directory_id in library.directory_db.get_all_directory_ids() {
        let pool = library.directory_db.get_pool(directory_id)?;
        let connection = pool.get()?;
        let found = if except_directory == Some(directory_id) {
            connection
                .query_row(
                    "SELECT 1 FROM images WHERE file_hash = ?1 AND id != ?2 LIMIT 1",
                    params![hash, except_image.unwrap_or(-1)],
                    |_| Ok(()),
                )
                .is_ok()
        } else {
            connection
                .query_row(
                    "SELECT 1 FROM images WHERE file_hash = ?1 LIMIT 1",
                    params![hash],
                    |_| Ok(()),
                )
                .is_ok()
        };
        if found {
            return Ok(true);
        }
    }
    Ok(false)
}

fn image_format(extension: &str) -> Result<image::ImageFormat, AppError> {
    match extension {
        "jpg" | "jpeg" => Ok(image::ImageFormat::Jpeg),
        "png" => Ok(image::ImageFormat::Png),
        "webp" => Ok(image::ImageFormat::WebP),
        "bmp" => Ok(image::ImageFormat::Bmp),
        "tiff" | "tif" => Ok(image::ImageFormat::Tiff),
        _ => Err(AppError::BadRequest(format!(
            "Cannot adjust .{} files",
            extension
        ))),
    }
}

fn adjustment_temp_path(path: &Path) -> PathBuf {
    path.with_file_name(format!(
        ".localbooru-adjust-{}-{}.localbooru-adjusting",
        std::process::id(),
        unique_suffix()
    ))
}

fn adjustment_backup_path(path: &Path) -> PathBuf {
    path.with_file_name(format!(
        ".localbooru-backup-{}-{}.localbooru-backup",
        std::process::id(),
        unique_suffix()
    ))
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn modified_at_rfc3339(metadata: &std::fs::Metadata) -> Option<String> {
    let modified = metadata.modified().ok()?;
    let duration = modified.duration_since(UNIX_EPOCH).ok()?;
    chrono::DateTime::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
        .map(|value| value.to_rfc3339())
}

fn sync_file_contents(path: &Path) -> std::io::Result<()> {
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)?
        .sync_all()
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn replace_file_if_hash_matches(
    replacement: &mut TempPath,
    destination: &Path,
    expected: &str,
) -> Result<(), AppError> {
    let installed_hash = importer::calculate_quick_hash(&replacement.path.to_string_lossy())?;
    exchange_files(&replacement.path, destination).map_err(|error| {
        AppError::Internal(format!("Failed to atomically exchange image: {}", error))
    })?;
    if let Err(validation_error) = ensure_path_hash(&replacement.path, expected) {
        return rollback_exchange(replacement, destination, &installed_hash, validation_error);
    }
    if let Err(remove_error) = std::fs::remove_file(&replacement.path) {
        return rollback_exchange(
            replacement,
            destination,
            &installed_hash,
            AppError::Internal(format!("Failed to remove replaced image: {}", remove_error)),
        );
    }
    replacement.disarm();
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn rollback_exchange(
    captured: &mut TempPath,
    destination: &Path,
    installed_hash: &str,
    cause: AppError,
) -> Result<(), AppError> {
    if let Err(error) = exchange_files(&captured.path, destination) {
        captured.disarm();
        return Err(AppError::Internal(format!(
            "Failed to restore captured image after {}: {}",
            cause, error
        )));
    }
    if ensure_path_hash(&captured.path, installed_hash).is_ok() {
        return Err(cause);
    }

    captured.disarm();
    Err(AppError::Internal(format!(
        "A newer external image was preserved at {} while restoring after {}",
        captured.path.display(),
        cause
    )))
}

#[cfg(target_os = "linux")]
fn exchange_files(first: &Path, second: &Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let first = CString::new(first.as_os_str().as_bytes())?;
    let second = CString::new(second.as_os_str().as_bytes())?;
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            first.as_ptr(),
            libc::AT_FDCWD,
            second.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(target_os = "macos")]
fn exchange_files(first: &Path, second: &Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let first = CString::new(first.as_os_str().as_bytes())?;
    let second = CString::new(second.as_os_str().as_bytes())?;
    let result = unsafe { libc::renamex_np(first.as_ptr(), second.as_ptr(), libc::RENAME_SWAP) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(target_os = "windows")]
fn replace_file_if_hash_matches(
    replacement: &mut TempPath,
    destination: &Path,
    expected: &str,
) -> Result<(), AppError> {
    let installed_hash = importer::calculate_quick_hash(&replacement.path.to_string_lossy())?;
    let mut captured = TempPath::new(destination.with_file_name(format!(
        ".localbooru-cas-{}-{}",
        std::process::id(),
        unique_suffix()
    )));
    if let Err(error) =
        replace_file_windows_with_backup(&replacement.path, destination, &captured.path)
    {
        restore_windows_partial_failure(&mut captured, destination)?;
        return Err(error);
    }
    replacement.disarm();

    if let Err(validation_error) = ensure_path_hash(&captured.path, expected) {
        return rollback_windows_capture(
            &mut captured,
            destination,
            &installed_hash,
            validation_error,
        );
    }
    if let Err(remove_error) = std::fs::remove_file(&captured.path) {
        return rollback_windows_capture(
            &mut captured,
            destination,
            &installed_hash,
            AppError::Internal(format!("Failed to remove replaced image: {}", remove_error)),
        );
    }
    captured.disarm();
    Ok(())
}

#[cfg(target_os = "windows")]
fn rollback_windows_capture(
    captured: &mut TempPath,
    destination: &Path,
    installed_hash: &str,
    cause: AppError,
) -> Result<(), AppError> {
    let mut displaced = TempPath::new(destination.with_file_name(format!(
        ".localbooru-rollback-{}-{}",
        std::process::id(),
        unique_suffix()
    )));
    if let Err(error) =
        replace_file_windows_with_backup(&captured.path, destination, &displaced.path)
    {
        captured.disarm();
        restore_windows_partial_failure(&mut displaced, destination)?;
        return Err(AppError::Internal(format!(
            "Failed to restore captured image after {}: {}",
            cause, error
        )));
    }
    captured.disarm();

    if ensure_path_hash(&displaced.path, installed_hash).is_ok() {
        return Err(cause);
    }

    displaced.disarm();
    Err(AppError::Internal(format!(
        "A newer external image was preserved at {} while restoring after {}",
        displaced.path.display(),
        cause
    )))
}

#[cfg(target_os = "windows")]
fn restore_windows_partial_failure(
    preserved: &mut TempPath,
    destination: &Path,
) -> Result<(), AppError> {
    if !destination.exists() && preserved.path.is_file() {
        if let Err(error) = replace_file(&preserved.path, destination) {
            preserved.disarm();
            return Err(AppError::Internal(format!(
                "Failed to restore image after partial Windows replacement: {} (preserved at {})",
                error,
                preserved.path.display()
            )));
        }
    }
    preserved.disarm();
    Ok(())
}

#[cfg(target_os = "windows")]
fn replace_file_windows_with_backup(
    replacement: &Path,
    destination: &Path,
    captured: &Path,
) -> Result<(), AppError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{ReplaceFileW, REPLACEFILE_WRITE_THROUGH};

    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let replacement: Vec<u16> = replacement
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let captured: Vec<u16> = captured.as_os_str().encode_wide().chain(Some(0)).collect();
    let result = unsafe {
        ReplaceFileW(
            destination.as_ptr(),
            replacement.as_ptr(),
            captured.as_ptr(),
            REPLACEFILE_WRITE_THROUGH,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if result == 0 {
        Err(AppError::Internal(format!(
            "Failed to atomically exchange image: {}",
            std::io::Error::last_os_error()
        )))
    } else {
        Ok(())
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn replace_file_if_hash_matches(
    replacement: &mut TempPath,
    destination: &Path,
    expected: &str,
) -> Result<(), AppError> {
    ensure_path_hash(destination, expected)?;
    replace_file(&replacement.path, destination).map_err(|error| {
        AppError::Internal(format!("Failed to atomically replace image: {}", error))
    })?;
    replacement.disarm();
    Ok(())
}

#[cfg(unix)]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(target_os = "windows")]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

fn validate_adjustments(adjustments: &ImageAdjustmentRequest) -> Result<(), AppError> {
    if !(-200..=200).contains(&adjustments.brightness) {
        return Err(AppError::BadRequest(
            "Brightness must be between -200 and +200".into(),
        ));
    }
    if !(-100..=100).contains(&adjustments.contrast) {
        return Err(AppError::BadRequest(
            "Contrast must be between -100 and +100".into(),
        ));
    }
    if !(-100..=100).contains(&adjustments.gamma) {
        return Err(AppError::BadRequest(
            "Gamma must be between -100 and +100".into(),
        ));
    }
    Ok(())
}

fn apply_adjustments_to_image(
    image: &DynamicImage,
    adjustments: &ImageAdjustmentRequest,
) -> DynamicImage {
    let has_alpha = image.color().has_alpha();
    let mut rgba = image.to_rgba8();
    let mut channels: Vec<f32> = rgba
        .pixels()
        .flat_map(|pixel| pixel.0[..3].iter().map(|value| *value as f32))
        .collect();

    if adjustments.brightness != 0 {
        let factor = (1.0 + adjustments.brightness as f32 / 100.0).max(0.0);
        for value in &mut channels {
            *value *= factor;
        }
    }
    if adjustments.contrast != 0 {
        let factor = (adjustments.contrast as f32 + 100.0) / 100.0;
        for value in &mut channels {
            *value = ((*value - 127.0) * factor) + 127.0;
        }
    }
    if adjustments.gamma != 0 {
        let exponent = 3.0_f32.powf(-adjustments.gamma as f32 / 100.0);
        for value in &mut channels {
            *value = (value.clamp(0.0, 255.0) / 255.0).powf(exponent) * 255.0;
        }
    }

    let dither_strength = if adjustments.gamma != 0 {
        0.5 + (adjustments.gamma.unsigned_abs() as f32 / 100.0) * 0.5
    } else {
        0.5
    };
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    for (index, value) in channels.iter_mut().enumerate() {
        let mut hasher = DefaultHasher::new();
        index.hash(&mut hasher);
        let noise = ((hasher.finish() % 1000) as f32 / 500.0 - 1.0) * dither_strength;
        *value += noise;
    }

    for (pixel, adjusted) in rgba.pixels_mut().zip(channels.chunks_exact(3)) {
        pixel.0[0] = adjusted[0].clamp(0.0, 255.0) as u8;
        pixel.0[1] = adjusted[1].clamp(0.0, 255.0) as u8;
        pixel.0[2] = adjusted[2].clamp(0.0, 255.0) as u8;
    }
    if has_alpha {
        DynamicImage::ImageRgba8(rgba)
    } else {
        let (width, height) = rgba.dimensions();
        let rgb: Vec<u8> = rgba
            .pixels()
            .flat_map(|pixel| pixel.0[..3].iter().copied())
            .collect();
        DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, rgb).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "localbooru-adjust-{}-{}-{}",
            name,
            std::process::id(),
            unique_suffix()
        ))
    }

    fn insert_image(
        library: &LibraryContext,
        directory_id: i64,
        image_id: i64,
        path: &Path,
        hash: &str,
    ) {
        let pool = library.directory_db.get_pool(directory_id).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (?1, 'image.png', ?2)",
                params![image_id, hash],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO image_files (image_id, original_path, file_extension) VALUES (?1, ?2, 'png')",
                params![image_id, path.to_string_lossy()],
            )
            .unwrap();
    }

    // AC: @identity-safe-image-adjustments ac-1
    #[test]
    fn exact_locator_resolves_duplicate_image_id_and_rejects_ambiguous_paths() {
        let root = test_root("locator");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let first = root.join("first.png");
        let second = root.join("second.png");
        DynamicImage::new_rgb8(2, 2).save(&first).unwrap();
        DynamicImage::new_rgb8(3, 3).save(&second).unwrap();
        insert_image(&library, 1, 9, &first, "first-hash");
        insert_image(&library, 2, 9, &second, "second-hash");
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 2,
        };
        assert_eq!(resolve_image(&state, &locator, 9).unwrap().path, second);
        let extra = root.join("extra.png");
        DynamicImage::new_rgb8(3, 3).save(&extra).unwrap();
        let pool = library.directory_db.get_pool(2).unwrap();
        pool.get()
            .unwrap()
            .execute(
                "INSERT INTO image_files (image_id, original_path, file_extension) VALUES (9, ?1, 'png')",
                params![extra.to_string_lossy()],
            )
            .unwrap();
        assert!(matches!(
            resolve_image(&state, &locator, 9),
            Err(AppError::BadRequest(_))
        ));
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-1
    #[cfg(unix)]
    #[test]
    fn exact_locator_rejects_symbolic_link_targets() {
        use std::os::unix::fs::symlink;
        let root = test_root("symlink");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let target = root.join("target.png");
        let link = root.join("link.png");
        DynamicImage::new_rgb8(2, 2).save(&target).unwrap();
        symlink(&target, &link).unwrap();
        insert_image(&library, 1, 1, &link, "hash");
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 1,
        };
        assert!(matches!(
            resolve_image(&state, &locator, 1),
            Err(AppError::BadRequest(_))
        ));
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-windows-preview
    #[test]
    fn durability_flush_uses_a_writable_file_handle() {
        let root = test_root("writable-flush");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("preview.webp");
        std::fs::write(&path, b"preview").unwrap();

        sync_file_contents(&path).unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"preview");
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-2
    #[test]
    fn concurrent_preview_generations_have_independent_exact_urls() {
        let locator = ImageLocatorQuery {
            library_id: "library&one".into(),
            directory_id: 1,
        };
        let first = preview_cache_filename(&locator, 9, "source", "adjust", "generation-a");
        let second = preview_cache_filename(&locator, 9, "source", "adjust", "generation-b");
        assert_ne!(first, second);
        let url = preview_url(&locator, 9, "source", "adjust", "generation-a");
        assert!(url.contains("library_id=library%26one"));
        assert!(url.contains("source_file_hash=source"));
        assert!(url.contains("preview_key=generation-a"));
    }

    // AC: @identity-safe-image-adjustments ac-3
    #[test]
    fn commit_failure_restores_original_bytes_and_database_hash() {
        let root = test_root("rollback");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let path = root.join("source.png");
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])))
            .save(&path)
            .unwrap();
        let original = std::fs::read(&path).unwrap();
        let old_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
        insert_image(&library, 4, 12, &path, &old_hash);
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 4,
        };
        let adjustments = ImageAdjustmentRequest {
            brightness: 20,
            contrast: 0,
            gamma: 0,
        };
        let candidate = root.join("candidate.png");
        apply_adjustments_to_image(&image::open(&path).unwrap(), &adjustments)
            .save(&candidate)
            .unwrap();
        let candidate_hash = importer::calculate_quick_hash(&candidate.to_string_lossy()).unwrap();
        let candidate_thumbnail = library
            .thumbnails_dir()
            .join(format!("{}.webp", candidate_hash));
        std::fs::create_dir_all(library.thumbnails_dir()).unwrap();
        std::fs::write(&candidate_thumbnail, b"shared-valid-thumbnail").unwrap();
        std::fs::remove_file(candidate).unwrap();

        let resolved = resolve_image(&state, &locator, 12).unwrap();
        assert!(
            apply_to_resolved_image(resolved, &locator, 12, &adjustments, &old_hash, true,)
                .is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), original);
        let pool = library.directory_db.get_pool(4).unwrap();
        let stored: String = pool
            .get()
            .unwrap()
            .query_row("SELECT file_hash FROM images WHERE id = 12", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored, old_hash);
        assert_eq!(
            std::fs::read(candidate_thumbnail).unwrap(),
            b"shared-valid-thumbnail"
        );
        assert!(std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .all(|entry| !entry
                .file_name()
                .to_string_lossy()
                .contains("localbooru-adjust")));
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-1
    // AC: @identity-safe-image-adjustments ac-3
    #[test]
    fn apply_rejects_externally_changed_source_when_database_hash_is_stale() {
        let root = test_root("external-apply-change");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let path = root.join("source.png");
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])))
            .save(&path)
            .unwrap();
        let old_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
        insert_image(&library, 4, 12, &path, &old_hash);
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([10, 20, 30])))
            .save(&path)
            .unwrap();
        let external_bytes = std::fs::read(&path).unwrap();
        assert_ne!(
            importer::calculate_quick_hash(&path.to_string_lossy()).unwrap(),
            old_hash
        );
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 4,
        };

        let result = apply_to_resolved_image(
            resolve_image(&state, &locator, 12).unwrap(),
            &locator,
            12,
            &ImageAdjustmentRequest {
                brightness: 20,
                contrast: 0,
                gamma: 0,
            },
            &old_hash,
            false,
        );

        assert!(matches!(result, Err(AppError::BadRequest(_))));
        assert_eq!(std::fs::read(&path).unwrap(), external_bytes);
        let stored: String = library
            .directory_db
            .get_pool(4)
            .unwrap()
            .get()
            .unwrap()
            .query_row("SELECT file_hash FROM images WHERE id = 12", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored, old_hash);
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-3
    #[test]
    fn rollback_preserves_write_after_adjusted_file_was_installed() {
        let root = test_root("external-post-replace");
        std::fs::create_dir_all(&root).unwrap();
        let destination = root.join("source.png");
        let backup_path = root.join("backup.png");
        let adjusted =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])));
        adjusted.save(&destination).unwrap();
        let installed_hash =
            importer::calculate_quick_hash(&destination.to_string_lossy()).unwrap();
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([40, 50, 60])))
            .save(&backup_path)
            .unwrap();
        let mut backup = TempPath::new(backup_path);
        let external =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([10, 20, 30])));
        external.save(&destination).unwrap();
        let external_bytes = std::fs::read(&destination).unwrap();

        let result = restore_original_if_hash_matches(&mut backup, &destination, &installed_hash);

        assert!(result.is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), external_bytes);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-3
    #[test]
    fn apply_preserves_external_write_at_final_replace_boundary() {
        let root = test_root("external-final-replace");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let path = root.join("source.png");
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])))
            .save(&path)
            .unwrap();
        let old_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
        insert_image(&library, 4, 12, &path, &old_hash);
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 4,
        };
        let external =
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([10, 20, 30])));
        let result = apply_to_resolved_image_with_hook(
            resolve_image(&state, &locator, 12).unwrap(),
            &locator,
            12,
            &ImageAdjustmentRequest {
                brightness: 20,
                contrast: 0,
                gamma: 0,
            },
            &old_hash,
            false,
            || external.save(&path).unwrap(),
        );
        let external_bytes = {
            let expected = root.join("external.png");
            external.save(&expected).unwrap();
            std::fs::read(expected).unwrap()
        };

        assert!(matches!(result, Err(AppError::BadRequest(_))));
        assert_eq!(std::fs::read(&path).unwrap(), external_bytes);
        let stored: String = library
            .directory_db
            .get_pool(4)
            .unwrap()
            .get()
            .unwrap()
            .query_row("SELECT file_hash FROM images WHERE id = 12", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored, old_hash);
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-1
    #[test]
    fn preview_decodes_the_same_bytes_that_were_hash_validated() {
        let root = test_root("preview-hash-open-race");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("source.png");
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])))
            .save(&path)
            .unwrap();
        let expected_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();

        let snapshot = create_validated_snapshot(&path, &root, &expected_hash).unwrap();
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([10, 20, 30])))
            .save(&path)
            .unwrap();
        let decoded = image::open(&snapshot.path).unwrap();

        assert_eq!(decoded.to_rgb8().get_pixel(0, 0).0, [80, 90, 100]);
        assert_ne!(
            importer::calculate_quick_hash(&path.to_string_lossy()).unwrap(),
            expected_hash
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-1
    #[tokio::test]
    async fn preview_rejects_externally_changed_source_when_database_hash_is_stale() {
        let root = test_root("external-preview-change");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let path = root.join("source.png");
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])))
            .save(&path)
            .unwrap();
        let old_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
        insert_image(&library, 4, 12, &path, &old_hash);
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([10, 20, 30])))
            .save(&path)
            .unwrap();
        assert_ne!(
            importer::calculate_quick_hash(&path.to_string_lossy()).unwrap(),
            old_hash
        );
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 4,
        };

        let result = preview_adjust(
            State(state.clone()),
            AxumPath(12),
            Query(locator),
            Json(ImageAdjustmentRequest {
                brightness: 20,
                contrast: 0,
                gamma: 0,
            }),
        )
        .await;

        assert!(matches!(result, Err(AppError::BadRequest(_))));
        assert!(!library.data_dir.join("preview_cache").exists());
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-apply-decode
    // AC: @identity-safe-image-adjustments ac-4
    #[tokio::test]
    async fn png_and_jpeg_apply_urls_serve_decode_immediately_and_reject_old_hashes() {
        use axum::body::{to_bytes, Body};
        use axum::http::Request;
        use tower::ServiceExt;

        for (extension, format) in [
            ("png", image::ImageFormat::Png),
            ("jpg", image::ImageFormat::Jpeg),
        ] {
            let root = test_root(&format!("apply-decode-{extension}"));
            let state = AppState::new(&root, 0).unwrap();
            let library = state.resolve_library(None).unwrap();
            let path = root.join(format!("source.{extension}"));
            DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([40, 50, 60])))
                .save_with_format(&path, format)
                .unwrap();
            let old_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
            insert_image(&library, 4, 12, &path, &old_hash);
            let locator = ImageLocatorQuery {
                library_id: library.uuid.clone(),
                directory_id: 4,
            };
            let response = apply_to_resolved_image(
                resolve_image(&state, &locator, 12).unwrap(),
                &locator,
                12,
                &ImageAdjustmentRequest {
                    brightness: 50,
                    contrast: 0,
                    gamma: 0,
                },
                &old_hash,
                false,
            )
            .unwrap();
            let new_hash = response["file_hash"].as_str().unwrap();
            let returned_url = response["url"].as_str().unwrap();
            assert_ne!(new_hash, old_hash);
            assert_eq!(
                response["filename"],
                format!("{}.{}", &new_hash[..16], extension)
            );
            assert!(returned_url.contains(&format!("file_hash={new_hash}")));

            let route_url = returned_url.strip_prefix("/api/images").unwrap();
            for _ in 0..2 {
                let served = crate::routes::images::router()
                    .with_state(state.clone())
                    .oneshot(
                        Request::builder()
                            .uri(route_url)
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(served.status(), StatusCode::OK);
                let bytes = to_bytes(served.into_body(), usize::MAX).await.unwrap();
                let decoded = image::load_from_memory(&bytes).unwrap();
                assert_eq!((decoded.width(), decoded.height()), (4, 6));
                assert!(decoded.to_rgb8().get_pixel(0, 0).0[0] > 40);
            }

            let stale_url = route_url.replace(new_hash, &old_hash);
            let stale = crate::routes::images::router()
                .with_state(state.clone())
                .oneshot(
                    Request::builder()
                        .uri(stale_url)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(stale.status(), StatusCode::NOT_FOUND);

            drop(state);
            let _ = std::fs::remove_dir_all(root);
        }
    }

    // AC: @identity-safe-image-adjustments ac-4
    #[test]
    fn successful_apply_refreshes_metadata_and_stale_hash_cannot_apply_again() {
        let root = test_root("metadata");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let path = root.join("source.png");
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(4, 6, image::Rgb([80, 90, 100])))
            .save(&path)
            .unwrap();
        let old_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
        insert_image(&library, 4, 12, &path, &old_hash);
        let shared_path = root.join("shared.png");
        std::fs::copy(&path, &shared_path).unwrap();
        insert_image(&library, 5, 99, &shared_path, &old_hash);
        let shared_thumbnail = library.thumbnails_dir().join(format!("{}.webp", old_hash));
        std::fs::create_dir_all(library.thumbnails_dir()).unwrap();
        std::fs::write(&shared_thumbnail, b"shared-old-thumbnail").unwrap();
        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 4,
        };
        let response = apply_to_resolved_image(
            resolve_image(&state, &locator, 12).unwrap(),
            &locator,
            12,
            &ImageAdjustmentRequest {
                brightness: 20,
                contrast: 0,
                gamma: 0,
            },
            &old_hash,
            false,
        )
        .unwrap();
        let new_hash = importer::calculate_quick_hash(&path.to_string_lossy()).unwrap();
        assert_eq!(response["file_hash"], new_hash);
        assert!(response["thumbnail_url"]
            .as_str()
            .unwrap()
            .contains(&format!("file_hash={}", new_hash)));
        let stale = resolve_image(&state, &locator, 12).unwrap();
        assert!(ensure_expected_hash(&stale, &old_hash).is_err());
        assert!(library
            .thumbnails_dir()
            .join(format!("{}.webp", new_hash))
            .is_file());
        assert_eq!(
            std::fs::read(shared_thumbnail).unwrap(),
            b"shared-old-thumbnail"
        );
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-4
    #[test]
    fn rgba_adjustments_preserve_alpha_channel() {
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            2,
            1,
            image::Rgba([80, 90, 100, 37]),
        ));
        let adjusted = apply_adjustments_to_image(
            &image,
            &ImageAdjustmentRequest {
                brightness: 20,
                contrast: 0,
                gamma: 0,
            },
        )
        .to_rgba8();
        assert_eq!(adjusted.get_pixel(0, 0).0[3], 37);
        assert_eq!(adjusted.get_pixel(1, 0).0[3], 37);
    }
}
