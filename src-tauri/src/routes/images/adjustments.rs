use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Json, Response};
use image::{DynamicImage, ImageBuffer, Rgb};
use rusqlite::params;
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
pub struct ImageLocatorQuery {
    pub library_id: String,
    pub directory_id: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreviewQuery {
    pub library_id: String,
    pub directory_id: i64,
    pub adjustment_hash: String,
}

struct ResolvedImage {
    library: Arc<LibraryContext>,
    path: PathBuf,
    file_hash: String,
}

/// POST /api/images/:image_id/preview-adjust — Generate adjustment preview.
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
    let cache_dir = resolved.library.data_dir.join("preview_cache");
    let file_path = resolved.path;
    let hash = adjustment_hash(&adjustments);
    let preview_filename = preview_cache_filename(&canonical_locator, image_id, &hash);
    let preview_url = preview_url(&canonical_locator, image_id, &hash);
    let brightness = adjustments.brightness;
    let contrast = adjustments.contrast;
    let gamma = adjustments.gamma;

    tokio::task::spawn_blocking(move || {
        std::fs::create_dir_all(&cache_dir)?;
        remove_locator_previews(&cache_dir, &canonical_locator, image_id)?;

        let img = image::open(&file_path)
            .map_err(|e| AppError::Internal(format!("Failed to open image: {}", e)))?;
        let adjusted = apply_adjustments_to_image(&img, &adjustments);
        adjusted
            .save_with_format(cache_dir.join(preview_filename), image::ImageFormat::WebP)
            .map_err(|e| AppError::Internal(format!("Failed to save preview: {}", e)))?;
        Ok::<_, AppError>(())
    })
    .await??;

    Ok(Json(json!({
        "preview_url": preview_url,
        "adjustment_hash": hash,
        "adjustments": {
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma
        }
    })))
}

/// GET /api/images/:image_id/preview — Serve one exact cached preview.
pub async fn get_preview(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(query): Query<PreviewQuery>,
) -> Result<Response, AppError> {
    validate_adjustment_hash(&query.adjustment_hash)?;
    let locator = ImageLocatorQuery {
        library_id: query.library_id,
        directory_id: query.directory_id,
    };
    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let resolved =
        tokio::task::spawn_blocking(move || resolve_image(&state_clone, &locator_clone, image_id))
            .await??;
    let canonical_locator = ImageLocatorQuery {
        library_id: resolved.library.uuid.clone(),
        directory_id: locator.directory_id,
    };
    let path = resolved
        .library
        .data_dir
        .join("preview_cache")
        .join(preview_cache_filename(
            &canonical_locator,
            image_id,
            &query.adjustment_hash,
        ));

    if !path.is_file() {
        return Err(AppError::NotFound("No matching preview found".into()));
    }

    let file = File::open(&path).await?;
    let metadata = file.metadata().await?;
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "image/webp")
        .header("Content-Length", metadata.len())
        .body(body)
        .unwrap())
}

/// DELETE /api/images/:image_id/preview — Discard previews for one exact locator.
pub async fn discard_preview(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let resolved =
        tokio::task::spawn_blocking(move || resolve_image(&state_clone, &locator_clone, image_id))
            .await??;
    let canonical_locator = ImageLocatorQuery {
        library_id: resolved.library.uuid.clone(),
        directory_id: locator.directory_id,
    };
    let cache_dir = resolved.library.data_dir.join("preview_cache");
    let deleted = tokio::task::spawn_blocking(move || {
        remove_locator_previews(&cache_dir, &canonical_locator, image_id)
    })
    .await??;

    Ok(Json(json!({"deleted": deleted})))
}

/// POST /api/images/:image_id/adjust — Apply adjustments to the original file.
pub async fn apply_adjust(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
    Json(adjustments): Json<ImageAdjustmentRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validate_adjustments(&adjustments)?;

    if adjustments.brightness == 0 && adjustments.contrast == 0 && adjustments.gamma == 0 {
        return Ok(Json(json!({
            "adjusted": false,
            "message": "No adjustments needed"
        })));
    }

    let state_clone = state.clone();
    let locator_clone = locator.clone();
    let result = tokio::task::spawn_blocking(move || {
        let resolved = resolve_image(&state_clone, &locator_clone, image_id)?;
        apply_to_resolved_image(resolved, &locator_clone, image_id, &adjustments)
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
    let row = conn.query_row(
        "SELECT f.original_path, i.file_hash
         FROM images i
         JOIN image_files f ON f.image_id = i.id
         WHERE i.id = ?1 AND f.file_exists = 1
         ORDER BY f.id LIMIT 1",
        params![image_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    );
    let (path, file_hash) = match row {
        Ok(value) => value,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return Err(AppError::NotFound(format!(
                "Image {} not found in library '{}' directory {}",
                image_id, library.uuid, locator.directory_id
            )))
        }
        Err(error) => return Err(error.into()),
    };
    let path = PathBuf::from(path);
    if !path.is_file() {
        return Err(AppError::NotFound("Image file not found on disk".into()));
    }

    drop(conn);
    Ok(ResolvedImage {
        library,
        path,
        file_hash,
    })
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
    adjustment_hash: &str,
) -> String {
    format!(
        "{}{}.webp",
        preview_cache_prefix(locator, image_id),
        adjustment_hash
    )
}

fn preview_url(locator: &ImageLocatorQuery, image_id: i64, adjustment_hash: &str) -> String {
    format!(
        "/api/images/{}/preview?library_id={}&directory_id={}&adjustment_hash={}",
        image_id, locator.library_id, locator.directory_id, adjustment_hash
    )
}

fn validate_adjustment_hash(hash: &str) -> Result<(), AppError> {
    if hash.len() != 16 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AppError::BadRequest("Invalid adjustment hash".into()));
    }
    Ok(())
}

fn remove_locator_previews(
    cache_dir: &Path,
    locator: &ImageLocatorQuery,
    image_id: i64,
) -> Result<usize, AppError> {
    let prefix = preview_cache_prefix(locator, image_id);
    let mut deleted = 0;
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error.into()),
    };

    for entry in entries.flatten() {
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".webp"))
        {
            std::fs::remove_file(entry.path())?;
            deleted += 1;
        }
    }
    Ok(deleted)
}

fn apply_to_resolved_image(
    resolved: ResolvedImage,
    locator: &ImageLocatorQuery,
    image_id: i64,
    adjustments: &ImageAdjustmentRequest,
) -> Result<serde_json::Value, AppError> {
    let ext = resolved
        .path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_lowercase();
    let editable = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"];
    if !editable.contains(&ext.as_str()) {
        return Err(AppError::BadRequest(format!(
            "Cannot adjust .{} files",
            ext
        )));
    }

    let img = image::open(&resolved.path)
        .map_err(|e| AppError::Internal(format!("Failed to open image: {}", e)))?;
    let adjusted = apply_adjustments_to_image(&img, adjustments);
    let temporary = adjustment_temp_path(&resolved.path, &ext);
    adjusted
        .save(&temporary)
        .map_err(|e| AppError::Internal(format!("Failed to save adjusted image: {}", e)))?;
    let original_permissions = std::fs::metadata(&resolved.path)?.permissions();
    std::fs::set_permissions(&temporary, original_permissions)?;

    let temporary_string = temporary.to_string_lossy();
    let new_hash = importer::calculate_quick_hash(&temporary_string)
        .map_err(|e| AppError::Internal(format!("Hash error: {}", e)))?;
    let (width, height) = importer::get_image_dimensions(&temporary_string)
        .ok_or_else(|| AppError::Internal("Failed to read adjusted image dimensions".into()))?;
    let perceptual_hash = importer::calculate_perceptual_hash(&temporary_string);
    let metadata = std::fs::metadata(&temporary)?;
    let file_size = metadata.len() as i64;
    let file_modified_at = modified_at_rfc3339(&metadata);
    if let Ok(file) = std::fs::OpenOptions::new().write(true).open(&temporary) {
        let _ = file.sync_all();
    }

    let pool = resolved
        .library
        .directory_db
        .get_pool(locator.directory_id)?;
    let mut conn = pool.get()?;
    let duplicate: Option<i64> = conn
        .query_row(
            "SELECT id FROM images WHERE file_hash = ?1 AND id != ?2",
            params![&new_hash, image_id],
            |row| row.get(0),
        )
        .ok();
    if duplicate.is_some() {
        let _ = std::fs::remove_file(&temporary);
        return Err(AppError::BadRequest(
            "Adjusted image duplicates another image in this directory".into(),
        ));
    }

    let thumbnails_dir = resolved.library.thumbnails_dir();
    std::fs::create_dir_all(&thumbnails_dir)?;
    let thumbnail_path = thumbnails_dir.join(format!("{}.webp", &new_hash[..16]));
    let thumbnail_temp = thumbnails_dir.join(format!(
        ".{}.{}-{}.tmp.webp",
        new_hash,
        std::process::id(),
        unique_suffix()
    ));
    if !importer::generate_thumbnail(&temporary_string, &thumbnail_temp.to_string_lossy(), 400) {
        let _ = std::fs::remove_file(&temporary);
        let _ = std::fs::remove_file(&thumbnail_temp);
        return Err(AppError::Internal(
            "Failed to generate adjusted image thumbnail".into(),
        ));
    }

    let canonical_filename = format!("{}.{}", &new_hash[..16], ext);
    let transaction = conn.transaction()?;
    transaction.execute(
        "UPDATE images
         SET filename = ?1, file_hash = ?2, perceptual_hash = ?3,
             width = ?4, height = ?5, file_size = ?6,
             file_modified_at = ?7, updated_at = datetime('now')
         WHERE id = ?8",
        params![
            canonical_filename,
            &new_hash,
            perceptual_hash,
            width as i32,
            height as i32,
            file_size,
            file_modified_at,
            image_id
        ],
    )?;

    if let Err(error) = commit_prepared_file(&thumbnail_temp, &thumbnail_path, replace_file) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = commit_prepared_file(&temporary, &resolved.path, replace_file) {
        let _ = std::fs::remove_file(&thumbnail_path);
        return Err(error);
    }
    transaction.commit()?;

    if resolved.file_hash != new_hash {
        let old_thumbnail = thumbnails_dir.join(format!(
            "{}.webp",
            &resolved.file_hash[..16.min(resolved.file_hash.len())]
        ));
        let _ = std::fs::remove_file(old_thumbnail);
    }

    let library_id = &resolved.library.uuid;
    Ok(json!({
        "adjusted": true,
        "brightness": adjustments.brightness,
        "contrast": adjustments.contrast,
        "gamma": adjustments.gamma,
        "file_hash": new_hash,
        "file_size": file_size,
        "width": width,
        "height": height,
        "file_modified_at": file_modified_at,
        "url": format!("/api/images/{}/file?directory_id={}&library_id={}", image_id, locator.directory_id, library_id),
        "thumbnail_url": format!("/api/images/{}/thumbnail?directory_id={}&library_id={}", image_id, locator.directory_id, library_id)
    }))
}

fn adjustment_temp_path(path: &Path, extension: &str) -> PathBuf {
    let name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    path.with_file_name(format!(
        ".{}.localbooru-adjust-{}-{}.{}",
        name,
        std::process::id(),
        unique_suffix(),
        extension
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

fn commit_prepared_file<F>(temporary: &Path, destination: &Path, replace: F) -> Result<(), AppError>
where
    F: FnOnce(&Path, &Path) -> std::io::Result<()>,
{
    if let Err(error) = replace(temporary, destination) {
        let _ = std::fs::remove_file(temporary);
        return Err(AppError::Internal(format!(
            "Failed to atomically replace file: {}",
            error
        )));
    }
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

fn validate_adjustments(adj: &ImageAdjustmentRequest) -> Result<(), AppError> {
    if !(-200..=200).contains(&adj.brightness) {
        return Err(AppError::BadRequest(
            "Brightness must be between -200 and +200".into(),
        ));
    }
    if !(-100..=100).contains(&adj.contrast) {
        return Err(AppError::BadRequest(
            "Contrast must be between -100 and +100".into(),
        ));
    }
    if !(-100..=100).contains(&adj.gamma) {
        return Err(AppError::BadRequest(
            "Gamma must be between -100 and +100".into(),
        ));
    }
    Ok(())
}

/// Apply brightness, contrast, and gamma adjustments to an image.
/// Ports the Python numpy/PIL implementation.
fn apply_adjustments_to_image(img: &DynamicImage, adj: &ImageAdjustmentRequest) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut buffer: Vec<f32> = rgb.as_raw().iter().map(|&v| v as f32).collect();

    if adj.brightness != 0 {
        let factor = (1.0 + adj.brightness as f32 / 100.0).max(0.0);
        for v in &mut buffer {
            *v *= factor;
        }
    }

    if adj.contrast != 0 {
        let factor = (adj.contrast as f32 + 100.0) / 100.0;
        for v in &mut buffer {
            *v = ((*v - 127.0) * factor) + 127.0;
        }
    }

    if adj.gamma != 0 {
        let exponent = 3.0_f32.powf(-adj.gamma as f32 / 100.0);
        for v in &mut buffer {
            *v = v.clamp(0.0, 255.0);
            *v = (*v / 255.0).powf(exponent) * 255.0;
        }
    }

    let dither_strength = if adj.gamma != 0 {
        0.5 + (adj.gamma.unsigned_abs() as f32 / 100.0) * 0.5
    } else {
        0.5
    };

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    for (i, v) in buffer.iter_mut().enumerate() {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();
        let noise = ((hash % 1000) as f32 / 500.0 - 1.0) * dither_strength;
        *v += noise;
    }

    let pixels: Vec<u8> = buffer.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();
    let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, pixels).unwrap();
    DynamicImage::ImageRgb8(img_buf)
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

    // AC: @identity-safe-image-adjustments ac-1
    #[test]
    fn exact_locator_resolves_duplicate_image_id_from_requested_directory() {
        let root = test_root("locator");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let first_path = root.join("first.png");
        let second_path = root.join("second.png");
        DynamicImage::new_rgb8(2, 2).save(&first_path).unwrap();
        DynamicImage::new_rgb8(3, 3).save(&second_path).unwrap();

        for (directory_id, path, hash) in [
            (1, &first_path, "first-hash"),
            (2, &second_path, "second-hash"),
        ] {
            let pool = library.directory_db.get_pool(directory_id).unwrap();
            let conn = pool.get().unwrap();
            conn.execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (9, 'image.png', ?1)",
                params![hash],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO image_files (image_id, original_path, file_extension) VALUES (9, ?1, 'png')",
                params![path.to_string_lossy()],
            )
            .unwrap();
        }

        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 2,
        };
        let resolved = resolve_image(&state, &locator, 9).unwrap();
        assert_eq!(resolved.path, second_path);
        assert_eq!(resolved.file_hash, "second-hash");
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-2
    #[test]
    fn preview_cache_file_is_exact_for_locator_and_adjustments() {
        let first = ImageLocatorQuery {
            library_id: "library-a".into(),
            directory_id: 1,
        };
        let second = ImageLocatorQuery {
            library_id: "library-a".into(),
            directory_id: 2,
        };
        let low = ImageAdjustmentRequest {
            brightness: 10,
            contrast: 0,
            gamma: 0,
        };
        let high = ImageAdjustmentRequest {
            brightness: 20,
            contrast: 0,
            gamma: 0,
        };

        assert_ne!(
            preview_cache_filename(&first, 9, &adjustment_hash(&low)),
            preview_cache_filename(&second, 9, &adjustment_hash(&low))
        );
        assert_ne!(
            preview_cache_filename(&first, 9, &adjustment_hash(&low)),
            preview_cache_filename(&first, 9, &adjustment_hash(&high))
        );
        assert!(preview_url(&first, 9, &adjustment_hash(&low))
            .contains("library_id=library-a&directory_id=1&adjustment_hash="));
    }

    // AC: @identity-safe-image-adjustments ac-4
    #[test]
    fn apply_refreshes_exact_database_metadata_hash_and_thumbnail() {
        let root = test_root("metadata");
        let state = AppState::new(&root, 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let image_path = root.join("source.png");
        DynamicImage::new_rgb8(4, 6).save(&image_path).unwrap();
        let old_hash = importer::calculate_quick_hash(&image_path.to_string_lossy()).unwrap();
        let pool = library.directory_db.get_pool(4).unwrap();
        let conn = pool.get().unwrap();
        conn.execute(
            "INSERT INTO images (id, filename, file_hash, width, height, file_size) VALUES (12, 'source.png', ?1, 1, 1, 1)",
            params![old_hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO image_files (image_id, original_path, file_extension) VALUES (12, ?1, 'png')",
            params![image_path.to_string_lossy()],
        )
        .unwrap();
        drop(conn);

        let locator = ImageLocatorQuery {
            library_id: library.uuid.clone(),
            directory_id: 4,
        };
        let resolved = resolve_image(&state, &locator, 12).unwrap();
        let response = apply_to_resolved_image(
            resolved,
            &locator,
            12,
            &ImageAdjustmentRequest {
                brightness: 20,
                contrast: 0,
                gamma: 0,
            },
        )
        .unwrap();

        let new_hash = importer::calculate_quick_hash(&image_path.to_string_lossy()).unwrap();
        let conn = pool.get().unwrap();
        let metadata: (String, i32, i32, i64, Option<String>) = conn
            .query_row(
                "SELECT file_hash, width, height, file_size, file_modified_at FROM images WHERE id = 12",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .unwrap();
        assert_eq!(metadata.0, new_hash);
        assert_eq!((metadata.1, metadata.2), (4, 6));
        assert_eq!(
            metadata.3,
            std::fs::metadata(&image_path).unwrap().len() as i64
        );
        assert!(metadata.4.is_some());
        assert!(library
            .thumbnails_dir()
            .join(format!("{}.webp", new_hash))
            .is_file());
        assert_eq!(
            response["url"],
            format!(
                "/api/images/12/file?directory_id=4&library_id={}",
                library.uuid
            )
        );
        assert_eq!(
            response["thumbnail_url"],
            format!(
                "/api/images/12/thumbnail?directory_id=4&library_id={}",
                library.uuid
            )
        );
        drop(conn);
        drop(state);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @identity-safe-image-adjustments ac-4
    #[test]
    fn failed_atomic_replace_preserves_original() {
        let root = test_root("atomic");
        std::fs::create_dir_all(&root).unwrap();
        let original = root.join("image.png");
        let temporary = root.join(".image.localbooru-adjust.png");
        std::fs::write(&original, b"original").unwrap();
        std::fs::write(&temporary, b"adjusted").unwrap();

        let result = commit_prepared_file(&temporary, &original, |_, _| {
            Err(std::io::Error::other("injected replacement failure"))
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(&original).unwrap(), b"original");
        assert!(!temporary.exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
