use std::path::PathBuf;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{Json, Response};
use image::{DynamicImage, ImageBuffer, Rgb};
use rusqlite::params;
use serde::Deserialize;
use serde_json::json;
use tokio::fs::File;

use crate::server::error::AppError;
use crate::server::state::AppState;

use super::helpers::find_image_directory;

#[derive(Debug, Clone, Deserialize)]
pub struct ImageAdjustmentRequest {
    #[serde(default)]
    pub brightness: i32,
    #[serde(default)]
    pub contrast: i32,
    #[serde(default)]
    pub gamma: i32,
}

/// POST /api/images/:image_id/preview-adjust — Generate adjustment preview.
pub async fn preview_adjust(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Json(adjustments): Json<ImageAdjustmentRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    validate_adjustments(&adjustments)?;

    let state_clone = state.clone();

    let file_path = tokio::task::spawn_blocking(move || {
        find_image_file_path(&state_clone, image_id)
    })
    .await??;

    let data_dir = state.data_dir().to_path_buf();
    let brightness = adjustments.brightness;
    let contrast = adjustments.contrast;
    let gamma = adjustments.gamma;

    // Generate preview in blocking task (image processing is CPU-bound)
    tokio::task::spawn_blocking(move || {
        let preview_cache_dir = data_dir.join("preview_cache");
        std::fs::create_dir_all(&preview_cache_dir)?;

        // Clean up existing previews for this image
        if let Ok(entries) = std::fs::read_dir(&preview_cache_dir) {
            let prefix = format!("{}_", image_id);
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(&prefix) {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }

        // Generate unique filename
        let adj_hash = format!(
            "{:x}",
            xxhash_rust::xxh64::xxh64(
                format!("{}_{}_{}", adjustments.brightness, adjustments.contrast, adjustments.gamma)
                    .as_bytes(),
                0,
            )
        );
        let preview_filename = format!("{}_{}.webp", image_id, &adj_hash[..8]);
        let preview_path = preview_cache_dir.join(&preview_filename);

        let img = image::open(&file_path)
            .map_err(|e| AppError::Internal(format!("Failed to open image: {}", e)))?;

        let adjusted = apply_adjustments_to_image(&img, &adjustments);

        adjusted
            .save_with_format(&preview_path, image::ImageFormat::WebP)
            .map_err(|e| AppError::Internal(format!("Failed to save preview: {}", e)))?;

        Ok::<_, AppError>(())
    })
    .await??;

    Ok(Json(json!({
        "preview_url": format!("/api/images/{}/preview", image_id),
        "adjustments": {
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma
        }
    })))
}

/// GET /api/images/:image_id/preview — Serve cached preview.
pub async fn get_preview(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
) -> Result<Response, AppError> {
    let preview_cache_dir = state.data_dir().join("preview_cache");
    let prefix = format!("{}_", image_id);

    // Find preview file
    let preview_path = if let Ok(entries) = std::fs::read_dir(&preview_cache_dir) {
        entries
            .flatten()
            .find(|e| {
                e.file_name()
                    .to_str()
                    .map(|n| n.starts_with(&prefix) && n.ends_with(".webp"))
                    .unwrap_or(false)
            })
            .map(|e| e.path())
    } else {
        None
    };

    match preview_path {
        Some(path) => {
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
        None => Err(AppError::NotFound("No preview found for this image".into())),
    }
}

/// DELETE /api/images/:image_id/preview — Discard cached preview.
pub async fn discard_preview(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
) -> Result<Json<serde_json::Value>, AppError> {
    let preview_cache_dir = state.data_dir().join("preview_cache");
    let prefix = format!("{}_", image_id);
    let mut deleted = 0;

    if let Ok(entries) = std::fs::read_dir(&preview_cache_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&prefix) {
                    if std::fs::remove_file(entry.path()).is_ok() {
                        deleted += 1;
                    }
                }
            }
        }
    }

    Ok(Json(json!({"deleted": deleted})))
}

/// POST /api/images/:image_id/adjust — Apply adjustments to the original file.
pub async fn apply_adjust(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
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

    let (file_path, file_hash) = tokio::task::spawn_blocking(move || {
        let path = find_image_file_path(&state_clone, image_id)?;
        let hash = find_image_file_hash(&state_clone, image_id)?;
        Ok::<_, AppError>((path, hash))
    })
    .await??;

    let adj_brightness = adjustments.brightness;
    let adj_contrast = adjustments.contrast;
    let adj_gamma = adjustments.gamma;

    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let editable = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"];
    if !editable.contains(&ext.as_str()) {
        return Err(AppError::BadRequest(format!(
            "Cannot adjust .{} files",
            ext
        )));
    }

    let thumbnails_dir = state.thumbnails_dir();

    tokio::task::spawn_blocking(move || {
        let img = image::open(&file_path)
            .map_err(|e| AppError::Internal(format!("Failed to open image: {}", e)))?;

        let adjusted = apply_adjustments_to_image(&img, &adjustments);

        // Save back to the same file
        adjusted
            .save(&file_path)
            .map_err(|e| AppError::Internal(format!("Failed to save adjusted image: {}", e)))?;

        // Regenerate thumbnail
        let thumb_name = format!("{}.webp", &file_hash[..16.min(file_hash.len())]);
        let thumb_path = thumbnails_dir.join(&thumb_name);
        if thumb_path.exists() {
            let _ = std::fs::remove_file(&thumb_path);
        }
        // Thumbnail regeneration will happen on next request

        Ok::<_, AppError>(())
    })
    .await??;

    Ok(Json(json!({
        "adjusted": true,
        "brightness": adj_brightness,
        "contrast": adj_contrast,
        "gamma": adj_gamma
    })))
}

// ─── Internal helpers ───────────────────────────────────────────────────────

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

fn find_image_file_path(state: &AppState, image_id: i64) -> Result<PathBuf, AppError> {
    // Search directory DBs
    if let Some(dir_id) = find_image_directory(state.directory_db(), image_id, None) {
        let dir_pool = state.directory_db().get_pool(dir_id)?;
        let dir_conn = dir_pool.get()?;
        if let Ok(path) = dir_conn.query_row(
            "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            let p = PathBuf::from(&path);
            if p.exists() {
                return Ok(p);
            }
        }
    }

    // Legacy main DB
    let main_conn = state.main_db().get()?;
    match main_conn.query_row(
        "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
        params![image_id],
        |row| row.get::<_, String>(0),
    ) {
        Ok(path) => {
            let p = PathBuf::from(&path);
            if p.exists() {
                return Ok(p);
            }
            Err(AppError::NotFound("Image file not found on disk".into()))
        }
        Err(_) => Err(AppError::NotFound("Image file not found".into())),
    }
}

fn find_image_file_hash(state: &AppState, image_id: i64) -> Result<String, AppError> {
    if let Some(dir_id) = find_image_directory(state.directory_db(), image_id, None) {
        let dir_pool = state.directory_db().get_pool(dir_id)?;
        let dir_conn = dir_pool.get()?;
        if let Ok(hash) = dir_conn.query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            return Ok(hash);
        }
    }

    let main_conn = state.main_db().get()?;
    main_conn
        .query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        )
        .map_err(|_| AppError::NotFound("Image not found".into()))
}

/// Apply brightness, contrast, and gamma adjustments to an image.
/// Ports the Python numpy/PIL implementation.
fn apply_adjustments_to_image(
    img: &DynamicImage,
    adj: &ImageAdjustmentRequest,
) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut buffer: Vec<f32> = rgb.as_raw().iter().map(|&v| v as f32).collect();

    // Brightness: multiplicative (matches CSS brightness filter)
    // slider -100 to +100 maps to 0.0 to 2.0 multiplier
    if adj.brightness != 0 {
        let factor = (1.0 + adj.brightness as f32 / 100.0).max(0.0);
        for v in &mut buffer {
            *v *= factor;
        }
    }

    // Contrast: ((value - 127) * (contrast + 100) / 100) + 127
    if adj.contrast != 0 {
        let factor = (adj.contrast as f32 + 100.0) / 100.0;
        for v in &mut buffer {
            *v = ((*v - 127.0) * factor) + 127.0;
        }
    }

    // Gamma: exponential mapping
    // slider -100 to +100 maps to exponent 3.0 to 0.33
    if adj.gamma != 0 {
        let exponent = 3.0_f32.powf(-adj.gamma as f32 / 100.0);
        for v in &mut buffer {
            *v = v.clamp(0.0, 255.0);
            *v = (*v / 255.0).powf(exponent) * 255.0;
        }
    }

    // Apply dithering to reduce banding
    let dither_strength = if adj.gamma != 0 {
        0.5 + (adj.gamma.unsigned_abs() as f32 / 100.0) * 0.5
    } else {
        0.5
    };

    // Simple deterministic dithering (ordered dither pattern instead of random for reproducibility)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    for (i, v) in buffer.iter_mut().enumerate() {
        // Use pixel index as seed for pseudo-random dithering
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let hash = hasher.finish();
        let noise = ((hash % 1000) as f32 / 500.0 - 1.0) * dither_strength;
        *v += noise;
    }

    // Clamp and convert back
    let pixels: Vec<u8> = buffer.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect();

    let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, pixels).unwrap();

    DynamicImage::ImageRgb8(img_buf)
}
