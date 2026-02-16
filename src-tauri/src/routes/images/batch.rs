use axum::extract::State;
use axum::response::Json;
use rusqlite::params;
use serde::Deserialize;
use serde_json::json;

use crate::server::error::AppError;
use crate::server::state::AppState;

use super::helpers::find_image_directory;

#[derive(Debug, Deserialize)]
pub struct BatchDeleteRequest {
    pub image_ids: Vec<i64>,
    #[serde(default)]
    pub delete_files: bool,
}

#[derive(Debug, Deserialize)]
pub struct BatchRetagRequest {
    pub image_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchAgeDetectRequest {
    pub image_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchMetadataExtractRequest {
    pub image_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchMoveRequest {
    pub image_ids: Vec<i64>,
    pub target_directory_id: i64,
}

/// POST /api/images/batch/delete — Delete multiple images.
pub async fn batch_delete(
    State(state): State<AppState>,
    Json(request): Json<BatchDeleteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let total_requested = request.image_ids.len();

    let (deleted, errors) = tokio::task::spawn_blocking(move || {
        let mut deleted = 0i64;
        let mut errors: Vec<serde_json::Value> = Vec::new();

        for image_id in &request.image_ids {
            match delete_single_image(&state_clone, *image_id, request.delete_files) {
                Ok(()) => deleted += 1,
                Err(e) => errors.push(json!({"id": image_id, "error": e.to_string()})),
            }
        }

        Ok::<_, AppError>((deleted, errors))
    })
    .await??;

    Ok(Json(json!({
        "deleted": deleted,
        "errors": errors,
        "total_requested": total_requested
    })))
}

fn delete_single_image(state: &AppState, image_id: i64, delete_files: bool) -> Result<(), AppError> {
    let mut file_hash: Option<String> = None;

    // Search directory DBs
    if let Some(dir_id) = find_image_directory(state.directory_db(), image_id, None) {
        let dir_pool = state.directory_db().get_pool(dir_id)?;
        let dir_conn = dir_pool.get()?;

        if let Ok(hash) = dir_conn.query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            file_hash = Some(hash);

            if delete_files {
                let mut stmt = dir_conn.prepare(
                    "SELECT original_path FROM image_files WHERE image_id = ?1",
                )?;
                let paths: Vec<String> = stmt
                    .query_map(params![image_id], |row| row.get(0))?
                    .filter_map(|r| r.ok())
                    .collect();
                for path in &paths {
                    let _ = std::fs::remove_file(path);
                }
            }

            dir_conn.execute("DELETE FROM image_tags WHERE image_id = ?1", params![image_id])?;
            dir_conn.execute("DELETE FROM image_files WHERE image_id = ?1", params![image_id])?;
            dir_conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
        }
    }

    // Try main DB if not found
    if file_hash.is_none() {
        let main_conn = state.main_db().get()?;
        match main_conn.query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(hash) => {
                file_hash = Some(hash);

                if delete_files {
                    let mut stmt = main_conn.prepare(
                        "SELECT original_path FROM image_files WHERE image_id = ?1",
                    )?;
                    let paths: Vec<String> = stmt
                        .query_map(params![image_id], |row| row.get(0))?
                        .filter_map(|r| r.ok())
                        .collect();
                    for path in &paths {
                        let _ = std::fs::remove_file(path);
                    }
                }

                main_conn.execute("DELETE FROM image_tags WHERE image_id = ?1", params![image_id])?;
                main_conn.execute("DELETE FROM image_files WHERE image_id = ?1", params![image_id])?;
                main_conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
            }
            Err(_) => return Err(AppError::NotFound("Image not found".into())),
        }
    }

    // Delete thumbnail
    if let Some(ref hash) = file_hash {
        let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
        let thumb_path = state.thumbnails_dir().join(&thumb_name);
        if thumb_path.exists() {
            let _ = std::fs::remove_file(&thumb_path);
        }

        // Delete video preview frames
        crate::services::video_preview::delete_preview_frames(state.data_dir(), hash);
    }

    Ok(())
}

/// POST /api/images/batch/retag — Queue retagging (stub — needs task queue from Phase 3).
pub async fn batch_retag(
    Json(request): Json<BatchRetagRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // TODO: Implement when task queue service is available (Phase 3)
    Ok(Json(json!({
        "queued": 0,
        "errors": [{"error": "Task queue not yet implemented"}],
        "total_requested": request.image_ids.len()
    })))
}

/// POST /api/images/batch/age-detect — Queue age detection (stub — needs addon system).
pub async fn batch_age_detect(
    Json(request): Json<BatchAgeDetectRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // TODO: Implement when addon system is available (Phase 5)
    Ok(Json(json!({
        "queued": 0,
        "errors": [{"error": "Age detection addon not available"}],
        "total_requested": request.image_ids.len()
    })))
}

/// POST /api/images/batch/extract-metadata — Queue metadata extraction for images.
pub async fn batch_extract_metadata(
    State(state): State<AppState>,
    Json(request): Json<BatchMetadataExtractRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    use crate::services::task_queue;

    let state_clone = state.clone();
    let total_requested = request.image_ids.len();

    let (queued, errors) = tokio::task::spawn_blocking(move || {
        let mut queued = 0i64;
        let mut errors: Vec<serde_json::Value> = Vec::new();

        for image_id in &request.image_ids {
            // Find the directory and file path for this image
            let (dir_id, image_path) =
                match find_image_path_for_metadata(&state_clone, *image_id) {
                    Ok(info) => info,
                    Err(e) => {
                        errors.push(json!({"id": image_id, "error": e.to_string()}));
                        continue;
                    }
                };

            let payload = json!({
                "image_id": image_id,
                "directory_id": dir_id,
                "image_path": image_path,
            });

            match task_queue::enqueue_task(
                &state_clone,
                task_queue::TASK_EXTRACT_METADATA,
                &payload,
                0,
                Some(*image_id),
            ) {
                Ok(Some(_)) => queued += 1,
                Ok(None) => {} // Duplicate, skip
                Err(e) => {
                    errors.push(json!({"id": image_id, "error": e.to_string()}));
                }
            }
        }

        Ok::<_, AppError>((queued, errors))
    })
    .await??;

    Ok(Json(json!({
        "queued": queued,
        "errors": errors,
        "total_requested": total_requested
    })))
}

/// Look up the directory ID and file path for an image (used by batch metadata).
fn find_image_path_for_metadata(
    state: &AppState,
    image_id: i64,
) -> Result<(Option<i64>, String), AppError> {
    // Search directory DBs first
    if let Some(dir_id) = find_image_directory(state.directory_db(), image_id, None) {
        let dir_pool = state.directory_db().get_pool(dir_id)?;
        let dir_conn = dir_pool.get()?;

        let path: String = dir_conn
            .query_row(
                "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
                params![image_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("No file path found for image".into()))?;

        return Ok((Some(dir_id), path));
    }

    // Try main DB
    let main_conn = state.main_db().get()?;
    let path: String = main_conn
        .query_row(
            "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
            params![image_id],
            |row| row.get(0),
        )
        .map_err(|_| AppError::NotFound("Image not found".into()))?;

    Ok((None, path))
}

/// POST /api/images/batch/move — Move images to a different directory.
pub async fn batch_move(
    State(state): State<AppState>,
    Json(request): Json<BatchMoveRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let total_requested = request.image_ids.len();

    let (moved, errors) = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Get target directory path
        let target_path: String = main_conn
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![request.target_directory_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Target directory not found".into()))?;

        if !std::path::Path::new(&target_path).is_dir() {
            return Err(AppError::BadRequest(
                "Target directory path does not exist on disk".into(),
            ));
        }

        let mut moved = 0i64;
        let mut errors: Vec<serde_json::Value> = Vec::new();

        for image_id in &request.image_ids {
            match move_single_image(&state_clone, *image_id, &target_path) {
                Ok(()) => moved += 1,
                Err(e) => errors.push(json!({"id": image_id, "error": e.to_string()})),
            }
        }

        Ok::<_, AppError>((moved, errors))
    })
    .await??;

    Ok(Json(json!({
        "moved": moved,
        "errors": errors,
        "total_requested": total_requested
    })))
}

fn move_single_image(state: &AppState, image_id: i64, target_path: &str) -> Result<(), AppError> {
    // Find the image in directory DBs
    if let Some(dir_id) = find_image_directory(state.directory_db(), image_id, None) {
        let dir_pool = state.directory_db().get_pool(dir_id)?;
        let dir_conn = dir_pool.get()?;

        let mut stmt = dir_conn.prepare(
            "SELECT id, original_path FROM image_files WHERE image_id = ?1",
        )?;
        let files: Vec<(i64, String)> = stmt
            .query_map(params![image_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        for (file_id, original_path) in &files {
            let src = std::path::Path::new(original_path);
            if !src.exists() {
                continue;
            }

            let filename = src.file_name().unwrap_or_default().to_string_lossy();
            let mut new_path = std::path::PathBuf::from(target_path).join(filename.as_ref());

            // Handle conflicts
            if new_path.exists() {
                let stem = src
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let ext = src
                    .extension()
                    .map(|e| format!(".{}", e.to_string_lossy()))
                    .unwrap_or_default();
                let mut counter = 1;
                while new_path.exists() {
                    new_path = std::path::PathBuf::from(target_path)
                        .join(format!("{}_{}{}", stem, counter, ext));
                    counter += 1;
                }
            }

            std::fs::rename(src, &new_path)
                .or_else(|_| {
                    // rename fails across filesystems, fall back to copy+delete
                    std::fs::copy(src, &new_path)?;
                    std::fs::remove_file(src)
                })
                .map_err(|e| AppError::Internal(format!("Failed to move file: {}", e)))?;

            // Update path in DB
            dir_conn.execute(
                "UPDATE image_files SET original_path = ?1 WHERE id = ?2",
                params![new_path.to_string_lossy().to_string(), file_id],
            )?;
        }

        return Ok(());
    }

    Err(AppError::NotFound("Image not found".into()))
}
