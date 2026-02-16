use std::path::{Path, PathBuf};

use axum::body::Body;
use axum::extract::{Multipart, Path as AxumPath, Query, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use rusqlite::params;
use serde::Deserialize;
use serde_json::json;
use tokio::fs::File;
use tokio_util::io::ReaderStream;
use tower::ServiceExt;
use tower_http::services::ServeFile;

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::importer;
use crate::services::video_preview;

use super::helpers::{find_image_directory, get_image_tags_from_directory};

#[derive(Debug, Deserialize)]
pub struct DirectoryQuery {
    pub directory_id: Option<i64>,
}

/// GET /api/images/media/file-info — Get file size info.
pub async fn get_file_info(
    State(state): State<AppState>,
    Query(q): Query<FileInfoQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let path = PathBuf::from(&q.path);
    let resolved = path
        .canonicalize()
        .map_err(|_| AppError::BadRequest("Invalid file path".into()))?;

    // Validate path is within a known watch directory
    let state_clone = state.clone();
    let resolved_clone = resolved.clone();
    let allowed = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let mut stmt = conn.prepare("SELECT path FROM watch_directories")?;
        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        for wd_path in &paths {
            let wd = PathBuf::from(wd_path);
            if let Ok(wd_resolved) = wd.canonicalize() {
                if resolved_clone.starts_with(&wd_resolved) {
                    return Ok::<bool, AppError>(true);
                }
            }
        }
        Ok(false)
    })
    .await??;

    if !allowed {
        return Err(AppError::Forbidden(
            "Path is not within a watched directory".into(),
        ));
    }

    if !resolved.exists() {
        return Err(AppError::NotFound("File not found".into()));
    }
    if !resolved.is_file() {
        return Err(AppError::BadRequest("Path is not a file".into()));
    }

    let metadata = tokio::fs::metadata(&resolved).await?;
    Ok(Json(json!({
        "size": metadata.len(),
        "path": resolved.to_string_lossy()
    })))
}

#[derive(Debug, Deserialize)]
pub struct FileInfoQuery {
    pub path: String,
}

/// GET /api/images/:image_id — Get single image details.
pub async fn get_image(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<DirectoryQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let directory_id = q.directory_id;

    let result = tokio::task::spawn_blocking(move || {
        // Try directory DB first
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;

                let image = dir_conn.query_row(
                    "SELECT id, filename, original_filename, file_hash, width, height,
                            file_size, duration, rating, is_favorite, prompt, negative_prompt,
                            model_name, sampler, seed, steps, cfg_scale, source_url,
                            num_faces, min_detected_age, max_detected_age, detected_ages,
                            age_detection_data, view_count, created_at, import_source
                     FROM images WHERE id = ?1",
                    params![image_id],
                    |row| {
                        Ok(json!({
                            "id": row.get::<_, i64>(0)?,
                            "filename": row.get::<_, String>(1)?,
                            "original_filename": row.get::<_, Option<String>>(2)?,
                            "file_hash": row.get::<_, String>(3)?,
                            "width": row.get::<_, Option<i32>>(4)?,
                            "height": row.get::<_, Option<i32>>(5)?,
                            "file_size": row.get::<_, Option<i64>>(6)?,
                            "rating": row.get::<_, String>(8)?,
                            "is_favorite": row.get::<_, bool>(9)?,
                            "view_count": row.get::<_, i32>(23)?,
                            "prompt": row.get::<_, Option<String>>(10)?,
                            "negative_prompt": row.get::<_, Option<String>>(11)?,
                            "model_name": row.get::<_, Option<String>>(12)?,
                            "seed": row.get::<_, Option<String>>(14)?,
                            "source_url": row.get::<_, Option<String>>(17)?,
                            "import_source": row.get::<_, Option<String>>(25)?,
                            "num_faces": row.get::<_, Option<i32>>(18)?,
                            "min_age": row.get::<_, Option<i32>>(19)?,
                            "max_age": row.get::<_, Option<i32>>(20)?,
                            "detected_ages": row.get::<_, Option<String>>(21)?,
                            "age_detection_data": row.get::<_, Option<String>>(22)?,
                            "created_at": row.get::<_, Option<String>>(24)?,
                        }))
                    },
                );

                if let Ok(mut data) = image {
                    // Increment view count
                    let _ = dir_conn.execute(
                        "UPDATE images SET view_count = view_count + 1 WHERE id = ?1",
                        params![image_id],
                    );

                    // Get tags
                    let tags = get_image_tags_from_directory(
                        &dir_pool,
                        state_clone.main_db(),
                        image_id,
                    )?;
                    data["tags"] = json!(tags);

                    // Get files
                    let mut stmt = dir_conn.prepare(
                        "SELECT original_path, file_exists, file_status FROM image_files WHERE image_id = ?1",
                    )?;
                    let files: Vec<serde_json::Value> = stmt
                        .query_map(params![image_id], |row| {
                            Ok(json!({
                                "path": row.get::<_, String>(0)?,
                                "exists": row.get::<_, bool>(1)?,
                                "status": row.get::<_, String>(2)?
                            }))
                        })?
                        .filter_map(|r| r.ok())
                        .collect();

                    data["files"] = json!(files);
                    data["file_status"] = if let Some(first) = files.first() {
                        first["status"].clone()
                    } else {
                        json!("unknown")
                    };
                    data["thumbnail_url"] =
                        json!(format!("/api/images/{}/thumbnail?directory_id={}", image_id, dir_id));
                    data["url"] =
                        json!(format!("/api/images/{}/file?directory_id={}", image_id, dir_id));

                    return Ok(data);
                }
            }
        }

        // Try to find in any directory DB
        if let Some(found_dir) = find_image_directory(state_clone.directory_db(), image_id, None) {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;

            if let Ok(mut data) = dir_conn.query_row(
                "SELECT id, filename, original_filename, file_hash, width, height,
                        file_size, duration, rating, is_favorite, prompt, negative_prompt,
                        model_name, sampler, seed, steps, cfg_scale, source_url,
                        num_faces, min_detected_age, max_detected_age, detected_ages,
                        age_detection_data, view_count, created_at, import_source
                 FROM images WHERE id = ?1",
                params![image_id],
                |row| {
                    Ok(json!({
                        "id": row.get::<_, i64>(0)?,
                        "filename": row.get::<_, String>(1)?,
                        "original_filename": row.get::<_, Option<String>>(2)?,
                        "file_hash": row.get::<_, String>(3)?,
                        "width": row.get::<_, Option<i32>>(4)?,
                        "height": row.get::<_, Option<i32>>(5)?,
                        "file_size": row.get::<_, Option<i64>>(6)?,
                        "rating": row.get::<_, String>(8)?,
                        "is_favorite": row.get::<_, bool>(9)?,
                        "view_count": row.get::<_, i32>(23)?,
                        "prompt": row.get::<_, Option<String>>(10)?,
                        "negative_prompt": row.get::<_, Option<String>>(11)?,
                        "model_name": row.get::<_, Option<String>>(12)?,
                        "seed": row.get::<_, Option<String>>(14)?,
                        "source_url": row.get::<_, Option<String>>(17)?,
                        "import_source": row.get::<_, Option<String>>(25)?,
                        "num_faces": row.get::<_, Option<i32>>(18)?,
                        "min_age": row.get::<_, Option<i32>>(19)?,
                        "max_age": row.get::<_, Option<i32>>(20)?,
                        "detected_ages": row.get::<_, Option<String>>(21)?,
                        "age_detection_data": row.get::<_, Option<String>>(22)?,
                        "created_at": row.get::<_, Option<String>>(24)?,
                    }))
                },
            ) {
                let _ = dir_conn.execute(
                    "UPDATE images SET view_count = view_count + 1 WHERE id = ?1",
                    params![image_id],
                );
                let tags = get_image_tags_from_directory(
                    &dir_pool,
                    state_clone.main_db(),
                    image_id,
                )?;
                data["tags"] = json!(tags);
                data["thumbnail_url"] = json!(format!(
                    "/api/images/{}/thumbnail?directory_id={}",
                    image_id, found_dir
                ));
                data["url"] = json!(format!(
                    "/api/images/{}/file?directory_id={}",
                    image_id, found_dir
                ));

                let mut fstmt = dir_conn.prepare(
                    "SELECT original_path, file_exists, file_status FROM image_files WHERE image_id = ?1",
                )?;
                let files: Vec<serde_json::Value> = fstmt
                    .query_map(params![image_id], |row| {
                        Ok(json!({
                            "path": row.get::<_, String>(0)?,
                            "exists": row.get::<_, bool>(1)?,
                            "status": row.get::<_, String>(2)?
                        }))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                data["files"] = json!(files);
                data["file_status"] = if let Some(first) = files.first() {
                    first["status"].clone()
                } else {
                    json!("unknown")
                };

                return Ok(data);
            }
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        let data = main_conn.query_row(
            "SELECT id, filename, original_filename, file_hash, width, height,
                    file_size, duration, rating, is_favorite, prompt, negative_prompt,
                    model_name, sampler, seed, steps, cfg_scale, source_url,
                    num_faces, min_detected_age, max_detected_age, detected_ages,
                    age_detection_data, view_count, created_at, import_source
             FROM images WHERE id = ?1",
            params![image_id],
            |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "filename": row.get::<_, String>(1)?,
                    "original_filename": row.get::<_, Option<String>>(2)?,
                    "file_hash": row.get::<_, String>(3)?,
                    "width": row.get::<_, Option<i32>>(4)?,
                    "height": row.get::<_, Option<i32>>(5)?,
                    "file_size": row.get::<_, Option<i64>>(6)?,
                    "rating": row.get::<_, String>(8)?,
                    "is_favorite": row.get::<_, bool>(9)?,
                    "view_count": row.get::<_, i32>(23)?,
                    "thumbnail_url": format!("/api/images/{}/thumbnail", row.get::<_, i64>(0)?),
                    "url": format!("/api/images/{}/file", row.get::<_, i64>(0)?),
                    "prompt": row.get::<_, Option<String>>(10)?,
                    "negative_prompt": row.get::<_, Option<String>>(11)?,
                    "model_name": row.get::<_, Option<String>>(12)?,
                    "seed": row.get::<_, Option<String>>(14)?,
                    "source_url": row.get::<_, Option<String>>(17)?,
                    "import_source": row.get::<_, Option<String>>(25)?,
                    "num_faces": row.get::<_, Option<i32>>(18)?,
                    "min_age": row.get::<_, Option<i32>>(19)?,
                    "max_age": row.get::<_, Option<i32>>(20)?,
                    "detected_ages": row.get::<_, Option<String>>(21)?,
                    "age_detection_data": row.get::<_, Option<String>>(22)?,
                    "created_at": row.get::<_, Option<String>>(24)?,
                    "tags": [],
                    "files": [],
                    "file_status": "unknown",
                }))
            },
        );

        match data {
            Ok(d) => {
                let _ = main_conn.execute(
                    "UPDATE images SET view_count = view_count + 1 WHERE id = ?1",
                    params![image_id],
                );
                Ok(d)
            }
            Err(_) => Err(AppError::NotFound("Image not found".into())),
        }
    })
    .await??;

    Ok(Json(result))
}

/// GET /api/images/:image_id/file — Serve the original image/video file.
///
/// Uses tower-http ServeFile for automatic Range header support (206 Partial Content),
/// enabling seeking in video files.
pub async fn get_image_file(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<DirectoryQuery>,
    request: Request,
) -> Result<Response, AppError> {
    let state_clone = state.clone();
    let directory_id = q.directory_id;

    // Find the file path
    let file_path: PathBuf = tokio::task::spawn_blocking(move || {
        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;

                if let Ok(path) = dir_conn.query_row(
                    "SELECT original_path FROM image_files WHERE image_id = ?1 LIMIT 1",
                    params![image_id],
                    |row| row.get::<_, String>(0),
                ) {
                    return Ok(PathBuf::from(path));
                }
            }
        }

        // Search all directory DBs
        if let Some(found_dir) =
            find_image_directory(state_clone.directory_db(), image_id, None)
        {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;
            if let Ok(path) = dir_conn.query_row(
                "SELECT original_path FROM image_files WHERE image_id = ?1 LIMIT 1",
                params![image_id],
                |row| row.get::<_, String>(0),
            ) {
                return Ok(PathBuf::from(path));
            }
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        match main_conn.query_row(
            "SELECT original_path FROM image_files WHERE image_id = ?1 LIMIT 1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(path) => Ok(PathBuf::from(path)),
            Err(_) => Err(AppError::NotFound("Image file not found".into())),
        }
    })
    .await??;

    // Check file availability — distinguish missing from drive offline
    match crate::services::file_tracker::check_file_availability(file_path.to_str().unwrap_or("")) {
        crate::services::file_tracker::FileStatus::Available => {}
        crate::services::file_tracker::FileStatus::DriveOffline => {
            return Err(AppError::ServiceUnavailable("Drive is offline".into()));
        }
        crate::services::file_tracker::FileStatus::Missing => {
            return Err(AppError::NotFound("File not found on disk".into()));
        }
    }

    // ServeFile handles Range headers, ETag, 206 Partial Content automatically
    let response = ServeFile::new(&file_path)
        .oneshot(request)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to serve file: {}", e)))?;

    Ok(response.into_response())
}

/// GET /api/images/:image_id/thumbnail — Serve the thumbnail.
pub async fn get_image_thumbnail(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<DirectoryQuery>,
) -> Result<Response, AppError> {
    let state_clone = state.clone();
    let directory_id = q.directory_id;

    // Find the file hash to locate thumbnail
    let file_hash: String = tokio::task::spawn_blocking(move || {
        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;
                if let Ok(hash) = dir_conn.query_row(
                    "SELECT file_hash FROM images WHERE id = ?1",
                    params![image_id],
                    |row| row.get::<_, String>(0),
                ) {
                    return Ok(hash);
                }
            }
        }

        // Search directory DBs
        if let Some(found_dir) =
            find_image_directory(state_clone.directory_db(), image_id, None)
        {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;
            if let Ok(hash) = dir_conn.query_row(
                "SELECT file_hash FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get::<_, String>(0),
            ) {
                return Ok(hash);
            }
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        match main_conn.query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(hash) => Ok(hash),
            Err(_) => Err(AppError::NotFound("Image not found".into())),
        }
    })
    .await??;

    // Serve thumbnail file
    let thumbnail_name = format!("{}.webp", &file_hash[..16.min(file_hash.len())]);
    let thumbnail_path = state.thumbnails_dir().join(&thumbnail_name);

    if thumbnail_path.exists() {
        serve_file_with_type(&thumbnail_path, "image/webp").await
    } else {
        // Try to regenerate thumbnail from original file
        let state_regen = state.clone();
        let dir_id = directory_id;
        let img_id = image_id;
        let thumb_dir = state.thumbnails_dir().clone();
        let hash = file_hash.clone();

        let regenerated = tokio::task::spawn_blocking(move || -> Result<bool, AppError> {
            // Find the original file path
            let original_path: Option<String> = if let Some(did) = dir_id {
                if state_regen.directory_db().db_exists(did) {
                    let dir_pool = state_regen.directory_db().get_pool(did)?;
                    let dir_conn = dir_pool.get()?;
                    dir_conn.query_row(
                        "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_status = 'available' LIMIT 1",
                        params![img_id],
                        |row| row.get(0),
                    ).ok()
                } else { None }
            } else { None };

            if let Some(ref path) = original_path {
                if Path::new(path).exists() {
                    let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
                    let thumb_output = thumb_dir.join(&thumb_name);
                    if crate::services::importer::generate_thumbnail(path, thumb_output.to_str().unwrap_or(""), 300) {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        }).await??;

        if regenerated {
            serve_file_with_type(&thumbnail_path, "image/webp").await
        } else {
            Err(AppError::NotFound("Thumbnail not found".into()))
        }
    }
}

/// POST /api/images/:image_id/favorite — Toggle favorite status.
pub async fn toggle_favorite(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<DirectoryQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let directory_id = q.directory_id;

    let is_favorite = tokio::task::spawn_blocking(move || -> Result<bool, AppError> {
        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;
                if let Ok(current) = dir_conn.query_row(
                    "SELECT is_favorite FROM images WHERE id = ?1",
                    params![image_id],
                    |row| row.get::<_, bool>(0),
                ) {
                    let new_val = !current;
                    dir_conn.execute(
                        "UPDATE images SET is_favorite = ?1 WHERE id = ?2",
                        params![new_val, image_id],
                    )?;
                    return Ok(new_val);
                }
            }
        }

        // Search directory DBs
        if let Some(found_dir) =
            find_image_directory(state_clone.directory_db(), image_id, None)
        {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;
            if let Ok(current) = dir_conn.query_row(
                "SELECT is_favorite FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get::<_, bool>(0),
            ) {
                let new_val = !current;
                dir_conn.execute(
                    "UPDATE images SET is_favorite = ?1 WHERE id = ?2",
                    params![new_val, image_id],
                )?;
                return Ok(new_val);
            }
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        let current: bool = main_conn
            .query_row(
                "SELECT is_favorite FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Image not found".into()))?;

        let new_val = !current;
        main_conn.execute(
            "UPDATE images SET is_favorite = ?1 WHERE id = ?2",
            params![new_val, image_id],
        )?;
        Ok(new_val)
    })
    .await??;

    Ok(Json(json!({"is_favorite": is_favorite})))
}

/// PATCH /api/images/:image_id/rating — Update image rating.
pub async fn update_rating(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<RatingQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let valid_ratings = ["pg", "pg13", "r", "x", "xxx"];
    if !valid_ratings.contains(&q.rating.as_str()) {
        return Err(AppError::BadRequest(format!(
            "Invalid rating: {}",
            q.rating
        )));
    }

    let rating = q.rating.clone();
    let state_clone = state.clone();
    let directory_id = q.directory_id;

    tokio::task::spawn_blocking(move || {
        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;
                let updated = dir_conn.execute(
                    "UPDATE images SET rating = ?1 WHERE id = ?2",
                    params![rating, image_id],
                )?;
                if updated > 0 {
                    return Ok(());
                }
            }
        }

        // Search directory DBs
        if let Some(found_dir) =
            find_image_directory(state_clone.directory_db(), image_id, None)
        {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;
            dir_conn.execute(
                "UPDATE images SET rating = ?1 WHERE id = ?2",
                params![rating, image_id],
            )?;
            return Ok(());
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        let updated = main_conn.execute(
            "UPDATE images SET rating = ?1 WHERE id = ?2",
            params![rating, image_id],
        )?;
        if updated == 0 {
            return Err(AppError::NotFound("Image not found".into()));
        }
        Ok(())
    })
    .await??;

    Ok(Json(json!({"rating": q.rating})))
}

#[derive(Debug, Deserialize)]
pub struct RatingQuery {
    pub rating: String,
    pub directory_id: Option<i64>,
}

/// DELETE /api/images/:image_id — Delete an image.
pub async fn delete_image(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<DeleteQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let directory_id = q.directory_id;
    let delete_file = q.delete_file;

    tokio::task::spawn_blocking(move || {
        let mut file_hash: Option<String> = None;

        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;

                // Get file hash and paths
                if let Ok(hash) = dir_conn.query_row(
                    "SELECT file_hash FROM images WHERE id = ?1",
                    params![image_id],
                    |row| row.get::<_, String>(0),
                ) {
                    file_hash = Some(hash);

                    if delete_file {
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

                    // Delete from directory DB (cascades to files, tags)
                    dir_conn.execute("DELETE FROM image_tags WHERE image_id = ?1", params![image_id])?;
                    dir_conn.execute("DELETE FROM image_files WHERE image_id = ?1", params![image_id])?;
                    dir_conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
                }
            }
        }

        // If not found in directory, try main DB
        if file_hash.is_none() {
            if let Some(found_dir) =
                find_image_directory(state_clone.directory_db(), image_id, None)
            {
                let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
                let dir_conn = dir_pool.get()?;

                if let Ok(hash) = dir_conn.query_row(
                    "SELECT file_hash FROM images WHERE id = ?1",
                    params![image_id],
                    |row| row.get::<_, String>(0),
                ) {
                    file_hash = Some(hash);

                    if delete_file {
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
        }

        if file_hash.is_none() {
            // Try main/legacy DB
            let main_conn = state_clone.main_db().get()?;
            match main_conn.query_row(
                "SELECT file_hash FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get::<_, String>(0),
            ) {
                Ok(hash) => {
                    file_hash = Some(hash);

                    if delete_file {
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
            let thumb_path = state_clone.thumbnails_dir().join(&thumb_name);
            if thumb_path.exists() {
                let _ = std::fs::remove_file(&thumb_path);
            }

            // Delete video preview frames
            video_preview::delete_preview_frames(state_clone.data_dir(), hash);
        }

        Ok(())
    })
    .await??;

    Ok(Json(json!({"deleted": true})))
}

#[derive(Debug, Deserialize)]
pub struct DeleteQuery {
    #[serde(default)]
    pub delete_file: bool,
    pub directory_id: Option<i64>,
}

/// POST /api/images/upload — Upload an image via multipart form data.
pub async fn upload_image(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut file_data: Option<(String, Vec<u8>)> = None;
    let mut auto_tag = false;
    let mut directory_id: Option<i64> = None;

    // Parse multipart fields
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let filename = field
                    .file_name()
                    .unwrap_or("upload")
                    .to_string();
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read file: {}", e)))?;
                file_data = Some((filename, data.to_vec()));
            }
            "auto_tag" => {
                let val = field
                    .text()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read auto_tag: {}", e)))?;
                auto_tag = val == "true" || val == "1";
            }
            "directory_id" => {
                let val = field
                    .text()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("Failed to read directory_id: {}", e)))?;
                directory_id = val.parse::<i64>().ok();
            }
            _ => {}
        }
    }

    let (filename, data) = file_data.ok_or_else(|| AppError::BadRequest("Missing file field".into()))?;

    // Determine the target directory ID
    let dir_id = match directory_id {
        Some(id) => id,
        None => {
            // Use the first available directory
            let dirs = state.directory_db().get_all_directory_ids();
            if dirs.is_empty() {
                return Err(AppError::BadRequest(
                    "No watch directories configured. Add a directory first.".into(),
                ));
            }
            dirs[0]
        }
    };

    // Verify directory exists
    if !state.directory_db().db_exists(dir_id) {
        return Err(AppError::BadRequest(format!(
            "Directory {} does not exist",
            dir_id
        )));
    }

    // Save to temp file
    let ext = std::path::Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");
    let tmp_path = std::env::temp_dir().join(format!("localbooru_upload_{}.{}", uuid::Uuid::new_v4(), ext));

    tokio::fs::write(&tmp_path, &data).await?;

    // Import the file
    let state_clone = state.clone();
    let tmp_path_str = tmp_path.to_string_lossy().to_string();

    let result = tokio::task::spawn_blocking(move || {
        importer::import_image(&state_clone, &tmp_path_str, dir_id)
    })
    .await??;

    // Clean up temp file
    let _ = tokio::fs::remove_file(&tmp_path).await;

    let image_id = result.image_id;
    let status_str = match result.status {
        importer::ImportStatus::Imported => "imported",
        importer::ImportStatus::Duplicate => "duplicate",
        importer::ImportStatus::Error => "error",
    };

    // Enqueue tagging task if requested and import was successful
    if auto_tag && result.status == importer::ImportStatus::Imported {
        if let Some(img_id) = image_id {
            let payload = json!({
                "image_id": img_id,
                "directory_id": dir_id,
            });
            let _ = crate::services::task_queue::enqueue_task(
                &state,
                crate::services::task_queue::TASK_TAG,
                &payload,
                0,
                Some(img_id),
            );
        }
    }

    Ok(Json(json!({
        "status": status_str,
        "image_id": image_id,
        "directory_id": dir_id,
        "filename": result.filename,
        "message": result.message,
    })))
}

/// GET /api/images/:image_id/preview-frames — Get list of video preview frame URLs.
pub async fn get_preview_frames(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(q): Query<DirectoryQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let directory_id = q.directory_id;

    // Look up file_hash, filename, and original_path
    let (file_hash, filename, original_path, found_dir_id) = tokio::task::spawn_blocking(move || {
        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;

                if let Ok((hash, fname)) = dir_conn.query_row(
                    "SELECT file_hash, filename FROM images WHERE id = ?1",
                    params![image_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                ) {
                    let path: Option<String> = dir_conn
                        .query_row(
                            "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_status = 'available' LIMIT 1",
                            params![image_id],
                            |row| row.get(0),
                        )
                        .ok();
                    return Ok((hash, fname, path, directory_id));
                }
            }
        }

        // Search directory DBs
        if let Some(found_dir) = find_image_directory(state_clone.directory_db(), image_id, None) {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;

            if let Ok((hash, fname)) = dir_conn.query_row(
                "SELECT file_hash, filename FROM images WHERE id = ?1",
                params![image_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            ) {
                let path: Option<String> = dir_conn
                    .query_row(
                        "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_status = 'available' LIMIT 1",
                        params![image_id],
                        |row| row.get(0),
                    )
                    .ok();
                return Ok((hash, fname, path, Some(found_dir)));
            }
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        let (hash, fname) = main_conn
            .query_row(
                "SELECT file_hash, filename FROM images WHERE id = ?1",
                params![image_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(|_| AppError::NotFound("Image not found".into()))?;

        let path: Option<String> = main_conn
            .query_row(
                "SELECT original_path FROM image_files WHERE image_id = ?1 LIMIT 1",
                params![image_id],
                |row| row.get(0),
            )
            .ok();

        Ok::<_, AppError>((hash, fname, path, None::<i64>))
    })
    .await??;

    // Check if this is a video
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();
    let video_extensions = ["webm", "mp4", "mov", "avi", "mkv"];
    if !video_extensions.contains(&ext.as_str()) {
        return Ok(Json(json!({
            "frames": [],
            "is_video": false,
        })));
    }

    // Check for existing preview frames
    let data_dir = state.data_dir().to_path_buf();
    let existing = video_preview::get_preview_frames(&data_dir, &file_hash);

    if !existing.is_empty() {
        let dir_param = match found_dir_id {
            Some(did) => format!("?directory_id={}", did),
            None => String::new(),
        };
        let frame_urls: Vec<String> = (0..existing.len())
            .map(|i| format!("/api/images/{}/preview-frame/{}{}", image_id, i, dir_param))
            .collect();

        return Ok(Json(json!({
            "frames": frame_urls,
            "is_video": true,
            "count": existing.len(),
            "generating": false,
        })));
    }

    // No frames exist yet -- trigger background generation if source file is available
    if let Some(ref path) = original_path {
        let file_status = crate::services::file_tracker::check_file_availability(path);
        if matches!(file_status, crate::services::file_tracker::FileStatus::Available) {
            let video_path = path.clone();
            let hash = file_hash.clone();
            let dd = data_dir.clone();
            tokio::task::spawn_blocking(move || {
                video_preview::generate_video_previews(&video_path, &hash, &dd, 8);
            });
        }
    }

    Ok(Json(json!({
        "frames": [],
        "is_video": true,
        "count": 0,
        "generating": true,
    })))
}

/// GET /api/images/:image_id/preview-frame/:frame_index — Serve a specific video preview frame.
pub async fn get_preview_frame(
    State(state): State<AppState>,
    AxumPath((image_id, frame_index)): AxumPath<(i64, usize)>,
    Query(q): Query<DirectoryQuery>,
) -> Result<Response, AppError> {
    if frame_index >= 8 {
        return Err(AppError::BadRequest(
            "Invalid frame index (must be 0-7)".into(),
        ));
    }

    let state_clone = state.clone();
    let directory_id = q.directory_id;

    // Look up file_hash
    let file_hash: String = tokio::task::spawn_blocking(move || {
        // Try directory DB
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                let dir_conn = dir_pool.get()?;
                if let Ok(hash) = dir_conn.query_row(
                    "SELECT file_hash FROM images WHERE id = ?1",
                    params![image_id],
                    |row| row.get::<_, String>(0),
                ) {
                    return Ok(hash);
                }
            }
        }

        // Search directory DBs
        if let Some(found_dir) = find_image_directory(state_clone.directory_db(), image_id, None) {
            let dir_pool = state_clone.directory_db().get_pool(found_dir)?;
            let dir_conn = dir_pool.get()?;
            if let Ok(hash) = dir_conn.query_row(
                "SELECT file_hash FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get::<_, String>(0),
            ) {
                return Ok(hash);
            }
        }

        // Legacy main DB
        let main_conn = state_clone.main_db().get()?;
        main_conn
            .query_row(
                "SELECT file_hash FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get::<_, String>(0),
            )
            .map_err(|_| AppError::NotFound("Image not found".into()))
    })
    .await??;

    // Construct frame path
    let preview_dir = video_preview::get_preview_dir(state.data_dir(), &file_hash);
    let frame_path = preview_dir.join(format!("frame_{}.webp", frame_index));

    if !frame_path.exists() {
        return Err(AppError::NotFound("Preview frame not found".into()));
    }

    serve_file_with_type(&frame_path, "image/webp").await
}

// ─── File serving helpers ───────────────────────────────────────────────────

async fn serve_file_with_type(path: &Path, content_type: &str) -> Result<Response, AppError> {
    if !path.exists() {
        return Err(AppError::NotFound("File not found".into()));
    }

    let file = File::open(path).await?;
    let metadata = file.metadata().await?;
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, metadata.len())
        .header(header::ACCEPT_RANGES, "bytes")
        .body(body)
        .unwrap())
}
