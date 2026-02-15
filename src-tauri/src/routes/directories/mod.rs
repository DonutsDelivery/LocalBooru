use axum::extract::{Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

use crate::db::schema::init_directory_db;
use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::task_queue;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_directories).post(add_directory))
        .route("/add-parent", post(add_parent_directory))
        .route("/bulk-delete", post(bulk_delete))
        .route("/{directory_id}", get(get_directory).patch(update_directory).delete(remove_directory))
        .route("/{directory_id}/scan", post(scan_directory))
}

// ─── Request models ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DirectoryCreate {
    pub path: String,
    pub name: Option<String>,
    #[serde(default = "default_true")]
    pub recursive: bool,
    #[serde(default = "default_true")]
    pub auto_tag: bool,
    #[serde(default)]
    pub auto_age_detect: bool,
}

#[derive(Deserialize)]
pub struct DirectoryUpdate {
    pub name: Option<String>,
    pub enabled: Option<bool>,
    pub recursive: Option<bool>,
    pub auto_tag: Option<bool>,
    pub auto_age_detect: Option<bool>,
    pub public_access: Option<bool>,
    pub show_images: Option<bool>,
    pub show_videos: Option<bool>,
}

#[derive(Deserialize)]
pub struct ParentDirectoryCreate {
    pub path: String,
    #[serde(default = "default_true")]
    pub recursive: bool,
    #[serde(default = "default_true")]
    pub auto_tag: bool,
    #[serde(default)]
    pub auto_age_detect: bool,
}

#[derive(Deserialize)]
pub struct BulkDeleteRequest {
    pub directory_ids: Vec<i64>,
    #[serde(default)]
    pub keep_images: bool,
}

#[derive(Deserialize)]
pub struct ScanOptions {
    #[serde(default)]
    pub clean_deleted: bool,
}

fn default_true() -> bool {
    true
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/directories — List all watch directories with stats.
async fn list_directories(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Get all directories
        let mut stmt = main_conn.prepare(
            "SELECT id, path, name, enabled, recursive, auto_tag, auto_age_detect,
                    last_scanned_at, created_at, public_access, show_images, show_videos,
                    parent_path, metadata_format
             FROM watch_directories ORDER BY created_at",
        )?;

        let dirs: Vec<Value> = stmt
            .query_map([], |row| {
                let dir_id: i64 = row.get(0)?;
                let path: String = row.get(1)?;
                let name: Option<String> = row.get(2)?;
                let enabled: bool = row.get(3)?;
                let recursive: bool = row.get(4)?;
                let auto_tag: bool = row.get(5)?;
                let auto_age_detect: bool = row.get(6)?;
                let last_scanned_at: Option<String> = row.get(7)?;
                let created_at: Option<String> = row.get(8)?;
                let public_access: bool = row.get::<_, Option<bool>>(9)?.unwrap_or(false);
                let show_images: bool = row.get::<_, Option<bool>>(10)?.unwrap_or(true);
                let show_videos: bool = row.get::<_, Option<bool>>(11)?.unwrap_or(true);
                let _parent_path: Option<String> = row.get(12)?;
                let metadata_format: Option<String> = row.get(13)?;

                // Get image count from per-directory DB
                let mut image_count: i64 = 0;
                let mut tagged_count: i64 = 0;
                let mut favorited_count: i64 = 0;

                if let Ok(dir_pool) = state_clone.directory_db().get_pool(dir_id) {
                    if let Ok(dir_conn) = dir_pool.get() {
                        image_count = dir_conn
                            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                            .unwrap_or(0);
                        tagged_count = dir_conn
                            .query_row(
                                "SELECT COUNT(DISTINCT image_id) FROM image_tags",
                                [],
                                |r| r.get(0),
                            )
                            .unwrap_or(0);
                        favorited_count = dir_conn
                            .query_row(
                                "SELECT COUNT(*) FROM images WHERE is_favorite = 1",
                                [],
                                |r| r.get(0),
                            )
                            .unwrap_or(0);
                    }
                }

                let display_name = name.unwrap_or_else(|| {
                    Path::new(&path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&path)
                        .to_string()
                });

                let path_exists = Path::new(&path).exists();

                let tagged_pct = if image_count > 0 {
                    (tagged_count as f64 / image_count as f64 * 100.0 * 10.0).round() / 10.0
                } else {
                    0.0
                };
                let fav_pct = if image_count > 0 {
                    (favorited_count as f64 / image_count as f64 * 100.0 * 10.0).round() / 10.0
                } else {
                    0.0
                };

                Ok(json!({
                    "id": dir_id,
                    "path": path,
                    "name": display_name,
                    "enabled": enabled,
                    "recursive": recursive,
                    "auto_tag": auto_tag,
                    "auto_age_detect": auto_age_detect,
                    "image_count": image_count,
                    "tagged_count": tagged_count,
                    "tagged_pct": tagged_pct,
                    "favorited_count": favorited_count,
                    "favorited_pct": fav_pct,
                    "path_exists": path_exists,
                    "last_scanned_at": last_scanned_at,
                    "created_at": created_at,
                    "public_access": public_access,
                    "show_images": show_images,
                    "show_videos": show_videos,
                    "metadata_format": metadata_format.unwrap_or_else(|| "auto".into()),
                    "uses_per_directory_db": true
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok::<_, AppError>(Json(json!({ "directories": dirs })))
    })
    .await?
}

/// POST /api/directories — Add a new watch directory.
async fn add_directory(
    State(state): State<AppState>,
    Json(data): Json<DirectoryCreate>,
) -> Result<Json<Value>, AppError> {
    let resolved_path = std::fs::canonicalize(&data.path)
        .map_err(|_| AppError::BadRequest(format!("Path does not exist: {}", data.path)))?;

    if !resolved_path.is_dir() {
        return Err(AppError::BadRequest(format!(
            "Path is not a directory: {}",
            data.path
        )));
    }

    let path_str = resolved_path.to_string_lossy().to_string();
    let name = data
        .name
        .unwrap_or_else(|| {
            resolved_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .to_string()
        });
    let recursive = data.recursive;
    let auto_tag = data.auto_tag;
    let auto_age_detect = data.auto_age_detect;

    let state_clone = state.clone();
    let name_clone = name.clone();
    let path_clone = path_str.clone();

    let dir_id = tokio::task::spawn_blocking(move || -> Result<i64, AppError> {
        let conn = state_clone.main_db().get()?;

        // Check for duplicates
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM watch_directories WHERE path = ?1",
                params![&path_clone],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);

        if exists {
            return Err(AppError::BadRequest("Directory already added".into()));
        }

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO watch_directories (path, name, enabled, recursive, auto_tag, auto_age_detect, created_at)
             VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6)",
            params![&path_clone, &name_clone, recursive, auto_tag, auto_age_detect, &now],
        )?;

        let dir_id = conn.last_insert_rowid();

        // Create per-directory database
        let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
        let dir_conn = dir_pool.get()?;
        init_directory_db(&dir_conn)?;

        // Queue initial scan
        task_queue::enqueue_task(
            &state_clone,
            task_queue::TASK_SCAN_DIRECTORY,
            &json!({
                "directory_id": dir_id,
                "directory_path": &path_clone
            }),
            2,
            None,
        )?;

        Ok(dir_id)
    })
    .await??;

    Ok(Json(json!({
        "id": dir_id,
        "path": path_str,
        "name": name,
        "message": "Directory added, initial scan queued"
    })))
}

/// POST /api/directories/add-parent — Add all subdirectories of a parent folder.
async fn add_parent_directory(
    State(state): State<AppState>,
    Json(data): Json<ParentDirectoryCreate>,
) -> Result<Json<Value>, AppError> {
    let parent_path = std::fs::canonicalize(&data.path)
        .map_err(|_| AppError::BadRequest(format!("Path does not exist: {}", data.path)))?;

    if !parent_path.is_dir() {
        return Err(AppError::BadRequest("Path is not a directory".into()));
    }

    // Get immediate subdirectories
    let mut subdirs: Vec<_> = std::fs::read_dir(&parent_path)
        .map_err(|e| AppError::Internal(format!("Failed to read directory: {}", e)))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();

    subdirs.sort();

    if subdirs.is_empty() {
        return Err(AppError::BadRequest(
            "No subdirectories found in the selected folder".into(),
        ));
    }

    let recursive = data.recursive;
    let auto_tag = data.auto_tag;
    let auto_age_detect = data.auto_age_detect;

    let state_clone = state.clone();

    let (added, skipped) = tokio::task::spawn_blocking(move || -> Result<(Vec<Value>, Vec<String>), AppError> {
        let conn = state_clone.main_db().get()?;
        let mut added = Vec::new();
        let mut skipped = Vec::new();

        for subdir in subdirs {
            let path_str = subdir.to_string_lossy().to_string();

            // Check for duplicates
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM watch_directories WHERE path = ?1",
                    params![&path_str],
                    |row| row.get::<_, i64>(0).map(|c| c > 0),
                )
                .unwrap_or(false);

            if exists {
                skipped.push(path_str);
                continue;
            }

            let name = subdir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .to_string();

            let now = chrono::Utc::now().to_rfc3339();
            conn.execute(
                "INSERT INTO watch_directories (path, name, enabled, recursive, auto_tag, auto_age_detect, parent_path, created_at)
                 VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7)",
                params![
                    &path_str,
                    &name,
                    recursive,
                    auto_tag,
                    auto_age_detect,
                    parent_path.to_string_lossy().as_ref(),
                    &now,
                ],
            )?;

            let dir_id = conn.last_insert_rowid();

            // Create per-directory DB
            let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
            let dir_conn = dir_pool.get()?;
            init_directory_db(&dir_conn)?;

            // Queue scan
            task_queue::enqueue_task(
                &state_clone,
                task_queue::TASK_SCAN_DIRECTORY,
                &json!({
                    "directory_id": dir_id,
                    "directory_path": &path_str
                }),
                2,
                None,
            )?;

            added.push(json!({
                "id": dir_id,
                "path": path_str,
                "name": name
            }));
        }

        Ok((added, skipped))
    })
    .await??;

    let added_count = added.len();
    let skipped_count = skipped.len();

    Ok(Json(json!({
        "added": added,
        "skipped": skipped,
        "message": format!("Added {} directories, skipped {} existing", added_count, skipped_count)
    })))
}

/// GET /api/directories/:directory_id — Get directory details.
async fn get_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let dir = conn.query_row(
            "SELECT id, path, name, enabled, recursive, auto_tag, auto_age_detect,
                    last_scanned_at, created_at, public_access, show_images, show_videos, metadata_format
             FROM watch_directories WHERE id = ?1",
            params![directory_id],
            |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "path": row.get::<_, String>(1)?,
                    "name": row.get::<_, Option<String>>(2)?,
                    "enabled": row.get::<_, bool>(3)?,
                    "recursive": row.get::<_, bool>(4)?,
                    "auto_tag": row.get::<_, bool>(5)?,
                    "auto_age_detect": row.get::<_, bool>(6)?,
                    "last_scanned_at": row.get::<_, Option<String>>(7)?,
                    "created_at": row.get::<_, Option<String>>(8)?,
                    "public_access": row.get::<_, Option<bool>>(9)?.unwrap_or(false),
                    "show_images": row.get::<_, Option<bool>>(10)?.unwrap_or(true),
                    "show_videos": row.get::<_, Option<bool>>(11)?.unwrap_or(true),
                    "metadata_format": row.get::<_, Option<String>>(12)?.unwrap_or_else(|| "auto".into()),
                    "path_exists": Path::new(&row.get::<_, String>(1)?).exists()
                }))
            },
        );

        match dir {
            Ok(d) => Ok(Json(d)),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(AppError::NotFound("Directory not found".into()))
            }
            Err(e) => Err(AppError::from(e)),
        }
    })
    .await?
}

/// PATCH /api/directories/:directory_id — Update directory settings.
async fn update_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
    Json(data): Json<DirectoryUpdate>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Build dynamic UPDATE query
        let mut sets = Vec::new();
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(name) = &data.name {
            sets.push("name = ?");
            sql_params.push(Box::new(name.clone()));
        }
        if let Some(enabled) = data.enabled {
            sets.push("enabled = ?");
            sql_params.push(Box::new(enabled));
        }
        if let Some(recursive) = data.recursive {
            sets.push("recursive = ?");
            sql_params.push(Box::new(recursive));
        }
        if let Some(auto_tag) = data.auto_tag {
            sets.push("auto_tag = ?");
            sql_params.push(Box::new(auto_tag));
        }
        if let Some(auto_age_detect) = data.auto_age_detect {
            sets.push("auto_age_detect = ?");
            sql_params.push(Box::new(auto_age_detect));
        }
        if let Some(public_access) = data.public_access {
            sets.push("public_access = ?");
            sql_params.push(Box::new(public_access));
        }
        if let Some(show_images) = data.show_images {
            sets.push("show_images = ?");
            sql_params.push(Box::new(show_images));
        }
        if let Some(show_videos) = data.show_videos {
            sets.push("show_videos = ?");
            sql_params.push(Box::new(show_videos));
        }

        if sets.is_empty() {
            return Err(AppError::BadRequest("No fields to update".into()));
        }

        sql_params.push(Box::new(directory_id));
        let sql = format!(
            "UPDATE watch_directories SET {} WHERE id = ?",
            sets.join(", ")
        );

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        let updated = conn.execute(&sql, param_refs.as_slice())?;

        if updated == 0 {
            return Err(AppError::NotFound("Directory not found".into()));
        }

        Ok(Json(json!({
            "id": directory_id,
            "updated": true
        })))
    })
    .await?
}

/// DELETE /api/directories/:directory_id — Remove a watch directory.
async fn remove_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
    Query(params): Query<DeleteParams>,
) -> Result<Json<Value>, AppError> {
    delete_directories(state, vec![directory_id], params.keep_images.unwrap_or(false)).await
}

#[derive(Deserialize)]
struct DeleteParams {
    keep_images: Option<bool>,
}

/// POST /api/directories/bulk-delete — Remove multiple directories.
async fn bulk_delete(
    State(state): State<AppState>,
    Json(data): Json<BulkDeleteRequest>,
) -> Result<Json<Value>, AppError> {
    delete_directories(state, data.directory_ids, data.keep_images).await
}

async fn delete_directories(
    state: AppState,
    directory_ids: Vec<i64>,
    keep_images: bool,
) -> Result<Json<Value>, AppError> {
    if directory_ids.is_empty() {
        return Ok(Json(json!({"deleted": 0, "images_removed": false, "image_count": 0})));
    }

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let mut total_images: i64 = 0;
        let mut all_hashes: Vec<String> = Vec::new();
        let mut images_removed = false;

        for &dir_id in &directory_ids {
            // Get image count and hashes from directory DB
            if let Ok(dir_pool) = state_clone.directory_db().get_pool(dir_id) {
                if let Ok(dir_conn) = dir_pool.get() {
                    let count: i64 = dir_conn
                        .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                        .unwrap_or(0);
                    total_images += count;

                    if !keep_images {
                        // Get file hashes for thumbnail cleanup
                        let mut hash_stmt = dir_conn
                            .prepare("SELECT file_hash FROM images WHERE file_hash IS NOT NULL")
                            .ok();
                        if let Some(ref mut stmt) = hash_stmt {
                            let hashes: Vec<String> = stmt
                                .query_map([], |row| row.get(0))
                                .ok()
                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                .unwrap_or_default();
                            all_hashes.extend(hashes);
                        }

                        // Decrement tag post_counts in main DB
                        let mut tag_stmt = dir_conn
                            .prepare(
                                "SELECT tag_id, COUNT(image_id) FROM image_tags GROUP BY tag_id",
                            )
                            .ok();
                        if let Some(ref mut stmt) = tag_stmt {
                            let tag_counts: Vec<(i64, i64)> = stmt
                                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                                .ok()
                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                .unwrap_or_default();

                            for (tag_id, count) in tag_counts {
                                let _ = conn.execute(
                                    "UPDATE tags SET post_count = MAX(0, post_count - ?1) WHERE id = ?2",
                                    params![count, tag_id],
                                );
                            }
                        }

                        images_removed = true;
                    }
                }
            }

            // Delete directory DB file
            let _ = state_clone.directory_db().delete_directory_db(dir_id);

            // Delete directory record from main DB
            conn.execute(
                "DELETE FROM watch_directories WHERE id = ?1",
                params![dir_id],
            )?;
        }

        // Clean up thumbnails
        if !all_hashes.is_empty() {
            let thumbnails_dir = state_clone.thumbnails_dir();
            for hash in &all_hashes {
                let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
                let _ = std::fs::remove_file(thumbnails_dir.join(&thumb_name));
            }
        }

        Ok::<_, AppError>(Json(json!({
            "deleted": directory_ids.len(),
            "images_removed": images_removed,
            "image_count": total_images
        })))
    })
    .await?
}

/// POST /api/directories/:directory_id/scan — Trigger a directory scan.
async fn scan_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
    body: Option<Json<ScanOptions>>,
) -> Result<Json<Value>, AppError> {
    let clean_deleted = body.map(|b| b.clean_deleted).unwrap_or(false);

    let state_clone = state.clone();
    let task_id = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Get directory path
        let dir_path: String = conn
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![directory_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Directory not found".into()))?;

        task_queue::enqueue_task(
            &state_clone,
            task_queue::TASK_SCAN_DIRECTORY,
            &json!({
                "directory_id": directory_id,
                "directory_path": dir_path,
                "clean_deleted": clean_deleted
            }),
            2,
            None,
        )
    })
    .await??;

    Ok(Json(json!({
        "task_id": task_id,
        "message": "Scan queued"
    })))
}
