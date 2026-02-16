use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{get, patch, post};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::db::schema::init_directory_db;
use crate::server::error::AppError;
use crate::server::middleware::AccessTier;
use crate::server::state::AppState;
use crate::services::{file_tracker, importer, metadata, task_queue};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_directories).post(add_directory))
        .route("/add-parent", post(add_parent_directory))
        .route("/bulk-delete", post(bulk_delete))
        .route("/bulk-repair", post(bulk_repair))
        .route("/bulk-verify", post(bulk_verify))
        .route("/{directory_id}", get(get_directory).patch(update_directory).delete(remove_directory))
        .route("/{directory_id}/scan", post(scan_directory))
        .route("/{directory_id}/repair", post(repair_directory))
        .route("/{directory_id}/verify", post(verify_directory))
        .route("/{directory_id}/prune", post(prune_directory))
        .route("/{directory_id}/path", patch(update_directory_path))
        .route("/{directory_id}/clean-deleted", post(clean_deleted))
        .route("/{directory_id}/comfyui-nodes", get(get_comfyui_nodes))
        .route("/{directory_id}/comfyui-config", patch(update_comfyui_config))
        .route("/{directory_id}/reextract-metadata", post(reextract_metadata))
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
    pub family_safe: Option<bool>,
    pub lan_visible: Option<bool>,
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

#[derive(Deserialize)]
pub struct BulkRepairRequest {
    pub directory_ids: Vec<i64>,
}

#[derive(Deserialize)]
pub struct BulkVerifyRequest {
    pub directory_ids: Vec<i64>,
}

#[derive(Deserialize)]
pub struct UpdatePathRequest {
    pub new_path: String,
}

#[derive(Deserialize)]
pub struct ComfyUIConfigUpdate {
    pub comfyui_prompt_node_ids: Option<String>,
    pub comfyui_negative_node_ids: Option<String>,
    pub metadata_format: Option<String>,
}

fn default_true() -> bool {
    true
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/directories — List all watch directories with stats.
///
/// Public IP clients only see directories with `public_access = true`.
async fn list_directories(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> Result<Json<Value>, AppError> {
    let client_ip = addr.ip();

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Get all directories
        let mut stmt = main_conn.prepare(
            "SELECT id, path, name, enabled, recursive, auto_tag, auto_age_detect,
                    last_scanned_at, created_at, public_access, show_images, show_videos,
                    parent_path, metadata_format, family_safe, lan_visible
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
                let family_safe: bool = row.get::<_, Option<bool>>(14)?.unwrap_or(true);
                let lan_visible: bool = row.get::<_, Option<bool>>(15)?.unwrap_or(true);

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
                    "family_safe": family_safe,
                    "lan_visible": lan_visible,
                    "uses_per_directory_db": true
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Filter directories based on access tier and family mode
        let tier = AccessTier::from_ip(&client_ip);
        let family_locked = state_clone.is_family_mode_locked();

        let filtered: Vec<Value> = dirs.into_iter().filter(|d| {
            // Family mode: hide non-family-safe when locked
            if family_locked && !d["family_safe"].as_bool().unwrap_or(true) {
                return false;
            }
            match tier {
                AccessTier::Localhost => true,
                AccessTier::LocalNetwork => d["lan_visible"].as_bool().unwrap_or(true),
                AccessTier::Public => d["public_access"].as_bool().unwrap_or(false),
            }
        }).collect();

        Ok::<_, AppError>(Json(json!({ "directories": filtered })))
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

    // Register with filesystem watcher
    if let Some(watcher) = state.directory_watcher() {
        watcher.add_directory(dir_id, &path_str, recursive);
    }

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

    // Register all newly added directories with filesystem watcher
    if let Some(watcher) = state.directory_watcher() {
        for dir in &added {
            if let (Some(id), Some(path)) = (dir["id"].as_i64(), dir["path"].as_str()) {
                watcher.add_directory(id, path, recursive);
            }
        }
    }

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
                    last_scanned_at, created_at, public_access, show_images, show_videos,
                    metadata_format, family_safe, lan_visible
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
                    "family_safe": row.get::<_, Option<bool>>(13)?.unwrap_or(true),
                    "lan_visible": row.get::<_, Option<bool>>(14)?.unwrap_or(true),
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
    // Track whether watcher-affecting fields are being changed
    let watcher_refresh_needed = data.enabled.is_some() || data.recursive.is_some();

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
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
        if let Some(family_safe) = data.family_safe {
            sets.push("family_safe = ?");
            sql_params.push(Box::new(family_safe));
        }
        if let Some(lan_visible) = data.lan_visible {
            sets.push("lan_visible = ?");
            sql_params.push(Box::new(lan_visible));
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
    .await?;

    // Refresh watcher if enabled/recursive changed (needs to re-add with new settings)
    if watcher_refresh_needed {
        if let Some(watcher) = state.directory_watcher() {
            watcher.refresh();
        }
    }

    result
}

/// DELETE /api/directories/:directory_id — Remove a watch directory.
async fn remove_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
    Query(params): Query<DeleteParams>,
) -> Result<Json<Value>, AppError> {
    // Remove from filesystem watcher before deletion
    if let Some(watcher) = state.directory_watcher() {
        watcher.remove_directory(directory_id);
    }

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
    // Remove all from filesystem watcher before deletion
    if let Some(watcher) = state.directory_watcher() {
        for &dir_id in &data.directory_ids {
            watcher.remove_directory(dir_id);
        }
    }

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
///
/// Validates that the directory path exists on disk before queuing the scan task.
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

        // Validate directory path exists on disk
        if !Path::new(&dir_path).exists() {
            return Err(AppError::BadRequest(format!(
                "Directory path does not exist on disk: {}",
                dir_path
            )));
        }

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

// ─── Verify ──────────────────────────────────────────────────────────────────

/// POST /api/directories/:directory_id/verify — Verify files exist at recorded paths.
///
/// For each image file record in the directory DB: check if the file exists on disk.
/// If the parent directory is gone, mark as "drive_offline". If the file is missing,
/// delete the record. Returns counts of verified/missing/offline.
async fn verify_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        verify_directory_inner(&state_clone, directory_id)
    })
    .await??;

    Ok(Json(result))
}

fn verify_directory_inner(state: &AppState, directory_id: i64) -> Result<Value, AppError> {
    // Ensure the directory exists in main DB
    let main_conn = state.main_db().get()?;
    let _dir_path: String = main_conn
        .query_row(
            "SELECT path FROM watch_directories WHERE id = ?1",
            params![directory_id],
            |row| row.get(0),
        )
        .map_err(|_| AppError::NotFound("Directory not found".into()))?;

    let stats = file_tracker::verify_directory_files(state, directory_id)?;

    Ok(json!({
        "directory_id": directory_id,
        "verified": stats.verified,
        "deleted": stats.deleted,
        "drive_offline": stats.drive_offline
    }))
}

/// POST /api/directories/bulk-verify — Verify files in multiple directories.
async fn bulk_verify(
    State(state): State<AppState>,
    Json(data): Json<BulkVerifyRequest>,
) -> Result<Json<Value>, AppError> {
    if data.directory_ids.is_empty() {
        return Ok(Json(json!({
            "results": [],
            "totals": { "verified": 0, "deleted": 0, "drive_offline": 0 }
        })));
    }

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut results: Vec<Value> = Vec::new();
        let mut totals_verified: i64 = 0;
        let mut totals_deleted: i64 = 0;
        let mut totals_offline: i64 = 0;

        for dir_id in &data.directory_ids {
            match verify_directory_inner(&state_clone, *dir_id) {
                Ok(r) => {
                    totals_verified += r["verified"].as_i64().unwrap_or(0);
                    totals_deleted += r["deleted"].as_i64().unwrap_or(0);
                    totals_offline += r["drive_offline"].as_i64().unwrap_or(0);
                    results.push(r);
                }
                Err(e) => {
                    log::warn!("[Verify] Directory {} failed: {}", dir_id, e);
                    results.push(json!({
                        "directory_id": dir_id,
                        "error": e.to_string()
                    }));
                }
            }
        }

        Ok::<_, AppError>(json!({
            "results": results,
            "totals": {
                "verified": totals_verified,
                "deleted": totals_deleted,
                "drive_offline": totals_offline
            },
            "message": format!(
                "Verified {} directories: {} OK, {} deleted, {} offline",
                results.len(), totals_verified, totals_deleted, totals_offline
            )
        }))
    })
    .await??;

    Ok(Json(result))
}

// ─── Prune ───────────────────────────────────────────────────────────────────

/// POST /api/directories/:directory_id/prune — Move non-favorited images to a dumpster folder.
///
/// Moves non-favorited image files to a "dumpster" subfolder within the directory,
/// preserving relative paths. Deletes DB records after moving.
async fn prune_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        let dir_path: String = main_conn
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![directory_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Directory not found".into()))?;

        if !Path::new(&dir_path).exists() {
            return Err(AppError::BadRequest("Directory path does not exist".into()));
        }

        let dumpster_dir = Path::new(&dir_path).join("dumpster");

        let dir_pool = state_clone.directory_db().get_pool(directory_id)?;
        let conn = dir_pool.get()?;

        // Get non-favorited images with their file paths
        let mut stmt = conn.prepare(
            "SELECT i.id, i.file_hash, f.id, f.original_path
             FROM images i
             JOIN image_files f ON f.image_id = i.id
             WHERE i.is_favorite = 0",
        )?;

        let files: Vec<(i64, Option<String>, i64, String)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut moved: i64 = 0;
        let mut failed: i64 = 0;
        let mut image_ids_to_delete: HashSet<i64> = HashSet::new();
        let mut file_ids_to_delete: Vec<i64> = Vec::new();
        let mut hashes_to_clean: Vec<String> = Vec::new();

        for (image_id, file_hash, file_id, original_path) in &files {
            let src = Path::new(original_path);
            if !src.exists() {
                // File already gone, just clean up DB record
                file_ids_to_delete.push(*file_id);
                image_ids_to_delete.insert(*image_id);
                if let Some(hash) = file_hash {
                    hashes_to_clean.push(hash.clone());
                }
                moved += 1;
                continue;
            }

            // Calculate relative path from directory root to preserve structure
            let relative = src
                .strip_prefix(&dir_path)
                .unwrap_or_else(|_| Path::new(src.file_name().unwrap_or_default()));

            let dest = dumpster_dir.join(relative);

            // Create parent directories in dumpster
            if let Some(parent) = dest.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    log::warn!("[Prune] Failed to create dumpster dir: {}", e);
                    failed += 1;
                    continue;
                }
            }

            // Move the file
            match std::fs::rename(src, &dest) {
                Ok(()) => {
                    file_ids_to_delete.push(*file_id);
                    image_ids_to_delete.insert(*image_id);
                    if let Some(hash) = file_hash {
                        hashes_to_clean.push(hash.clone());
                    }
                    moved += 1;
                }
                Err(e) => {
                    // rename may fail across filesystems; try copy+delete
                    match std::fs::copy(src, &dest).and_then(|_| std::fs::remove_file(src)) {
                        Ok(()) => {
                            file_ids_to_delete.push(*file_id);
                            image_ids_to_delete.insert(*image_id);
                            if let Some(hash) = file_hash {
                                hashes_to_clean.push(hash.clone());
                            }
                            moved += 1;
                        }
                        Err(_) => {
                            log::warn!("[Prune] Failed to move {}: {}", original_path, e);
                            failed += 1;
                        }
                    }
                }
            }
        }

        // Delete DB records for moved files
        for file_id in &file_ids_to_delete {
            let _ = conn.execute("DELETE FROM image_files WHERE id = ?1", params![file_id]);
        }

        // Delete images that have no remaining file references
        for image_id in &image_ids_to_delete {
            let remaining: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM image_files WHERE image_id = ?1",
                    params![image_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            if remaining == 0 {
                let _ = conn.execute("DELETE FROM images WHERE id = ?1", params![image_id]);
            }
        }

        // Clean thumbnails for deleted images
        let thumbnails_dir = state_clone.thumbnails_dir();
        for hash in &hashes_to_clean {
            let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
            let _ = std::fs::remove_file(thumbnails_dir.join(&thumb_name));
        }

        Ok::<_, AppError>(json!({
            "directory_id": directory_id,
            "moved": moved,
            "failed": failed,
            "dumpster_path": dumpster_dir.to_string_lossy()
        }))
    })
    .await??;

    Ok(Json(result))
}

// ─── Update Path ─────────────────────────────────────────────────────────────

/// PATCH /api/directories/:directory_id/path — Update the directory's filesystem path.
///
/// Updates the path in both the main DB watch_directories table and updates
/// all image file_path prefixes in the directory DB.
async fn update_directory_path(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
    Json(data): Json<UpdatePathRequest>,
) -> Result<Json<Value>, AppError> {
    let new_resolved = std::fs::canonicalize(&data.new_path)
        .map_err(|_| AppError::BadRequest(format!("New path does not exist: {}", data.new_path)))?;

    if !new_resolved.is_dir() {
        return Err(AppError::BadRequest(format!(
            "New path is not a directory: {}",
            data.new_path
        )));
    }

    let new_path_str = new_resolved.to_string_lossy().to_string();

    let state_clone = state.clone();
    let new_path_clone = new_path_str.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Get old path
        let (old_path, recursive): (String, bool) = main_conn
            .query_row(
                "SELECT path, recursive FROM watch_directories WHERE id = ?1",
                params![directory_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| AppError::NotFound("Directory not found".into()))?;

        if old_path == new_path_clone {
            return Ok::<_, AppError>(json!({
                "directory_id": directory_id,
                "old_path": old_path,
                "new_path": new_path_clone,
                "files_updated": 0,
                "message": "Path unchanged"
            }));
        }

        // Check for duplicate
        let dup_exists: bool = main_conn
            .query_row(
                "SELECT COUNT(*) FROM watch_directories WHERE path = ?1 AND id != ?2",
                params![&new_path_clone, directory_id],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);

        if dup_exists {
            return Err(AppError::BadRequest(
                "Another directory already uses this path".into(),
            ));
        }

        // Update main DB
        main_conn.execute(
            "UPDATE watch_directories SET path = ?1 WHERE id = ?2",
            params![&new_path_clone, directory_id],
        )?;

        // Update file path prefixes in directory DB
        let dir_pool = state_clone.directory_db().get_pool(directory_id)?;
        let conn = dir_pool.get()?;

        // Replace old_path prefix with new_path in all image_files.original_path
        let old_prefix = if old_path.ends_with('/') {
            old_path.clone()
        } else {
            format!("{}/", old_path)
        };
        let new_prefix = if new_path_clone.ends_with('/') {
            new_path_clone.clone()
        } else {
            format!("{}/", new_path_clone)
        };

        let files_updated = conn.execute(
            "UPDATE image_files SET original_path = ?1 || substr(original_path, ?2)
             WHERE original_path LIKE ?3",
            params![
                &new_prefix,
                old_prefix.len() as i64 + 1,
                format!("{}%", old_prefix)
            ],
        )?;

        // Update watcher: remove old, add new
        if let Some(watcher) = state_clone.directory_watcher() {
            watcher.remove_directory(directory_id);
            watcher.add_directory(directory_id, &new_path_clone, recursive);
        }

        Ok::<_, AppError>(json!({
            "directory_id": directory_id,
            "old_path": old_path,
            "new_path": new_path_clone,
            "files_updated": files_updated,
            "message": format!("Path updated, {} file references updated", files_updated)
        }))
    })
    .await??;

    Ok(Json(result))
}

// ─── Clean Deleted ───────────────────────────────────────────────────────────

/// POST /api/directories/:directory_id/clean-deleted — Remove records for files that
/// no longer exist on disk, without doing a full rescan.
async fn clean_deleted(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Verify directory exists in main DB
        let main_conn = state_clone.main_db().get()?;
        let _dir_path: String = main_conn
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![directory_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Directory not found".into()))?;

        let removed = file_tracker::clean_deleted_files(&state_clone, directory_id)?;

        Ok::<_, AppError>(json!({
            "directory_id": directory_id,
            "removed": removed,
            "message": format!("Removed {} records for missing files", removed)
        }))
    })
    .await??;

    Ok(Json(result))
}

// ─── Repair ─────────────────────────────────────────────────────────────────

/// Repair a single directory: fix moved file paths, remove truly missing files.
///
/// 1. Walks the filesystem to build a filename->path map
/// 2. For each DB record: if path exists mark valid, else try filename match, else delete
fn repair_directory_inner(
    state: &AppState,
    directory_id: i64,
) -> Result<Value, AppError> {
    let main_conn = state.main_db().get()?;

    let (dir_path, recursive): (String, bool) = main_conn
        .query_row(
            "SELECT path, recursive FROM watch_directories WHERE id = ?1",
            params![directory_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| AppError::NotFound("Directory not found".into()))?;

    if !Path::new(&dir_path).exists() {
        return Err(AppError::BadRequest("Directory path does not exist".into()));
    }

    // Step 1: Scan filesystem and build filename->path map.
    // Only keep unambiguous matches (filenames that appear exactly once).
    let mut name_to_paths: HashMap<String, Vec<String>> = HashMap::new();

    let walker: Box<dyn Iterator<Item = walkdir::DirEntry>> = if recursive {
        Box::new(walkdir::WalkDir::new(&dir_path).into_iter().filter_map(|e| e.ok()))
    } else {
        Box::new(
            walkdir::WalkDir::new(&dir_path)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok()),
        )
    };

    for entry in walker {
        if !entry.file_type().is_file() {
            continue;
        }
        if !importer::is_media_file(entry.path()) {
            continue;
        }
        let filename = entry.file_name().to_string_lossy().to_string();
        name_to_paths
            .entry(filename)
            .or_default()
            .push(entry.path().to_string_lossy().to_string());
    }

    // Unambiguous: only filenames with exactly one path
    let name_to_path: HashMap<&str, &str> = name_to_paths
        .iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(k, v)| (k.as_str(), v[0].as_str()))
        .collect();

    // Step 2: Check each DB record
    let dir_pool = state.directory_db().get_pool(directory_id)?;
    let conn = dir_pool.get()?;

    let mut stmt = conn.prepare("SELECT id, image_id, original_path FROM image_files")?;
    let files: Vec<(i64, i64, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();

    let mut valid: i64 = 0;
    let mut repaired: i64 = 0;
    let mut removed: i64 = 0;

    for (file_id, image_id, original_path) in files {
        if Path::new(&original_path).exists() {
            conn.execute(
                "UPDATE image_files SET file_exists = 1, file_status = 'available' WHERE id = ?1",
                params![file_id],
            )?;
            valid += 1;
            continue;
        }

        // Try to find by filename match
        let filename = Path::new(&original_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        if let Some(&new_path) = name_to_path.get(filename) {
            conn.execute(
                "UPDATE image_files SET original_path = ?1, file_exists = 1, file_status = 'available' WHERE id = ?2",
                params![new_path, file_id],
            )?;
            repaired += 1;
        } else {
            // Truly missing — check if last reference
            let other_count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM image_files WHERE image_id = ?1 AND id != ?2",
                    params![image_id, file_id],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            conn.execute("DELETE FROM image_files WHERE id = ?1", params![file_id])?;

            if other_count == 0 {
                // Delete image and thumbnail
                if let Ok(hash) = conn.query_row(
                    "SELECT file_hash FROM images WHERE id = ?1",
                    params![image_id],
                    |row| row.get::<_, String>(0),
                ) {
                    let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
                    let _ = std::fs::remove_file(state.thumbnails_dir().join(&thumb_name));
                }
                conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
            }

            removed += 1;
        }
    }

    Ok(json!({
        "directory_id": directory_id,
        "files_on_disk": name_to_path.len(),
        "valid": valid,
        "repaired": repaired,
        "removed": removed
    }))
}

/// POST /api/directories/:directory_id/repair — Repair file paths in a single directory.
async fn repair_directory(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        repair_directory_inner(&state_clone, directory_id)
    })
    .await??;

    Ok(Json(result))
}

/// POST /api/directories/bulk-repair — Repair all selected directories + clean orphan thumbnails.
async fn bulk_repair(
    State(state): State<AppState>,
    Json(data): Json<BulkRepairRequest>,
) -> Result<Json<Value>, AppError> {
    if data.directory_ids.is_empty() {
        return Ok(Json(json!({
            "results": [],
            "totals": { "valid": 0, "repaired": 0, "removed": 0, "orphan_thumbnails": 0 }
        })));
    }

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut results: Vec<Value> = Vec::new();
        let mut totals_valid: i64 = 0;
        let mut totals_repaired: i64 = 0;
        let mut totals_removed: i64 = 0;

        for dir_id in &data.directory_ids {
            match repair_directory_inner(&state_clone, *dir_id) {
                Ok(r) => {
                    totals_valid += r["valid"].as_i64().unwrap_or(0);
                    totals_repaired += r["repaired"].as_i64().unwrap_or(0);
                    totals_removed += r["removed"].as_i64().unwrap_or(0);
                    results.push(r);
                }
                Err(e) => {
                    log::warn!("[Repair] Directory {} failed: {}", dir_id, e);
                }
            }
        }

        // Clean orphan thumbnails after all repairs
        let orphan_thumbs = clean_orphan_thumbnails(&state_clone);

        Ok::<_, AppError>(json!({
            "results": results,
            "totals": {
                "valid": totals_valid,
                "repaired": totals_repaired,
                "removed": totals_removed,
                "orphan_thumbnails": orphan_thumbs
            },
            "message": format!(
                "Repaired {} directories: {} OK, {} fixed, {} removed, {} orphan thumbnails cleaned",
                results.len(), totals_valid, totals_repaired, totals_removed, orphan_thumbs
            )
        }))
    })
    .await??;

    Ok(Json(result))
}

/// Scan the thumbnails directory and remove any .webp files that don't correspond
/// to any image in any directory database.
fn clean_orphan_thumbnails(state: &AppState) -> i64 {
    let thumbnails_dir = state.thumbnails_dir();
    if !thumbnails_dir.exists() {
        return 0;
    }

    // Collect all known thumbnail filenames from all directory DBs
    let mut known_thumbs: HashSet<String> = HashSet::new();

    let dir_ids = state.directory_db().get_all_directory_ids();
    for dir_id in dir_ids {
        if let Ok(pool) = state.directory_db().get_pool(dir_id) {
            if let Ok(conn) = pool.get() {
                if let Ok(mut stmt) =
                    conn.prepare("SELECT file_hash FROM images WHERE file_hash IS NOT NULL")
                {
                    let hashes: Vec<String> = stmt
                        .query_map([], |row| row.get(0))
                        .ok()
                        .map(|r| r.filter_map(|r| r.ok()).collect())
                        .unwrap_or_default();

                    for hash in hashes {
                        let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
                        known_thumbs.insert(thumb_name);
                    }
                }
            }
        }
    }

    // Scan thumbnails directory and remove orphans
    let mut removed: i64 = 0;
    if let Ok(entries) = std::fs::read_dir(&thumbnails_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.ends_with(".webp") && !known_thumbs.contains(&filename) {
                if std::fs::remove_file(entry.path()).is_ok() {
                    removed += 1;
                }
            }
        }
    }

    if removed > 0 {
        log::info!(
            "[Repair] Cleaned {} orphan thumbnails ({} known)",
            removed,
            known_thumbs.len()
        );
    }

    removed
}

// ─── ComfyUI Node Discovery ─────────────────────────────────────────────────

/// GET /api/directories/:directory_id/comfyui-nodes — Scan sample images to discover ComfyUI nodes.
///
/// Reads up to 5 sample PNG images from the directory, extracts PNG text chunks
/// looking for the "prompt" chunk (ComfyUI metadata JSON), parses it, and returns
/// a list of all node IDs with their class types and whether they contain text.
async fn get_comfyui_nodes(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Get directory path and current config
        let (dir_path, recursive, prompt_ids_str, negative_ids_str): (String, bool, Option<String>, Option<String>) = main_conn
            .query_row(
                "SELECT path, recursive, comfyui_prompt_node_ids, comfyui_negative_node_ids
                 FROM watch_directories WHERE id = ?1",
                params![directory_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .map_err(|_| AppError::NotFound("Directory not found".into()))?;

        if !Path::new(&dir_path).exists() {
            return Err(AppError::BadRequest("Directory path does not exist".into()));
        }

        // Parse current config for the response
        let current_prompt_ids: Vec<String> = prompt_ids_str
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.split(',').map(|id| id.trim().to_string()).filter(|id| !id.is_empty()).collect())
            .unwrap_or_default();
        let current_negative_ids: Vec<String> = negative_ids_str
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.split(',').map(|id| id.trim().to_string()).filter(|id| !id.is_empty()).collect())
            .unwrap_or_default();

        // Walk directory to find PNG files (up to 5 samples)
        let walker: Box<dyn Iterator<Item = walkdir::DirEntry>> = if recursive {
            Box::new(walkdir::WalkDir::new(&dir_path).into_iter().filter_map(|e| e.ok()))
        } else {
            Box::new(
                walkdir::WalkDir::new(&dir_path)
                    .max_depth(1)
                    .into_iter()
                    .filter_map(|e| e.ok()),
            )
        };

        let mut sample_pngs: Vec<String> = Vec::new();
        for entry in walker {
            if !entry.file_type().is_file() {
                continue;
            }
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            if ext.as_deref() == Some("png") {
                sample_pngs.push(entry.path().to_string_lossy().to_string());
                if sample_pngs.len() >= 5 {
                    break;
                }
            }
        }

        // Scan each sample for ComfyUI metadata and collect unique nodes
        let text_keys = ["text", "string", "prompt", "clip_l", "clip_g", "positive", "negative"];
        let mut seen_nodes: HashSet<String> = HashSet::new();
        let mut nodes: Vec<Value> = Vec::new();

        for png_path in &sample_pngs {
            let chunks = match metadata::extract_png_text_chunks(png_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let prompt_json = match chunks.get("prompt") {
                Some(p) => p,
                None => continue,
            };

            let prompt_data: serde_json::Value = match serde_json::from_str(prompt_json) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let obj = match prompt_data.as_object() {
                Some(o) => o,
                None => continue,
            };

            for (node_id, node) in obj {
                if seen_nodes.contains(node_id) {
                    continue;
                }
                seen_nodes.insert(node_id.clone());

                let class_type = node
                    .get("class_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown")
                    .to_string();

                // Check if this node has any text inputs
                let mut has_text = false;
                let mut sample_text: Option<String> = None;
                let mut field_name: Option<String> = None;

                if let Some(inputs) = node.get("inputs").and_then(|i| i.as_object()) {
                    for &key in &text_keys {
                        if let Some(text) = inputs.get(key).and_then(|v| v.as_str()) {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                has_text = true;
                                field_name = Some(key.to_string());
                                // Truncate sample text for display
                                sample_text = Some(if trimmed.len() > 200 {
                                    format!("{}...", &trimmed[..200])
                                } else {
                                    trimmed.to_string()
                                });
                                break;
                            }
                        }
                    }
                }

                nodes.push(json!({
                    "node_id": node_id,
                    "node_type": class_type,
                    "has_text": has_text,
                    "field": field_name,
                    "sample_text": sample_text,
                }));
            }
        }

        // Sort nodes: text-containing nodes first, then by node_id
        nodes.sort_by(|a, b| {
            let a_has = a["has_text"].as_bool().unwrap_or(false);
            let b_has = b["has_text"].as_bool().unwrap_or(false);
            b_has.cmp(&a_has).then_with(|| {
                let a_id = a["node_id"].as_str().unwrap_or("");
                let b_id = b["node_id"].as_str().unwrap_or("");
                a_id.cmp(b_id)
            })
        });

        Ok::<_, AppError>(Json(json!({
            "nodes": nodes,
            "samples_scanned": sample_pngs.len(),
            "current_config": {
                "comfyui_prompt_node_ids": current_prompt_ids,
                "comfyui_negative_node_ids": current_negative_ids
            }
        })))
    })
    .await??;

    Ok(result)
}

// ─── ComfyUI Config Update ──────────────────────────────────────────────────

/// PATCH /api/directories/:directory_id/comfyui-config — Update ComfyUI metadata configuration.
///
/// Accepts `comfyui_prompt_node_ids`, `comfyui_negative_node_ids`, and `metadata_format`.
/// Validates metadata_format is one of "auto", "a1111", "comfyui", "none".
async fn update_comfyui_config(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
    Json(data): Json<ComfyUIConfigUpdate>,
) -> Result<Json<Value>, AppError> {
    // Validate metadata_format
    if let Some(ref format) = data.metadata_format {
        let valid_formats = ["auto", "a1111", "comfyui", "none"];
        if !valid_formats.contains(&format.as_str()) {
            return Err(AppError::BadRequest(format!(
                "Invalid metadata_format '{}'. Must be one of: auto, a1111, comfyui, none",
                format
            )));
        }
    }

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Build dynamic UPDATE query
        let mut sets = Vec::new();
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(ref prompt_ids) = data.comfyui_prompt_node_ids {
            sets.push("comfyui_prompt_node_ids = ?");
            sql_params.push(Box::new(prompt_ids.clone()));
        }
        if let Some(ref negative_ids) = data.comfyui_negative_node_ids {
            sets.push("comfyui_negative_node_ids = ?");
            sql_params.push(Box::new(negative_ids.clone()));
        }
        if let Some(ref format) = data.metadata_format {
            sets.push("metadata_format = ?");
            sql_params.push(Box::new(format.clone()));
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

        Ok::<_, AppError>(Json(json!({
            "id": directory_id,
            "updated": true,
            "comfyui_prompt_node_ids": data.comfyui_prompt_node_ids,
            "comfyui_negative_node_ids": data.comfyui_negative_node_ids,
            "metadata_format": data.metadata_format
        })))
    })
    .await??;

    Ok(result)
}

// ─── Re-extract Metadata ─────────────────────────────────────────────────────

/// POST /api/directories/:directory_id/reextract-metadata — Queue metadata re-extraction for all images.
///
/// For each image in the directory DB, enqueues a TASK_EXTRACT_METADATA task
/// with the image_id, directory_id, and image path.
async fn reextract_metadata(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Verify directory exists
        let _dir_path: String = main_conn
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![directory_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Directory not found".into()))?;

        // Get all images with their file paths from the directory DB
        let dir_pool = state_clone.directory_db().get_pool(directory_id)?;
        let dir_conn = dir_pool.get()?;

        let mut stmt = dir_conn.prepare(
            "SELECT i.id, f.original_path
             FROM images i
             JOIN image_files f ON f.image_id = i.id
             WHERE f.file_exists = 1"
        )?;

        let images: Vec<(i64, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        let mut queued: i64 = 0;
        for (image_id, image_path) in &images {
            match task_queue::enqueue_task(
                &state_clone,
                task_queue::TASK_EXTRACT_METADATA,
                &json!({
                    "image_id": image_id,
                    "directory_id": directory_id,
                    "image_path": image_path
                }),
                1,  // lower priority than scans
                Some(*image_id),  // dedupe by image_id
            ) {
                Ok(Some(_)) => queued += 1,
                Ok(None) => {} // duplicate, skip
                Err(e) => {
                    log::warn!(
                        "[ReextractMetadata] Failed to enqueue task for image #{}: {}",
                        image_id, e
                    );
                }
            }
        }

        Ok::<_, AppError>(Json(json!({
            "directory_id": directory_id,
            "queued": queued,
            "total_images": images.len(),
            "message": format!("Queued metadata extraction for {} images", queued)
        })))
    })
    .await??;

    Ok(result)
}
