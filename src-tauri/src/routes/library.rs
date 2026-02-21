use std::convert::Infallible;

use axum::extract::{Path as AxumPath, Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Json;
use axum::routing::{delete, get, post};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::{file_tracker, task_queue};

pub fn router() -> Router<AppState> {
    Router::new()
        // Stats
        .route("/stats", get(library_stats))
        .route("/untagged", get(untagged_count))
        // SSE events
        .route("/events", get(library_events_stream))
        // Task queue management
        .route("/queue", get(queue_status))
        .route("/queue/paused", get(queue_paused))
        .route("/queue/pause", post(pause_queue))
        .route("/queue/resume", post(resume_queue))
        .route("/queue/retry-failed", post(retry_failed))
        .route("/queue/pending", delete(clear_pending))
        .route(
            "/queue/pending/directory/{directory_id}",
            delete(clear_directory_pending),
        )
        .route("/clear-duplicate-tasks", post(clear_duplicates))
        .route("/clear-pending-tasks", delete(clear_all_pending))
        // Batch task enqueue
        .route("/tag-untagged", post(tag_untagged))
        .route("/detect-ages", post(detect_ages))
        // Maintenance
        .route("/clean-missing", post(clean_missing))
        .route("/regenerate-thumbnails", post(regenerate_thumbnails))
        .route("/verify-files", post(verify_files))
        .route("/import-file", post(import_file))
        .route("/file-missing", post(file_missing))
}

// ─── Stats ───────────────────────────────────────────────────────────────────

/// GET /api/library/stats — Library-wide statistics.
async fn library_stats(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;
        let mut total: i64 = 0;
        let mut favorites: i64 = 0;
        let mut missing: i64 = 0;
        let mut by_rating: serde_json::Map<String, Value> = serde_json::Map::new();

        // Aggregate stats from all per-directory databases
        let dir_ids = state_clone.directory_db().get_all_directory_ids();
        for dir_id in &dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(*dir_id) {
                if let Ok(conn) = pool.get() {
                    let count: i64 = conn
                        .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                        .unwrap_or(0);
                    total += count;

                    let fav: i64 = conn
                        .query_row(
                            "SELECT COUNT(*) FROM images WHERE is_favorite = 1",
                            [],
                            |r| r.get(0),
                        )
                        .unwrap_or(0);
                    favorites += fav;

                    // Rating counts
                    if let Ok(mut stmt) =
                        conn.prepare("SELECT rating, COUNT(*) FROM images GROUP BY rating")
                    {
                        let ratings: Vec<(String, i64)> = stmt
                            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                            .ok()
                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                            .unwrap_or_default();
                        for (rating, count) in ratings {
                            let key = if rating.is_empty() {
                                "unrated".to_string()
                            } else {
                                rating
                            };
                            let entry = by_rating.entry(key).or_insert(json!(0));
                            *entry =
                                json!(entry.as_i64().unwrap_or(0) + count);
                        }
                    }

                    // Missing files
                    let miss: i64 = conn
                        .query_row(
                            "SELECT COUNT(*) FROM image_files WHERE file_status = 'missing'",
                            [],
                            |r| r.get(0),
                        )
                        .unwrap_or(0);
                    missing += miss;
                }
            }
        }

        // Tags (always from main DB)
        let total_tags: i64 = main_conn
            .query_row("SELECT COUNT(*) FROM tags", [], |r| r.get(0))
            .unwrap_or(0);

        // Watch directories
        let total_dirs: i64 = main_conn
            .query_row("SELECT COUNT(*) FROM watch_directories", [], |r| r.get(0))
            .unwrap_or(0);

        // Task queue
        let pending: i64 = main_conn
            .query_row(
                "SELECT COUNT(*) FROM task_queue WHERE status = 'pending'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let processing: i64 = main_conn
            .query_row(
                "SELECT COUNT(*) FROM task_queue WHERE status = 'processing'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        Ok::<_, AppError>(Json(json!({
            "total_images": total,
            "favorites": favorites,
            "total_tags": total_tags,
            "watch_directories": total_dirs,
            "missing_files": missing,
            "by_rating": by_rating,
            "queue": {
                "pending": pending,
                "processing": processing
            }
        })))
    })
    .await?
}

/// GET /api/library/untagged — Count of untagged images.
async fn untagged_count(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let mut untagged: i64 = 0;
        let dir_ids = state_clone.directory_db().get_all_directory_ids();

        for dir_id in &dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(*dir_id) {
                if let Ok(conn) = pool.get() {
                    let count: i64 = conn
                        .query_row(
                            "SELECT COUNT(*) FROM images WHERE id NOT IN (SELECT DISTINCT image_id FROM image_tags)",
                            [],
                            |r| r.get(0),
                        )
                        .unwrap_or(0);
                    untagged += count;
                }
            }
        }

        Ok::<_, AppError>(Json(json!({ "untagged_count": untagged })))
    })
    .await?
}

// ─── SSE Events ──────────────────────────────────────────────────────────────

/// GET /api/library/events — Server-Sent Events stream for real-time updates.
async fn library_events_stream(
    State(state): State<AppState>,
) -> Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>> {
    let stream = state
        .events()
        .expect("events initialized")
        .library
        .sse_stream();

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ─── Task Queue Management ──────────────────────────────────────────────────

/// GET /api/library/queue — Queue status.
async fn queue_status(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Count by status
        let mut by_status: serde_json::Map<String, Value> = serde_json::Map::new();
        {
            let mut stmt =
                conn.prepare("SELECT status, COUNT(*) FROM task_queue GROUP BY status")?;
            let rows: Vec<(String, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .ok()
                .map(|r| r.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();
            for (status, count) in rows {
                by_status.insert(status, json!(count));
            }
        }

        // Count pending/processing by type
        let mut by_type: serde_json::Map<String, Value> = serde_json::Map::new();
        {
            let mut stmt = conn.prepare(
                "SELECT task_type, COUNT(*) FROM task_queue WHERE status IN ('pending', 'processing') GROUP BY task_type",
            )?;
            let rows: Vec<(String, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .ok()
                .map(|r| r.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();
            for (task_type, count) in rows {
                by_type.insert(task_type, json!(count));
            }
        }

        // Recent failed tasks
        let failed_tasks: Vec<Value> = {
            let mut stmt = conn.prepare(
                "SELECT id, task_type, error_message, attempts FROM task_queue WHERE status = 'failed' ORDER BY completed_at DESC LIMIT 10",
            )?;
            stmt.query_map([], |row| {
                    Ok(json!({
                        "id": row.get::<_, i64>(0)?,
                        "type": row.get::<_, String>(1)?,
                        "error": row.get::<_, Option<String>>(2)?,
                        "attempts": row.get::<_, i64>(3)?
                    }))
                })
                .ok()
                .map(|r| r.filter_map(|r| r.ok()).collect())
                .unwrap_or_default()
        };

        Ok::<_, AppError>(Json(json!({
            "by_status": by_status,
            "pending_by_type": by_type,
            "recent_failures": failed_tasks
        })))
    })
    .await?
}

/// GET /api/library/queue/paused — Check if queue is paused.
async fn queue_paused(State(state): State<AppState>) -> Json<Value> {
    let paused = state
        .task_queue()
        .map(|tq| tq.is_paused())
        .unwrap_or(false);
    Json(json!({ "paused": paused }))
}

/// POST /api/library/queue/pause — Pause the task queue.
async fn pause_queue(State(state): State<AppState>) -> Json<Value> {
    if let Some(tq) = state.task_queue() {
        tq.pause();
    }
    Json(json!({ "paused": true }))
}

/// POST /api/library/queue/resume — Resume the task queue.
async fn resume_queue(State(state): State<AppState>) -> Json<Value> {
    if let Some(tq) = state.task_queue() {
        tq.resume();
    }
    Json(json!({ "paused": false }))
}

/// POST /api/library/queue/retry-failed — Retry all failed tasks.
async fn retry_failed(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let updated = conn.execute(
            "UPDATE task_queue SET status = 'pending', attempts = 0, error_message = NULL WHERE status = 'failed'",
            [],
        )?;
        Ok::<_, AppError>(Json(json!({ "retried": updated })))
    })
    .await?
}

/// DELETE /api/library/queue/pending — Clear all pending tasks.
async fn clear_pending(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let deleted = conn.execute(
            "DELETE FROM task_queue WHERE status = 'pending'",
            [],
        )?;
        Ok::<_, AppError>(Json(json!({ "cleared": deleted })))
    })
    .await?
}

/// DELETE /api/library/queue/pending/directory/:directory_id — Clear pending tasks for a directory.
async fn clear_directory_pending(
    State(state): State<AppState>,
    AxumPath(directory_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let mut stmt = conn.prepare(
            "SELECT id, payload FROM task_queue WHERE task_type = 'tag' AND status = 'pending'",
        )?;

        let tasks: Vec<(i64, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .ok()
            .map(|r| r.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        let mut cleared = 0;
        for (task_id, payload_str) in tasks {
            if let Ok(payload) = serde_json::from_str::<Value>(&payload_str) {
                if payload["directory_id"].as_i64() == Some(directory_id) {
                    conn.execute(
                        "DELETE FROM task_queue WHERE id = ?1",
                        params![task_id],
                    )?;
                    cleared += 1;
                }
            }
        }

        Ok::<_, AppError>(Json(json!({ "cleared": cleared })))
    })
    .await?
}

/// POST /api/library/clear-duplicate-tasks — Remove duplicate pending tasks.
async fn clear_duplicates(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let removed = task_queue::clear_duplicate_tasks(&state_clone)?;
        Ok::<_, AppError>(Json(json!({ "duplicates_removed": removed })))
    })
    .await?
}

#[derive(Deserialize)]
struct ClearPendingParams {
    task_type: Option<String>,
}

/// DELETE /api/library/clear-pending-tasks — Clear pending tasks, optionally by type.
async fn clear_all_pending(
    State(state): State<AppState>,
    Query(params): Query<ClearPendingParams>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let cancelled = if let Some(task_type) = &params.task_type {
            conn.execute(
                "UPDATE task_queue SET status = 'failed', error_message = 'Cancelled by user' WHERE status = 'pending' AND task_type = ?1",
                params![task_type],
            )?
        } else {
            conn.execute(
                "UPDATE task_queue SET status = 'failed', error_message = 'Cancelled by user' WHERE status = 'pending'",
                [],
            )?
        };

        Ok::<_, AppError>(Json(json!({ "cancelled": cancelled })))
    })
    .await?
}

// ─── Batch task enqueue ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct TagUntaggedParams {
    directory_id: Option<i64>,
}

/// POST /api/library/tag-untagged — Enqueue TASK_TAG tasks for all untagged images.
async fn tag_untagged(
    State(state): State<AppState>,
    Query(params): Query<TagUntaggedParams>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let dir_ids = if let Some(dir_id) = params.directory_id {
            vec![dir_id]
        } else {
            state_clone.directory_db().get_all_directory_ids()
        };
        let mut queued: i64 = 0;

        for dir_id in dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(dir_id) {
                if let Ok(conn) = pool.get() {
                    // Find images with no tags
                    let mut stmt = conn.prepare(
                        "SELECT id FROM images WHERE id NOT IN (SELECT DISTINCT image_id FROM image_tags)",
                    )?;

                    let image_ids: Vec<i64> = stmt
                        .query_map([], |row| row.get(0))
                        .ok()
                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                        .unwrap_or_default();

                    for image_id in image_ids {
                        let payload = json!({
                            "image_id": image_id,
                            "directory_id": dir_id
                        });
                        if let Ok(Some(_)) = task_queue::enqueue_task(
                            &state_clone,
                            task_queue::TASK_TAG,
                            &payload,
                            5, // low priority
                            Some(image_id),
                        ) {
                            queued += 1;
                        }
                    }
                }
            }
        }

        Ok::<_, AppError>(Json(json!({ "queued": queued })))
    })
    .await?
}

/// POST /api/library/detect-ages — Enqueue TASK_AGE_DETECT tasks for all images.
async fn detect_ages(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let dir_ids = state_clone.directory_db().get_all_directory_ids();
        let mut queued: i64 = 0;

        for dir_id in dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(dir_id) {
                if let Ok(conn) = pool.get() {
                    // Enqueue age detection for all images
                    let mut stmt = conn.prepare("SELECT id FROM images")?;

                    let image_ids: Vec<i64> = stmt
                        .query_map([], |row| row.get(0))
                        .ok()
                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                        .unwrap_or_default();

                    for image_id in image_ids {
                        let payload = json!({
                            "image_id": image_id,
                            "directory_id": dir_id
                        });
                        if let Ok(Some(_)) = task_queue::enqueue_task(
                            &state_clone,
                            task_queue::TASK_AGE_DETECT,
                            &payload,
                            5, // low priority
                            Some(image_id),
                        ) {
                            queued += 1;
                        }
                    }
                }
            }
        }

        Ok::<_, AppError>(Json(json!({ "queued": queued })))
    })
    .await?
}

// ─── Maintenance ─────────────────────────────────────────────────────────────

/// POST /api/library/clean-missing — Remove images with missing files.
async fn clean_missing(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let lib = state_clone.library_manager().primary().clone();
        let dir_ids = state_clone.directory_db().get_all_directory_ids();
        let mut total_removed: i64 = 0;

        for dir_id in dir_ids {
            match file_tracker::clean_deleted_files(&lib, dir_id) {
                Ok(removed) => total_removed += removed,
                Err(e) => log::error!("[Maintenance] Clean failed for dir {}: {}", dir_id, e),
            }
        }

        Ok::<_, AppError>(Json(json!({
            "removed": total_removed,
            "message": format!("Removed {} missing file references", total_removed)
        })))
    })
    .await?
}

#[derive(Deserialize)]
struct RegenerateThumbnailsParams {
    directory_id: Option<i64>,
}

/// POST /api/library/regenerate-thumbnails — Regenerate missing thumbnails.
async fn regenerate_thumbnails(
    State(state): State<AppState>,
    Query(params): Query<RegenerateThumbnailsParams>,
) -> Result<Json<Value>, AppError> {
    use crate::services::importer;

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let thumbnails_dir = state_clone.thumbnails_dir();
        std::fs::create_dir_all(&thumbnails_dir).ok();

        let dir_ids = if let Some(dir_id) = params.directory_id {
            vec![dir_id]
        } else {
            state_clone.directory_db().get_all_directory_ids()
        };

        let mut missing = 0i64;
        let mut regenerated = 0i64;
        let mut failed = 0i64;

        for dir_id in dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(dir_id) {
                if let Ok(conn) = pool.get() {
                    let mut stmt = conn
                        .prepare(
                            "SELECT i.file_hash, f.original_path FROM images i
                             JOIN image_files f ON f.image_id = i.id
                             WHERE f.file_exists = 1 AND i.file_hash IS NOT NULL",
                        )
                        .ok();

                    if let Some(ref mut stmt) = stmt {
                        let files: Vec<(String, String)> = stmt
                            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                            .ok()
                            .map(|r| r.filter_map(|r| r.ok()).collect())
                            .unwrap_or_default();

                        for (file_hash, file_path) in files {
                            let thumb_name =
                                format!("{}.webp", &file_hash[..16.min(file_hash.len())]);
                            let thumb_path = thumbnails_dir.join(&thumb_name);

                            if thumb_path.exists() {
                                continue;
                            }
                            missing += 1;

                            let ok = if importer::is_video_file(&file_path) {
                                importer::generate_video_thumbnail(
                                    &file_path,
                                    &thumb_path.to_string_lossy(),
                                    400,
                                )
                            } else {
                                importer::generate_thumbnail(
                                    &file_path,
                                    &thumb_path.to_string_lossy(),
                                    400,
                                )
                            };

                            if ok {
                                regenerated += 1;
                            } else {
                                failed += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok::<_, AppError>(Json(json!({
            "missing": missing,
            "regenerated": regenerated,
            "failed": failed
        })))
    })
    .await?
}

/// POST /api/library/verify-files — Queue file verification task.
async fn verify_files(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let task_id = task_queue::enqueue_task(
            &state_clone,
            task_queue::TASK_VERIFY_FILES,
            &json!({}),
            1,
            None,
        )?;
        Ok::<_, AppError>(Json(json!({
            "task_id": task_id,
            "message": "File verification queued"
        })))
    })
    .await?
}

#[derive(Deserialize)]
struct ImportFileRequest {
    file_path: String,
    watch_directory_id: Option<i64>,
}

/// POST /api/library/import-file — Import a single file.
async fn import_file(
    State(state): State<AppState>,
    Json(data): Json<ImportFileRequest>,
) -> Result<Json<Value>, AppError> {
    use crate::services::importer;

    let directory_id = data
        .watch_directory_id
        .ok_or_else(|| AppError::BadRequest("watch_directory_id required".into()))?;

    let state_clone = state.clone();
    let lib = state.library_manager().primary().clone();
    let file_path = data.file_path.clone();
    tokio::task::spawn_blocking(move || {
        let result = importer::import_image(&state_clone, &lib, &file_path, directory_id, false)?;
        Ok::<_, AppError>(Json(json!({
            "status": match result.status {
                importer::ImportStatus::Imported => "imported",
                importer::ImportStatus::Duplicate => "duplicate",
                importer::ImportStatus::Error => "error",
            },
            "image_id": result.image_id,
            "directory_id": result.directory_id,
            "filename": result.filename,
            "message": result.message
        })))
    })
    .await?
}

#[derive(Deserialize)]
struct FileMissingRequest {
    file_path: String,
}

/// POST /api/library/file-missing — Mark a file as missing.
async fn file_missing(
    State(state): State<AppState>,
    Json(data): Json<FileMissingRequest>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let file_path = data.file_path.clone();
    tokio::task::spawn_blocking(move || {
        let lib = state_clone.library_manager().primary().clone();
        let dir_ids = state_clone.directory_db().get_all_directory_ids();
        for dir_id in dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(dir_id) {
                if let Ok(conn) = pool.get() {
                    let exists: bool = conn
                        .query_row(
                            "SELECT COUNT(*) FROM image_files WHERE original_path = ?1",
                            params![&file_path],
                            |row| row.get::<_, i64>(0).map(|c| c > 0),
                        )
                        .unwrap_or(false);

                    if exists {
                        file_tracker::mark_file_missing(&lib, &file_path, dir_id)?;
                        return Ok(Json(json!({ "marked_missing": true })));
                    }
                }
            }
        }
        Ok::<_, AppError>(Json(json!({ "marked_missing": false, "message": "File not found in any directory" })))
    })
    .await?
}
