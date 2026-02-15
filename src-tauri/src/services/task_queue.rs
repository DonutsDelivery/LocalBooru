use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rusqlite::params;
use serde_json::Value;
use tokio::sync::Notify;

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::events::event_type;
use crate::services::file_tracker;

// ─── Task types (match Python TaskType enum) ─────────────────────────────────

pub const TASK_TAG: &str = "tag";
pub const TASK_SCAN_DIRECTORY: &str = "scan_directory";
pub const TASK_VERIFY_FILES: &str = "verify_files";
pub const TASK_AGE_DETECT: &str = "age_detect";
pub const TASK_EXTRACT_METADATA: &str = "extract_metadata";

pub const STATUS_PENDING: &str = "pending";
pub const STATUS_PROCESSING: &str = "processing";
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_FAILED: &str = "failed";

// ─── BackgroundTaskQueue ─────────────────────────────────────────────────────

/// Background task queue that processes pending tasks from the database.
pub struct BackgroundTaskQueue {
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl BackgroundTaskQueue {
    pub fn new() -> Self {
        Self {
            paused: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
        log::info!("[TaskQueue] Paused");
    }

    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
        self.notify.notify_one(); // Wake the worker
        log::info!("[TaskQueue] Resumed");
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Wake the worker to check for new tasks immediately.
    pub fn wake(&self) {
        self.notify.notify_one();
    }

    /// Start the background worker loop.
    pub fn start(&self, state: AppState) {
        if self.running.load(Ordering::SeqCst) {
            log::warn!("[TaskQueue] Already running");
            return;
        }

        self.running.store(true, Ordering::SeqCst);
        let paused = self.paused.clone();
        let running = self.running.clone();
        let notify = self.notify.clone();

        // Reset stuck tasks on startup
        if let Err(e) = reset_stuck_tasks(&state) {
            log::error!("[TaskQueue] Failed to reset stuck tasks: {}", e);
        }

        tokio::spawn(async move {
            log::info!("[TaskQueue] Worker started");

            loop {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                if paused.load(Ordering::SeqCst) {
                    // Wait until resumed or notified
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }

                // Try to process one task
                match process_next_task(&state).await {
                    Ok(true) => {
                        // Processed a task, immediately check for more
                        continue;
                    }
                    Ok(false) => {
                        // No tasks available, wait for notification or timeout
                        tokio::select! {
                            _ = notify.notified() => {},
                            _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                        }
                    }
                    Err(e) => {
                        log::error!("[TaskQueue] Error processing task: {}", e);
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                }
            }

            log::info!("[TaskQueue] Worker stopped");
        });
    }

    /// Stop the background worker.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.notify.notify_one();
    }
}

impl Default for BackgroundTaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Task processing ─────────────────────────────────────────────────────────

/// Reset any tasks stuck in "processing" state (from previous crash).
fn reset_stuck_tasks(state: &AppState) -> Result<(), AppError> {
    let conn = state.main_db().get()?;
    let updated = conn.execute(
        "UPDATE task_queue SET status = ?1 WHERE status = ?2",
        params![STATUS_PENDING, STATUS_PROCESSING],
    )?;
    if updated > 0 {
        log::info!("[TaskQueue] Reset {} stuck tasks to pending", updated);
    }
    Ok(())
}

/// Try to fetch and process the next pending task.
/// Returns true if a task was processed, false if queue is empty.
async fn process_next_task(state: &AppState) -> Result<bool, AppError> {
    let state_clone = state.clone();

    // Fetch next task (blocking DB operation)
    let task_info = tokio::task::spawn_blocking(move || -> Result<Option<(i64, String, String)>, AppError> {
        let conn = state_clone.main_db().get()?;

        // Fetch highest-priority pending task
        let result = conn.query_row(
            "SELECT id, task_type, payload FROM task_queue
             WHERE status = ?1
             ORDER BY priority DESC, created_at ASC
             LIMIT 1",
            params![STATUS_PENDING],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        );

        match result {
            Ok(task) => {
                // Mark as processing
                conn.execute(
                    "UPDATE task_queue SET status = ?1, started_at = datetime('now') WHERE id = ?2",
                    params![STATUS_PROCESSING, task.0],
                )?;
                Ok(Some(task))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AppError::from(e)),
        }
    })
    .await??;

    let (task_id, task_type, payload_str) = match task_info {
        Some(t) => t,
        None => return Ok(false),
    };

    log::info!("[TaskQueue] Processing task #{} ({})", task_id, task_type);

    let payload: Value = serde_json::from_str(&payload_str).unwrap_or_default();

    // Dispatch to handler
    let result = execute_task(state, &task_type, &payload).await;

    // Update task status
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || -> Result<(), AppError> {
        let conn = state_clone.main_db().get()?;
        match result {
            Ok(()) => {
                conn.execute(
                    "UPDATE task_queue SET status = ?1, completed_at = datetime('now') WHERE id = ?2",
                    params![STATUS_COMPLETED, task_id],
                )?;
            }
            Err(e) => {
                let error_msg = e.to_string();
                conn.execute(
                    "UPDATE task_queue SET status = ?1, error_message = ?2, attempts = attempts + 1, completed_at = datetime('now') WHERE id = ?3",
                    params![STATUS_FAILED, &error_msg, task_id],
                )?;
                log::error!("[TaskQueue] Task #{} failed: {}", task_id, error_msg);
            }
        }
        Ok(())
    })
    .await??;

    // Broadcast task completion event
    if let Some(events) = state.events() {
        events.library.broadcast(
            event_type::TASK_COMPLETED,
            serde_json::json!({ "task_id": task_id, "task_type": task_type }),
        );
    }

    Ok(true)
}

/// Execute a task based on its type.
async fn execute_task(state: &AppState, task_type: &str, payload: &Value) -> Result<(), AppError> {
    match task_type {
        TASK_SCAN_DIRECTORY => {
            let directory_id = payload["directory_id"]
                .as_i64()
                .ok_or_else(|| AppError::Internal("Missing directory_id".into()))?;
            let directory_path = payload["directory_path"]
                .as_str()
                .ok_or_else(|| AppError::Internal("Missing directory_path".into()))?
                .to_string();
            let clean_deleted = payload["clean_deleted"].as_bool().unwrap_or(false);

            let state_clone = state.clone();
            tokio::task::spawn_blocking(move || {
                let stats = file_tracker::scan_directory(
                    &state_clone,
                    directory_id,
                    &directory_path,
                    true, // recursive
                    clean_deleted,
                )?;
                log::info!(
                    "[TaskQueue] Scan complete: {} found, {} imported, {} duplicates, {} errors",
                    stats.found,
                    stats.imported,
                    stats.duplicates,
                    stats.errors
                );
                Ok::<_, AppError>(())
            })
            .await??;
        }

        TASK_VERIFY_FILES => {
            let directory_id = payload["directory_id"].as_i64();
            let state_clone = state.clone();
            tokio::task::spawn_blocking(move || {
                if let Some(dir_id) = directory_id {
                    let stats = file_tracker::verify_directory_files(&state_clone, dir_id)?;
                    log::info!(
                        "[TaskQueue] Verify complete: {} verified, {} deleted, {} offline",
                        stats.verified,
                        stats.deleted,
                        stats.drive_offline
                    );
                }
                Ok::<_, AppError>(())
            })
            .await??;
        }

        TASK_TAG | TASK_AGE_DETECT => {
            // These tasks require addon services (Python sidecar).
            // For now, log and skip — will be handled when addon system is built.
            log::info!(
                "[TaskQueue] Skipping {} task (addon not available)",
                task_type
            );
        }

        TASK_EXTRACT_METADATA => {
            // Metadata extraction will be implemented when we add the metadata service.
            // For fast imports, this completes the deferred work.
            let image_id = payload["image_id"].as_i64();
            let directory_id = payload["directory_id"].as_i64();
            let complete_import = payload["complete_import"].as_bool().unwrap_or(false);

            if complete_import {
                if let (Some(img_id), Some(dir_id)) = (image_id, directory_id) {
                    let image_path = payload["image_path"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();
                    let state_clone = state.clone();

                    tokio::task::spawn_blocking(move || {
                        complete_fast_import(&state_clone, img_id, dir_id, &image_path)
                    })
                    .await??;
                }
            }
            // Actual metadata extraction (EXIF, ComfyUI, etc.) deferred to later phase
        }

        _ => {
            log::warn!("[TaskQueue] Unknown task type: {}", task_type);
        }
    }

    Ok(())
}

/// Complete a fast import by calculating full hash, dimensions, and thumbnail.
fn complete_fast_import(
    state: &AppState,
    image_id: i64,
    directory_id: i64,
    image_path: &str,
) -> Result<(), AppError> {
    use crate::services::importer;

    if !std::path::Path::new(image_path).exists() {
        return Ok(());
    }

    let dir_pool = state.directory_db().get_pool(directory_id)?;
    let conn = dir_pool.get()?;

    // Calculate full hash
    let full_hash = importer::calculate_file_hash(image_path)
        .map_err(|e| AppError::Internal(format!("Hash error: {}", e)))?;

    // Get dimensions
    let dims = importer::get_image_dimensions(image_path);

    // Update the image record
    if let Some((w, h)) = dims {
        conn.execute(
            "UPDATE images SET file_hash = ?1, width = ?2, height = ?3 WHERE id = ?4",
            params![&full_hash, w as i32, h as i32, image_id],
        )?;
    } else {
        conn.execute(
            "UPDATE images SET file_hash = ?1 WHERE id = ?2",
            params![&full_hash, image_id],
        )?;
    }

    // Generate thumbnail if missing
    let thumbnails_dir = state.thumbnails_dir();
    std::fs::create_dir_all(&thumbnails_dir).ok();
    let thumb_name = format!("{}.webp", &full_hash[..16.min(full_hash.len())]);
    let thumb_path = thumbnails_dir.join(&thumb_name);

    if !thumb_path.exists() {
        let is_video = importer::is_video_file(image_path);
        if is_video {
            importer::generate_video_thumbnail(image_path, &thumb_path.to_string_lossy(), 400);
        } else {
            importer::generate_thumbnail(image_path, &thumb_path.to_string_lossy(), 400);
        }
    }

    Ok(())
}

// ─── Enqueue helpers ─────────────────────────────────────────────────────────

/// Enqueue a task to the background queue.
///
/// If `dedupe_key` is provided, skips creating the task if one with the same
/// image_id is already pending for that task type.
pub fn enqueue_task(
    state: &AppState,
    task_type: &str,
    payload: &Value,
    priority: i32,
    dedupe_key: Option<i64>,
) -> Result<Option<i64>, AppError> {
    let conn = state.main_db().get()?;

    // Deduplication check
    if let Some(image_id) = dedupe_key {
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM task_queue WHERE task_type = ?1 AND status = ?2 AND payload LIKE ?3",
                params![task_type, STATUS_PENDING, format!("%\"image_id\":{}%", image_id)],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);

        if exists {
            return Ok(None);
        }
    }

    let payload_str = serde_json::to_string(payload).unwrap_or_default();
    conn.execute(
        "INSERT INTO task_queue (task_type, payload, status, priority, created_at) VALUES (?1, ?2, ?3, ?4, datetime('now'))",
        params![task_type, &payload_str, STATUS_PENDING, priority],
    )?;

    let task_id = conn.last_insert_rowid();

    // Wake the task queue worker
    if let Some(tq) = state.task_queue() {
        tq.wake();
    }

    Ok(Some(task_id))
}

/// Clear duplicate pending tasks, keeping the oldest for each image_id.
pub fn clear_duplicate_tasks(state: &AppState) -> Result<i64, AppError> {
    let conn = state.main_db().get()?;
    let removed = conn.execute(
        "DELETE FROM task_queue WHERE id NOT IN (
            SELECT MIN(id) FROM task_queue WHERE status = ?1 GROUP BY task_type, payload
        ) AND status = ?1",
        params![STATUS_PENDING],
    )?;
    Ok(removed as i64)
}
