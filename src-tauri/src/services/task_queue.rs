use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rusqlite::params;
use serde_json::Value;
use tokio::sync::{Notify, Semaphore};

use crate::addons::manager::AddonStatus;
use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::events::event_type;
use crate::services::file_tracker;

// ─── Task types (match Python TaskType enum) ─────────────────────────────────

pub const TASK_TAG: &str = "tag";
pub const TASK_SCAN_DIRECTORY: &str = "scan_directory";
pub const TASK_VERIFY_FILES: &str = "verify_files";
pub const TASK_UPLOAD: &str = "upload";
pub const TASK_AGE_DETECT: &str = "age_detect";
pub const TASK_EXTRACT_METADATA: &str = "extract_metadata";

pub const STATUS_PENDING: &str = "pending";
pub const STATUS_PROCESSING: &str = "processing";
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_FAILED: &str = "failed";

/// Maximum number of retry attempts before a task is permanently failed.
const MAX_RETRIES: i64 = 3;

/// Default number of concurrent task workers.
const DEFAULT_WORKERS: usize = 2;

/// Settings key for configuring worker concurrency.
const SETTINGS_WORKERS_KEY: &str = "task_queue_workers";

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
        self.notify.notify_waiters(); // Wake all workers
        log::info!("[TaskQueue] Resumed");
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Wake the workers to check for new tasks immediately.
    pub fn wake(&self) {
        self.notify.notify_waiters();
    }

    /// Start the background worker pool and tag guardian.
    pub fn start(&self, state: AppState) {
        if self.running.load(Ordering::SeqCst) {
            log::warn!("[TaskQueue] Already running");
            return;
        }

        self.running.store(true, Ordering::SeqCst);

        // Reset stuck tasks on startup
        if let Err(e) = reset_stuck_tasks(&state) {
            log::error!("[TaskQueue] Failed to reset stuck tasks: {}", e);
        }

        // Read worker count from settings
        let num_workers = read_worker_count(&state);

        let semaphore = Arc::new(Semaphore::new(num_workers));

        log::info!(
            "[TaskQueue] Starting {} worker(s)",
            num_workers
        );

        // Spawn N worker tasks
        for worker_id in 0..num_workers {
            let paused = self.paused.clone();
            let running = self.running.clone();
            let notify = self.notify.clone();
            let sem = semaphore.clone();
            let state = state.clone();

            tokio::spawn(async move {
                log::info!("[TaskQueue] Worker {} started", worker_id);

                loop {
                    if !running.load(Ordering::SeqCst) {
                        break;
                    }

                    if paused.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        continue;
                    }

                    // Acquire semaphore permit to limit concurrency
                    let _permit = match sem.acquire().await {
                        Ok(p) => p,
                        Err(_) => break, // Semaphore closed
                    };

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
                            log::error!(
                                "[TaskQueue] Worker {} error: {}",
                                worker_id, e
                            );
                            tokio::time::sleep(Duration::from_secs(2)).await;
                        }
                    }
                }

                log::info!("[TaskQueue] Worker {} stopped", worker_id);
            });
        }

        // Spawn tag guardian background task
        {
            let running = self.running.clone();
            let state = state.clone();

            tokio::spawn(async move {
                log::info!("[TaskQueue] Tag guardian started");

                loop {
                    // Sleep first — give the system time to start up
                    tokio::time::sleep(Duration::from_secs(300)).await;

                    if !running.load(Ordering::SeqCst) {
                        break;
                    }

                    if let Err(e) = run_tag_guardian(&state).await {
                        log::error!("[TaskQueue] Tag guardian error: {}", e);
                    }
                }

                log::info!("[TaskQueue] Tag guardian stopped");
            });
        }
    }

    /// Stop the background worker.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
}

impl Default for BackgroundTaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Read the configured worker count from the settings DB table.
/// Falls back to DEFAULT_WORKERS if not set or invalid.
fn read_worker_count(state: &AppState) -> usize {
    let conn = match state.main_db().get() {
        Ok(c) => c,
        Err(_) => return DEFAULT_WORKERS,
    };

    let result = conn.query_row(
        "SELECT value FROM settings WHERE key = ?1",
        params![SETTINGS_WORKERS_KEY],
        |row| row.get::<_, String>(0),
    );

    match result {
        Ok(val) => val.parse::<usize>().unwrap_or(DEFAULT_WORKERS).max(1),
        Err(_) => DEFAULT_WORKERS,
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
    let task_info = tokio::task::spawn_blocking(move || -> Result<Option<(i64, String, String, i64)>, AppError> {
        let conn = state_clone.main_db().get()?;

        // Fetch highest-priority pending task
        let result = conn.query_row(
            "SELECT id, task_type, payload, COALESCE(attempts, 0) FROM task_queue
             WHERE status = ?1
             ORDER BY priority DESC, created_at ASC
             LIMIT 1",
            params![STATUS_PENDING],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
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

    let (task_id, task_type, payload_str, attempts) = match task_info {
        Some(t) => t,
        None => return Ok(false),
    };

    log::info!(
        "[TaskQueue] Processing task #{} ({}) [attempt {}]",
        task_id, task_type, attempts + 1
    );

    let payload: Value = serde_json::from_str(&payload_str).unwrap_or_default();

    // Dispatch to handler
    let result = execute_task(state, &task_type, &payload).await;

    // Update task status
    let state_clone = state.clone();
    let task_type_clone = task_type.clone();
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
                let new_attempts = attempts + 1;

                if new_attempts < MAX_RETRIES {
                    // Retry: set back to pending with incremented attempts
                    conn.execute(
                        "UPDATE task_queue SET status = ?1, error_message = ?2, attempts = ?3 WHERE id = ?4",
                        params![STATUS_PENDING, &error_msg, new_attempts, task_id],
                    )?;

                    // Exponential backoff: 1s, 2s, 4s (2^(attempt-1) seconds)
                    let backoff = Duration::from_secs(1u64 << (new_attempts - 1));

                    log::warn!(
                        "[TaskQueue] Task #{} ({}) failed (attempt {}/{}), retrying in {:?}: {}",
                        task_id, task_type_clone, new_attempts, MAX_RETRIES, backoff, error_msg
                    );

                    // Schedule the retry by spawning a delayed wake
                    let state_for_wake = state_clone.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(backoff).await;
                        if let Some(tq) = state_for_wake.task_queue() {
                            tq.wake();
                        }
                    });
                } else {
                    // Permanently failed after all retries
                    conn.execute(
                        "UPDATE task_queue SET status = ?1, error_message = ?2, attempts = ?3, completed_at = datetime('now') WHERE id = ?4",
                        params![STATUS_FAILED, &error_msg, new_attempts, task_id],
                    )?;
                    log::error!(
                        "[TaskQueue] Task #{} ({}) permanently failed after {} attempts: {}",
                        task_id, task_type_clone, new_attempts, error_msg
                    );
                }
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

        TASK_TAG => {
            execute_tag_task(state, payload).await?;
        }

        TASK_UPLOAD => {
            let image_id = payload["image_id"]
                .as_i64()
                .ok_or_else(|| AppError::Internal("Missing image_id in upload task".into()))?;
            let directory_id = payload["directory_id"]
                .as_i64()
                .ok_or_else(|| AppError::Internal("Missing directory_id in upload task".into()))?;

            log::info!(
                "[TaskQueue] Upload task processed for image #{} in directory #{}",
                image_id, directory_id
            );
        }

        TASK_AGE_DETECT => {
            execute_age_detect_task(state, payload).await?;
        }

        TASK_EXTRACT_METADATA => {
            let image_id = payload["image_id"].as_i64();
            let directory_id = payload["directory_id"].as_i64();
            let complete_import = payload["complete_import"].as_bool().unwrap_or(false);

            // Resolve image path from payload or directory DB
            let mut image_path = payload["image_path"]
                .as_str()
                .unwrap_or("")
                .to_string();

            if image_path.is_empty() {
                if let (Some(img_id), Some(dir_id)) = (image_id, directory_id) {
                    let state_clone = state.clone();
                    let resolved = tokio::task::spawn_blocking(move || -> Option<String> {
                        let pool = state_clone.directory_db().get_pool(dir_id).ok()?;
                        let conn = pool.get().ok()?;
                        conn.query_row(
                            "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
                            rusqlite::params![img_id],
                            |row| row.get::<_, String>(0),
                        ).ok()
                    }).await.unwrap_or(None);

                    if let Some(path) = resolved {
                        image_path = path;
                    }
                }
            }

            // Complete fast import first if requested (hash, dimensions, thumbnail)
            if complete_import {
                if let (Some(img_id), Some(dir_id)) = (image_id, directory_id) {
                    let path = image_path.clone();
                    let state_clone = state.clone();

                    tokio::task::spawn_blocking(move || {
                        complete_fast_import(&state_clone, img_id, dir_id, &path)
                    })
                    .await??;
                }
            }

            // Extract AI generation metadata (A1111 / ComfyUI)
            if let Some(img_id) = image_id {
                if !image_path.is_empty() && std::path::Path::new(&image_path).exists() {
                    let state_clone = state.clone();
                    let dir_id = directory_id;

                    tokio::task::spawn_blocking(move || {
                        run_metadata_extraction(&state_clone, img_id, dir_id, &image_path)
                    })
                    .await??;
                }
            }
        }

        _ => {
            log::warn!("[TaskQueue] Unknown task type: {}", task_type);
        }
    }

    Ok(())
}

// ─── TASK_TAG handler ────────────────────────────────────────────────────────

/// Execute a tag task by sending the image to the auto-tagger addon sidecar.
///
/// Expected payload: { "image_id": N, "directory_id": N, "image_path": "..." }
///
/// The auto-tagger addon (port 18001) receives a POST to /predict with the
/// image file path and returns predicted tags with confidence scores.
/// If the addon is not running, the task is skipped (not failed).
async fn execute_tag_task(state: &AppState, payload: &Value) -> Result<(), AppError> {
    let image_id = payload["image_id"]
        .as_i64()
        .ok_or_else(|| AppError::Internal("Missing image_id in tag task".into()))?;
    let directory_id = payload["directory_id"].as_i64();
    let mut image_path = payload["image_path"]
        .as_str()
        .unwrap_or("")
        .to_string();

    // Resolve image path from directory DB if not in the payload
    if image_path.is_empty() {
        if let Some(dir_id) = directory_id {
            let state_clone = state.clone();
            let resolved = tokio::task::spawn_blocking(move || -> Option<String> {
                let pool = state_clone.directory_db().get_pool(dir_id).ok()?;
                let conn = pool.get().ok()?;
                conn.query_row(
                    "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
                    rusqlite::params![image_id],
                    |row| row.get::<_, String>(0),
                ).ok()
            }).await.unwrap_or(None);

            if let Some(path) = resolved {
                image_path = path;
            }
        }
    }

    if image_path.is_empty() || !std::path::Path::new(&image_path).exists() {
        log::warn!(
            "[TaskQueue] Tag task for image #{}: file not found at '{}'",
            image_id, image_path
        );
        return Ok(());
    }

    // Skip video files — the auto-tagger only handles images
    let ext = std::path::Path::new(&image_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if matches!(ext.as_str(), "mp4" | "webm" | "mkv" | "avi" | "mov") {
        return Ok(());
    }

    // Check if auto-tagger addon is running
    let addon_status = state.addon_manager().get_addon_status("auto-tagger");
    if addon_status != AddonStatus::Running {
        log::info!(
            "[TaskQueue] Skipping tag task for image #{} (auto-tagger addon not running, status: {:?})",
            image_id, addon_status
        );
        return Ok(());
    }

    let base_url = match state.addon_manager().addon_url("auto-tagger") {
        Some(url) => url,
        None => {
            log::warn!(
                "[TaskQueue] auto-tagger addon URL unavailable for image #{}",
                image_id
            );
            return Ok(());
        }
    };

    // Send image path to the auto-tagger sidecar
    let predict_url = format!("{}/predict", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| AppError::Internal(format!("Failed to create HTTP client: {}", e)))?;

    let response = client
        .post(&predict_url)
        .json(&serde_json::json!({
            "file_path": image_path,
            "image_id": image_id,
        }))
        .send()
        .await;

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            log::warn!(
                "[TaskQueue] Failed to reach auto-tagger for image #{}: {}",
                image_id, e
            );
            return Ok(());
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        log::warn!(
            "[TaskQueue] auto-tagger returned {} for image #{}: {}",
            status, image_id, body
        );
        return Ok(());
    }

    // Parse the response — expected: { "tags": [{"name": "...", "confidence": 0.9, "category": "..."}, ...] }
    let result: Value = response.json().await.map_err(|e| {
        AppError::Internal(format!(
            "Failed to parse auto-tagger response for image #{}: {}",
            image_id, e
        ))
    })?;

    let tags = match result.get("tags").and_then(|t| t.as_array()) {
        Some(t) => t,
        None => {
            log::info!(
                "[TaskQueue] auto-tagger returned no tags for image #{}",
                image_id
            );
            return Ok(());
        }
    };

    if tags.is_empty() {
        log::info!(
            "[TaskQueue] auto-tagger returned empty tags for image #{}",
            image_id
        );
        return Ok(());
    }

    // Insert tags into the database
    let state_clone = state.clone();
    let tags_owned: Vec<Value> = tags.clone();
    let dir_id = directory_id;

    tokio::task::spawn_blocking(move || -> Result<(), AppError> {
        let main_conn = state_clone.main_db().get()?;
        let mut inserted = 0u32;

        for tag_val in &tags_owned {
            let tag_name = match tag_val.get("name").and_then(|n| n.as_str()) {
                Some(n) => n.trim().to_lowercase(),
                None => continue,
            };
            if tag_name.is_empty() {
                continue;
            }

            let confidence = tag_val
                .get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.0);
            let category = tag_val
                .get("category")
                .and_then(|c| c.as_str())
                .unwrap_or("general");

            // Ensure the tag exists in the global tags table (upsert)
            main_conn.execute(
                "INSERT OR IGNORE INTO tags (name, category) VALUES (?1, ?2)",
                params![&tag_name, category],
            )?;

            let tag_id: i64 = main_conn.query_row(
                "SELECT id FROM tags WHERE name = ?1",
                params![&tag_name],
                |row| row.get(0),
            )?;

            // Insert into directory-level image_tags if we have a directory
            if let Some(did) = dir_id {
                if let Ok(dir_pool) = state_clone.directory_db().get_pool(did) {
                    if let Ok(dir_conn) = dir_pool.get() {
                        dir_conn.execute(
                            "INSERT OR IGNORE INTO image_tags (image_id, tag_id, confidence, is_manual)
                             VALUES (?1, ?2, ?3, 0)",
                            params![image_id, tag_id, confidence],
                        )?;
                    }
                }
            }

            // Also insert into main DB image_tags
            main_conn.execute(
                "INSERT OR IGNORE INTO image_tags (image_id, tag_id, confidence, is_manual)
                 VALUES (?1, ?2, ?3, 0)",
                params![image_id, tag_id, confidence],
            )?;

            // Update tag post count
            main_conn.execute(
                "UPDATE tags SET post_count = (
                    SELECT COUNT(*) FROM image_tags WHERE tag_id = ?1
                ) WHERE id = ?1",
                params![tag_id],
            )?;

            inserted += 1;
        }

        log::info!(
            "[TaskQueue] Tagged image #{}: {} tags inserted",
            image_id, inserted
        );

        Ok(())
    })
    .await??;

    Ok(())
}

// ─── TASK_AGE_DETECT handler ─────────────────────────────────────────────────

/// Execute an age detection task by sending the image to the age-detector sidecar.
///
/// Expected payload: { "image_id": N, "directory_id": N }
/// The sidecar returns: { "num_faces": N, "faces": [...], "min_age": N, "max_age": N, "detected_ages": "..." }
/// Results are written to the directory DB's images table.
async fn execute_age_detect_task(state: &AppState, payload: &Value) -> Result<(), AppError> {
    let image_id = payload["image_id"]
        .as_i64()
        .ok_or_else(|| AppError::Internal("Missing image_id in age_detect task".into()))?;
    let directory_id = payload["directory_id"].as_i64();

    // Resolve image path
    let mut image_path = payload["image_path"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if image_path.is_empty() {
        if let Some(dir_id) = directory_id {
            let state_clone = state.clone();
            let resolved = tokio::task::spawn_blocking(move || -> Option<String> {
                let pool = state_clone.directory_db().get_pool(dir_id).ok()?;
                let conn = pool.get().ok()?;
                conn.query_row(
                    "SELECT original_path FROM image_files WHERE image_id = ?1 AND file_exists = 1 LIMIT 1",
                    rusqlite::params![image_id],
                    |row| row.get::<_, String>(0),
                ).ok()
            }).await.unwrap_or(None);

            if let Some(path) = resolved {
                image_path = path;
            }
        }
    }

    if image_path.is_empty() || !std::path::Path::new(&image_path).exists() {
        log::warn!(
            "[TaskQueue] Age detect task for image #{}: file not found at '{}'",
            image_id, image_path
        );
        return Ok(());
    }

    // Skip video files
    let ext = std::path::Path::new(&image_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if matches!(ext.as_str(), "mp4" | "webm" | "mkv" | "avi" | "mov") {
        return Ok(());
    }

    // Check if age-detector addon is running
    let addon_status = state.addon_manager().get_addon_status("age-detector");
    if addon_status != AddonStatus::Running {
        log::info!(
            "[TaskQueue] Skipping age detect for image #{} (age-detector not running)",
            image_id
        );
        return Ok(());
    }

    let base_url = match state.addon_manager().addon_url("age-detector") {
        Some(url) => url,
        None => return Ok(()),
    };

    let detect_url = format!("{}/detect", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| AppError::Internal(format!("HTTP client error: {}", e)))?;

    let response = client
        .post(&detect_url)
        .json(&serde_json::json!({ "file_path": image_path }))
        .send()
        .await;

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            log::warn!("[TaskQueue] Failed to reach age-detector for image #{}: {}", image_id, e);
            return Ok(());
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        log::warn!("[TaskQueue] age-detector returned {} for image #{}: {}", status, image_id, body);
        return Ok(());
    }

    let result: Value = response.json().await.map_err(|e| {
        AppError::Internal(format!("Failed to parse age-detector response for image #{}: {}", image_id, e))
    })?;

    let num_faces = result["num_faces"].as_i64().unwrap_or(0);
    let min_age = result["min_age"].as_i64();
    let max_age = result["max_age"].as_i64();
    let detected_ages = result["detected_ages"].as_str().unwrap_or("").to_string();
    let detection_data = serde_json::to_string(&result).unwrap_or_default();

    // Write results to directory DB
    if let Some(dir_id) = directory_id {
        let state_clone = state.clone();
        tokio::task::spawn_blocking(move || -> Result<(), AppError> {
            let pool = state_clone.directory_db().get_pool(dir_id)?;
            let conn = pool.get()?;
            conn.execute(
                "UPDATE images SET num_faces = ?1, min_detected_age = ?2, max_detected_age = ?3,
                 detected_ages = ?4, age_detection_data = ?5 WHERE id = ?6",
                params![num_faces, min_age, max_age, &detected_ages, &detection_data, image_id],
            )?;
            log::info!(
                "[TaskQueue] Age detection for image #{}: {} faces, ages {}",
                image_id, num_faces, detected_ages
            );
            Ok(())
        }).await??;
    }

    Ok(())
}

// ─── Tag guardian ────────────────────────────────────────────────────────────

/// Periodic background task that finds images with no tags and re-queues
/// tag extraction tasks for them.
///
/// Scans each directory database for images that have zero entries in
/// image_tags. For each untagged image, enqueues a TASK_TAG task (respecting
/// the directory's auto_tag setting).
async fn run_tag_guardian(state: &AppState) -> Result<(), AppError> {
    // Check if auto-tagger addon is running; no point queueing if it won't be handled
    let addon_status = state.addon_manager().get_addon_status("auto-tagger");
    if addon_status != AddonStatus::Running {
        log::debug!("[TagGuardian] auto-tagger addon not running, skipping sweep");
        return Ok(());
    }

    let state_clone = state.clone();

    let queued = tokio::task::spawn_blocking(move || -> Result<u32, AppError> {
        let main_conn = state_clone.main_db().get()?;

        // Get all enabled directories with auto_tag enabled
        let mut stmt = main_conn.prepare(
            "SELECT id, path, auto_tag FROM watch_directories WHERE enabled = 1"
        )?;

        let dirs: Vec<(i64, String, bool)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, bool>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let mut total_queued = 0u32;

        for (dir_id, dir_path, auto_tag) in &dirs {
            if !auto_tag {
                continue;
            }

            // Get the directory database
            let dir_pool = match state_clone.directory_db().get_pool(*dir_id) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let dir_conn = match dir_pool.get() {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Find images with no tags in this directory (exclude video files)
            let mut img_stmt = dir_conn.prepare(
                "SELECT i.id, if2.original_path
                 FROM images i
                 LEFT JOIN image_tags it ON i.id = it.image_id
                 LEFT JOIN image_files if2 ON i.id = if2.image_id
                 WHERE it.image_id IS NULL
                 AND if2.file_exists = 1
                 AND LOWER(if2.original_path) NOT LIKE '%.mp4'
                 AND LOWER(if2.original_path) NOT LIKE '%.webm'
                 AND LOWER(if2.original_path) NOT LIKE '%.mkv'
                 AND LOWER(if2.original_path) NOT LIKE '%.avi'
                 AND LOWER(if2.original_path) NOT LIKE '%.mov'
                 LIMIT 50"
            )?;

            let untagged: Vec<(i64, String)> = img_stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                    ))
                })?
                .filter_map(|r| r.ok())
                .collect();

            for (image_id, image_path) in &untagged {
                let payload = serde_json::json!({
                    "image_id": image_id,
                    "directory_id": dir_id,
                    "image_path": image_path,
                    "directory_path": dir_path,
                });

                match enqueue_task(&state_clone, TASK_TAG, &payload, 0, Some(*image_id)) {
                    Ok(Some(_)) => total_queued += 1,
                    Ok(None) => {} // Already queued (dedup)
                    Err(e) => {
                        log::warn!(
                            "[TagGuardian] Failed to enqueue tag task for image #{}: {}",
                            image_id, e
                        );
                    }
                }
            }
        }

        Ok(total_queued)
    })
    .await??;

    if queued > 0 {
        log::info!("[TagGuardian] Queued {} tag tasks for untagged images", queued);
        // Wake workers to process the new tasks
        if let Some(tq) = state.task_queue() {
            tq.wake();
        }
    }

    Ok(())
}

// ─── Metadata / import helpers ───────────────────────────────────────────────

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

/// Run metadata extraction for a single image.
///
/// Looks up the directory's metadata_format and ComfyUI node ID configuration,
/// then calls the metadata service to extract and save to DB.
fn run_metadata_extraction(
    state: &AppState,
    image_id: i64,
    directory_id: Option<i64>,
    image_path: &str,
) -> Result<(), AppError> {
    use crate::services::metadata;

    // Get directory-level configuration for metadata extraction
    let (format_hint, comfyui_prompt_ids, comfyui_negative_ids) =
        get_directory_metadata_config(state, directory_id)?;

    // Get the appropriate database connection
    let conn;
    let pool;
    if let Some(dir_id) = directory_id {
        pool = state.directory_db().get_pool(dir_id)?;
        conn = pool.get()?;
    } else {
        conn = state.main_db().get()?;
    }

    let prompt_refs: Vec<String> = comfyui_prompt_ids;
    let negative_refs: Vec<String> = comfyui_negative_ids;

    let prompt_slice: Option<&[String]> = if prompt_refs.is_empty() {
        None
    } else {
        Some(&prompt_refs)
    };
    let negative_slice: Option<&[String]> = if negative_refs.is_empty() {
        None
    } else {
        Some(&negative_refs)
    };

    match metadata::extract_and_save_metadata(
        &conn,
        image_id,
        image_path,
        prompt_slice,
        negative_slice,
        &format_hint,
    ) {
        Ok(result) => {
            match result.status.as_str() {
                "success" => {
                    log::info!(
                        "[TaskQueue] Metadata extracted for image #{} (format: {})",
                        image_id,
                        result
                            .metadata
                            .as_ref()
                            .and_then(|m| m.source_format.as_deref())
                            .unwrap_or("unknown")
                    );
                }
                "no_metadata" => {
                    log::debug!(
                        "[TaskQueue] No AI metadata found in image #{}",
                        image_id
                    );
                }
                "config_mismatch" => {
                    log::info!(
                        "[TaskQueue] Metadata config mismatch for image #{}: {}",
                        image_id,
                        result.message.unwrap_or_default()
                    );
                }
                _ => {}
            }
            Ok(())
        }
        Err(e) => {
            log::warn!(
                "[TaskQueue] Metadata extraction error for image #{}: {}",
                image_id,
                e
            );
            // Non-fatal: don't fail the whole task for metadata issues
            Ok(())
        }
    }
}

/// Read the directory's metadata configuration (format hint, ComfyUI node IDs).
fn get_directory_metadata_config(
    state: &AppState,
    directory_id: Option<i64>,
) -> Result<(String, Vec<String>, Vec<String>), AppError> {
    let dir_id = match directory_id {
        Some(id) => id,
        None => return Ok(("auto".into(), vec![], vec![])),
    };

    let main_conn = state.main_db().get()?;

    let result = main_conn.query_row(
        "SELECT metadata_format, comfyui_prompt_node_ids, comfyui_negative_node_ids
         FROM watch_directories WHERE id = ?1",
        params![dir_id],
        |row| {
            Ok((
                row.get::<_, Option<String>>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        },
    );

    match result {
        Ok((format, prompt_ids_str, negative_ids_str)) => {
            let format_hint = format.unwrap_or_else(|| "auto".into());

            // Parse comma-separated node IDs
            let prompt_ids = parse_node_ids(prompt_ids_str.as_deref());
            let negative_ids = parse_node_ids(negative_ids_str.as_deref());

            Ok((format_hint, prompt_ids, negative_ids))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(("auto".into(), vec![], vec![])),
        Err(e) => Err(AppError::from(e)),
    }
}

/// Parse a comma-separated or JSON array string of node IDs into a Vec<String>.
fn parse_node_ids(raw: Option<&str>) -> Vec<String> {
    match raw {
        None => vec![],
        Some(s) if s.trim().is_empty() => vec![],
        Some(s) => {
            let trimmed = s.trim();
            // Try JSON array first: ["3", "5"]
            if trimmed.starts_with('[') {
                if let Ok(ids) = serde_json::from_str::<Vec<String>>(trimmed) {
                    return ids.into_iter().filter(|id| !id.is_empty()).collect();
                }
            }
            // Fall back to comma-separated: "3,5" or "3, 5"
            trimmed
                .split(',')
                .map(|id| id.trim().to_string())
                .filter(|id| !id.is_empty())
                .collect()
        }
    }
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

    // Deduplication check — use boundary-aware patterns to avoid false positives
    // (e.g. image_id 1 must not match image_id 10 or 100).
    // JSON payloads can have `"image_id":N,` (followed by comma) or `"image_id":N}`
    // (last key in object), with optional spaces around the colon.
    if let Some(image_id) = dedupe_key {
        // Build patterns that match the image_id with a boundary character after it:
        // - comma (more keys follow): "image_id":N,...
        // - closing brace (last key):  "image_id":N}
        // Also handle optional space after the colon.
        let pat_comma = format!("%\"image_id\":{},%", image_id);
        let pat_comma_sp = format!("%\"image_id\": {},%", image_id);
        let pat_brace = format!("%\"image_id\":{}}}%", image_id);
        let pat_brace_sp = format!("%\"image_id\": {}}}%", image_id);
        let patterns = [pat_comma, pat_comma_sp, pat_brace, pat_brace_sp];

        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM task_queue
                 WHERE task_type = ?1 AND status = ?2
                 AND (payload LIKE ?3 OR payload LIKE ?4 OR payload LIKE ?5 OR payload LIKE ?6)",
                params![
                    task_type,
                    STATUS_PENDING,
                    &patterns[0],
                    &patterns[1],
                    &patterns[2],
                    &patterns[3],
                ],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);

        if exists {
            return Ok(None);
        }
    }

    let payload_str = serde_json::to_string(payload).unwrap_or_default();
    conn.execute(
        "INSERT INTO task_queue (task_type, payload, status, priority, attempts, created_at) VALUES (?1, ?2, ?3, ?4, 0, datetime('now'))",
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
