use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify::event::{ModifyKind, RenameMode};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rusqlite::params;
use tokio::sync::mpsc;

use crate::server::state::AppState;
use crate::services::file_tracker;
use crate::services::importer;

/// Manages filesystem watchers for all watch directories.
pub struct DirectoryWatcher {
    state: AppState,
    /// Active watcher handles by directory_id.
    watches: Arc<Mutex<HashMap<i64, WatchHandle>>>,
    /// Shutdown signal sender.
    shutdown_tx: Option<mpsc::Sender<()>>,
}

struct WatchHandle {
    _watcher: RecommendedWatcher,
    _path: PathBuf,
    recursive: bool,
}

impl DirectoryWatcher {
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            watches: Arc::new(Mutex::new(HashMap::new())),
            shutdown_tx: None,
        }
    }

    /// Start watching all enabled directories.
    pub fn start(&mut self) {
        let state = self.state.clone();
        let watches = self.watches.clone();
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        tokio::spawn(async move {
            // Load enabled directories from main DB
            let directories = match load_enabled_directories(&state) {
                Ok(dirs) => dirs,
                Err(e) => {
                    log::error!("[Watcher] Failed to load directories: {}", e);
                    return;
                }
            };

            log::info!(
                "[Watcher] Starting watches for {} directories",
                directories.len()
            );

            for (dir_id, dir_path, recursive) in &directories {
                if let Err(e) = add_watch(&state, &watches, *dir_id, dir_path, *recursive) {
                    log::error!(
                        "[Watcher] Failed to watch directory {} ({}): {}",
                        dir_id,
                        dir_path,
                        e
                    );
                }
            }

            // Run startup scan for recently modified files
            for (dir_id, dir_path, recursive) in &directories {
                startup_scan(&state, *dir_id, dir_path, *recursive);
            }

            // Wait for shutdown
            shutdown_rx.recv().await;
            log::info!("[Watcher] Shutting down");
        });
    }

    /// Stop all watchers.
    pub fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.try_send(());
        }
        if let Ok(mut w) = self.watches.lock() {
            w.clear();
        }
    }

    /// Add a watch for a single directory.
    pub fn add_directory(&self, directory_id: i64, path: &str, recursive: bool) {
        if let Err(e) = add_watch(&self.state, &self.watches, directory_id, path, recursive) {
            log::error!(
                "[Watcher] Failed to add watch for directory {}: {}",
                directory_id,
                e
            );
        }
    }

    /// Remove a watch for a directory.
    pub fn remove_directory(&self, directory_id: i64) {
        if let Ok(mut w) = self.watches.lock() {
            if w.remove(&directory_id).is_some() {
                log::info!("[Watcher] Removed watch for directory {}", directory_id);
            }
        }
    }

    /// Re-read enabled directories from the DB and sync watches.
    ///
    /// Adds watches for newly-enabled directories and removes watches for
    /// disabled/deleted ones. Call this after updating directory settings
    /// (e.g., enabling/disabling, changing recursive flag).
    pub fn refresh(&self) {
        let directories = match load_enabled_directories(&self.state) {
            Ok(dirs) => dirs,
            Err(e) => {
                log::error!("[Watcher] refresh: failed to load directories: {}", e);
                return;
            }
        };

        let new_ids: HashMap<i64, (String, bool)> = directories
            .into_iter()
            .map(|(id, path, recursive)| (id, (path, recursive)))
            .collect();

        let mut watches = match self.watches.lock() {
            Ok(w) => w,
            Err(_) => return,
        };

        // Remove watches for directories no longer enabled
        let stale_ids: Vec<i64> = watches
            .keys()
            .filter(|id| !new_ids.contains_key(id))
            .copied()
            .collect();

        for id in &stale_ids {
            if watches.remove(id).is_some() {
                log::info!("[Watcher] refresh: removed stale watch for directory {}", id);
            }
        }

        // Detect directories whose recursive flag changed — need to re-add
        let changed_ids: Vec<i64> = watches
            .iter()
            .filter_map(|(id, handle)| {
                new_ids.get(id).and_then(|(_, new_recursive)| {
                    if handle.recursive != *new_recursive {
                        Some(*id)
                    } else {
                        None
                    }
                })
            })
            .collect();

        for id in &changed_ids {
            if watches.remove(id).is_some() {
                log::info!(
                    "[Watcher] refresh: removed watch for directory {} (recursive flag changed)",
                    id
                );
            }
        }

        // Collect directories that need a watch added (new or changed)
        // (We need to drop the lock before calling add_watch since it also locks)
        let to_add: Vec<(i64, String, bool)> = new_ids
            .into_iter()
            .filter(|(id, _)| !watches.contains_key(id))
            .map(|(id, (path, recursive))| (id, path, recursive))
            .collect();

        drop(watches);

        for (dir_id, dir_path, recursive) in &to_add {
            if let Err(e) = add_watch(&self.state, &self.watches, *dir_id, dir_path, *recursive) {
                log::error!(
                    "[Watcher] refresh: failed to add watch for directory {} ({}): {}",
                    dir_id,
                    dir_path,
                    e
                );
            }
        }

        log::info!(
            "[Watcher] refresh: watches synced (removed {}, re-added {}, added {}).",
            stale_ids.len(),
            changed_ids.len(),
            to_add.len().saturating_sub(changed_ids.len())
        );
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn load_enabled_directories(state: &AppState) -> Result<Vec<(i64, String, bool)>, String> {
    let conn = state
        .main_db()
        .get()
        .map_err(|e| format!("DB error: {}", e))?;

    let mut stmt = conn
        .prepare("SELECT id, path, recursive FROM watch_directories WHERE enabled = 1")
        .map_err(|e| format!("Query error: {}", e))?;

    let dirs: Vec<(i64, String, bool)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(|e| format!("Query error: {}", e))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(dirs)
}

fn add_watch(
    state: &AppState,
    watches: &Arc<Mutex<HashMap<i64, WatchHandle>>>,
    directory_id: i64,
    path: &str,
    recursive: bool,
) -> Result<(), String> {
    let dir_path = PathBuf::from(path);
    if !dir_path.exists() {
        return Err(format!("Directory does not exist: {}", path));
    }

    let state_clone = state.clone();
    let dir_id = directory_id;
    // Capture the tokio runtime handle so we can spawn from the notify thread
    let rt_handle = tokio::runtime::Handle::current();

    let mut watcher = notify::recommended_watcher(move |event: Result<Event, notify::Error>| {
        match event {
            Ok(ev) => handle_fs_event(&state_clone, dir_id, ev, &rt_handle),
            Err(e) => log::error!("[Watcher] Error in directory {}: {}", dir_id, e),
        }
    })
    .map_err(|e| format!("Failed to create watcher: {}", e))?;

    let mode = if recursive {
        RecursiveMode::Recursive
    } else {
        RecursiveMode::NonRecursive
    };

    watcher
        .watch(&dir_path, mode)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    // notify's inotify backend doesn't follow symlinks when recursively
    // discovering subdirectories. Walk the tree ourselves and add explicit
    // watches for any symlinked directories so their contents are monitored.
    if recursive {
        for entry in walkdir::WalkDir::new(&dir_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path_is_symlink() && entry.file_type().is_dir() {
                if let Err(e) = watcher.watch(entry.path(), RecursiveMode::Recursive) {
                    log::warn!(
                        "[Watcher] Failed to watch symlinked dir {}: {}",
                        entry.path().display(),
                        e
                    );
                }
            }
        }
    }

    if let Ok(mut w) = watches.lock() {
        w.insert(
            directory_id,
            WatchHandle {
                _watcher: watcher,
                _path: dir_path,
                recursive,
            },
        );
    }

    log::info!(
        "[Watcher] Watching directory {} ({}) [{}]",
        directory_id,
        path,
        if recursive { "recursive" } else { "flat" }
    );

    Ok(())
}

fn handle_fs_event(state: &AppState, directory_id: i64, event: Event, rt: &tokio::runtime::Handle) {
    match event.kind {
        // New file created
        EventKind::Create(_) => {
            for path in &event.paths {
                if path.is_file() && importer::is_media_file(path) {
                    let file_path = path.to_string_lossy().to_string();
                    let state_clone = state.clone();

                    // Debounced import — wait for file to stabilize
                    rt.spawn(async move {
                        debounced_import(state_clone, directory_id, file_path).await;
                    });
                }
            }
        }

        // File removed
        EventKind::Remove(_) => {
            for path in &event.paths {
                let file_path = path.to_string_lossy().to_string();
                let state_clone = state.clone();

                rt.spawn(async move {
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        file_tracker::mark_file_missing(&state_clone, &file_path, directory_id)
                    })
                    .await
                    {
                        log::error!("[Watcher] Error marking file missing: {}", e);
                    }
                });
            }
        }

        // File moved/renamed TO this directory — treat as new file
        EventKind::Modify(ModifyKind::Name(RenameMode::To)) => {
            for path in &event.paths {
                if path.is_file() && importer::is_media_file(path) {
                    let file_path = path.to_string_lossy().to_string();
                    let state_clone = state.clone();

                    rt.spawn(async move {
                        debounced_import(state_clone, directory_id, file_path).await;
                    });
                }
            }
        }

        // File moved/renamed FROM this directory — treat as removal
        EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
            for path in &event.paths {
                let file_path = path.to_string_lossy().to_string();
                let state_clone = state.clone();

                rt.spawn(async move {
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        file_tracker::mark_file_missing(&state_clone, &file_path, directory_id)
                    })
                    .await
                    {
                        log::error!("[Watcher] Error marking moved-away file: {}", e);
                    }
                });
            }
        }

        // Other modifications — import untracked media files
        EventKind::Modify(_) => {
            for path in &event.paths {
                if path.is_file() && importer::is_media_file(path) {
                    let file_path = path.to_string_lossy().to_string();
                    let state_clone = state.clone();

                    rt.spawn(async move {
                        // Check if already tracked before importing
                        let already_tracked = tokio::task::spawn_blocking({
                            let state = state_clone.clone();
                            let fp = file_path.clone();
                            move || -> bool {
                                if let Ok(pool) = state.directory_db().get_pool(directory_id) {
                                    if let Ok(conn) = pool.get() {
                                        return conn
                                            .query_row(
                                                "SELECT COUNT(*) FROM image_files WHERE original_path = ?1",
                                                params![&fp],
                                                |row| row.get::<_, i64>(0),
                                            )
                                            .unwrap_or(0)
                                            > 0;
                                    }
                                }
                                false
                            }
                        })
                        .await
                        .unwrap_or(false);

                        if !already_tracked {
                            debounced_import(state_clone, directory_id, file_path).await;
                        }
                    });
                }
            }
        }

        _ => {}
    }
}

/// Wait for a file to stabilize (not being written to), then import it with retry.
async fn debounced_import(state: AppState, directory_id: i64, file_path: String) {
    // Wait 1 second for file to settle
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Check file size stability
    let stable = tokio::task::spawn_blocking({
        let fp = file_path.clone();
        move || {
            let size1 = std::fs::metadata(&fp).map(|m| m.len()).ok();
            std::thread::sleep(Duration::from_millis(500));
            let size2 = std::fs::metadata(&fp).map(|m| m.len()).ok();
            size1.is_some() && size1 == size2
        }
    })
    .await
    .unwrap_or(false);

    if !stable {
        // File still being written, try again later
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    // Import with retry (3 attempts, 500ms backoff) for database busy errors
    let mut last_err: Option<String> = None;

    for attempt in 0..3u32 {
        let state_clone = state.clone();
        let fp = file_path.clone();

        match tokio::task::spawn_blocking(move || {
            importer::import_image(&state_clone, &fp, directory_id)
        })
        .await
        {
            Ok(Ok(result)) => {
                if result.status == importer::ImportStatus::Imported {
                    log::info!(
                        "[Watcher] Imported: {}",
                        std::path::Path::new(&file_path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&file_path)
                    );
                }
                return;
            }
            Ok(Err(e)) => {
                let msg = format!("{}", e);
                if (msg.contains("database is locked") || msg.contains("database is busy"))
                    && attempt < 2
                {
                    let backoff_ms = 500 * (attempt as u64 + 1);
                    log::warn!(
                        "[Watcher] DB busy importing {}, retrying in {}ms (attempt {}/3)",
                        file_path,
                        backoff_ms,
                        attempt + 1
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    last_err = Some(msg);
                    continue;
                }
                log::error!("[Watcher] Import error for {}: {}", file_path, e);
                return;
            }
            Err(e) => {
                log::error!("[Watcher] Task error: {}", e);
                return;
            }
        }
    }

    if let Some(err) = last_err {
        log::error!(
            "[Watcher] Import failed after 3 retries for {}: {}",
            file_path,
            err
        );
    }
}

/// Scan for files modified since the last scan time.
///
/// Respects the directory's `recursive` flag — only walks subdirectories
/// if recursive=true. Skips directories that already have a pending
/// `scan_directory` task to avoid duplicate work. Updates `last_scanned_at`
/// after completion.
fn startup_scan(state: &AppState, directory_id: i64, dir_path: &str, recursive: bool) {
    let conn = match state.main_db().get() {
        Ok(c) => c,
        Err(_) => return,
    };

    // ── Duplicate scan prevention ────────────────────────────────────────
    // Check if there's already a pending scan_directory task for this directory
    let has_pending_scan: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM task_queue
             WHERE task_type = 'scan_directory'
               AND status IN ('pending', 'processing')
               AND payload LIKE ?1",
            params![format!("%\"directory_id\":{}%", directory_id)],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        > 0;

    if has_pending_scan {
        log::info!(
            "[Watcher] Skipping startup scan for directory {} — scan task already queued",
            directory_id
        );
        return;
    }

    // Get last_scanned_at
    let last_scanned: Option<String> = conn
        .query_row(
            "SELECT last_scanned_at FROM watch_directories WHERE id = ?1",
            params![directory_id],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    if last_scanned.is_none() {
        // Never scanned — skip startup scan (full scan should be triggered separately)
        return;
    }

    // Parse timestamp
    let last_dt = match last_scanned.and_then(|s| {
        chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S%.f%z")
            .or_else(|_| chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S"))
            .ok()
    }) {
        Some(dt) => dt,
        None => return,
    };

    let dir_path_owned = dir_path.to_string();
    let dir_path = std::path::Path::new(&dir_path_owned);
    if !dir_path.exists() {
        return;
    }

    // ── Recursive flag respect ───────────────────────────────────────────
    // Only walk subdirectories if the directory has recursive=true.
    let mut new_files = Vec::new();

    let walker = walkdir::WalkDir::new(dir_path)
        .follow_links(true)
        .max_depth(if recursive { usize::MAX } else { 1 });

    for entry in walker.into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        if !importer::is_media_file(entry.path()) {
            continue;
        }

        // Check if modified after last scan
        if let Ok(metadata) = entry.metadata() {
            if let Ok(modified) = metadata.modified() {
                let mod_dt = chrono::DateTime::<chrono::Utc>::from(modified).naive_utc();
                if mod_dt > last_dt {
                    new_files.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }

    if new_files.is_empty() {
        // No new files, but still update last_scanned_at
        update_last_scanned_at(state, directory_id);
        return;
    }

    log::info!(
        "[Watcher] Startup: found {} modified files in directory {}",
        new_files.len(),
        directory_id
    );

    let state_clone = state.clone();
    tokio::spawn(async move {
        for file_path in new_files {
            let sc = state_clone.clone();
            let fp = file_path.clone();
            let _ = tokio::task::spawn_blocking(move || {
                importer::import_image(&sc, &fp, directory_id)
            })
            .await;
        }

        // ── Update last_scanned_at after scan completes ──────────────────
        update_last_scanned_at(&state_clone, directory_id);
    });
}

/// Update the `last_scanned_at` timestamp for a directory to now.
fn update_last_scanned_at(state: &AppState, directory_id: i64) {
    let now = chrono::Utc::now().to_rfc3339();
    if let Ok(conn) = state.main_db().get() {
        match conn.execute(
            "UPDATE watch_directories SET last_scanned_at = ?1 WHERE id = ?2",
            params![&now, directory_id],
        ) {
            Ok(_) => {
                log::debug!(
                    "[Watcher] Updated last_scanned_at for directory {}",
                    directory_id
                );
            }
            Err(e) => {
                log::error!(
                    "[Watcher] Failed to update last_scanned_at for directory {}: {}",
                    directory_id,
                    e
                );
            }
        }
    }
}
