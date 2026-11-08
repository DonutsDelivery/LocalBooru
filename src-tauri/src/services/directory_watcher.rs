use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use notify::event::{ModifyKind, RenameMode};
use notify::{Config, Event, EventKind, PollWatcher, RecursiveMode, Watcher};
use rusqlite::params;
use tokio::sync::mpsc;

use serde_json::json;

use crate::db::library::LibraryContext;
use crate::db::schema::init_directory_db;
use crate::server::state::AppState;
use crate::services::file_tracker;
use crate::services::importer;
use crate::services::music;
use crate::services::task_queue;

/// Check if a path is inside a "dumpster" subfolder (pruned images).
fn is_in_dumpster(path: &std::path::Path) -> bool {
    path.components().any(|c| c.as_os_str() == "dumpster")
}

/// Manages filesystem watchers for all watch directories.
pub struct DirectoryWatcher {
    state: AppState,
    /// Active watcher handles by directory_id.
    watches: Arc<WatchMap>,
    /// Parent directories being polled for new subdirectories.
    parent_dirs: Arc<Mutex<HashMap<String, ParentSettings>>>,
    /// Shutdown signal sender.
    shutdown_tx: Option<mpsc::Sender<()>>,
}

type WatchMap = Mutex<HashMap<i64, WatchHandle>>;
type EventSink = Arc<dyn Fn(Result<Event, notify::Error>) + Send + Sync>;

struct WatchHandle {
    watcher: Box<dyn Watcher + Send>,
    /// Poll watchers for symlinked subdirectories on filesystems without
    /// usable inotify (FUSE, network mounts).
    extra_watchers: Vec<Box<dyn Watcher + Send>>,
    sink: EventSink,
    /// True when `watcher` is a poll watcher (it follows symlinks on its own).
    polling: bool,
    /// Symlinked subdirectories that already have an explicit watch.
    linked_dirs: HashSet<PathBuf>,
    _path: PathBuf,
    recursive: bool,
}

impl WatchHandle {
    /// Explicitly watch a symlinked subdirectory. Returns true if it is new.
    ///
    /// inotify only learns about subdirectories from `ISDIR` events, which
    /// symlinks never produce, so links created after the watch started would
    /// otherwise stay unwatched until restart.
    fn watch_symlinked_dir(&mut self, link: &Path) -> Result<bool, String> {
        if self.polling || self.linked_dirs.contains(link) {
            return Ok(false);
        }
        if self.linked_dirs.len() >= MAX_SYMLINK_WATCHES {
            return Err(format!(
                "more than {} symlinked directories, not watching {}",
                MAX_SYMLINK_WATCHES,
                link.display()
            ));
        }
        if needs_polling(link) {
            let mut poll = new_watcher(true, &self.sink)?;
            poll.watch(link, RecursiveMode::Recursive)
                .map_err(|e| format!("Failed to poll {}: {}", link.display(), e))?;
            self.extra_watchers.push(poll);
        } else {
            self.watcher
                .watch(link, RecursiveMode::Recursive)
                .map_err(|e| format!("Failed to watch {}: {}", link.display(), e))?;
        }
        self.linked_dirs.insert(link.to_path_buf());
        Ok(true)
    }
}

/// Settings inherited by new subdirectories created under a parent.
#[derive(Clone)]
struct ParentSettings {
    recursive: bool,
    auto_tag: bool,
    auto_age_detect: bool,
    show_images: bool,
    show_videos: bool,
    show_music: bool,
    library: Arc<LibraryContext>,
}

/// How often to poll parent directories for new subdirectories (seconds).
const PARENT_POLL_INTERVAL_SECS: u64 = 30;

/// How often directories on FUSE/network filesystems are polled for changes
/// (seconds). inotify never reports changes made outside the local mount.
const POLL_WATCH_INTERVAL_SECS: u64 = if cfg!(test) { 1 } else { 60 };

/// Upper bound on explicitly watched symlinked subdirectories per directory.
const MAX_SYMLINK_WATCHES: usize = 256;

impl DirectoryWatcher {
    pub fn new(state: AppState) -> Self {
        Self {
            state,
            watches: Arc::new(Mutex::new(HashMap::new())),
            parent_dirs: Arc::new(Mutex::new(HashMap::new())),
            shutdown_tx: None,
        }
    }

    /// Start watching all enabled directories.
    pub fn start(&mut self) {
        let state = self.state.clone();
        let watches = self.watches.clone();
        let parent_dirs = self.parent_dirs.clone();
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Watch registration, recursive WalkDir scans, and SQLite backfills are
        // synchronous. Keep the startup lifecycle on Tokio's blocking pool so
        // it cannot starve Axum or the frontend health-check command.
        tokio::task::spawn_blocking(move || {
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

            for (dir_id, dir_path, recursive, lib) in &directories {
                if let Err(e) =
                    add_watch(&state, &watches, *dir_id, dir_path, *recursive, lib.clone())
                {
                    log::error!(
                        "[Watcher] Failed to watch directory {} ({}): {}",
                        dir_id,
                        dir_path,
                        e
                    );
                }
            }

            // Load parent directories for polling
            match load_parent_directories(&state) {
                Ok(parents) => {
                    if !parents.is_empty() {
                        log::info!(
                            "[Watcher] Registering {} parent directories for polling",
                            parents.len()
                        );
                        if let Ok(mut pd) = parent_dirs.lock() {
                            for (parent_path, settings) in parents {
                                pd.insert(parent_path, settings);
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("[Watcher] Failed to load parent directories: {}", e);
                }
            }

            // Start parent directory polling loop
            let poll_state = state.clone();
            let poll_watches = watches.clone();
            let poll_parents = parent_dirs.clone();
            tokio::spawn(async move {
                poll_parent_directories(poll_state, poll_watches, poll_parents).await;
            });

            // Backfill cached counts for dirs with 0 (one-time for existing large DBs after adding the columns)
            // Runs in background so it doesn't block startup; updates image/tagged/fav counts from per-dir DBs
            let backfill_dirs = directories.clone();
            let _backfill_state = state.clone(); // kept for future if needed
            tokio::task::spawn_blocking(move || {
                for (dir_id, _p, _r, lib) in &backfill_dirs {
                    if let Ok(main_conn) = lib.main_pool.get() {
                        let cur: i64 = main_conn.query_row(
                            "SELECT COALESCE(image_count, 0) FROM watch_directories WHERE id = ?1",
                            params![*dir_id], |r| r.get(0)
                        ).unwrap_or(0);
                        if cur == 0 {
                            if let Ok(dpool) = lib.directory_db.get_pool(*dir_id) {
                                if let Ok(dconn) = dpool.get() {
                                    let ic: i64 = dconn
                                        .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                                        .unwrap_or(0);
                                    let tc: i64 = dconn
                                        .query_row(
                                            "SELECT COUNT(DISTINCT image_id) FROM image_tags",
                                            [],
                                            |r| r.get(0),
                                        )
                                        .unwrap_or(0);
                                    let fc: i64 = dconn
                                        .query_row(
                                            "SELECT COUNT(*) FROM images WHERE is_favorite=1",
                                            [],
                                            |r| r.get(0),
                                        )
                                        .unwrap_or(0);
                                    let _ = main_conn.execute(
                                        "UPDATE watch_directories SET image_count=?1, tagged_count=?2, favorited_count=?3 WHERE id=?4",
                                        params![ic, tc, fc, *dir_id]
                                    );
                                }
                            }
                        }
                    }
                }
            });

            // Run startup scan for recently modified files
            for (dir_id, dir_path, recursive, lib) in &directories {
                startup_scan(&state, lib, *dir_id, dir_path, *recursive);
            }

            // Wait for shutdown
            shutdown_rx.blocking_recv();
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
        if let Ok(mut pd) = self.parent_dirs.lock() {
            pd.clear();
        }
    }

    /// Add a watch for a single directory (defaults to primary library).
    pub fn add_directory(&self, directory_id: i64, path: &str, recursive: bool) {
        let lib = self.state.library_manager().primary().clone();
        self.add_directory_for_library(directory_id, path, recursive, lib);
    }

    /// Add a watch for a directory in a specific library.
    pub fn add_directory_for_library(
        &self,
        directory_id: i64,
        path: &str,
        recursive: bool,
        library: Arc<LibraryContext>,
    ) {
        if let Err(e) = add_watch(
            &self.state,
            &self.watches,
            directory_id,
            path,
            recursive,
            library,
        ) {
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

    /// Register a parent directory for polling.
    ///
    /// New subdirectories found during polling are automatically registered as
    /// watch directories with the given settings, a per-directory DB is created,
    /// and a scan task is queued. Uses polling instead of inotify so it works
    /// on mounted/network drives.
    pub fn watch_parent(
        &self,
        parent_path: &str,
        recursive: bool,
        auto_tag: bool,
        auto_age_detect: bool,
        show_images: bool,
        show_videos: bool,
        show_music: bool,
        library: Arc<LibraryContext>,
    ) {
        let settings = ParentSettings {
            recursive,
            auto_tag,
            auto_age_detect,
            show_images,
            show_videos,
            show_music,
            library,
        };
        if let Ok(mut pd) = self.parent_dirs.lock() {
            pd.insert(parent_path.to_string(), settings);
            log::info!(
                "[ParentWatcher] Registered parent directory for polling: {}",
                parent_path
            );
        }
    }

    /// Remove a parent directory from the polling watcher.
    pub fn remove_parent(&self, parent_path: &str) {
        if let Ok(mut pd) = self.parent_dirs.lock() {
            if pd.remove(parent_path).is_some() {
                log::info!("[ParentWatcher] Removed parent directory: {}", parent_path);
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

        let new_ids: HashMap<i64, (String, bool, Arc<LibraryContext>)> = directories
            .into_iter()
            .map(|(id, path, recursive, lib)| (id, (path, recursive, lib)))
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
                log::info!(
                    "[Watcher] refresh: removed stale watch for directory {}",
                    id
                );
            }
        }

        // Detect directories whose recursive flag changed — need to re-add
        let changed_ids: Vec<i64> = watches
            .iter()
            .filter_map(|(id, handle)| {
                new_ids.get(id).and_then(|(_, new_recursive, _)| {
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
        let to_add: Vec<(i64, String, bool, Arc<LibraryContext>)> = new_ids
            .into_iter()
            .filter(|(id, _)| !watches.contains_key(id))
            .map(|(id, (path, recursive, lib))| (id, path, recursive, lib))
            .collect();

        drop(watches);

        for (dir_id, dir_path, recursive, lib) in &to_add {
            if let Err(e) = add_watch(
                &self.state,
                &self.watches,
                *dir_id,
                dir_path,
                *recursive,
                lib.clone(),
            ) {
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

fn load_enabled_directories(
    state: &AppState,
) -> Result<Vec<(i64, String, bool, Arc<LibraryContext>)>, String> {
    let mut all_dirs = Vec::new();

    for lib in state.library_manager().all_mounted() {
        let conn = lib
            .main_pool
            .get()
            .map_err(|e| format!("DB error for library '{}': {}", lib.name, e))?;

        let mut stmt = conn
            .prepare("SELECT id, path, recursive FROM watch_directories WHERE enabled = 1")
            .map_err(|e| format!("Query error: {}", e))?;

        let dirs: Vec<(i64, String, bool)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .map_err(|e| format!("Query error: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        for (id, path, recursive) in dirs {
            all_dirs.push((id, path, recursive, lib.clone()));
        }
    }

    Ok(all_dirs)
}

fn add_watch(
    state: &AppState,
    watches: &Arc<WatchMap>,
    directory_id: i64,
    path: &str,
    recursive: bool,
    library: Arc<LibraryContext>,
) -> Result<(), String> {
    let dir_path = PathBuf::from(path);
    if !dir_path.exists() {
        return Err(format!("Directory does not exist: {}", path));
    }

    let state_clone = state.clone();
    let lib_clone = library.clone();
    let dir_id = directory_id;
    // Capture the tokio runtime handle so we can spawn from the notify thread
    let rt_handle = tokio::runtime::Handle::current();
    let weak_watches = Arc::downgrade(watches);
    let polling = needs_polling(&dir_path);

    let sink: EventSink = Arc::new(move |event: Result<Event, notify::Error>| match event {
        Ok(ev) => {
            if !polling && recursive {
                let links = new_symlinked_dirs(&ev);
                if !links.is_empty() {
                    let state = state_clone.clone();
                    let lib = lib_clone.clone();
                    let weak = weak_watches.clone();
                    // Registering a watch waits on the notify thread that is
                    // running this callback, so hand it to another thread.
                    rt_handle.spawn_blocking(move || {
                        register_new_symlinked_dirs(&state, &lib, &weak, dir_id, links);
                    });
                }
            }
            handle_fs_event(&state_clone, &lib_clone, dir_id, ev, &rt_handle)
        }
        Err(e) => log::error!("[Watcher] Error in directory {}: {}", dir_id, e),
    });

    let handle = start_watch(dir_path, recursive, polling, sink)?;

    if let Ok(mut w) = watches.lock() {
        w.insert(directory_id, handle);
    }

    log::info!(
        "[Watcher] Watching directory {} ({}) [{}{}]",
        directory_id,
        path,
        if recursive { "recursive" } else { "flat" },
        if polling { ", polling" } else { "" }
    );

    Ok(())
}

/// Create the watcher(s) for one directory.
fn start_watch(
    dir_path: PathBuf,
    recursive: bool,
    polling: bool,
    sink: EventSink,
) -> Result<WatchHandle, String> {
    let mut watcher = new_watcher(polling, &sink)?;

    let mode = if recursive {
        RecursiveMode::Recursive
    } else {
        RecursiveMode::NonRecursive
    };

    watcher
        .watch(&dir_path, mode)
        .map_err(|e| format!("Failed to start watching: {}", e))?;

    let mut handle = WatchHandle {
        watcher,
        extra_watchers: Vec::new(),
        sink,
        polling,
        linked_dirs: HashSet::new(),
        _path: dir_path.clone(),
        recursive,
    };

    // Give every symlinked subdirectory an explicit watch, using polling for
    // the ones that live on a filesystem inotify cannot observe.
    if recursive && !polling {
        for link in collect_symlinked_dirs(&dir_path) {
            if let Err(e) = handle.watch_symlinked_dir(&link) {
                log::warn!("[Watcher] {}", e);
            }
        }
    }

    Ok(handle)
}

/// inotify on local filesystems, polling where inotify cannot see changes.
fn new_watcher(polling: bool, sink: &EventSink) -> Result<Box<dyn Watcher + Send>, String> {
    let sink = sink.clone();
    let handler = move |event: Result<Event, notify::Error>| sink(event);
    if polling {
        let config = Config::default().with_poll_interval(Duration::from_secs(POLL_WATCH_INTERVAL_SECS));
        PollWatcher::new(handler, config)
            .map(|w| Box::new(w) as Box<dyn Watcher + Send>)
            .map_err(|e| format!("Failed to create poll watcher: {}", e))
    } else {
        notify::recommended_watcher(handler)
            .map(|w| Box::new(w) as Box<dyn Watcher + Send>)
            .map_err(|e| format!("Failed to create watcher: {}", e))
    }
}

/// Symlinks to directories anywhere under `root` (walkdir skips link loops).
fn collect_symlinked_dirs(root: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path_is_symlink() && e.file_type().is_dir())
        .map(|e| e.into_path())
        .collect()
}

/// Paths in a create/rename-in event that are symlinks to directories.
fn new_symlinked_dirs(event: &Event) -> Vec<PathBuf> {
    if !matches!(
        event.kind,
        EventKind::Create(_)
            | EventKind::Modify(ModifyKind::Name(RenameMode::To | RenameMode::Both | RenameMode::Any))
    ) {
        return Vec::new();
    }
    event
        .paths
        .iter()
        .filter(|p| p.is_symlink() && p.is_dir())
        .cloned()
        .collect()
}

/// Watch symlinked directories that appeared after the watch was started and
/// queue a scan for the files already inside them.
fn register_new_symlinked_dirs(
    state: &AppState,
    lib: &Arc<LibraryContext>,
    watches: &Weak<WatchMap>,
    directory_id: i64,
    links: Vec<PathBuf>,
) {
    let Some(watches) = watches.upgrade() else {
        return;
    };
    let mut added = false;
    let dir_path = {
        let Ok(mut map) = watches.lock() else {
            return;
        };
        let Some(handle) = map.get_mut(&directory_id) else {
            return;
        };
        for link in &links {
            match handle.watch_symlinked_dir(link) {
                Ok(true) => {
                    added = true;
                    log::info!("[Watcher] Watching new symlinked directory {}", link.display());
                }
                Ok(false) => {}
                Err(e) => log::warn!("[Watcher] {}", e),
            }
        }
        handle._path.clone()
    };

    if added {
        if let Err(e) = task_queue::enqueue_task(
            state,
            task_queue::TASK_SCAN_DIRECTORY,
            &json!({
                "directory_id": directory_id,
                "directory_path": dir_path.to_string_lossy(),
                "library_id": lib.uuid,
                "recursive": true,
                "fast_import": true,
            }),
            task_queue::PRIORITY_INDEX,
            None,
        ) {
            log::error!(
                "[Watcher] Failed to queue scan for new symlinked directory in {}: {}",
                directory_id,
                e
            );
        }
    }
}

/// Whether inotify cannot be relied on for `path` because it lives on a FUSE or
/// network filesystem (changes made outside this mount never raise events).
fn needs_polling(path: &Path) -> bool {
    #[cfg(target_os = "linux")]
    {
        let Ok(mountinfo) = std::fs::read_to_string("/proc/self/mountinfo") else {
            return false;
        };
        let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        return mount_fs_type(&mountinfo, &resolved)
            .map(|t| fs_type_needs_polling(&t))
            .unwrap_or(false);
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        false
    }
}

/// Filesystem type of the mount containing `path`, from /proc/self/mountinfo text.
fn mount_fs_type(mountinfo: &str, path: &Path) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for line in mountinfo.lines() {
        let fields: Vec<&str> = line.split(' ').collect();
        let Some(sep) = fields.iter().position(|f| *f == "-") else {
            continue;
        };
        let (Some(mount_point), Some(fs_type)) = (fields.get(4), fields.get(sep + 1)) else {
            continue;
        };
        let mount_point = PathBuf::from(unescape_mountinfo(mount_point));
        if path.starts_with(&mount_point) {
            let depth = mount_point.components().count();
            // Later entries win ties: they are mounted over earlier ones.
            if best.as_ref().map_or(true, |(d, _)| depth >= *d) {
                best = Some((depth, fs_type.to_string()));
            }
        }
    }
    best.map(|(_, t)| t)
}

/// mountinfo escapes space, tab, newline and backslash as octal (`\040`).
fn unescape_mountinfo(field: &str) -> String {
    let bytes = field.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() && bytes[i + 1..i + 4].iter().all(|b| (b'0'..=b'7').contains(b)) {
            let v = (bytes[i + 1] - b'0') * 64 + (bytes[i + 2] - b'0') * 8 + (bytes[i + 3] - b'0');
            out.push(v);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn fs_type_needs_polling(fs_type: &str) -> bool {
    matches!(
        fs_type,
        "nfs" | "nfs4" | "cifs" | "smb3" | "smbfs" | "9p" | "afs" | "ceph" | "davfs" | "fuse"
    ) || (fs_type.starts_with("fuse.") && fs_type != "fuse.portal")
}

/// Load distinct parent paths and their settings from the DB.
fn load_parent_directories(state: &AppState) -> Result<Vec<(String, ParentSettings)>, String> {
    let mut parents = Vec::new();

    for lib in state.library_manager().all_mounted() {
        let conn = lib
            .main_pool
            .get()
            .map_err(|e| format!("DB error for library '{}': {}", lib.name, e))?;

        // Get distinct parent_paths with the settings from the first child directory
        let mut stmt = conn
            .prepare(
                "SELECT parent_path, recursive, auto_tag, auto_age_detect,
                        show_images, show_videos, show_music
                 FROM watch_directories
                 WHERE parent_path IS NOT NULL AND enabled = 1
                 GROUP BY parent_path",
            )
            .map_err(|e| format!("Query error: {}", e))?;

        let rows: Vec<(String, bool, bool, bool, bool, bool, bool)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?))
            })
            .map_err(|e| format!("Query error: {}", e))?
            .filter_map(|r| r.ok())
            .collect();

        for (parent_path, recursive, auto_tag, auto_age_detect, show_images, show_videos, show_music) in rows {
            parents.push((
                parent_path,
                ParentSettings {
                    recursive,
                    auto_tag,
                    auto_age_detect,
                    show_images,
                    show_videos,
                    show_music,
                    library: lib.clone(),
                },
            ));
        }
    }

    Ok(parents)
}

/// Polling loop: periodically scans parent directories for new subdirectories.
/// Uses polling instead of inotify so it works on mounted/network/FUSE drives.
async fn poll_parent_directories(
    state: AppState,
    watches: Arc<WatchMap>,
    parent_dirs: Arc<Mutex<HashMap<String, ParentSettings>>>,
) {
    loop {
        tokio::time::sleep(Duration::from_secs(PARENT_POLL_INTERVAL_SECS)).await;

        // Snapshot current parent dirs
        let parents: Vec<(String, ParentSettings)> = match parent_dirs.lock() {
            Ok(pd) => pd.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Err(_) => continue,
        };

        if parents.is_empty() {
            continue;
        }

        for (parent_path, settings) in parents {
            let dir_path = PathBuf::from(&parent_path);
            if !dir_path.exists() {
                continue;
            }

            // Read immediate subdirectories
            let subdirs = match std::fs::read_dir(&dir_path) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect::<Vec<_>>(),
                Err(_) => continue,
            };

            for subdir in subdirs {
                // Canonicalize to match how add_directory stores paths
                let canonical = std::fs::canonicalize(&subdir).unwrap_or(subdir.clone());
                let subdir_path = canonical.to_string_lossy().to_string();
                let subdir_name = subdir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("Unknown")
                    .to_string();

                // Check if already registered in ANY library
                let sp = subdir_path.clone();
                let state_ref = state.clone();
                let already_exists = tokio::task::spawn_blocking({
                    move || -> bool {
                        for lib in state_ref.library_manager().all_mounted() {
                            if let Ok(conn) = lib.main_pool.get() {
                                let found = conn
                                    .query_row(
                                        "SELECT COUNT(*) FROM watch_directories WHERE path = ?1",
                                        params![&sp],
                                        |row| row.get::<_, i64>(0).map(|c| c > 0),
                                    )
                                    .unwrap_or(false);
                                if found {
                                    return true;
                                }
                            }
                        }
                        false
                    }
                })
                .await
                .unwrap_or(false);
                let lib = settings.library.clone();

                if already_exists {
                    continue;
                }

                log::info!(
                    "[ParentWatcher] New subdirectory detected: {} (in {})",
                    subdir_name,
                    parent_path
                );

                // Add the new subdirectory as a watch directory
                let recursive = settings.recursive;
                let auto_tag = settings.auto_tag;
                let auto_age_detect = settings.auto_age_detect;
                let show_images = settings.show_images;
                let show_videos = settings.show_videos;
                let show_music = settings.show_music;
                let state_clone = state.clone();
                let watches_clone = watches.clone();
                let parent_path_clone = parent_path.clone();
                let subdir_path_clone = subdir_path.clone();
                let subdir_name_clone = subdir_name.clone();

                let result = tokio::task::spawn_blocking({
                    let lib = lib.clone();
                    let state = state_clone.clone();
                    move || -> Result<i64, String> {
                        let conn = lib.main_pool.get().map_err(|e| e.to_string())?;

                        let now = chrono::Utc::now().to_rfc3339();
                        conn.execute(
                            "INSERT INTO watch_directories (path, name, enabled, recursive, auto_tag, auto_age_detect, show_images, show_videos, show_music, parent_path, created_at)
                             VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                            params![
                                &subdir_path_clone,
                                &subdir_name_clone,
                                recursive,
                                auto_tag,
                                auto_age_detect,
                                show_images,
                                show_videos,
                                show_music,
                                &parent_path_clone,
                                &now,
                            ],
                        ).map_err(|e| e.to_string())?;

                        let dir_id = conn.last_insert_rowid();

                        // Create per-directory database
                        let dir_pool = lib.directory_db.get_pool(dir_id).map_err(|e| e.to_string())?;
                        let dir_conn = dir_pool.get().map_err(|e| e.to_string())?;
                        init_directory_db(&dir_conn).map_err(|e| e.to_string())?;

                        // Queue scan task
                        task_queue::enqueue_task(
                            &state,
                            task_queue::TASK_SCAN_DIRECTORY,
                            &json!({
                                "directory_id": dir_id,
                                "directory_path": &subdir_path_clone,
                                "library_id": lib.uuid,
                                "recursive": recursive,
                                "fast_import": true
                            }),
                            task_queue::PRIORITY_INDEX,
                            None,
                        ).map_err(|e| e.to_string())?;

                        Ok(dir_id)
                    }
                })
                .await;

                match result {
                    Ok(Ok(dir_id)) => {
                        log::info!(
                            "[ParentWatcher] Auto-added directory {} as id {}",
                            subdir_path,
                            dir_id
                        );

                        // Grant asset:// read access to the auto-discovered subdir.
                        state_clone.allow_asset_dir(&subdir_path);

                        // Register with the file watcher
                        if let Err(e) = add_watch(
                            &state_clone,
                            &watches_clone,
                            dir_id,
                            &subdir_path,
                            recursive,
                            lib,
                        ) {
                            log::error!(
                                "[ParentWatcher] Failed to add file watcher for {}: {}",
                                subdir_path,
                                e
                            );
                        }

                        // Emit event so frontend updates
                        if let Some(events) = state_clone.events() {
                            events.library.broadcast(
                                "directory_added",
                                json!({
                                    "id": dir_id,
                                    "path": subdir_path,
                                    "name": subdir_name,
                                }),
                            );
                        }
                    }
                    Ok(Err(e)) => {
                        log::error!(
                            "[ParentWatcher] Failed to auto-add directory {}: {}",
                            subdir_path,
                            e
                        );
                    }
                    Err(e) => {
                        log::error!("[ParentWatcher] Task error: {}", e);
                    }
                }
            }
        }
    }
}

fn handle_fs_event(
    state: &AppState,
    lib: &Arc<LibraryContext>,
    directory_id: i64,
    event: Event,
    rt: &tokio::runtime::Handle,
) {
    let media = match file_tracker::directory_media_settings(lib, directory_id) {
        Ok(media) => media,
        Err(error) => {
            log::warn!("[Watcher] Cannot read directory media settings: {}", error);
            return;
        }
    };
    // Audio shares watched folders with images, but has a separate index and
    // never enters the image import or tagging pipeline.
    for path in &event.paths {
        if !media.music || !music::is_audio_file(path) || is_in_dumpster(path) {
            continue;
        }
        let path = path.clone();
        let lib = lib.clone();
        let removed = matches!(
            event.kind,
            EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(RenameMode::From))
        );
        rt.spawn(async move {
            if !removed {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            let result = tokio::task::spawn_blocking(move || {
                if removed {
                    music::mark_audio_missing(&lib, &path)
                } else {
                    if !file_tracker::directory_media_settings(&lib, directory_id)?.music {
                        return Ok(());
                    }
                    music::index_audio_file(&lib, directory_id, &path)
                }
            })
            .await;
            if let Ok(Err(error)) = result {
                log::warn!("[Watcher] Music index error: {}", error);
            }
        });
    }
    match event.kind {
        // New file created
        EventKind::Create(_) => {
            for path in &event.paths {
                if path.is_file() && media.allows_gallery_file(path) && !is_in_dumpster(path) {
                    let file_path = path.to_string_lossy().to_string();
                    let state_clone = state.clone();
                    let lib_clone = lib.clone();

                    // Debounced import — wait for file to stabilize
                    rt.spawn(async move {
                        debounced_import(state_clone, lib_clone, directory_id, file_path).await;
                    });
                }
            }
        }

        // File removed
        EventKind::Remove(_) => {
            for path in &event.paths {
                let file_path = path.to_string_lossy().to_string();
                let lib_clone = lib.clone();

                rt.spawn(async move {
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        file_tracker::mark_file_missing(&lib_clone, &file_path, directory_id)
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
                if path.is_file() && media.allows_gallery_file(path) && !is_in_dumpster(path) {
                    let file_path = path.to_string_lossy().to_string();
                    let state_clone = state.clone();
                    let lib_clone = lib.clone();

                    rt.spawn(async move {
                        debounced_import(state_clone, lib_clone, directory_id, file_path).await;
                    });
                }
            }
        }

        // File moved/renamed FROM this directory — treat as removal
        EventKind::Modify(ModifyKind::Name(RenameMode::From)) => {
            for path in &event.paths {
                let file_path = path.to_string_lossy().to_string();
                let lib_clone = lib.clone();

                rt.spawn(async move {
                    if let Err(e) = tokio::task::spawn_blocking(move || {
                        file_tracker::mark_file_missing(&lib_clone, &file_path, directory_id)
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
                if path.is_file() && media.allows_gallery_file(path) && !is_in_dumpster(path) {
                    let file_path = path.to_string_lossy().to_string();
                    let state_clone = state.clone();
                    let lib_clone = lib.clone();

                    rt.spawn(async move {
                        // Check if already tracked before importing
                        let already_tracked = tokio::task::spawn_blocking({
                            let lib = lib_clone.clone();
                            let fp = file_path.clone();
                            move || -> bool {
                                if let Ok(pool) = lib.directory_db.get_pool(directory_id) {
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
                            debounced_import(state_clone, lib_clone, directory_id, file_path).await;
                        }
                    });
                }
            }
        }

        _ => {}
    }
}

/// Wait for a file to stabilize (not being written to), then import it with retry.
async fn debounced_import(
    state: AppState,
    lib: Arc<LibraryContext>,
    directory_id: i64,
    file_path: String,
) {
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
        let lib_clone = lib.clone();
        let fp = file_path.clone();

        match tokio::task::spawn_blocking(move || {
            let media = file_tracker::directory_media_settings(&lib_clone, directory_id)?;
            if !media.allows_gallery_file(std::path::Path::new(&fp)) {
                return Ok(None);
            }
            importer::import_image(&state_clone, &lib_clone, &fp, directory_id, true).map(Some)
        })
        .await
        {
            Ok(Ok(Some(result))) => {
                if result.status == importer::ImportStatus::Imported {
                    log::info!(
                        "[Watcher] Imported: {}",
                        std::path::Path::new(&file_path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(&file_path)
                    );
                    let _ = task_queue::enqueue_task(
                        &state,
                        task_queue::TASK_COMPLETE_IMPORTS,
                        &json!({
                            "directory_id": directory_id,
                            "library_id": lib.uuid,
                        }),
                        task_queue::PRIORITY_THUMBNAILS,
                        None,
                    );
                }
                return;
            }
            Ok(Ok(None)) => return,
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
fn startup_scan(
    state: &AppState,
    lib: &Arc<LibraryContext>,
    directory_id: i64,
    dir_path: &str,
    recursive: bool,
) {
    let start = std::time::Instant::now();
    if !std::path::Path::new(dir_path).is_dir() {
        return;
    }

    let result = task_queue::enqueue_task(
        state,
        task_queue::TASK_SCAN_DIRECTORY,
        &json!({
            "directory_id": directory_id,
            "directory_path": dir_path,
            "library_id": lib.uuid,
            "recursive": recursive,
            "clean_deleted": true,
            "fast_import": true,
        }),
        task_queue::PRIORITY_INDEX,
        None,
    );

    match result {
        Ok(Some(_)) => log::info!(
            "[Watcher] Queued startup reconciliation for directory {} in {:?}",
            directory_id,
            start.elapsed()
        ),
        Ok(None) => log::debug!(
            "[Watcher] Startup reconciliation already queued for directory {}",
            directory_id
        ),
        Err(e) => log::error!(
            "[Watcher] Failed to queue startup reconciliation for directory {}: {}",
            directory_id,
            e
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "localbooru-watcher-{}-{}",
            name,
            uuid::Uuid::new_v4()
        ))
    }

    // AC: @safe-development-startup ac-reconciliation-preserved
    #[test]
    fn startup_queues_fast_reconciliation_for_never_scanned_directory() {
        let data_dir = temp_test_dir("startup-reconcile");
        let watch_dir = data_dir.join("media");
        std::fs::create_dir_all(&watch_dir).unwrap();

        let state = AppState::new(&data_dir, 0).unwrap();
        let lib = state.library_manager().primary().clone();
        let directory_id = {
            let conn = lib.main_pool.get().unwrap();
            conn.execute(
                "INSERT INTO watch_directories (path, name, recursive) VALUES (?1, 'Media', 0)",
                params![watch_dir.to_string_lossy()],
            )
            .unwrap();
            conn.last_insert_rowid()
        };

        startup_scan(
            &state,
            &lib,
            directory_id,
            watch_dir.to_str().unwrap(),
            false,
        );

        let payload: String = state
            .main_db()
            .get()
            .unwrap()
            .query_row(
                "SELECT payload FROM task_queue WHERE task_type = 'scan_directory'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let payload: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(payload["directory_id"], directory_id);
        assert_eq!(payload["recursive"], false);
        assert_eq!(payload["clean_deleted"], true);
        assert_eq!(payload["fast_import"], true);

        drop(state);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    fn channel_sink() -> (EventSink, std::sync::mpsc::Receiver<Event>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let tx = Mutex::new(tx);
        let sink: EventSink = Arc::new(move |ev| {
            if let Ok(ev) = ev {
                let _ = tx.lock().unwrap().send(ev);
            }
        });
        (sink, rx)
    }

    /// Wait for an event touching a path that satisfies `pred`.
    fn wait_for_path(
        rx: &std::sync::mpsc::Receiver<Event>,
        secs: u64,
        pred: impl Fn(&Event, &Path) -> bool,
    ) -> Option<PathBuf> {
        let deadline = std::time::Instant::now() + Duration::from_secs(secs);
        while let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) {
            if let Ok(ev) = rx.recv_timeout(left) {
                if let Some(p) = ev.paths.iter().find(|p| pred(&ev, p)) {
                    return Some(p.clone());
                }
            }
        }
        None
    }

    #[test]
    fn mount_fs_type_picks_deepest_mount_and_flags_fuse() {
        let mountinfo = "\
24 1 259:2 / / rw - ext4 /dev/root rw
30 24 0:40 / /mnt/storage rw - ext4 /dev/mapper/storage rw
31 30 0:50 / /mnt/storage/decrypted rw,nosuid - fuse.gocryptfs /mnt/storage/encrypted-storage rw
32 24 0:51 / /mnt/my\\040share rw - cifs //nas/share rw
33 24 0:52 / /run/user/1000/doc rw - fuse.portal portal rw
";
        let fs = |p: &str| mount_fs_type(mountinfo, Path::new(p)).unwrap();
        assert_eq!(fs("/home/user/pics"), "ext4");
        assert_eq!(fs("/mnt/storage/media"), "ext4");
        assert_eq!(fs("/mnt/storage/decrypted"), "fuse.gocryptfs");
        assert_eq!(fs("/mnt/storage/decrypted/a/b"), "fuse.gocryptfs");
        // A sibling that merely shares a name prefix is not inside the mount.
        assert_eq!(fs("/mnt/storage/decrypted-other"), "ext4");
        assert_eq!(fs("/mnt/my share/x"), "cifs");

        assert!(fs_type_needs_polling("fuse.gocryptfs"));
        assert!(fs_type_needs_polling("nfs4"));
        assert!(fs_type_needs_polling("cifs"));
        assert!(!fs_type_needs_polling("ext4"));
        assert!(!fs_type_needs_polling("btrfs"));
        assert!(!fs_type_needs_polling("fuseblk"));
        assert!(!fs_type_needs_polling("fuse.portal"));
    }

    #[test]
    fn collect_symlinked_dirs_finds_links_and_survives_loops() {
        let root = temp_test_dir("collect-links");
        let outside = temp_test_dir("collect-links-target");
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        // Loop back to the root, and a symlink to a file (not a directory).
        std::os::unix::fs::symlink(&root, root.join("real").join("loop")).unwrap();
        std::fs::write(outside.join("f.jpg"), b"x").unwrap();
        std::os::unix::fs::symlink(outside.join("f.jpg"), root.join("file-link")).unwrap();

        let links = collect_symlinked_dirs(&root);
        assert!(links.contains(&root.join("link")), "{links:?}");
        assert!(!links.contains(&root.join("file-link")), "{links:?}");

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn existing_symlinked_subdir_reports_events_under_link_path() {
        let root = temp_test_dir("existing-link");
        let outside = temp_test_dir("existing-link-target");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

        let (sink, rx) = channel_sink();
        let _handle = start_watch(root.clone(), true, false, sink).unwrap();

        std::fs::write(outside.join("new.jpg"), b"x").unwrap();
        let hit = wait_for_path(&rx, 5, |_, p| p.ends_with("new.jpg"));
        assert_eq!(hit, Some(root.join("link").join("new.jpg")));

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn symlinked_subdir_created_after_start_gets_watched() {
        let root = temp_test_dir("late-link");
        let outside = temp_test_dir("late-link-target");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        let (sink, rx) = channel_sink();
        let mut handle = start_watch(root.clone(), true, false, sink).unwrap();

        let link = root.join("late");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        // The creation event is what triggers registration in production.
        let created = wait_for_path(&rx, 5, |ev, p| {
            *p == link && !new_symlinked_dirs(ev).is_empty()
        });
        assert_eq!(created.as_ref(), Some(&link));
        assert_eq!(handle.watch_symlinked_dir(&link), Ok(true));
        assert_eq!(handle.watch_symlinked_dir(&link), Ok(false));

        std::fs::write(outside.join("new.jpg"), b"x").unwrap();
        let hit = wait_for_path(&rx, 5, |_, p| p.ends_with("new.jpg"));
        assert_eq!(hit, Some(link.join("new.jpg")));

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn polling_watcher_detects_files_in_symlinked_subdir() {
        let root = temp_test_dir("poll-link");
        let outside = temp_test_dir("poll-link-target");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

        let (sink, rx) = channel_sink();
        let _handle = start_watch(root.clone(), true, true, sink).unwrap();

        std::fs::write(outside.join("new.jpg"), b"x").unwrap();
        let hit = wait_for_path(&rx, 10, |_, p| p.ends_with("new.jpg"));
        assert_eq!(hit, Some(root.join("link").join("new.jpg")));

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }
}
