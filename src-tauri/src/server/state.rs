use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::RwLock;

use crate::addons::manager::AddonManager;
use crate::db::pool::DbPool;
use crate::db::directory_db::DirectoryDbManager;
use crate::db::library::{LibraryContext, LibraryManager};
use crate::routes::cast::CastState;
use crate::routes::migration::{SharedMigrationState, create_migration_state};
use crate::routes::models::{ModelRegistry, create_model_registry};
use crate::routes::network::{HandshakeManager, SharedHandshakeManager};
use crate::routes::share::{ShareSessions, create_share_sessions};
use crate::routes::svp_web::{WebDownloadRegistry, create_download_registry};
use crate::services::directory_watcher::DirectoryWatcher;
use crate::services::events::{SharedEvents, create_events};
use crate::services::rate_limit::RateLimiter;
use crate::services::task_queue::BackgroundTaskQueue;
use crate::services::transcode::TranscodeManager;

/// Shared application state available to all axum handlers.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    /// Library manager (primary + auxiliary libraries)
    library_manager: LibraryManager,
    /// Server port
    port: u16,
    /// Per-install JWT signing secret (loaded from or generated into settings.json)
    jwt_secret: String,
    /// Event broadcasters (SSE)
    events: SharedEvents,
    /// Background task queue
    task_queue: Arc<BackgroundTaskQueue>,
    /// Addon manager (sidecar lifecycle + registry)
    addon_manager: AddonManager,
    /// Transcode manager (FFmpeg HLS streaming)
    transcode_manager: TranscodeManager,
    /// Rate limiter (in-memory, per-IP sliding window)
    rate_limiter: Arc<RateLimiter>,
    /// Active media share sessions (token -> session)
    share_sessions: ShareSessions,
    /// Cast/Chromecast session state
    cast_state: Arc<RwLock<CastState>>,
    /// Active web video downloads (yt-dlp), keyed by download_id
    web_download_registry: WebDownloadRegistry,
    /// ML model download state registry, keyed by model name
    model_registry: ModelRegistry,
    /// Data migration state (per-directory <-> main DB)
    migration_state: SharedMigrationState,
    /// Network handshake nonce manager (SSL pinning / QR verification)
    handshake_manager: SharedHandshakeManager,
    /// Shared HTTP client (connection pool reused across requests)
    http_client: reqwest::Client,
    /// Directory watcher (set after AppState construction to break circular dep)
    directory_watcher: std::sync::OnceLock<Arc<DirectoryWatcher>>,
    /// Family mode lock state (true = locked, hides non-family-safe content)
    family_mode_locked: AtomicBool,
}

/// Load the JWT secret from `settings.json` in `data_dir`, or generate a new
/// one if absent. The secret is persisted so tokens survive server restarts.
fn load_or_generate_jwt_secret(data_dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let settings_path = data_dir.join("settings.json");

    // Try to load existing secret from settings.json
    if settings_path.exists() {
        let contents = std::fs::read_to_string(&settings_path)?;
        if let Ok(mut obj) = serde_json::from_str::<serde_json::Value>(&contents) {
            if let Some(secret) = obj.get("jwt_secret").and_then(|v| v.as_str()) {
                if !secret.is_empty() {
                    return Ok(secret.to_owned());
                }
            }

            // settings.json exists but has no jwt_secret — generate and merge
            let secret = generate_jwt_secret();
            obj.as_object_mut()
                .ok_or("settings.json is not a JSON object")?
                .insert("jwt_secret".into(), serde_json::Value::String(secret.clone()));
            std::fs::write(&settings_path, serde_json::to_string_pretty(&obj)?)?;
            return Ok(secret);
        }
    }

    // No settings.json at all — create one with just the secret
    let secret = generate_jwt_secret();
    let obj = serde_json::json!({ "jwt_secret": secret });
    std::fs::write(&settings_path, serde_json::to_string_pretty(&obj)?)?;
    Ok(secret)
}

/// Generate a random 32-byte hex-encoded JWT secret.
fn generate_jwt_secret() -> String {
    use rand::Rng;
    let bytes: [u8; 32] = rand::thread_rng().gen();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Determine whether family mode should start locked based on settings.json.
/// Returns true if family_mode.enabled && family_mode.auto_lock_on_start.
fn load_family_mode_initial_lock(data_dir: &Path) -> bool {
    let settings_path = data_dir.join("settings.json");
    if !settings_path.exists() {
        return false;
    }
    let contents = match std::fs::read_to_string(&settings_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let obj: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let fm = match obj.get("family_mode") {
        Some(v) => v,
        None => return false,
    };
    let enabled = fm.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    let auto_lock = fm.get("auto_lock_on_start").and_then(|v| v.as_bool()).unwrap_or(true);
    enabled && auto_lock
}

impl AppState {
    /// Create new AppState, initializing database pools and schema.
    pub fn new(data_dir: &Path, port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        // Ensure data directory exists
        std::fs::create_dir_all(data_dir)?;

        // Create primary library context (opens/creates DB, loads UUID)
        let primary = LibraryContext::open(data_dir, "Local Library")?;
        let library_manager = LibraryManager::new(primary);

        // Auto-mount libraries registered with auto_mount = 1
        {
            let conn = library_manager.primary().main_pool.get()?;
            let mut stmt = conn.prepare(
                "SELECT uuid, name, path FROM mounted_libraries WHERE auto_mount = 1 ORDER BY mount_order"
            )?;
            let libraries: Vec<(String, String, String)> = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?.filter_map(|r| r.ok()).collect();
            drop(stmt);
            drop(conn);

            for (uuid, name, path) in libraries {
                let lib_path = PathBuf::from(&path);
                match LibraryContext::open(&lib_path, &name) {
                    Ok(ctx) => {
                        if ctx.uuid != uuid {
                            log::warn!(
                                "[Libraries] UUID mismatch for '{}': expected {}, got {}",
                                name, uuid, ctx.uuid
                            );
                        }
                        library_manager.mount(ctx);
                        // Update last_mounted_at timestamp
                        if let Ok(conn) = library_manager.primary().main_pool.get() {
                            let _ = conn.execute(
                                "UPDATE mounted_libraries SET last_mounted_at = datetime('now') WHERE uuid = ?1",
                                rusqlite::params![uuid],
                            );
                        }
                        log::info!("[Libraries] Auto-mounted library '{}' from {}", name, path);
                    }
                    Err(e) => {
                        log::warn!(
                            "[Libraries] Failed to auto-mount library '{}' at {}: {}",
                            name, path, e
                        );
                    }
                }
            }
        }

        // Load or generate per-install JWT secret
        let jwt_secret = load_or_generate_jwt_secret(data_dir)?;

        // Create event broadcasters
        let events = create_events();

        // Create task queue
        let task_queue = Arc::new(BackgroundTaskQueue::new());

        // Create addon manager
        let addon_manager = AddonManager::new(data_dir);

        // Create transcode manager
        let transcode_manager = TranscodeManager::new();

        // Create rate limiter
        let rate_limiter = Arc::new(RateLimiter::new());

        // Create share sessions map
        let share_sessions = create_share_sessions();

        // Create cast state
        let cast_state = Arc::new(RwLock::new(CastState::new()));

        // Create web download registry (SVP web video / yt-dlp)
        let web_download_registry = create_download_registry();

        // Create ML model registry
        let model_registry = create_model_registry();

        // Create migration state
        let migration_state = create_migration_state();

        // Create handshake nonce manager
        let handshake_manager = Arc::new(HandshakeManager::new());

        // Create shared HTTP client (connection pool reused across requests)
        let http_client = reqwest::Client::new();

        // Determine initial family mode lock state from settings
        let family_mode_locked = load_family_mode_initial_lock(data_dir);

        Ok(Self {
            inner: Arc::new(AppStateInner {
                library_manager,
                port,
                jwt_secret,
                events,
                task_queue,
                addon_manager,
                transcode_manager,
                rate_limiter,
                share_sessions,
                cast_state,
                web_download_registry,
                model_registry,
                migration_state,
                handshake_manager,
                http_client,
                directory_watcher: std::sync::OnceLock::new(),
                family_mode_locked: AtomicBool::new(family_mode_locked),
            }),
        })
    }

    // ── Primary library backward-compatible accessors ────────────────────────

    /// Get the main library database pool (primary library).
    pub fn main_db(&self) -> &DbPool {
        &self.inner.library_manager.primary().main_pool
    }

    /// Get the directory database manager (primary library).
    pub fn directory_db(&self) -> &DirectoryDbManager {
        &self.inner.library_manager.primary().directory_db
    }

    /// Get the data directory path (primary library).
    pub fn data_dir(&self) -> &Path {
        &self.inner.library_manager.primary().data_dir
    }

    /// Get the thumbnails directory path (primary library).
    pub fn thumbnails_dir(&self) -> PathBuf {
        self.inner.library_manager.primary().thumbnails_dir()
    }

    // ── Multi-library accessors ─────────────────────────────────────────────

    /// Get the library manager.
    pub fn library_manager(&self) -> &LibraryManager {
        &self.inner.library_manager
    }

    /// Resolve a library by UUID. Returns the primary library when `library_id`
    /// is `None` or `"primary"`. Returns 404 error if the library is not found
    /// or not mounted.
    pub fn resolve_library(
        &self,
        library_id: Option<&str>,
    ) -> Result<Arc<LibraryContext>, crate::server::error::AppError> {
        match library_id {
            None | Some("primary") => Ok(self.inner.library_manager.primary().clone()),
            Some(uuid) => self.inner.library_manager.get(uuid).ok_or_else(|| {
                crate::server::error::AppError::NotFound(format!(
                    "Library '{}' not found or not mounted",
                    uuid
                ))
            }),
        }
    }

    // ── Other accessors (unchanged) ─────────────────────────────────────────

    /// Get the server port.
    pub fn port(&self) -> u16 {
        self.inner.port
    }

    /// Get the per-install JWT signing secret.
    pub fn jwt_secret(&self) -> &str {
        &self.inner.jwt_secret
    }

    /// Get the event broadcasters.
    pub fn events(&self) -> Option<&SharedEvents> {
        Some(&self.inner.events)
    }

    /// Get the background task queue.
    pub fn task_queue(&self) -> Option<&BackgroundTaskQueue> {
        Some(&*self.inner.task_queue)
    }

    /// Get the task queue Arc (for starting the worker).
    pub fn task_queue_arc(&self) -> Arc<BackgroundTaskQueue> {
        self.inner.task_queue.clone()
    }

    /// Get the addon manager.
    pub fn addon_manager(&self) -> &AddonManager {
        &self.inner.addon_manager
    }

    /// Get the transcode manager.
    pub fn transcode_manager(&self) -> &TranscodeManager {
        &self.inner.transcode_manager
    }

    /// Get the rate limiter.
    pub fn rate_limiter(&self) -> &RateLimiter {
        &self.inner.rate_limiter
    }

    /// Get the active share sessions map.
    pub fn share_sessions(&self) -> &ShareSessions {
        &self.inner.share_sessions
    }

    /// Get the cast/Chromecast session state.
    pub fn cast_state(&self) -> &Arc<RwLock<CastState>> {
        &self.inner.cast_state
    }

    /// Get the web download registry (SVP web video / yt-dlp downloads).
    pub fn web_download_registry(&self) -> &WebDownloadRegistry {
        &self.inner.web_download_registry
    }

    /// Get the ML model download registry.
    pub fn model_registry(&self) -> &ModelRegistry {
        &self.inner.model_registry
    }

    /// Get the migration state (per-directory <-> main DB migration).
    pub fn migration_state(&self) -> &SharedMigrationState {
        &self.inner.migration_state
    }

    /// Get the network handshake nonce manager.
    pub fn handshake_manager(&self) -> &SharedHandshakeManager {
        &self.inner.handshake_manager
    }

    /// Get the shared HTTP client (reuses connection pool across requests).
    pub fn http_client(&self) -> &reqwest::Client {
        &self.inner.http_client
    }

    /// Set the directory watcher (called once after AppState construction).
    pub fn set_directory_watcher(&self, watcher: Arc<DirectoryWatcher>) {
        let _ = self.inner.directory_watcher.set(watcher);
    }

    /// Get the directory watcher, if set.
    pub fn directory_watcher(&self) -> Option<&Arc<DirectoryWatcher>> {
        self.inner.directory_watcher.get()
    }

    /// Check if family mode is currently locked.
    pub fn is_family_mode_locked(&self) -> bool {
        self.inner.family_mode_locked.load(Ordering::Relaxed)
    }

    /// Set the family mode lock state.
    pub fn set_family_mode_locked(&self, locked: bool) {
        self.inner.family_mode_locked.store(locked, Ordering::Relaxed);
    }

    /// Check if local network access is enabled in settings.json.
    /// Used to determine whether to bind to 0.0.0.0 or 127.0.0.1.
    pub fn is_lan_enabled(&self) -> bool {
        let settings_path = self.inner.library_manager.primary().data_dir.join("settings.json");
        let contents = match std::fs::read_to_string(&settings_path) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let obj: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(v) => v,
            Err(_) => return false,
        };
        obj.get("network")
            .and_then(|n| n.get("local_network_enabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}
