use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::addons::manager::AddonManager;
use crate::db::pool::{create_main_pool, DbPool};
use crate::db::directory_db::DirectoryDbManager;
use crate::db::schema::init_main_db;
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
    /// Main library database pool
    main_pool: DbPool,
    /// Per-directory database manager
    directory_db: DirectoryDbManager,
    /// Data directory path (e.g. ~/.localbooru)
    data_dir: PathBuf,
    /// Server port
    port: u16,
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
    /// Directory watcher (set after AppState construction to break circular dep)
    directory_watcher: std::sync::OnceLock<Arc<DirectoryWatcher>>,
}

impl AppState {
    /// Create new AppState, initializing database pools and schema.
    pub fn new(data_dir: &Path, port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        // Ensure directories exist
        std::fs::create_dir_all(data_dir)?;
        std::fs::create_dir_all(data_dir.join("thumbnails"))?;
        std::fs::create_dir_all(data_dir.join("directories"))?;

        // Create main database pool and initialize schema
        let main_pool = create_main_pool(data_dir)?;
        {
            let conn = main_pool.get()?;
            init_main_db(&conn)?;
        }

        // Create directory database manager
        let directory_db = DirectoryDbManager::new(data_dir);

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

        Ok(Self {
            inner: Arc::new(AppStateInner {
                main_pool,
                directory_db,
                data_dir: data_dir.to_path_buf(),
                port,
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
                directory_watcher: std::sync::OnceLock::new(),
            }),
        })
    }

    /// Get the main library database pool.
    pub fn main_db(&self) -> &DbPool {
        &self.inner.main_pool
    }

    /// Get the directory database manager.
    pub fn directory_db(&self) -> &DirectoryDbManager {
        &self.inner.directory_db
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.inner.data_dir
    }

    /// Get the thumbnails directory path.
    pub fn thumbnails_dir(&self) -> PathBuf {
        self.inner.data_dir.join("thumbnails")
    }

    /// Get the server port.
    pub fn port(&self) -> u16 {
        self.inner.port
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

    /// Set the directory watcher (called once after AppState construction).
    pub fn set_directory_watcher(&self, watcher: Arc<DirectoryWatcher>) {
        let _ = self.inner.directory_watcher.set(watcher);
    }

    /// Get the directory watcher, if set.
    pub fn directory_watcher(&self) -> Option<&Arc<DirectoryWatcher>> {
        self.inner.directory_watcher.get()
    }
}
