use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::addons::manager::AddonManager;
use crate::db::pool::{create_main_pool, DbPool};
use crate::db::directory_db::DirectoryDbManager;
use crate::db::schema::init_main_db;
use crate::services::events::{SharedEvents, create_events};
use crate::services::task_queue::BackgroundTaskQueue;

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

        Ok(Self {
            inner: Arc::new(AppStateInner {
                main_pool,
                directory_db,
                data_dir: data_dir.to_path_buf(),
                port,
                events,
                task_queue,
                addon_manager,
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
}
