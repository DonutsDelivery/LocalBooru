use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::db::pool::{create_main_pool, DbPool};
use crate::db::directory_db::DirectoryDbManager;
use crate::db::schema::init_main_db;

/// Shared application state available to all axum handlers.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    /// Main library database pool
    pub main_pool: DbPool,
    /// Per-directory database manager
    pub directory_db: DirectoryDbManager,
    /// Data directory path (e.g. ~/.localbooru)
    pub data_dir: PathBuf,
    /// Server port
    pub port: u16,
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

        Ok(Self {
            inner: Arc::new(AppStateInner {
                main_pool,
                directory_db,
                data_dir: data_dir.to_path_buf(),
                port,
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
}
