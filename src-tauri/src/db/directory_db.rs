use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::db::pool::{create_directory_pool, DbPool};
use crate::db::schema::init_directory_db;

/// Manages per-directory SQLite databases for image data.
///
/// Each WatchDirectory gets its own database file at:
///     {data_dir}/directories/{directory_id}.db
///
/// This allows instant directory deletion by simply removing the file.
pub struct DirectoryDbManager {
    pools: Mutex<HashMap<i64, DbPool>>,
    directories_path: PathBuf,
}

impl DirectoryDbManager {
    pub fn new(data_dir: &Path) -> Self {
        let directories_path = data_dir.join("directories");
        fs::create_dir_all(&directories_path).ok();
        Self {
            pools: Mutex::new(HashMap::new()),
            directories_path,
        }
    }

    /// Get the database file path for a directory.
    pub fn db_path(&self, directory_id: i64) -> PathBuf {
        self.directories_path.join(format!("{}.db", directory_id))
    }

    /// Get or create a connection pool for a directory database.
    /// Ensures the schema is initialized on first access (handles empty DBs
    /// from mounted libraries that may have files but no tables).
    pub fn get_pool(&self, directory_id: i64) -> Result<DbPool, Box<dyn std::error::Error>> {
        let mut pools = self.pools.lock().unwrap();
        if let Some(pool) = pools.get(&directory_id) {
            return Ok(pool.clone());
        }
        let path = self.db_path(directory_id);
        let pool = create_directory_pool(&path)?;
        // Ensure schema exists (idempotent — uses CREATE TABLE IF NOT EXISTS)
        let conn = pool.get()?;
        init_directory_db(&conn)?;
        drop(conn);
        pools.insert(directory_id, pool.clone());
        Ok(pool)
    }

    /// Create a new directory database with schema.
    pub fn create_directory_db(&self, directory_id: i64) -> Result<(), Box<dyn std::error::Error>> {
        let pool = self.get_pool(directory_id)?;
        let conn = pool.get()?;
        init_directory_db(&conn)?;
        log::info!("[DirectoryDB] Created database for directory {}", directory_id);
        Ok(())
    }

    /// Delete a directory database file.
    pub fn delete_directory_db(&self, directory_id: i64) -> Result<(), Box<dyn std::error::Error>> {
        // Remove from pool cache
        {
            let mut pools = self.pools.lock().unwrap();
            pools.remove(&directory_id);
        }

        // Delete database files
        let db_path = self.db_path(directory_id);
        if db_path.exists() {
            fs::remove_file(&db_path)?;
            log::info!("[DirectoryDB] Deleted database for directory {}", directory_id);
        }
        // Also clean up WAL and SHM files
        let wal_path = db_path.with_extension("db-wal");
        let shm_path = db_path.with_extension("db-shm");
        if wal_path.exists() {
            fs::remove_file(&wal_path)?;
        }
        if shm_path.exists() {
            fs::remove_file(&shm_path)?;
        }
        Ok(())
    }

    /// Check if a directory database exists.
    pub fn db_exists(&self, directory_id: i64) -> bool {
        self.db_path(directory_id).exists()
    }

    /// Ensure a directory database exists, creating it if needed.
    pub fn ensure_db_exists(&self, directory_id: i64) -> Result<(), Box<dyn std::error::Error>> {
        if !self.db_exists(directory_id) {
            self.create_directory_db(directory_id)?;
        }
        Ok(())
    }

    /// Get list of all directory IDs that have databases.
    pub fn get_all_directory_ids(&self) -> Vec<i64> {
        let mut ids = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.directories_path) {
            for entry in entries.flatten() {
                if let Some(stem) = entry.path().file_stem() {
                    if let Some(stem_str) = stem.to_str() {
                        if let Ok(id) = stem_str.parse::<i64>() {
                            // Only count .db files
                            if entry.path().extension().and_then(|e| e.to_str()) == Some("db") {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
        ids
    }
}
