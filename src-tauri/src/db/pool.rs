use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;
use std::path::Path;

pub type DbPool = Pool<SqliteConnectionManager>;

/// Custom initializer that sets SQLite pragmas on every new connection.
#[derive(Debug)]
struct PragmaCustomizer;

impl r2d2::CustomizeConnection<Connection, rusqlite::Error> for PragmaCustomizer {
    fn on_acquire(&self, conn: &mut Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(
            "PRAGMA busy_timeout = 60000;
             PRAGMA journal_mode = WAL;
             PRAGMA cache_size = -64000;
             PRAGMA synchronous = NORMAL;
             PRAGMA mmap_size = 268435456;
             PRAGMA foreign_keys = ON;",
        )?;
        Ok(())
    }
}

/// Create a connection pool for the main library database.
pub fn create_main_pool(data_dir: &Path) -> Result<DbPool, Box<dyn std::error::Error>> {
    let db_path = data_dir.join("library.db");
    let manager = SqliteConnectionManager::file(&db_path);
    let pool = Pool::builder()
        .max_size(10)
        .min_idle(Some(1))
        .connection_customizer(Box::new(PragmaCustomizer))
        .build(manager)?;
    Ok(pool)
}

/// Create a connection pool for a per-directory database.
pub fn create_directory_pool(db_path: &Path) -> Result<DbPool, Box<dyn std::error::Error>> {
    let manager = SqliteConnectionManager::file(db_path);
    let pool = Pool::builder()
        .max_size(5)
        .min_idle(Some(0))
        .connection_customizer(Box::new(PragmaCustomizer))
        .build(manager)?;
    Ok(pool)
}
