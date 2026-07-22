use rusqlite::Connection;

/// A single database migration.
pub struct Migration {
    /// Human-readable description of what this migration does.
    pub description: &'static str,
    /// SQL statements to execute. Each migration runs inside a transaction.
    pub sql: &'static str,
}

/// Ensure the schema_version table exists, then return the current version.
/// Version 0 means no migrations have been applied yet.
fn get_schema_version(conn: &Connection) -> Result<i64, rusqlite::Error> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")?;
    let version: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);
    Ok(version)
}

/// Run all migrations that haven't been applied yet.
///
/// Each migration is wrapped in a transaction. If a migration fails due to
/// a benign error (e.g., `ALTER TABLE ADD COLUMN` on a column that already
/// exists), the error is caught and the migration is marked as applied.
pub fn run_migrations(conn: &Connection, migrations: &[Migration]) -> Result<(), rusqlite::Error> {
    let current_version = get_schema_version(conn)?;

    for (i, migration) in migrations.iter().enumerate() {
        let version = (i + 1) as i64;
        if version <= current_version {
            continue;
        }

        log::info!(
            "[Migration] Applying v{}: {}",
            version,
            migration.description
        );

        // Run each migration in a savepoint so failures are isolated
        match conn.execute_batch(&format!(
            "BEGIN;\n{}\nINSERT INTO schema_version (version) VALUES ({});\nCOMMIT;",
            migration.sql, version
        )) {
            Ok(()) => {
                log::info!("[Migration] v{} applied successfully", version);
            }
            Err(e) => {
                // Check for benign errors (duplicate column, table/index already exists)
                let msg = e.to_string();
                if msg.contains("duplicate column name") || msg.contains("already exists") {
                    log::info!(
                        "[Migration] v{} skipped (already applied): {}",
                        version,
                        msg
                    );
                    // Rollback the failed transaction, then record the version
                    conn.execute_batch("ROLLBACK;")?;
                    conn.execute(
                        "INSERT OR IGNORE INTO schema_version (version) VALUES (?1)",
                        rusqlite::params![version],
                    )?;
                } else {
                    // Real error: rollback and propagate
                    conn.execute_batch("ROLLBACK;").ok();
                    log::error!("[Migration] v{} failed: {}", version, e);
                    return Err(e);
                }
            }
        }
    }

    Ok(())
}

// ─── Main DB migrations ─────────────────────────────────────────────────────

/// Migrations for the main library database.
/// Each entry corresponds to a schema version (1-indexed).
pub static MAIN_MIGRATIONS: &[Migration] = &[
    // v1: Add missing index on collection_items.image_id
    Migration {
        description: "Add index on collection_items.image_id",
        sql: "CREATE INDEX IF NOT EXISTS idx_collection_items_image_id ON collection_items(image_id);",
    },
    // v2: Add family_safe and lan_visible to watch_directories
    Migration {
        description: "Add family_safe and lan_visible to watch_directories",
        sql: "ALTER TABLE watch_directories ADD COLUMN family_safe INTEGER NOT NULL DEFAULT 1;\
              ALTER TABLE watch_directories ADD COLUMN lan_visible INTEGER NOT NULL DEFAULT 1;",
    },
    // v3: Add directory_id to watch_history for visibility filtering
    Migration {
        description: "Add directory_id to watch_history for visibility filtering",
        sql: "ALTER TABLE watch_history ADD COLUMN directory_id INTEGER;",
    },
    // v4: Fix NULL attempts in task_queue (column may have been added without NOT NULL)
    Migration {
        description: "Fix NULL attempts in task_queue",
        sql: "UPDATE task_queue SET attempts = 0 WHERE attempts IS NULL;",
    },
    // v5: Add file_extension column + index to image_files for fast media type filtering
    Migration {
        description: "Add file_extension column and index to image_files (main DB)",
        sql: "ALTER TABLE image_files ADD COLUMN file_extension TEXT;\
              CREATE INDEX IF NOT EXISTS idx_image_files_file_extension ON image_files(file_extension);\
              UPDATE image_files SET file_extension = \
                CASE \
                  WHEN LOWER(original_path) LIKE '%.png' THEN 'png' \
                  WHEN LOWER(original_path) LIKE '%.jpg' THEN 'jpg' \
                  WHEN LOWER(original_path) LIKE '%.jpeg' THEN 'jpeg' \
                  WHEN LOWER(original_path) LIKE '%.gif' THEN 'gif' \
                  WHEN LOWER(original_path) LIKE '%.webp' THEN 'webp' \
                  WHEN LOWER(original_path) LIKE '%.bmp' THEN 'bmp' \
                  WHEN LOWER(original_path) LIKE '%.tiff' THEN 'tiff' \
                  WHEN LOWER(original_path) LIKE '%.tif' THEN 'tif' \
                  WHEN LOWER(original_path) LIKE '%.webm' THEN 'webm' \
                  WHEN LOWER(original_path) LIKE '%.mp4' THEN 'mp4' \
                  WHEN LOWER(original_path) LIKE '%.mov' THEN 'mov' \
                  WHEN LOWER(original_path) LIKE '%.avi' THEN 'avi' \
                  WHEN LOWER(original_path) LIKE '%.mkv' THEN 'mkv' \
                END \
              WHERE file_extension IS NULL;",
    },
    // v6: Add cached image/tagged/favorited counts to watch_directories for fast directory listing at startup/large DBs
    Migration {
        description: "Add cached counts to watch_directories for fast /directories responses",
        sql: "ALTER TABLE watch_directories ADD COLUMN image_count INTEGER NOT NULL DEFAULT 0;\
              ALTER TABLE watch_directories ADD COLUMN tagged_count INTEGER NOT NULL DEFAULT 0;\
              ALTER TABLE watch_directories ADD COLUMN favorited_count INTEGER NOT NULL DEFAULT 0;",
    },
    // v7-v9: One statement per migration keeps fresh and existing schemas compatible.
    Migration {
        description: "Add curation original path to image_files (main DB)",
        sql: "ALTER TABLE image_files ADD COLUMN curation_original_path TEXT;",
    },
    Migration {
        description: "Add curation discarded timestamp to image_files (main DB)",
        sql: "ALTER TABLE image_files ADD COLUMN curation_discarded_at TEXT;",
    },
    Migration {
        description: "Index curation discard state (main DB)",
        sql: "CREATE INDEX IF NOT EXISTS idx_image_files_curation_discarded_at ON image_files(curation_discarded_at);",
    },
    // v10: Watch history uses exact library/directory/image identity.
    Migration {
        description: "Use composite identity for watch history",
        sql: "ALTER TABLE watch_history RENAME TO watch_history_legacy;\
              CREATE TABLE watch_history (\
                  id INTEGER PRIMARY KEY AUTOINCREMENT,\
                  image_id INTEGER NOT NULL,\
                  playback_position REAL NOT NULL DEFAULT 0.0,\
                  duration REAL NOT NULL DEFAULT 0.0,\
                  completed INTEGER NOT NULL DEFAULT 0,\
                  last_watched TEXT NOT NULL DEFAULT (datetime('now')),\
                  created_at TEXT NOT NULL DEFAULT (datetime('now')),\
                  directory_id INTEGER,\
                  library_id TEXT,\
                  UNIQUE(library_id, directory_id, image_id)\
              );\
              INSERT INTO watch_history (\
                  id, image_id, playback_position, duration, completed, last_watched, created_at, directory_id, library_id\
              ) SELECT id, image_id, playback_position, duration, completed, last_watched, created_at, directory_id, NULL \
                FROM watch_history_legacy;\
              DROP TABLE watch_history_legacy;\
              CREATE INDEX IF NOT EXISTS idx_watch_history_image_id ON watch_history(image_id);\
              CREATE INDEX IF NOT EXISTS idx_watch_history_completed ON watch_history(completed);\
              CREATE INDEX IF NOT EXISTS idx_watch_history_locator ON watch_history(library_id, directory_id, image_id);",
    },
    // v11-v12: Persist retry eligibility separately so fresh schemas can skip
    // the duplicate column while still applying the claim index migration.
    Migration {
        description: "Add durable task retry eligibility",
        sql: "ALTER TABLE task_queue ADD COLUMN next_attempt_at TEXT;",
    },
    Migration {
        description: "Index eligible task claims",
        sql: "CREATE INDEX IF NOT EXISTS idx_task_queue_claim
              ON task_queue(status, priority DESC, COALESCE(next_attempt_at, created_at), created_at);",
    },
];

/// Run all pending migrations on the main library database.
pub fn run_main_migrations(conn: &Connection) -> Result<(), rusqlite::Error> {
    run_migrations(conn, MAIN_MIGRATIONS)
}

// ─── Directory DB migrations ────────────────────────────────────────────────

/// Migrations for per-directory databases.
/// Each entry corresponds to a schema version (1-indexed).
pub static DIRECTORY_MIGRATIONS: &[Migration] = &[
    // v1: Add index on image_files.file_status (existed in main DB but was missing here)
    Migration {
        description: "Add index on image_files.file_status for directory DB",
        sql: "CREATE INDEX IF NOT EXISTS idx_image_files_file_status ON image_files(file_status);",
    },
    // v2: Add file_extension column + index for fast media type filtering
    Migration {
        description: "Add file_extension column and index to image_files",
        sql: "ALTER TABLE image_files ADD COLUMN file_extension TEXT;\
              CREATE INDEX IF NOT EXISTS idx_image_files_file_extension ON image_files(file_extension);\
              UPDATE image_files SET file_extension = \
                CASE \
                  WHEN LOWER(original_path) LIKE '%.png' THEN 'png' \
                  WHEN LOWER(original_path) LIKE '%.jpg' THEN 'jpg' \
                  WHEN LOWER(original_path) LIKE '%.jpeg' THEN 'jpeg' \
                  WHEN LOWER(original_path) LIKE '%.gif' THEN 'gif' \
                  WHEN LOWER(original_path) LIKE '%.webp' THEN 'webp' \
                  WHEN LOWER(original_path) LIKE '%.bmp' THEN 'bmp' \
                  WHEN LOWER(original_path) LIKE '%.tiff' THEN 'tiff' \
                  WHEN LOWER(original_path) LIKE '%.tif' THEN 'tif' \
                  WHEN LOWER(original_path) LIKE '%.webm' THEN 'webm' \
                  WHEN LOWER(original_path) LIKE '%.mp4' THEN 'mp4' \
                  WHEN LOWER(original_path) LIKE '%.mov' THEN 'mov' \
                  WHEN LOWER(original_path) LIKE '%.avi' THEN 'avi' \
                  WHEN LOWER(original_path) LIKE '%.mkv' THEN 'mkv' \
                END \
              WHERE file_extension IS NULL;",
    },
    // v3-v5: One statement per migration keeps fresh and existing schemas compatible.
    Migration {
        description: "Add curation original path to image_files",
        sql: "ALTER TABLE image_files ADD COLUMN curation_original_path TEXT;",
    },
    Migration {
        description: "Add curation discarded timestamp to image_files",
        sql: "ALTER TABLE image_files ADD COLUMN curation_discarded_at TEXT;",
    },
    Migration {
        description: "Index curation discard state",
        sql: "CREATE INDEX IF NOT EXISTS idx_image_files_curation_discarded_at ON image_files(curation_discarded_at);",
    },
];

/// Run all pending migrations on a per-directory database.
pub fn run_directory_migrations(conn: &Connection) -> Result<(), rusqlite::Error> {
    run_migrations(conn, DIRECTORY_MIGRATIONS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // AC: @identity-safe-image-adjustments ac-canonical-entry
    fn watch_history_migration_preserves_legacy_rows_and_allows_composite_identity() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             INSERT INTO schema_version (version) VALUES (9);
             CREATE TABLE watch_history (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 image_id INTEGER NOT NULL UNIQUE,
                 playback_position REAL NOT NULL DEFAULT 0.0,
                 duration REAL NOT NULL DEFAULT 0.0,
                 completed INTEGER NOT NULL DEFAULT 0,
                 last_watched TEXT NOT NULL DEFAULT (datetime('now')),
                 created_at TEXT NOT NULL DEFAULT (datetime('now')),
                 directory_id INTEGER
             );
             CREATE TABLE task_queue (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 task_type TEXT NOT NULL,
                 payload TEXT,
                 status TEXT NOT NULL DEFAULT 'pending',
                 priority INTEGER NOT NULL DEFAULT 0,
                 attempts INTEGER NOT NULL DEFAULT 0,
                 error_message TEXT,
                 created_at TEXT NOT NULL DEFAULT (datetime('now')),
                 started_at TEXT,
                 completed_at TEXT
             );
             INSERT INTO watch_history
                 (image_id, playback_position, duration, completed, directory_id)
             VALUES (12, 5.0, 10.0, 0, 1);",
        )
        .unwrap();

        crate::db::schema::init_main_db(&conn).unwrap();

        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(watch_history)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(columns.contains(&"library_id".to_string()));
        let legacy: (Option<String>, i64, i64, f64) = conn
            .query_row(
                "SELECT library_id, directory_id, image_id, playback_position FROM watch_history",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(legacy, (None, 1, 12, 5.0));

        conn.execute(
            "INSERT INTO watch_history (library_id, directory_id, image_id) VALUES ('library-a', 1, 12)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO watch_history (library_id, directory_id, image_id) VALUES ('library-b', 1, 12)",
            [],
        )
        .unwrap();
    }

    #[test]
    fn curation_migrations_work_for_existing_directory_schema() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE image_files (
                id INTEGER PRIMARY KEY,
                original_path TEXT NOT NULL,
                file_status TEXT NOT NULL
            );",
        )
        .unwrap();
        run_directory_migrations(&conn).unwrap();
        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(image_files)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(columns.contains(&"curation_original_path".to_string()));
        assert!(columns.contains(&"curation_discarded_at".to_string()));
    }

    #[test]
    fn curation_migrations_tolerate_current_fresh_schema() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE image_files (
                id INTEGER PRIMARY KEY,
                original_path TEXT NOT NULL,
                file_status TEXT NOT NULL,
                file_extension TEXT,
                curation_original_path TEXT,
                curation_discarded_at TEXT
            );",
        )
        .unwrap();
        run_directory_migrations(&conn).unwrap();
        let index_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_image_files_curation_discarded_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(index_count, 1);
    }

    #[test]
    // AC: @durable-task-retry-scheduling ac-restart-safe
    fn existing_queue_initializes_before_retry_migration_runs() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             INSERT INTO schema_version (version) VALUES (10);
             CREATE TABLE task_queue (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 task_type TEXT NOT NULL,
                 payload TEXT,
                 status TEXT NOT NULL DEFAULT 'pending',
                 priority INTEGER NOT NULL DEFAULT 0,
                 attempts INTEGER NOT NULL DEFAULT 0,
                 error_message TEXT,
                 created_at TEXT NOT NULL DEFAULT (datetime('now')),
                 started_at TEXT,
                 completed_at TEXT
             );",
        )
        .unwrap();

        crate::db::schema::init_main_db(&conn).unwrap();

        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(task_queue)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(columns.contains(&"next_attempt_at".to_string()));
        let index_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_task_queue_claim'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(index_count, 1);
    }

    #[test]
    // AC: @durable-task-retry-scheduling ac-restart-safe
    fn task_retry_migration_adds_persisted_eligibility_and_claim_index() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
             INSERT INTO schema_version (version) VALUES (10);
             CREATE TABLE task_queue (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 task_type TEXT NOT NULL,
                 payload TEXT,
                 status TEXT NOT NULL DEFAULT 'pending',
                 priority INTEGER NOT NULL DEFAULT 0,
                 attempts INTEGER NOT NULL DEFAULT 0,
                 error_message TEXT,
                 created_at TEXT NOT NULL DEFAULT (datetime('now')),
                 started_at TEXT,
                 completed_at TEXT
             );
             INSERT INTO task_queue (task_type, status) VALUES ('tag', 'pending');",
        )
        .unwrap();

        run_main_migrations(&conn).unwrap();

        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(task_queue)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(columns.contains(&"next_attempt_at".to_string()));
        let index_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_task_queue_claim'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(index_count, 1);
        let next_attempt: Option<String> = conn
            .query_row("SELECT next_attempt_at FROM task_queue", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(next_attempt.is_none());
    }
}
