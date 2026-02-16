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
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)",
    )?;
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
                if msg.contains("duplicate column name")
                    || msg.contains("already exists")
                {
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
                    log::error!(
                        "[Migration] v{} failed: {}",
                        version,
                        e
                    );
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
];

/// Run all pending migrations on a per-directory database.
pub fn run_directory_migrations(conn: &Connection) -> Result<(), rusqlite::Error> {
    run_migrations(conn, DIRECTORY_MIGRATIONS)
}
