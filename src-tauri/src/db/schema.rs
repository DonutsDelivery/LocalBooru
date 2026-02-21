use rusqlite::Connection;

use crate::db::migrations::{run_main_migrations, run_directory_migrations};

/// Initialize the main library database schema.
/// Creates all tables and runs migrations (idempotent).
pub fn init_main_db(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        -- Tags (global)
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL DEFAULT 'general'
                CHECK(category IN ('general','character','copyright','artist','meta')),
            post_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
        CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
        CREATE INDEX IF NOT EXISTS idx_tag_post_count ON tags(post_count DESC);
        CREATE INDEX IF NOT EXISTS idx_tag_name_post_count ON tags(name, post_count DESC);

        -- Tag aliases
        CREATE TABLE IF NOT EXISTS tag_aliases (
            alias TEXT PRIMARY KEY,
            target_tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE
        );

        -- Watch directories
        CREATE TABLE IF NOT EXISTS watch_directories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            name TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            recursive INTEGER NOT NULL DEFAULT 1,
            auto_tag INTEGER NOT NULL DEFAULT 1,
            auto_age_detect INTEGER NOT NULL DEFAULT 0,
            last_scanned_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            comfyui_prompt_node_ids TEXT,
            comfyui_negative_node_ids TEXT,
            metadata_format TEXT NOT NULL DEFAULT 'auto',
            parent_path TEXT,
            public_access INTEGER NOT NULL DEFAULT 0,
            show_images INTEGER NOT NULL DEFAULT 1,
            show_videos INTEGER NOT NULL DEFAULT 1,
            family_safe INTEGER NOT NULL DEFAULT 1,
            lan_visible INTEGER NOT NULL DEFAULT 1
        );

        -- Images (main/legacy — per-directory DBs are primary now)
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT,
            file_hash TEXT NOT NULL UNIQUE,
            perceptual_hash TEXT,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            duration REAL,
            rating TEXT NOT NULL DEFAULT 'pg'
                CHECK(rating IN ('pg','pg13','r','x','xxx')),
            prompt TEXT,
            negative_prompt TEXT,
            model_name TEXT,
            sampler TEXT,
            seed TEXT,
            steps INTEGER,
            cfg_scale REAL,
            source_url TEXT,
            num_faces INTEGER,
            min_detected_age INTEGER,
            max_detected_age INTEGER,
            detected_ages TEXT,
            age_detection_data TEXT,
            is_favorite INTEGER NOT NULL DEFAULT 0,
            import_source TEXT,
            view_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT,
            file_created_at TEXT,
            file_modified_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);
        CREATE INDEX IF NOT EXISTS idx_images_perceptual_hash ON images(perceptual_hash);
        CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating);
        CREATE INDEX IF NOT EXISTS idx_images_is_favorite ON images(is_favorite);
        CREATE INDEX IF NOT EXISTS idx_images_min_detected_age ON images(min_detected_age);
        CREATE INDEX IF NOT EXISTS idx_images_max_detected_age ON images(max_detected_age);
        CREATE INDEX IF NOT EXISTS idx_images_file_created_at ON images(file_created_at);
        CREATE INDEX IF NOT EXISTS idx_images_file_modified_at ON images(file_modified_at);

        -- Image files (track original file locations)
        CREATE TABLE IF NOT EXISTS image_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            original_path TEXT NOT NULL UNIQUE,
            file_exists INTEGER NOT NULL DEFAULT 1,
            file_status TEXT NOT NULL DEFAULT 'available'
                CHECK(file_status IN ('available','missing','drive_offline','unknown')),
            last_verified_at TEXT,
            watch_directory_id INTEGER REFERENCES watch_directories(id) ON DELETE SET NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_image_files_image_id ON image_files(image_id);
        CREATE INDEX IF NOT EXISTS idx_image_files_original_path ON image_files(original_path);
        CREATE INDEX IF NOT EXISTS idx_image_files_file_status ON image_files(file_status);

        -- Image-tag many-to-many
        CREATE TABLE IF NOT EXISTS image_tags (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            confidence REAL,
            is_manual INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (image_id, tag_id)
        );

        CREATE INDEX IF NOT EXISTS idx_image_tags_tag_id ON image_tags(tag_id);

        -- Task queue
        CREATE TABLE IF NOT EXISTS task_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL
                CHECK(task_type IN ('tag','scan_directory','verify_files','upload','age_detect','extract_metadata')),
            payload TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','processing','completed','failed','cancelled')),
            priority INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            started_at TEXT,
            completed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
        CREATE INDEX IF NOT EXISTS idx_task_queue_task_type ON task_queue(task_type);

        -- Collections
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            cover_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
            item_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT
        );

        -- Collection items
        CREATE TABLE IF NOT EXISTS collection_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            sort_order INTEGER NOT NULL DEFAULT 0,
            added_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(collection_id, image_id)
        );

        CREATE INDEX IF NOT EXISTS idx_collection_items_collection_id ON collection_items(collection_id);
        CREATE INDEX IF NOT EXISTS idx_collection_items_image_id ON collection_items(image_id);

        -- Watch history
        -- Note: image_id has no FK constraint because images live in per-directory
        -- SQLite databases, not in the main DB. Cross-database FKs aren't possible.
        CREATE TABLE IF NOT EXISTS watch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL UNIQUE,
            playback_position REAL NOT NULL DEFAULT 0.0,
            duration REAL NOT NULL DEFAULT 0.0,
            completed INTEGER NOT NULL DEFAULT 0,
            last_watched TEXT NOT NULL DEFAULT (datetime('now')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            directory_id INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_watch_history_image_id ON watch_history(image_id);
        CREATE INDEX IF NOT EXISTS idx_watch_history_completed ON watch_history(completed);

        -- Users (for network access auth)
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            access_level TEXT NOT NULL DEFAULT 'local_network'
                CHECK(access_level IN ('localhost','local_network','public')),
            can_write INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_login TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

        -- Mounted libraries (multi-library support)
        CREATE TABLE IF NOT EXISTS mounted_libraries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            auto_mount INTEGER NOT NULL DEFAULT 1,
            mount_order INTEGER NOT NULL DEFAULT 0,
            last_mounted_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        ",
    )?;

    // Run incremental migrations for existing databases
    run_main_migrations(conn)?;

    Ok(())
}

/// Initialize a per-directory database schema.
/// Same image/image_files/image_tags structure but isolated per directory.
pub fn init_directory_db(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "
        -- Images in this directory
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT,
            file_hash TEXT NOT NULL UNIQUE,
            perceptual_hash TEXT,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            duration REAL,
            rating TEXT NOT NULL DEFAULT 'pg'
                CHECK(rating IN ('pg','pg13','r','x','xxx')),
            prompt TEXT,
            negative_prompt TEXT,
            model_name TEXT,
            sampler TEXT,
            seed TEXT,
            steps INTEGER,
            cfg_scale REAL,
            source_url TEXT,
            num_faces INTEGER,
            min_detected_age INTEGER,
            max_detected_age INTEGER,
            detected_ages TEXT,
            age_detection_data TEXT,
            is_favorite INTEGER NOT NULL DEFAULT 0,
            import_source TEXT,
            view_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT,
            file_created_at TEXT,
            file_modified_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_images_file_hash ON images(file_hash);
        CREATE INDEX IF NOT EXISTS idx_images_perceptual_hash ON images(perceptual_hash);
        CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating);
        CREATE INDEX IF NOT EXISTS idx_images_is_favorite ON images(is_favorite);
        CREATE INDEX IF NOT EXISTS idx_images_min_detected_age ON images(min_detected_age);
        CREATE INDEX IF NOT EXISTS idx_images_max_detected_age ON images(max_detected_age);
        CREATE INDEX IF NOT EXISTS idx_images_file_created_at ON images(file_created_at);
        CREATE INDEX IF NOT EXISTS idx_images_file_modified_at ON images(file_modified_at);

        -- Image files (track original file locations)
        CREATE TABLE IF NOT EXISTS image_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            original_path TEXT NOT NULL UNIQUE,
            file_exists INTEGER NOT NULL DEFAULT 1,
            file_status TEXT NOT NULL DEFAULT 'available'
                CHECK(file_status IN ('available','missing','drive_offline','unknown')),
            last_verified_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_image_files_image_id ON image_files(image_id);
        CREATE INDEX IF NOT EXISTS idx_image_files_original_path ON image_files(original_path);

        -- Image-tag associations (tag_id references global tags table in main DB)
        CREATE TABLE IF NOT EXISTS image_tags (
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            tag_id INTEGER NOT NULL,
            confidence REAL,
            is_manual INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (image_id, tag_id)
        );

        CREATE INDEX IF NOT EXISTS idx_image_tags_tag_id ON image_tags(tag_id);
        ",
    )?;

    // Run incremental migrations for existing databases
    run_directory_migrations(conn)?;

    Ok(())
}
