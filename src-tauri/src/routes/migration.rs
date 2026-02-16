use std::convert::Infallible;
use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use chrono::Utc;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::events::migration_event_type;

// ---- Migration state types --------------------------------------------------

/// Migration direction: per-directory DBs -> main DB, or main DB -> per-directory DBs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MigrationDirection {
    /// Consolidate per-directory databases into the main database.
    ToMainDb,
    /// Distribute main database records into per-directory databases.
    ToPerDirectory,
}

/// Migration mode: migrate all directories or only selected ones.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MigrationMode {
    Full,
    Selective,
}

/// Overall status of the migration subsystem.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MigrationStatus {
    Idle,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Progress details for a running migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationProgress {
    pub total: i64,
    pub completed: i64,
    pub current_directory: Option<String>,
    pub current_directory_id: Option<i64>,
    pub directories_total: i64,
    pub directories_completed: i64,
}

/// Shared migration state, accessible from route handlers and the background task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationState {
    pub status: MigrationStatus,
    pub direction: MigrationDirection,
    pub mode: Option<MigrationMode>,
    pub directory_ids: Option<Vec<i64>>,
    pub progress: Option<MigrationProgress>,
    pub errors: Vec<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

impl Default for MigrationState {
    fn default() -> Self {
        Self {
            status: MigrationStatus::Idle,
            direction: MigrationDirection::ToMainDb,
            mode: None,
            directory_ids: None,
            progress: None,
            errors: Vec::new(),
            started_at: None,
            completed_at: None,
        }
    }
}

/// Thread-safe shared migration state.
pub type SharedMigrationState = Arc<RwLock<MigrationState>>;

/// Create a new default shared migration state.
pub fn create_migration_state() -> SharedMigrationState {
    Arc::new(RwLock::new(MigrationState::default()))
}

// ---- Request/response models ------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ValidateRequest {
    pub mode: MigrationMode,
    #[serde(default)]
    pub directory_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct StartRequest {
    pub mode: MigrationMode,
    #[serde(default)]
    pub directory_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteSourceRequest {
    #[allow(dead_code)]
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct ImportValidateRequest {
    #[allow(dead_code)]
    pub mode: String,
    #[serde(default)]
    pub directory_ids: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct ImportStartRequest {
    #[allow(dead_code)]
    pub mode: String,
    #[serde(default)]
    pub directory_ids: Vec<i64>,
}

// ---- Router -----------------------------------------------------------------

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(get_migration_status))
        .route("/directories", get(get_migration_directories))
        .route("/validate", post(validate_migration))
        .route("/start", post(start_migration))
        .route("/status", get(get_migration_progress))
        .route("/stop", post(stop_migration))
        .route("/cleanup", post(cleanup_migration))
        .route("/verify", post(verify_migration))
        .route("/delete-source", post(delete_source))
        .route("/import/validate", post(validate_import))
        .route("/import/start", post(start_import))
        .route("/events", get(migration_events))
}

// ---- Route handlers ---------------------------------------------------------

/// GET /settings/migration - Return current migration status.
///
/// Returns the current storage mode, migration status, and progress (if running).
async fn get_migration_status(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let migration = state.migration_state().read().await;

    // Determine current storage mode by checking if per-directory DBs exist
    let current_mode = {
        let dir_ids = state.directory_db().get_all_directory_ids();
        if dir_ids.is_empty() {
            "main_db"
        } else {
            "per_directory"
        }
    };

    let progress = migration.progress.as_ref().map(|p| {
        json!({
            "total": p.total,
            "completed": p.completed,
            "current_directory": p.current_directory,
            "current_directory_id": p.current_directory_id,
            "directories_total": p.directories_total,
            "directories_completed": p.directories_completed,
        })
    });

    Ok(Json(json!({
        "current_mode": current_mode,
        "status": migration.status,
        "direction": migration.direction,
        "progress": progress,
        "errors": migration.errors,
        "started_at": migration.started_at,
        "completed_at": migration.completed_at,
    })))
}

/// GET /settings/migration/directories - List watch directories with image counts and sizes.
///
/// Returns per-directory info useful for selective migration planning.
async fn get_migration_directories(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        let mut stmt = main_conn.prepare(
            "SELECT id, path, name FROM watch_directories ORDER BY id",
        )?;

        let directories: Vec<Value> = stmt
            .query_map([], |row| {
                let dir_id: i64 = row.get(0)?;
                let path: String = row.get(1)?;
                let name: Option<String> = row.get(2)?;

                let display_name = name.unwrap_or_else(|| {
                    Path::new(&path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&path)
                        .to_string()
                });

                // Query the per-directory database for image count and total file size
                let (image_count, total_size) =
                    if let Ok(dir_pool) = state_clone.directory_db().get_pool(dir_id) {
                        if let Ok(dir_conn) = dir_pool.get() {
                            let count: i64 = dir_conn
                                .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                                .unwrap_or(0);
                            let size: i64 = dir_conn
                                .query_row(
                                    "SELECT COALESCE(SUM(file_size), 0) FROM images",
                                    [],
                                    |r| r.get(0),
                                )
                                .unwrap_or(0);
                            (count, size)
                        } else {
                            (0, 0)
                        }
                    } else {
                        (0, 0)
                    };

                let path_exists = Path::new(&path).exists();
                let db_exists = state_clone.directory_db().db_exists(dir_id);

                Ok(json!({
                    "id": dir_id,
                    "path": path,
                    "name": display_name,
                    "image_count": image_count,
                    "total_file_size": total_size,
                    "path_exists": path_exists,
                    "db_exists": db_exists,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok::<_, AppError>(json!({ "directories": directories }))
    })
    .await??;

    Ok(Json(result))
}

/// POST /settings/migration/validate - Dry-run validation before migration.
///
/// Checks that the requested directories exist, their DBs are accessible,
/// files are reachable, and estimates the total data size.
async fn validate_migration(
    State(state): State<AppState>,
    Json(body): Json<ValidateRequest>,
) -> Result<Json<Value>, AppError> {
    // Check that migration is not already running
    {
        let migration = state.migration_state().read().await;
        if migration.status == MigrationStatus::Running {
            return Err(AppError::BadRequest(
                "A migration is already in progress".into(),
            ));
        }
    }

    let state_clone = state.clone();
    let mode = body.mode.clone();
    let requested_ids = body.directory_ids.clone();

    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;
        let mut errors: Vec<String> = Vec::new();
        let mut estimated_size: i64 = 0;
        let mut image_count: i64 = 0;

        // Determine which directories to validate
        let dir_ids: Vec<i64> = if mode == MigrationMode::Selective {
            if requested_ids.is_empty() {
                errors.push("Selective mode requires at least one directory_id".into());
                return Ok::<_, AppError>(json!({
                    "valid": false,
                    "errors": errors,
                    "estimated_size": 0,
                    "image_count": 0,
                }));
            }
            requested_ids
        } else {
            // Full mode: get all directory IDs
            let mut stmt = main_conn.prepare("SELECT id FROM watch_directories")?;
            let ids: Vec<i64> = stmt.query_map([], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            ids
        };

        if dir_ids.is_empty() {
            errors.push("No directories found to migrate".into());
            return Ok::<_, AppError>(json!({
                "valid": false,
                "errors": errors,
                "estimated_size": 0,
                "image_count": 0,
            }));
        }

        for dir_id in &dir_ids {
            // Check directory exists in main DB
            let dir_info: Result<(String, Option<String>), _> = main_conn.query_row(
                "SELECT path, name FROM watch_directories WHERE id = ?1",
                params![dir_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            );

            let (dir_path, dir_name) = match dir_info {
                Ok(info) => info,
                Err(_) => {
                    errors.push(format!("Directory ID {} not found in database", dir_id));
                    continue;
                }
            };

            let label = dir_name.unwrap_or_else(|| dir_path.clone());

            // Check filesystem path exists
            if !Path::new(&dir_path).exists() {
                errors.push(format!(
                    "Directory '{}' path does not exist: {}",
                    label, dir_path
                ));
            }

            // Check per-directory DB is accessible
            if !state_clone.directory_db().db_exists(*dir_id) {
                errors.push(format!(
                    "Directory '{}' (id={}) has no per-directory database",
                    label, dir_id
                ));
                continue;
            }

            match state_clone.directory_db().get_pool(*dir_id) {
                Ok(pool) => match pool.get() {
                    Ok(conn) => {
                        let count: i64 = conn
                            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                            .unwrap_or(0);
                        let size: i64 = conn
                            .query_row(
                                "SELECT COALESCE(SUM(file_size), 0) FROM images",
                                [],
                                |r| r.get(0),
                            )
                            .unwrap_or(0);

                        image_count += count;
                        estimated_size += size;

                        // Verify sample files are accessible
                        let mut file_stmt = conn
                            .prepare(
                                "SELECT original_path FROM image_files WHERE file_exists = 1 LIMIT 5",
                            )?;
                        let sample_paths: Vec<String> = file_stmt
                            .query_map([], |row| row.get(0))?
                            .filter_map(|r| r.ok())
                            .collect();

                        let mut inaccessible = 0;
                        for path in &sample_paths {
                            if !Path::new(path).exists() {
                                inaccessible += 1;
                            }
                        }
                        if inaccessible > 0 {
                            errors.push(format!(
                                "Directory '{}': {} of {} sampled files are inaccessible",
                                label,
                                inaccessible,
                                sample_paths.len()
                            ));
                        }
                    }
                    Err(e) => {
                        errors.push(format!(
                            "Directory '{}' DB connection failed: {}",
                            label, e
                        ));
                    }
                },
                Err(e) => {
                    errors.push(format!(
                        "Directory '{}' DB pool creation failed: {}",
                        label, e
                    ));
                }
            }
        }

        let valid = errors.is_empty();
        Ok::<_, AppError>(json!({
            "valid": valid,
            "errors": errors,
            "estimated_size": estimated_size,
            "image_count": image_count,
            "directory_count": dir_ids.len(),
        }))
    })
    .await??;

    Ok(Json(result))
}

/// POST /settings/migration/start - Start migration in a background tokio task.
///
/// Copies image records from per-directory databases into the main database.
/// Emits progress events via the EventBroadcaster and updates MigrationState.
async fn start_migration(
    State(state): State<AppState>,
    Json(body): Json<StartRequest>,
) -> Result<Json<Value>, AppError> {
    // Check that no migration is already running
    {
        let migration = state.migration_state().read().await;
        if migration.status == MigrationStatus::Running {
            return Err(AppError::BadRequest(
                "A migration is already in progress".into(),
            ));
        }
    }

    let mode = body.mode.clone();
    let requested_ids = body.directory_ids.clone();

    // Determine directory IDs to migrate
    let dir_ids: Vec<i64> = if mode == MigrationMode::Selective {
        if requested_ids.is_empty() {
            return Err(AppError::BadRequest(
                "Selective mode requires at least one directory_id".into(),
            ));
        }
        requested_ids
    } else {
        let main_conn = state.main_db().get()?;
        let mut stmt = main_conn.prepare("SELECT id FROM watch_directories")?;
        let ids: Vec<i64> = stmt.query_map([], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .collect();
        ids
    };

    if dir_ids.is_empty() {
        return Err(AppError::BadRequest("No directories to migrate".into()));
    }

    // Initialize migration state
    {
        let mut migration = state.migration_state().write().await;
        migration.status = MigrationStatus::Running;
        migration.direction = MigrationDirection::ToMainDb;
        migration.mode = Some(mode.clone());
        migration.directory_ids = Some(dir_ids.clone());
        migration.progress = Some(MigrationProgress {
            total: 0,
            completed: 0,
            current_directory: None,
            current_directory_id: None,
            directories_total: dir_ids.len() as i64,
            directories_completed: 0,
        });
        migration.errors.clear();
        migration.started_at = Some(Utc::now().to_rfc3339());
        migration.completed_at = None;
    }

    // Broadcast start event
    if let Some(events) = state.events() {
        events.migration.broadcast(
            migration_event_type::STARTED,
            json!({
                "mode": mode,
                "directory_count": dir_ids.len(),
            }),
        );
    }

    // Spawn the background migration task
    let state_bg = state.clone();
    tokio::spawn(async move {
        run_migration(state_bg, dir_ids).await;
    });

    Ok(Json(json!({
        "success": true,
        "message": "Migration started",
    })))
}

/// Result from migrating a single directory (returned by spawn_blocking).
struct DirectoryMigrationResult {
    migrated: i64,
    skipped: i64,
    errors: Vec<String>,
}

/// Background migration task.
///
/// For each selected directory, reads all records from the per-directory DB
/// and inserts them into the main DB, using file_hash to avoid duplicates.
/// All DB work is done inside spawn_blocking to keep the rusqlite Connection
/// off the async runtime (Connection is not Send).
async fn run_migration(state: AppState, dir_ids: Vec<i64>) {
    let mut total_migrated: i64 = 0;
    let mut total_skipped: i64 = 0;
    let mut dirs_completed: i64 = 0;

    // First pass: count total images across all directories for progress
    let state_count = state.clone();
    let dir_ids_count = dir_ids.clone();
    let total_images: i64 = tokio::task::spawn_blocking(move || {
        let mut total = 0i64;
        for &dir_id in &dir_ids_count {
            if let Ok(pool) = state_count.directory_db().get_pool(dir_id) {
                if let Ok(conn) = pool.get() {
                    let count: i64 = conn
                        .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                        .unwrap_or(0);
                    total += count;
                }
            }
        }
        total
    })
    .await
    .unwrap_or(0);

    // Update total in progress
    {
        let mut migration = state.migration_state().write().await;
        if let Some(ref mut progress) = migration.progress {
            progress.total = total_images;
        }
    }

    for &dir_id in &dir_ids {
        // Check for cancellation
        {
            let migration = state.migration_state().read().await;
            if migration.status == MigrationStatus::Cancelled {
                log::info!("[Migration] Cancelled by user");
                return;
            }
        }

        // Get directory label (name or path) from main DB
        let state_info = state.clone();
        let dir_info = tokio::task::spawn_blocking(move || {
            let main_conn = state_info.main_db().get().ok()?;
            main_conn
                .query_row(
                    "SELECT path, name FROM watch_directories WHERE id = ?1",
                    params![dir_id],
                    |row| {
                        Ok((
                            row.get::<_, String>(0).ok(),
                            row.get::<_, Option<String>>(1).ok(),
                        ))
                    },
                )
                .ok()
        })
        .await
        .ok()
        .flatten();

        let label = match dir_info {
            Some((Some(path), name)) => name.flatten().unwrap_or_else(|| {
                Path::new(&path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&path)
                    .to_string()
            }),
            _ => {
                let msg = format!("Directory {} not found, skipping", dir_id);
                log::warn!("[Migration] {}", msg);
                let mut migration = state.migration_state().write().await;
                migration.errors.push(msg);
                dirs_completed += 1;
                continue;
            }
        };

        // Update progress with current directory
        {
            let mut migration = state.migration_state().write().await;
            if let Some(ref mut progress) = migration.progress {
                progress.current_directory = Some(label.clone());
                progress.current_directory_id = Some(dir_id);
            }
        }

        // Broadcast progress
        if let Some(events) = state.events() {
            events.migration.broadcast(
                migration_event_type::PROGRESS,
                json!({
                    "directory_id": dir_id,
                    "directory_name": &label,
                    "total": total_images,
                    "completed": total_migrated,
                    "directories_completed": dirs_completed,
                }),
            );
        }

        // Run the actual DB migration for this directory inside spawn_blocking
        let state_migrate = state.clone();
        let label_clone = label.clone();
        let result = tokio::task::spawn_blocking(move || {
            migrate_single_directory(&state_migrate, dir_id, &label_clone)
        })
        .await;

        match result {
            Ok(dir_result) => {
                total_migrated += dir_result.migrated;
                total_skipped += dir_result.skipped;

                if !dir_result.errors.is_empty() {
                    let mut migration = state.migration_state().write().await;
                    migration.errors.extend(dir_result.errors);
                }
            }
            Err(e) => {
                let msg = format!("Migration task for directory '{}' panicked: {}", label, e);
                log::error!("[Migration] {}", msg);
                let mut migration = state.migration_state().write().await;
                migration.errors.push(msg);
            }
        }

        dirs_completed += 1;

        // Update progress
        {
            let mut migration = state.migration_state().write().await;
            if let Some(ref mut progress) = migration.progress {
                progress.completed = total_migrated;
                progress.directories_completed = dirs_completed;
            }
        }

        log::info!(
            "[Migration] Completed directory '{}' ({}/{})",
            label,
            dirs_completed,
            dir_ids.len()
        );
    }

    // Mark migration as completed
    {
        let mut migration = state.migration_state().write().await;
        migration.status = MigrationStatus::Completed;
        migration.completed_at = Some(Utc::now().to_rfc3339());
        if let Some(ref mut progress) = migration.progress {
            progress.current_directory = None;
            progress.current_directory_id = None;
        }
    }

    // Broadcast completion
    if let Some(events) = state.events() {
        events.migration.broadcast(
            migration_event_type::COMPLETED,
            json!({
                "total_migrated": total_migrated,
                "total_skipped": total_skipped,
                "directories_completed": dirs_completed,
            }),
        );
    }

    log::info!(
        "[Migration] Complete: {} migrated, {} skipped, {} directories",
        total_migrated,
        total_skipped,
        dirs_completed
    );
}

/// Synchronous migration of a single directory's data into the main DB.
/// Runs inside spawn_blocking so rusqlite connections stay on the blocking pool.
fn migrate_single_directory(
    state: &AppState,
    dir_id: i64,
    label: &str,
) -> DirectoryMigrationResult {
    let mut migrated: i64 = 0;
    let mut skipped: i64 = 0;
    let mut errors: Vec<String> = Vec::new();

    // Get per-directory DB connection
    let dir_pool = match state.directory_db().get_pool(dir_id) {
        Ok(p) => p,
        Err(e) => {
            errors.push(format!("Failed to open DB for directory '{}': {}", label, e));
            return DirectoryMigrationResult { migrated, skipped, errors };
        }
    };
    let dir_conn = match dir_pool.get() {
        Ok(c) => c,
        Err(e) => {
            errors.push(format!("Failed to get connection for directory '{}': {}", label, e));
            return DirectoryMigrationResult { migrated, skipped, errors };
        }
    };

    // Get main DB connection
    let main_conn = match state.main_db().get() {
        Ok(c) => c,
        Err(e) => {
            errors.push(format!("Failed to get main DB connection: {}", e));
            return DirectoryMigrationResult { migrated, skipped, errors };
        }
    };

    // Read all images from this directory DB
    let mut stmt = match dir_conn.prepare(
        "SELECT id, filename, original_filename, file_hash, perceptual_hash,
                width, height, file_size, duration, rating, prompt, negative_prompt,
                model_name, sampler, seed, steps, cfg_scale, source_url,
                num_faces, min_detected_age, max_detected_age, detected_ages,
                age_detection_data, is_favorite, import_source, view_count,
                created_at, updated_at, file_created_at, file_modified_at
         FROM images",
    ) {
        Ok(s) => s,
        Err(e) => {
            errors.push(format!("Failed to prepare query for '{}': {}", label, e));
            return DirectoryMigrationResult { migrated, skipped, errors };
        }
    };

    let images: Vec<ImageRow> = stmt
        .query_map([], |row| {
            Ok(ImageRow {
                id: row.get(0)?,
                filename: row.get(1)?,
                original_filename: row.get(2)?,
                file_hash: row.get(3)?,
                perceptual_hash: row.get(4)?,
                width: row.get(5)?,
                height: row.get(6)?,
                file_size: row.get(7)?,
                duration: row.get(8)?,
                rating: row.get(9)?,
                prompt: row.get(10)?,
                negative_prompt: row.get(11)?,
                model_name: row.get(12)?,
                sampler: row.get(13)?,
                seed: row.get(14)?,
                steps: row.get(15)?,
                cfg_scale: row.get(16)?,
                source_url: row.get(17)?,
                num_faces: row.get(18)?,
                min_detected_age: row.get(19)?,
                max_detected_age: row.get(20)?,
                detected_ages: row.get(21)?,
                age_detection_data: row.get(22)?,
                is_favorite: row.get(23)?,
                import_source: row.get(24)?,
                view_count: row.get(25)?,
                created_at: row.get(26)?,
                updated_at: row.get(27)?,
                file_created_at: row.get(28)?,
                file_modified_at: row.get(29)?,
            })
        })
        .ok()
        .map(|r| r.filter_map(|r| r.ok()).collect())
        .unwrap_or_default();

    // Process each image
    for image in &images {
        // Check if image already exists in main DB by file_hash
        let existing: Option<i64> = main_conn
            .query_row(
                "SELECT id FROM images WHERE file_hash = ?1",
                params![&image.file_hash],
                |row| row.get(0),
            )
            .ok();

        let main_image_id = if let Some(existing_id) = existing {
            skipped += 1;
            existing_id
        } else {
            // Insert image into main DB
            match main_conn.execute(
                "INSERT INTO images (
                    filename, original_filename, file_hash, perceptual_hash,
                    width, height, file_size, duration, rating, prompt, negative_prompt,
                    model_name, sampler, seed, steps, cfg_scale, source_url,
                    num_faces, min_detected_age, max_detected_age, detected_ages,
                    age_detection_data, is_favorite, import_source, view_count,
                    created_at, updated_at, file_created_at, file_modified_at
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                    ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
                    ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29
                )",
                params![
                    &image.filename,
                    &image.original_filename,
                    &image.file_hash,
                    &image.perceptual_hash,
                    &image.width,
                    &image.height,
                    &image.file_size,
                    &image.duration,
                    &image.rating,
                    &image.prompt,
                    &image.negative_prompt,
                    &image.model_name,
                    &image.sampler,
                    &image.seed,
                    &image.steps,
                    &image.cfg_scale,
                    &image.source_url,
                    &image.num_faces,
                    &image.min_detected_age,
                    &image.max_detected_age,
                    &image.detected_ages,
                    &image.age_detection_data,
                    &image.is_favorite,
                    &image.import_source,
                    &image.view_count,
                    &image.created_at,
                    &image.updated_at,
                    &image.file_created_at,
                    &image.file_modified_at,
                ],
            ) {
                Ok(_) => main_conn.last_insert_rowid(),
                Err(e) => {
                    errors.push(format!(
                        "Failed to insert image '{}' (hash={}): {}",
                        image.filename, image.file_hash, e
                    ));
                    migrated += 1;
                    continue;
                }
            }
        };

        // Migrate image_files for this image
        let dir_image_id = image.id;
        if let Ok(mut file_stmt) = dir_conn.prepare(
            "SELECT original_path, file_exists, file_status, last_verified_at, created_at
             FROM image_files WHERE image_id = ?1",
        ) {
            let files: Vec<(String, bool, String, Option<String>, String)> = file_stmt
                .query_map(params![dir_image_id], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })
                .ok()
                .map(|r| r.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();

            for (original_path, file_exists, file_status, last_verified_at, created_at) in &files {
                let file_exists_in_main: bool = main_conn
                    .query_row(
                        "SELECT COUNT(*) FROM image_files WHERE original_path = ?1",
                        params![original_path],
                        |row| row.get::<_, i64>(0).map(|c| c > 0),
                    )
                    .unwrap_or(false);

                if !file_exists_in_main {
                    let _ = main_conn.execute(
                        "INSERT INTO image_files (
                            image_id, original_path, file_exists, file_status,
                            last_verified_at, watch_directory_id, created_at
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![
                            main_image_id,
                            original_path,
                            file_exists,
                            file_status,
                            last_verified_at,
                            dir_id,
                            created_at,
                        ],
                    );
                }
            }
        }

        // Migrate image_tags for this image
        if let Ok(mut tag_stmt) = dir_conn.prepare(
            "SELECT tag_id, confidence, is_manual FROM image_tags WHERE image_id = ?1",
        ) {
            let tags: Vec<(i64, Option<f64>, bool)> = tag_stmt
                .query_map(params![dir_image_id], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })
                .ok()
                .map(|r| r.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();

            for (tag_id, confidence, is_manual) in &tags {
                let _ = main_conn.execute(
                    "INSERT OR IGNORE INTO image_tags (image_id, tag_id, confidence, is_manual)
                     VALUES (?1, ?2, ?3, ?4)",
                    params![main_image_id, tag_id, confidence, is_manual],
                );
            }
        }

        migrated += 1;
    }

    DirectoryMigrationResult { migrated, skipped, errors }
}

/// GET /settings/migration/status - Return current migration progress details.
async fn get_migration_progress(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let migration = state.migration_state().read().await;

    let result = match &migration.progress {
        Some(progress) => json!({
            "status": migration.status,
            "total": progress.total,
            "completed": progress.completed,
            "current_directory": progress.current_directory,
            "current_directory_id": progress.current_directory_id,
            "directories_total": progress.directories_total,
            "directories_completed": progress.directories_completed,
            "errors": migration.errors,
            "started_at": migration.started_at,
            "completed_at": migration.completed_at,
        }),
        None => json!({
            "status": migration.status,
            "total": 0,
            "completed": 0,
            "current_directory": null,
            "current_directory_id": null,
            "directories_total": 0,
            "directories_completed": 0,
            "errors": migration.errors,
            "started_at": migration.started_at,
            "completed_at": migration.completed_at,
        }),
    };

    Ok(Json(result))
}

/// POST /settings/migration/stop - Cancel a running migration.
///
/// Sets the migration status to Cancelled, which the background task checks
/// periodically and will stop processing further records.
async fn stop_migration(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let mut migration = state.migration_state().write().await;

    if migration.status != MigrationStatus::Running {
        return Err(AppError::BadRequest(
            "No migration is currently running".into(),
        ));
    }

    migration.status = MigrationStatus::Cancelled;
    migration.completed_at = Some(Utc::now().to_rfc3339());

    // Broadcast cancellation
    if let Some(events) = state.events() {
        events.migration.broadcast(
            migration_event_type::ERROR,
            json!({ "error": "Migration cancelled by user" }),
        );
    }

    Ok(Json(json!({
        "success": true,
        "message": "Migration cancellation requested",
    })))
}

/// POST /settings/migration/cleanup - Remove partial data from a failed/cancelled migration.
///
/// Removes images from the main DB that were inserted during the migration
/// but whose source directory DB records still exist (i.e., the per-directory
/// DB is still the source of truth). Only runs when status is failed or cancelled.
async fn cleanup_migration(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    {
        let migration = state.migration_state().read().await;
        if migration.status == MigrationStatus::Running {
            return Err(AppError::BadRequest(
                "Cannot cleanup while migration is running".into(),
            ));
        }
    }

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Find images in main DB that have a watch_directory_id in image_files
        // (these were migrated from per-directory DBs). Remove them if the
        // source directory DB still has the record.
        let mut removed: i64 = 0;

        let dir_ids = state_clone.directory_db().get_all_directory_ids();

        for dir_id in &dir_ids {
            if let Ok(dir_pool) = state_clone.directory_db().get_pool(*dir_id) {
                if let Ok(dir_conn) = dir_pool.get() {
                    // Get all file_hashes from the directory DB
                    let mut stmt = dir_conn.prepare(
                        "SELECT file_hash FROM images WHERE file_hash IS NOT NULL",
                    )?;
                    let hashes: Vec<String> = stmt
                        .query_map([], |row| row.get(0))?
                        .filter_map(|r| r.ok())
                        .collect();

                    for hash in &hashes {
                        // Check if this hash exists in main DB
                        let main_image_id: Option<i64> = main_conn
                            .query_row(
                                "SELECT id FROM images WHERE file_hash = ?1",
                                params![hash],
                                |row| row.get(0),
                            )
                            .ok();

                        if let Some(image_id) = main_image_id {
                            // Delete image_tags, image_files, then image from main DB
                            main_conn.execute(
                                "DELETE FROM image_tags WHERE image_id = ?1",
                                params![image_id],
                            )?;
                            main_conn.execute(
                                "DELETE FROM image_files WHERE image_id = ?1",
                                params![image_id],
                            )?;
                            main_conn.execute(
                                "DELETE FROM images WHERE id = ?1",
                                params![image_id],
                            )?;
                            removed += 1;
                        }
                    }
                }
            }
        }

        Ok::<_, AppError>(json!({
            "success": true,
            "removed": removed,
            "message": format!("Cleaned up {} migrated records from main database", removed),
        }))
    })
    .await??;

    // Reset migration state to idle
    {
        let mut migration = state.migration_state().write().await;
        *migration = MigrationState::default();
    }

    Ok(Json(result))
}

/// POST /settings/migration/verify - Verify that migration completed correctly.
///
/// Compares record counts between source (per-directory DBs) and target (main DB)
/// to confirm all records were migrated successfully.
async fn verify_migration(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;

        // Count images in main DB
        let main_image_count: i64 = main_conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .unwrap_or(0);

        let main_file_count: i64 = main_conn
            .query_row("SELECT COUNT(*) FROM image_files", [], |r| r.get(0))
            .unwrap_or(0);

        let main_tag_count: i64 = main_conn
            .query_row("SELECT COUNT(*) FROM image_tags", [], |r| r.get(0))
            .unwrap_or(0);

        // Count images across all per-directory DBs
        let mut dir_image_count: i64 = 0;
        let mut dir_file_count: i64 = 0;
        let mut dir_tag_count: i64 = 0;
        let mut dir_details: Vec<Value> = Vec::new();

        let dir_ids = state_clone.directory_db().get_all_directory_ids();

        for dir_id in &dir_ids {
            if let Ok(pool) = state_clone.directory_db().get_pool(*dir_id) {
                if let Ok(conn) = pool.get() {
                    let images: i64 = conn
                        .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                        .unwrap_or(0);
                    let files: i64 = conn
                        .query_row("SELECT COUNT(*) FROM image_files", [], |r| r.get(0))
                        .unwrap_or(0);
                    let tags: i64 = conn
                        .query_row("SELECT COUNT(*) FROM image_tags", [], |r| r.get(0))
                        .unwrap_or(0);

                    dir_image_count += images;
                    dir_file_count += files;
                    dir_tag_count += tags;

                    dir_details.push(json!({
                        "directory_id": dir_id,
                        "images": images,
                        "files": files,
                        "tags": tags,
                    }));
                }
            }
        }

        let images_match = main_image_count >= dir_image_count;
        let files_match = main_file_count >= dir_file_count;
        let tags_match = main_tag_count >= dir_tag_count;
        let verified = images_match && files_match && tags_match;

        Ok::<_, AppError>(json!({
            "verified": verified,
            "main_db": {
                "images": main_image_count,
                "files": main_file_count,
                "tags": main_tag_count,
            },
            "directory_dbs": {
                "images": dir_image_count,
                "files": dir_file_count,
                "tags": dir_tag_count,
            },
            "directory_details": dir_details,
            "images_match": images_match,
            "files_match": files_match,
            "tags_match": tags_match,
        }))
    })
    .await??;

    Ok(Json(result))
}

/// POST /settings/migration/delete-source - Delete source per-directory databases after migration.
///
/// After a successful migration to the main DB, this removes the per-directory
/// database files that are no longer needed. The `mode` field indicates the
/// migration direction but the action is the same: delete all per-directory DBs
/// whose data has already been migrated into the main DB.
async fn delete_source(
    State(state): State<AppState>,
    Json(_body): Json<DeleteSourceRequest>,
) -> Result<Json<Value>, AppError> {
    // Only allow deletion when migration is completed (not running)
    {
        let migration = state.migration_state().read().await;
        if migration.status == MigrationStatus::Running {
            return Err(AppError::BadRequest(
                "Cannot delete source data while migration is running".into(),
            ));
        }
    }

    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || {
        let dir_ids = state_clone.directory_db().get_all_directory_ids();
        let mut deleted: i64 = 0;
        let mut errors: Vec<String> = Vec::new();

        for dir_id in &dir_ids {
            match state_clone.directory_db().delete_directory_db(*dir_id) {
                Ok(()) => {
                    deleted += 1;
                    log::info!("[Migration] Deleted source database for directory {}", dir_id);
                }
                Err(e) => {
                    errors.push(format!(
                        "Failed to delete database for directory {}: {}",
                        dir_id, e
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok::<_, AppError>(json!({
                "success": true,
                "deleted": deleted,
                "message": format!("Deleted {} source database(s)", deleted),
            }))
        } else {
            Ok::<_, AppError>(json!({
                "success": false,
                "deleted": deleted,
                "error": errors.join("; "),
                "message": format!("Deleted {} database(s) with {} error(s)", deleted, errors.len()),
            }))
        }
    })
    .await??;

    Ok(Json(result))
}

/// POST /settings/migration/import/validate - Validate import of directories into existing database.
///
/// Similar to validate_migration but for additive imports where the destination
/// database already has data. Checks that requested directories exist and their
/// per-directory databases are accessible.
async fn validate_import(
    State(state): State<AppState>,
    Json(body): Json<ImportValidateRequest>,
) -> Result<Json<Value>, AppError> {
    // Check that migration is not already running
    {
        let migration = state.migration_state().read().await;
        if migration.status == MigrationStatus::Running {
            return Err(AppError::BadRequest(
                "A migration is already in progress".into(),
            ));
        }
    }

    let state_clone = state.clone();
    let requested_ids = body.directory_ids.clone();

    let result = tokio::task::spawn_blocking(move || {
        let main_conn = state_clone.main_db().get()?;
        let mut errors: Vec<String> = Vec::new();
        let mut estimated_size: i64 = 0;
        let mut image_count: i64 = 0;

        // Determine which directories to validate
        let dir_ids: Vec<i64> = if requested_ids.is_empty() {
            // All directories
            let mut stmt = main_conn.prepare("SELECT id FROM watch_directories")?;
            let ids: Vec<i64> = stmt.query_map([], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            ids
        } else {
            requested_ids
        };

        if dir_ids.is_empty() {
            errors.push("No directories found to import".into());
            return Ok::<_, AppError>(json!({
                "valid": false,
                "errors": errors,
                "estimated_size": 0,
                "image_count": 0,
            }));
        }

        for dir_id in &dir_ids {
            // Check directory exists in main DB
            let dir_info: Result<(String, Option<String>), _> = main_conn.query_row(
                "SELECT path, name FROM watch_directories WHERE id = ?1",
                params![dir_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            );

            let (dir_path, dir_name) = match dir_info {
                Ok(info) => info,
                Err(_) => {
                    errors.push(format!("Directory ID {} not found in database", dir_id));
                    continue;
                }
            };

            let label = dir_name.unwrap_or_else(|| {
                Path::new(&dir_path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&dir_path)
                    .to_string()
            });

            // Check filesystem path exists
            if !Path::new(&dir_path).exists() {
                errors.push(format!(
                    "Directory '{}' path does not exist: {}",
                    label, dir_path
                ));
            }

            // Check per-directory DB is accessible
            if !state_clone.directory_db().db_exists(*dir_id) {
                errors.push(format!(
                    "Directory '{}' (id={}) has no per-directory database",
                    label, dir_id
                ));
                continue;
            }

            match state_clone.directory_db().get_pool(*dir_id) {
                Ok(pool) => match pool.get() {
                    Ok(conn) => {
                        let count: i64 = conn
                            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                            .unwrap_or(0);
                        let size: i64 = conn
                            .query_row(
                                "SELECT COALESCE(SUM(file_size), 0) FROM images",
                                [],
                                |r| r.get(0),
                            )
                            .unwrap_or(0);

                        image_count += count;
                        estimated_size += size;
                    }
                    Err(e) => {
                        errors.push(format!(
                            "Directory '{}' DB connection failed: {}",
                            label, e
                        ));
                    }
                },
                Err(e) => {
                    errors.push(format!(
                        "Directory '{}' DB pool creation failed: {}",
                        label, e
                    ));
                }
            }
        }

        let valid = errors.is_empty();
        Ok::<_, AppError>(json!({
            "valid": valid,
            "errors": errors,
            "estimated_size": estimated_size,
            "image_count": image_count,
            "directory_count": dir_ids.len(),
        }))
    })
    .await??;

    Ok(Json(result))
}

/// POST /settings/migration/import/start - Start importing directories into the existing main database.
///
/// Similar to start_migration but explicitly additive: merges records from
/// per-directory databases into the main database without requiring the
/// destination to be empty. Uses file_hash to skip duplicates.
async fn start_import(
    State(state): State<AppState>,
    Json(body): Json<ImportStartRequest>,
) -> Result<Json<Value>, AppError> {
    // Check that no migration is already running
    {
        let migration = state.migration_state().read().await;
        if migration.status == MigrationStatus::Running {
            return Err(AppError::BadRequest(
                "A migration is already in progress".into(),
            ));
        }
    }

    let requested_ids = body.directory_ids.clone();

    // Determine directory IDs to import
    let dir_ids: Vec<i64> = if requested_ids.is_empty() {
        let main_conn = state.main_db().get()?;
        let mut stmt = main_conn.prepare("SELECT id FROM watch_directories")?;
        let ids: Vec<i64> = stmt.query_map([], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .collect();
        ids
    } else {
        requested_ids
    };

    if dir_ids.is_empty() {
        return Err(AppError::BadRequest("No directories to import".into()));
    }

    // Initialize migration state (reuse the same state tracker)
    {
        let mut migration = state.migration_state().write().await;
        migration.status = MigrationStatus::Running;
        migration.direction = MigrationDirection::ToMainDb;
        migration.mode = Some(MigrationMode::Selective);
        migration.directory_ids = Some(dir_ids.clone());
        migration.progress = Some(MigrationProgress {
            total: 0,
            completed: 0,
            current_directory: None,
            current_directory_id: None,
            directories_total: dir_ids.len() as i64,
            directories_completed: 0,
        });
        migration.errors.clear();
        migration.started_at = Some(Utc::now().to_rfc3339());
        migration.completed_at = None;
    }

    // Broadcast start event
    if let Some(events) = state.events() {
        events.migration.broadcast(
            migration_event_type::STARTED,
            json!({
                "mode": "import",
                "directory_count": dir_ids.len(),
            }),
        );
    }

    // Spawn the background migration task (reuses the same migration logic)
    let state_bg = state.clone();
    tokio::spawn(async move {
        run_migration(state_bg, dir_ids).await;
    });

    Ok(Json(json!({
        "success": true,
        "message": "Import started",
    })))
}

/// GET /settings/migration/events - SSE stream for real-time migration progress.
///
/// Uses the migration EventBroadcaster to stream progress events to connected
/// clients in real time.
async fn migration_events(
    State(state): State<AppState>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, AppError> {
    let events = state
        .events()
        .ok_or_else(|| AppError::Internal("Event system not available".into()))?;

    let stream = events.migration.sse_stream();

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

// ---- Internal types ---------------------------------------------------------

/// Intermediate struct for reading image rows from per-directory DBs.
struct ImageRow {
    id: i64,
    filename: String,
    original_filename: Option<String>,
    file_hash: String,
    perceptual_hash: Option<String>,
    width: Option<i32>,
    height: Option<i32>,
    file_size: Option<i64>,
    duration: Option<f64>,
    rating: String,
    prompt: Option<String>,
    negative_prompt: Option<String>,
    model_name: Option<String>,
    sampler: Option<String>,
    seed: Option<String>,
    steps: Option<i32>,
    cfg_scale: Option<f64>,
    source_url: Option<String>,
    num_faces: Option<i32>,
    min_detected_age: Option<i32>,
    max_detected_age: Option<i32>,
    detected_ages: Option<String>,
    age_detection_data: Option<String>,
    is_favorite: bool,
    import_source: Option<String>,
    view_count: i32,
    created_at: String,
    updated_at: Option<String>,
    file_created_at: Option<String>,
    file_modified_at: Option<String>,
}
