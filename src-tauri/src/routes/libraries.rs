use axum::extract::{Path as AxumPath, State};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::db::library::LibraryContext;
use crate::server::error::AppError;
use crate::server::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_libraries).post(add_library))
        .route("/{uuid}", get(get_library).patch(update_library).delete(remove_library))
        .route("/{uuid}/mount", post(mount_library))
        .route("/{uuid}/unmount", post(unmount_library))
}

// ─── Request/Response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct AddLibraryRequest {
    path: String,
    name: String,
    #[serde(default = "default_true")]
    auto_mount: bool,
    #[serde(default)]
    create_new: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct UpdateLibraryRequest {
    name: Option<String>,
    auto_mount: Option<bool>,
}

// ─── Route handlers ─────────────────────────────────────────────────────────

/// GET /api/libraries — List all registered libraries with status and stats.
async fn list_libraries(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let library_manager = state_clone.library_manager();

        // Primary library info
        let primary = library_manager.primary();
        let primary_stats = get_library_stats(&state_clone, &primary)?;

        let mut libraries = vec![json!({
            "uuid": primary.uuid,
            "name": primary.name,
            "path": primary.data_dir.display().to_string(),
            "is_primary": true,
            "mounted": true,
            "accessible": true,
            "auto_mount": true,
            "stats": primary_stats
        })];

        // Registered auxiliary libraries
        let mut stmt = conn.prepare(
            "SELECT uuid, name, path, auto_mount, mount_order, last_mounted_at, created_at
             FROM mounted_libraries ORDER BY mount_order, created_at"
        )?;

        let rows: Vec<(String, String, String, bool, i32, Option<String>, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        for (uuid, name, path, auto_mount, _mount_order, last_mounted_at, created_at) in rows {
            let mounted = library_manager.is_mounted(&uuid);
            let accessible = std::path::Path::new(&path).exists();

            let stats = if mounted {
                if let Some(lib) = library_manager.get(&uuid) {
                    get_library_stats(&state_clone, &lib).ok()
                } else {
                    None
                }
            } else {
                None
            };

            libraries.push(json!({
                "uuid": uuid,
                "name": name,
                "path": path,
                "is_primary": false,
                "mounted": mounted,
                "accessible": accessible,
                "auto_mount": auto_mount,
                "last_mounted_at": last_mounted_at,
                "created_at": created_at,
                "stats": stats
            }));
        }

        Ok::<_, AppError>(Json(json!({ "libraries": libraries })))
    })
    .await?
}

/// POST /api/libraries — Register an existing library by path, or create a new one.
async fn add_library(
    State(state): State<AppState>,
    Json(data): Json<AddLibraryRequest>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let lib_path = std::path::PathBuf::from(&data.path);

        // Open or create the library
        let ctx = if data.create_new {
            LibraryContext::create(&lib_path, &data.name)
                .map_err(|e| AppError::Internal(format!("Failed to create library: {}", e)))?
        } else {
            // Validate the path looks like a library
            if !lib_path.exists() {
                return Err(AppError::BadRequest(format!(
                    "Path does not exist: {}",
                    data.path
                )));
            }
            if !lib_path.join("library.db").exists() {
                return Err(AppError::BadRequest(format!(
                    "No library.db found at: {}",
                    data.path
                )));
            }
            LibraryContext::open(&lib_path, &data.name)
                .map_err(|e| AppError::Internal(format!("Failed to open library: {}", e)))?
        };

        let uuid = ctx.uuid.clone();

        // Check for duplicate UUID
        {
            let conn = state_clone.main_db().get()?;
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM mounted_libraries WHERE uuid = ?1",
                    params![&uuid],
                    |row| row.get::<_, i64>(0).map(|c| c > 0),
                )
                .unwrap_or(false);

            if exists {
                return Err(AppError::BadRequest(format!(
                    "Library with UUID '{}' is already registered",
                    uuid
                )));
            }

            // Also check this isn't the primary library's path
            let primary = state_clone.library_manager().primary();
            if lib_path == primary.data_dir {
                return Err(AppError::BadRequest(
                    "Cannot register the primary library as an auxiliary library".into(),
                ));
            }
        }

        // Register in mounted_libraries table
        {
            let conn = state_clone.main_db().get()?;
            let next_order: i32 = conn
                .query_row(
                    "SELECT COALESCE(MAX(mount_order), 0) + 1 FROM mounted_libraries",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            conn.execute(
                "INSERT INTO mounted_libraries (uuid, name, path, auto_mount, mount_order, last_mounted_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))",
                params![uuid, data.name, data.path, data.auto_mount, next_order],
            )?;
        }

        // Mount immediately
        state_clone.library_manager().mount(ctx);

        Ok::<_, AppError>(Json(json!({
            "uuid": uuid,
            "name": data.name,
            "path": data.path,
            "mounted": true,
            "message": if data.create_new { "Library created and mounted" } else { "Library registered and mounted" }
        })))
    })
    .await?
}

/// GET /api/libraries/:uuid — Single library details.
async fn get_library(
    State(state): State<AppState>,
    AxumPath(uuid): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let library_manager = state_clone.library_manager();

        // Check primary
        let primary = library_manager.primary();
        if uuid == "primary" || uuid == primary.uuid {
            let stats = get_library_stats(&state_clone, &primary)?;
            return Ok(Json(json!({
                "uuid": primary.uuid,
                "name": primary.name,
                "path": primary.data_dir.display().to_string(),
                "is_primary": true,
                "mounted": true,
                "accessible": true,
                "auto_mount": true,
                "stats": stats
            })));
        }

        // Check registered libraries
        let conn = state_clone.main_db().get()?;
        let result = conn.query_row(
            "SELECT uuid, name, path, auto_mount, mount_order, last_mounted_at, created_at
             FROM mounted_libraries WHERE uuid = ?1",
            params![&uuid],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, bool>(3)?,
                    row.get::<_, i32>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                ))
            },
        );

        match result {
            Ok((uuid, name, path, auto_mount, _mount_order, last_mounted_at, created_at)) => {
                let mounted = library_manager.is_mounted(&uuid);
                let accessible = std::path::Path::new(&path).exists();

                let stats = if mounted {
                    library_manager
                        .get(&uuid)
                        .and_then(|lib| get_library_stats(&state_clone, &lib).ok())
                } else {
                    None
                };

                Ok(Json(json!({
                    "uuid": uuid,
                    "name": name,
                    "path": path,
                    "is_primary": false,
                    "mounted": mounted,
                    "accessible": accessible,
                    "auto_mount": auto_mount,
                    "last_mounted_at": last_mounted_at,
                    "created_at": created_at,
                    "stats": stats
                })))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(AppError::NotFound(format!("Library '{}' not found", uuid)))
            }
            Err(e) => Err(AppError::Internal(format!("Database error: {}", e))),
        }
    })
    .await?
}

/// POST /api/libraries/:uuid/mount — Mount a registered library.
async fn mount_library(
    State(state): State<AppState>,
    AxumPath(uuid): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let library_manager = state_clone.library_manager();

        // Can't mount what's already mounted
        if library_manager.is_mounted(&uuid) {
            return Ok(Json(json!({
                "mounted": true,
                "message": "Library is already mounted"
            })));
        }

        // Look up library in registry
        let conn = state_clone.main_db().get()?;
        let (name, path) = conn
            .query_row(
                "SELECT name, path FROM mounted_libraries WHERE uuid = ?1",
                params![&uuid],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(|_| AppError::NotFound(format!("Library '{}' not registered", uuid)))?;

        let lib_path = std::path::PathBuf::from(&path);
        let ctx = LibraryContext::open(&lib_path, &name)
            .map_err(|e| AppError::Internal(format!("Failed to open library: {}", e)))?;

        library_manager.mount(ctx);

        // Update last_mounted_at
        let _ = conn.execute(
            "UPDATE mounted_libraries SET last_mounted_at = datetime('now') WHERE uuid = ?1",
            params![&uuid],
        );

        Ok::<_, AppError>(Json(json!({
            "mounted": true,
            "message": format!("Library '{}' mounted", name)
        })))
    })
    .await?
}

/// POST /api/libraries/:uuid/unmount — Unmount a library (close its database pools).
async fn unmount_library(
    State(state): State<AppState>,
    AxumPath(uuid): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let library_manager = state.library_manager();

    if uuid == "primary" || uuid == library_manager.primary().uuid {
        return Err(AppError::BadRequest("Cannot unmount the primary library".into()));
    }

    let was_mounted = library_manager.unmount(&uuid);

    Ok(Json(json!({
        "mounted": false,
        "was_mounted": was_mounted,
        "message": if was_mounted { "Library unmounted" } else { "Library was not mounted" }
    })))
}

/// PATCH /api/libraries/:uuid — Update library name or auto_mount.
async fn update_library(
    State(state): State<AppState>,
    AxumPath(uuid): AxumPath<String>,
    Json(data): Json<UpdateLibraryRequest>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        if uuid == "primary" || uuid == state_clone.library_manager().primary().uuid {
            return Err(AppError::BadRequest(
                "Cannot modify the primary library via this endpoint".into(),
            ));
        }

        let conn = state_clone.main_db().get()?;

        // Check it exists
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM mounted_libraries WHERE uuid = ?1",
                params![&uuid],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);

        if !exists {
            return Err(AppError::NotFound(format!("Library '{}' not found", uuid)));
        }

        if let Some(ref name) = data.name {
            conn.execute(
                "UPDATE mounted_libraries SET name = ?1 WHERE uuid = ?2",
                params![name, &uuid],
            )?;
        }

        if let Some(auto_mount) = data.auto_mount {
            conn.execute(
                "UPDATE mounted_libraries SET auto_mount = ?1 WHERE uuid = ?2",
                params![auto_mount, &uuid],
            )?;
        }

        Ok::<_, AppError>(Json(json!({
            "uuid": uuid,
            "updated": true
        })))
    })
    .await?
}

/// DELETE /api/libraries/:uuid — Unregister a library (unmount + remove from registry).
async fn remove_library(
    State(state): State<AppState>,
    AxumPath(uuid): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let library_manager = state_clone.library_manager();

        if uuid == "primary" || uuid == library_manager.primary().uuid {
            return Err(AppError::BadRequest("Cannot remove the primary library".into()));
        }

        // Unmount if mounted
        library_manager.unmount(&uuid);

        // Remove from registry
        let conn = state_clone.main_db().get()?;
        let deleted = conn.execute(
            "DELETE FROM mounted_libraries WHERE uuid = ?1",
            params![&uuid],
        )?;

        if deleted == 0 {
            return Err(AppError::NotFound(format!("Library '{}' not found", uuid)));
        }

        Ok::<_, AppError>(Json(json!({
            "uuid": uuid,
            "removed": true,
            "message": "Library unregistered and unmounted"
        })))
    })
    .await?
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Get basic stats for a library (total images, directories, tags).
fn get_library_stats(
    _state: &AppState,
    lib: &LibraryContext,
) -> Result<Value, AppError> {
    let main_conn = lib.main_pool.get()?;

    let total_tags: i64 = main_conn
        .query_row("SELECT COUNT(*) FROM tags", [], |r| r.get(0))
        .unwrap_or(0);

    let total_dirs: i64 = main_conn
        .query_row("SELECT COUNT(*) FROM watch_directories", [], |r| r.get(0))
        .unwrap_or(0);

    // Count images across directory DBs
    let mut total_images: i64 = 0;
    let dir_ids = lib.directory_db.get_all_directory_ids();
    for dir_id in &dir_ids {
        if let Ok(pool) = lib.directory_db.get_pool(*dir_id) {
            if let Ok(conn) = pool.get() {
                let count: i64 = conn
                    .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
                    .unwrap_or(0);
                total_images += count;
            }
        }
    }

    Ok(json!({
        "total_images": total_images,
        "directories": total_dirs,
        "tags": total_tags
    }))
}
