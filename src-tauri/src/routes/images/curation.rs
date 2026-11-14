use std::path::{Path, PathBuf};

use axum::extract::{Path as AxumPath, State};
use axum::response::Json;
use axum::Json as JsonBody;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::server::error::AppError;
use crate::server::state::AppState;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageLocator {
    pub image_id: i64,
    pub directory_id: i64,
    pub library_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FavoriteRequest {
    pub is_favorite: bool,
    pub directory_id: i64,
    pub library_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DiscardRequest {
    pub directory_id: i64,
    pub library_id: Option<String>,
    pub dumpster_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RestoreRequest {
    pub directory_id: i64,
    pub library_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UnfavoriteRequest {
    pub items: Vec<ImageLocator>,
}

fn io_error(context: &str, error: impl std::fmt::Display) -> AppError {
    AppError::Internal(format!("{}: {}", context, error))
}

fn move_file(source: &Path, destination: &Path) -> Result<(), AppError> {
    if destination.exists() {
        return Err(AppError::BadRequest(format!(
            "Refusing to overwrite existing file: {}",
            destination.display()
        )));
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| io_error("Failed to create dumpster directory", error))?;
    }
    // Link installation is atomic and refuses an existing destination. Unlike
    // rename, it cannot overwrite a file created after the collision check.
    if std::fs::hard_link(source, destination).is_err() {
        // Cross-device filesystems may not support links. Reserve the output
        // exclusively before copying so this fallback also refuses collisions.
        let mut input = std::fs::File::open(source)?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)
            .map_err(|error| io_error("Failed to reserve media destination", error))?;
        let copied = (|| -> Result<(), std::io::Error> {
            std::io::copy(&mut input, &mut output)?;
            output.set_permissions(input.metadata()?.permissions())?;
            output.sync_all()
        })();
        if let Err(error) = copied {
            drop(output);
            let cleanup = std::fs::remove_file(destination);
            return Err(io_error(
                "Failed to copy media",
                match cleanup {
                    Ok(()) => error.to_string(),
                    Err(cleanup) => format!("{error}; incomplete destination retained: {cleanup}"),
                },
            ));
        }
    }
    if let Err(error) = std::fs::remove_file(source) {
        let cleanup = std::fs::remove_file(destination);
        return Err(io_error(
            "Failed to remove original media",
            match cleanup {
                Ok(()) => error.to_string(),
                Err(cleanup) => format!("{error}; destination copy retained: {cleanup}"),
            },
        ));
    }
    Ok(())
}

type TrackedMove = (i64, PathBuf, PathBuf);

fn discard_moves(
    conn: &mut rusqlite::Connection,
    moves: &[TrackedMove],
    mut mover: impl FnMut(&Path, &Path) -> Result<(), AppError>,
) -> Result<usize, AppError> {
    // Reject missing inputs and collisions before changing any file or row.
    let mut destinations = std::collections::HashSet::new();
    for (_, source, destination) in moves {
        if !source.try_exists()? {
            return Err(AppError::NotFound(format!(
                "Media is missing: {}",
                source.display()
            )));
        }
        if destination.try_exists()? || !destinations.insert(destination.clone()) {
            return Err(AppError::BadRequest(format!(
                "Discard destination already exists: {}",
                destination.display()
            )));
        }
    }

    // Publish undo information before a removal notification can reach the
    // watcher. A stale selection must not move files whose rows disappeared.
    let tx = conn.transaction()?;
    for (id, source, destination) in moves {
        let changed = tx.execute(
            "UPDATE image_files SET original_path = ?1, curation_original_path = ?2,
             curation_discarded_at = datetime('now')
             WHERE id = ?3 AND original_path = ?2 AND curation_discarded_at IS NULL",
            params![destination.to_string_lossy(), source.to_string_lossy(), id],
        )?;
        if changed != 1 {
            return Err(AppError::NotFound(
                "Curation file changed before discard".into(),
            ));
        }
    }
    tx.commit()?;

    let mut completed: Vec<usize> = Vec::new();
    for (index, (_, source, destination)) in moves.iter().enumerate() {
        if let Err(error) = mover(source, destination) {
            let mut rollback_failed = std::collections::HashSet::new();
            let mut failures = Vec::new();
            for &done in completed.iter().rev() {
                let (id, original, dumped) = &moves[done];
                if let Err(rollback_error) = mover(dumped, original) {
                    rollback_failed.insert(*id);
                    failures.push(rollback_error.to_string());
                }
            }
            let rollback = conn.transaction()?;
            for (id, original, _) in moves {
                // Never clear the recovery marker for a file still in the
                // dumpster. Untouched/successfully rolled-back files are live.
                if !rollback_failed.contains(id) && original.try_exists()? {
                    rollback.execute(
                        "UPDATE image_files SET original_path = ?1,
                         curation_original_path = NULL, curation_discarded_at = NULL WHERE id = ?2",
                        params![original.to_string_lossy(), id],
                    )?;
                }
            }
            rollback.commit()?;
            if !failures.is_empty() {
                return Err(AppError::Internal(format!(
                    "{error}; discard rollback incomplete; retained files can be restored: {}",
                    failures.join("; ")
                )));
            }
            return Err(error);
        }
        completed.push(index);
    }
    Ok(moves.len())
}

fn rollback_restore_moves(
    conn: &rusqlite::Connection,
    moves: &[TrackedMove],
    completed: &[usize],
    mover: &mut impl FnMut(&Path, &Path) -> Result<(), AppError>,
    error: AppError,
) -> AppError {
    let mut failures = Vec::new();
    for &index in completed.iter().rev() {
        let (id, dumped, original) = &moves[index];
        if let Err(rollback_error) = mover(original, dumped) {
            failures.push(rollback_error.to_string());
            if original.try_exists().unwrap_or(false) && !dumped.try_exists().unwrap_or(true) {
                // Serve the surviving original while retaining its undo marker.
                // A later restore recognizes this already-restored row.
                if let Err(db_error) = conn.execute(
                    "UPDATE image_files SET original_path = ?1 WHERE id = ?2",
                    params![original.to_string_lossy(), id],
                ) {
                    failures.push(db_error.to_string());
                }
            }
        }
    }
    if failures.is_empty() {
        error
    } else {
        AppError::Internal(format!(
            "{error}; restore rollback incomplete: {}",
            failures.join("; ")
        ))
    }
}

fn restore_moves(
    conn: &mut rusqlite::Connection,
    moves: &[TrackedMove],
    mut mover: impl FnMut(&Path, &Path) -> Result<(), AppError>,
) -> Result<usize, AppError> {
    let mut completed = Vec::new();
    let update = (|| -> Result<(), AppError> {
        // Keep watcher imports from creating another reference at the restored
        // location before this transaction publishes its original row there.
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        // Inspect the entire batch before moving the first file. Interrupted
        // discards/restores may already have the original in its final location.
        let mut pending = Vec::new();
        for (index, (_, dumped, original)) in moves.iter().enumerate() {
            let dumped_exists = dumped.try_exists()?;
            let original_exists = original.try_exists()?;
            if original_exists && (!dumped_exists || dumped == original) {
                continue;
            }
            if original_exists {
                return Err(AppError::BadRequest(format!(
                    "Cannot restore because both discarded and original files exist: {} and {}",
                    dumped.display(),
                    original.display()
                )));
            }
            if !dumped_exists {
                return Err(AppError::NotFound(format!(
                    "Discarded file is missing: {}",
                    dumped.display()
                )));
            }
            pending.push(index);
        }
        for index in pending {
            let (_, dumped, original) = &moves[index];
            mover(dumped, original)?;
            completed.push(index);
        }
        for (id, _, original) in moves {
            if tx.execute(
                "UPDATE image_files SET original_path = ?1, curation_original_path = NULL,
                 curation_discarded_at = NULL WHERE id = ?2 AND curation_discarded_at IS NOT NULL",
                params![original.to_string_lossy(), id],
            )? != 1
            {
                return Err(AppError::NotFound(
                    "Curation file changed before restore".into(),
                ));
            }
        }
        tx.commit()?;
        Ok(())
    })();
    if let Err(error) = update {
        return Err(rollback_restore_moves(
            conn, moves, &completed, &mut mover, error,
        ));
    }
    Ok(moves.len())
}

fn set_favorite_value(
    state: &AppState,
    locator: &ImageLocator,
    value: bool,
) -> Result<bool, AppError> {
    let lib = state.resolve_library(locator.library_id.as_deref())?;
    if !lib.directory_db.db_exists(locator.directory_id) {
        return Err(AppError::NotFound("Directory database not found".into()));
    }
    let pool = lib.directory_db.get_pool(locator.directory_id)?;
    let conn = pool.get()?;
    let previous: bool = conn
        .query_row(
            "SELECT is_favorite FROM images WHERE id = ?1",
            params![locator.image_id],
            |row| row.get(0),
        )
        .map_err(|_| AppError::NotFound("Image not found".into()))?;
    if previous == value {
        return Ok(value);
    }
    conn.execute(
        "UPDATE images SET is_favorite = ?1 WHERE id = ?2",
        params![value, locator.image_id],
    )?;
    let delta = if value { 1 } else { -1 };
    let main_conn = lib.main_pool.get()?;
    main_conn.execute(
        "UPDATE watch_directories
         SET favorited_count = MAX(0, favorited_count + ?1)
         WHERE id = ?2",
        params![delta, locator.directory_id],
    )?;
    Ok(value)
}

// AC: @curation-game ac-3, ac-5
pub async fn set_favorite(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    JsonBody(body): JsonBody<FavoriteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let locator = ImageLocator {
        image_id,
        directory_id: body.directory_id,
        library_id: body.library_id,
    };
    let value = body.is_favorite;
    let state_clone = state.clone();
    let is_favorite =
        tokio::task::spawn_blocking(move || set_favorite_value(&state_clone, &locator, value))
            .await??;
    Ok(Json(
        json!({"image_id": image_id, "is_favorite": is_favorite}),
    ))
}

// AC: @curation-game ac-4
pub async fn discard(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    JsonBody(body): JsonBody<DiscardRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let moved = tokio::task::spawn_blocking(move || -> Result<usize, AppError> {
        let lib = state_clone.resolve_library(body.library_id.as_deref())?;
        let main_conn = lib.main_pool.get()?;
        let watch_root: String = main_conn
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![body.directory_id],
                |row| row.get(0),
            )
            .map_err(|_| AppError::NotFound("Watch directory not found".into()))?;
        let watch_root = PathBuf::from(watch_root);
        let pool = lib.directory_db.get_pool(body.directory_id)?;
        let mut conn = pool.get()?;
        let rows: Vec<(i64, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, original_path FROM image_files
                 WHERE image_id = ?1 AND curation_discarded_at IS NULL",
            )?;
            let result = stmt
                .query_map(params![image_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            result
        };
        if rows.is_empty() {
            return Err(AppError::NotFound(
                "Image file not found or already discarded".into(),
            ));
        }

        let custom_root = body.dumpster_path.filter(|path| !path.trim().is_empty());
        let moves: Vec<(i64, PathBuf, PathBuf)> = rows
            .into_iter()
            .map(|(file_id, source)| {
                let source = PathBuf::from(source);
                let relative = source
                    .strip_prefix(&watch_root)
                    .ok()
                    .filter(|path| !path.as_os_str().is_empty())
                    .map(Path::to_path_buf)
                    .or_else(|| source.file_name().map(PathBuf::from))
                    .ok_or_else(|| AppError::BadRequest("Invalid source path".into()))?;
                let destination = if let Some(root) = custom_root.as_deref() {
                    PathBuf::from(root)
                        .join(&lib.uuid)
                        .join(body.directory_id.to_string())
                        .join(relative)
                } else {
                    watch_root.join("dumpster").join(relative)
                };
                Ok((file_id, source, destination))
            })
            .collect::<Result<Vec<_>, AppError>>()?;

        discard_moves(&mut conn, &moves, move_file)
    })
    .await??;

    Ok(Json(
        json!({"image_id": image_id, "discarded": true, "moved_files": moved}),
    ))
}

// AC: @curation-game ac-5
pub async fn restore(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    JsonBody(body): JsonBody<RestoreRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let restored = tokio::task::spawn_blocking(move || -> Result<usize, AppError> {
        let lib = state_clone.resolve_library(body.library_id.as_deref())?;
        let pool = lib.directory_db.get_pool(body.directory_id)?;
        let mut conn = pool.get()?;
        let rows: Vec<(i64, String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, original_path, curation_original_path FROM image_files
                 WHERE image_id = ?1 AND curation_discarded_at IS NOT NULL",
            )?;
            let result = stmt
                .query_map(params![image_id], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            result
        };
        if rows.is_empty() {
            return Err(AppError::NotFound("Discarded image file not found".into()));
        }
        let moves: Vec<(i64, PathBuf, PathBuf)> = rows
            .into_iter()
            .map(|(id, dumped, original)| (id, PathBuf::from(dumped), PathBuf::from(original)))
            .collect();
        restore_moves(&mut conn, &moves, move_file)
    })
    .await??;

    Ok(Json(
        json!({"image_id": image_id, "restored": true, "restored_files": restored}),
    ))
}

// AC: @curation-game ac-7
pub async fn unfavorite_many(
    State(state): State<AppState>,
    JsonBody(body): JsonBody<UnfavoriteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    if body.items.len() > 400 {
        return Err(AppError::BadRequest(
            "At most 400 items may be updated".into(),
        ));
    }
    let state_clone = state.clone();
    let updated = tokio::task::spawn_blocking(move || -> Result<usize, AppError> {
        let mut count = 0;
        for locator in &body.items {
            set_favorite_value(&state_clone, locator, false)?;
            count += 1;
        }
        Ok(count)
    })
    .await??;
    Ok(Json(json!({"updated": updated})))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        root: PathBuf,
        moves: Vec<TrackedMove>,
    }

    impl Fixture {
        fn new(count: usize) -> Self {
            let root =
                std::env::temp_dir().join(format!("curation-regression-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let conn = rusqlite::Connection::open(root.join("metadata.sqlite")).unwrap();
            conn.execute_batch(
                "CREATE TABLE image_files (
                id INTEGER PRIMARY KEY, original_path TEXT NOT NULL UNIQUE,
                curation_original_path TEXT, curation_discarded_at TEXT)",
            )
            .unwrap();
            let mut moves = Vec::new();
            for index in 1..=count {
                let source = root.join(format!("source-{index}.png"));
                let dumped = root.join("dumpster").join(format!("source-{index}.png"));
                std::fs::write(&source, format!("synthetic-media-{index}")).unwrap();
                conn.execute(
                    "INSERT INTO image_files(id, original_path) VALUES (?1, ?2)",
                    params![index as i64, source.to_string_lossy()],
                )
                .unwrap();
                moves.push((index as i64, source, dumped));
            }
            Self { root, moves }
        }

        fn library(&self) -> (crate::db::library::LibraryContext, PathBuf) {
            let lib =
                crate::db::library::LibraryContext::create(&self.root.join("library"), "Synthetic")
                    .unwrap();
            let pool = lib.directory_db.get_pool(1).unwrap();
            let conn = pool.get().unwrap();
            conn.execute(
                "INSERT INTO images(id, filename, file_hash) VALUES (1, 'synthetic.png', ?1)",
                params!["0123456789abcdef0123456789abcdef"],
            )
            .unwrap();
            for (id, original, _) in &self.moves {
                conn.execute(
                    "INSERT INTO image_files(id, image_id, original_path) VALUES (?1, 1, ?2)",
                    params![id, original.to_string_lossy()],
                )
                .unwrap();
            }
            let thumb = lib.thumbnails_dir().join("0123456789abcdef.webp");
            std::fs::write(&thumb, b"synthetic-thumbnail").unwrap();
            drop(conn);
            drop(pool);
            (lib, thumb)
        }

        fn connection(&self) -> rusqlite::Connection {
            rusqlite::Connection::open(self.root.join("metadata.sqlite")).unwrap()
        }

        fn restore_plan(&self, conn: &rusqlite::Connection) -> Vec<TrackedMove> {
            let mut statement = conn
                .prepare(
                    "SELECT id, original_path, curation_original_path
                FROM image_files WHERE curation_discarded_at IS NOT NULL ORDER BY id",
                )
                .unwrap();
            let rows = statement
                .query_map([], |row| {
                    Ok((
                        row.get(0)?,
                        PathBuf::from(row.get::<_, String>(1)?),
                        PathBuf::from(row.get::<_, String>(2)?),
                    ))
                })
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            rows
        }

        fn assert_originals(&self, conn: &rusqlite::Connection) {
            for (id, original, _) in &self.moves {
                assert_eq!(
                    std::fs::read(original).unwrap(),
                    format!("synthetic-media-{id}").as_bytes()
                );
                let row: (String, bool) = conn.query_row(
                    "SELECT original_path, curation_discarded_at IS NOT NULL FROM image_files WHERE id=?1",
                    params![id], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
                assert_eq!(row, (original.to_string_lossy().into_owned(), false));
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.root) {
                if !std::thread::panicking() {
                    panic!("Synthetic fixture cleanup failed: {error}");
                }
            }
        }
    }

    fn forced_failure() -> AppError {
        AppError::Internal("injected move failure".into())
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn curation_discard_and_undo_persist_real_files_and_rows() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        assert_eq!(
            discard_moves(&mut conn, &fixture.moves, move_file).unwrap(),
            2
        );
        drop(conn);
        let mut conn = fixture.connection();
        let plan = fixture.restore_plan(&conn);
        assert_eq!(plan.len(), 2);
        for (_, original, dumped) in &fixture.moves {
            assert!(!original.exists());
            assert!(dumped.exists());
        }
        assert_eq!(restore_moves(&mut conn, &plan, move_file).unwrap(), 2);
        fixture.assert_originals(&conn);
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn curation_failed_discard_rolls_back_all_successful_moves() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        let mut calls = 0;
        assert!(discard_moves(&mut conn, &fixture.moves, |from, to| {
            calls += 1;
            if calls == 2 {
                Err(forced_failure())
            } else {
                move_file(from, to)
            }
        })
        .is_err());
        assert_eq!(calls, 3);
        fixture.assert_originals(&conn);
        assert!(fixture.moves.iter().all(|(_, _, dumped)| !dumped.exists()));
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn curation_failed_discard_rollback_keeps_the_surviving_copy_restorable() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        let mut calls = 0;
        assert!(discard_moves(&mut conn, &fixture.moves, |from, to| {
            calls += 1;
            if calls >= 2 {
                Err(forced_failure())
            } else {
                move_file(from, to)
            }
        })
        .is_err());
        assert!(!fixture.moves[0].1.exists());
        assert_eq!(
            std::fs::read(&fixture.moves[0].2).unwrap(),
            b"synthetic-media-1"
        );
        let plan = fixture.restore_plan(&conn);
        assert_eq!(
            plan,
            vec![(1, fixture.moves[0].2.clone(), fixture.moves[0].1.clone())]
        );
        restore_moves(&mut conn, &plan, move_file).unwrap();
        fixture.assert_originals(&conn);
    }

    // AC: @curation-game ac-5
    #[test]
    fn curation_restore_preflights_a_missing_later_file_before_any_moves() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        discard_moves(&mut conn, &fixture.moves, move_file).unwrap();
        std::fs::remove_file(&fixture.moves[1].2).unwrap();
        let plan = fixture.restore_plan(&conn);
        let mut calls = 0;
        assert!(restore_moves(&mut conn, &plan, |from, to| {
            calls += 1;
            move_file(from, to)
        })
        .is_err());
        assert_eq!(calls, 0);
        assert!(fixture.moves[0].2.exists());
        assert!(!fixture.moves[0].1.exists());
        assert_eq!(fixture.restore_plan(&conn).len(), 2);
    }

    // AC: @curation-game ac-5
    #[test]
    fn curation_restore_reconciles_publication_interrupted_before_file_moves() {
        let fixture = Fixture::new(2);
        let conn = fixture.connection();
        for (id, source, dumped) in &fixture.moves {
            conn.execute(
                "UPDATE image_files SET original_path=?1, curation_original_path=?2,
                curation_discarded_at='interrupted' WHERE id=?3",
                params![dumped.to_string_lossy(), source.to_string_lossy(), id],
            )
            .unwrap();
        }
        drop(conn);
        let mut conn = fixture.connection();
        let plan = fixture.restore_plan(&conn);
        restore_moves(&mut conn, &plan, |_, _| {
            panic!("Already-present originals must not move")
        })
        .unwrap();
        fixture.assert_originals(&conn);
    }

    // AC: @curation-game ac-5
    #[test]
    fn curation_restore_failed_rollback_records_actual_path_and_can_retry() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        discard_moves(&mut conn, &fixture.moves, move_file).unwrap();
        let plan = fixture.restore_plan(&conn);
        let mut calls = 0;
        assert!(restore_moves(&mut conn, &plan, |from, to| {
            calls += 1;
            if calls >= 2 {
                Err(forced_failure())
            } else {
                move_file(from, to)
            }
        })
        .is_err());
        let retry = fixture.restore_plan(&conn);
        assert_eq!(
            retry[0],
            (1, fixture.moves[0].1.clone(), fixture.moves[0].1.clone())
        );
        assert!(fixture.moves[0].1.exists());
        assert!(fixture.moves[1].2.exists());
        restore_moves(&mut conn, &retry, move_file).unwrap();
        fixture.assert_originals(&conn);
    }

    // AC: @curation-game ac-5
    #[test]
    fn curation_restore_database_failure_rolls_files_back_without_losing_undo() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        discard_moves(&mut conn, &fixture.moves, move_file).unwrap();
        conn.execute_batch("CREATE TRIGGER reject_restore BEFORE UPDATE ON image_files
            WHEN NEW.curation_discarded_at IS NULL BEGIN SELECT RAISE(ABORT, 'injected DB failure'); END;").unwrap();
        let plan = fixture.restore_plan(&conn);
        assert!(restore_moves(&mut conn, &plan, move_file).is_err());
        assert!(fixture
            .moves
            .iter()
            .all(|(_, original, dumped)| !original.exists() && dumped.exists()));
        assert_eq!(fixture.restore_plan(&conn).len(), 2);
        conn.execute_batch("DROP TRIGGER reject_restore").unwrap();
        restore_moves(&mut conn, &plan, move_file).unwrap();
        fixture.assert_originals(&conn);
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn curation_collision_and_stale_rows_fail_before_physical_changes() {
        let fixture = Fixture::new(2);
        let mut conn = fixture.connection();
        std::fs::create_dir_all(fixture.moves[1].2.parent().unwrap()).unwrap();
        std::fs::write(&fixture.moves[1].2, b"existing-destination").unwrap();
        assert!(discard_moves(&mut conn, &fixture.moves, |_, _| panic!(
            "Collision must be preflighted"
        ))
        .is_err());
        fixture.assert_originals(&conn);
        assert_eq!(
            std::fs::read(&fixture.moves[1].2).unwrap(),
            b"existing-destination"
        );
        std::fs::remove_file(&fixture.moves[1].2).unwrap();
        conn.execute("DELETE FROM image_files WHERE id=2", [])
            .unwrap();
        assert!(discard_moves(&mut conn, &fixture.moves, |_, _| panic!(
            "Stale rows must not move files"
        ))
        .is_err());
        let flags: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM image_files WHERE curation_discarded_at IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(flags, 0);
        assert!(fixture
            .moves
            .iter()
            .all(|(_, original, dumped)| original.exists() && !dumped.exists()));
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn curation_watcher_preserves_undo_and_thumbnail_then_handles_real_deletion() {
        let fixture = Fixture::new(1);
        let (lib, thumb) = fixture.library();
        let pool = lib.directory_db.get_pool(1).unwrap();
        let mut conn = pool.get().unwrap();
        discard_moves(&mut conn, &fixture.moves, |from, to| {
            move_file(from, to)?;
            crate::services::file_tracker::mark_file_missing(&lib, from.to_str().unwrap(), 1)?;
            let held = fixture.root.join("held.png");
            move_file(to, &held)?;
            crate::services::file_tracker::mark_file_missing(&lib, to.to_str().unwrap(), 1)?;
            move_file(&held, to)
        })
        .unwrap();
        assert!(thumb.exists());
        let plan = fixture.restore_plan(&conn);
        assert_eq!(plan.len(), 1);
        restore_moves(&mut conn, &plan, move_file).unwrap();
        crate::services::file_tracker::mark_file_missing(
            &lib,
            fixture.moves[0].1.to_str().unwrap(),
            1,
        )
        .unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM images", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert!(thumb.exists());
        std::fs::remove_file(&fixture.moves[0].1).unwrap();
        crate::services::file_tracker::mark_file_missing(
            &lib,
            fixture.moves[0].1.to_str().unwrap(),
            1,
        )
        .unwrap();
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM images", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert!(!thumb.exists());
    }

    // AC: @curation-game ac-5
    #[test]
    fn curation_restore_serializes_watcher_import_before_publishing_original_path() {
        let fixture = Fixture::new(1);
        let mut conn = fixture.connection();
        discard_moves(&mut conn, &fixture.moves, move_file).unwrap();
        let plan = fixture.restore_plan(&conn);
        restore_moves(&mut conn, &plan, |from, to| {
            move_file(from, to)?;
            let competitor = fixture.connection();
            competitor.busy_timeout(std::time::Duration::ZERO).unwrap();
            let error = competitor
                .execute(
                    "INSERT INTO image_files(id, original_path) VALUES (99, ?1)",
                    params![to.to_string_lossy()],
                )
                .unwrap_err();
            assert!(matches!(error, rusqlite::Error::SqliteFailure(code, _)
                if code.code == rusqlite::ErrorCode::DatabaseBusy));
            Ok(())
        })
        .unwrap();
        fixture.assert_originals(&conn);
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM image_files", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn curation_undo_cannot_interleave_with_an_unfinished_discard() {
        let fixture = Fixture::new(1);
        let mut conn = fixture.connection();
        discard_moves(&mut conn, &fixture.moves, |from, to| {
            // A second client can see the published undo row before the file
            // moves. It must not mistake a live discard for interrupted work.
            let mut competing = fixture.connection();
            let plan = fixture.restore_plan(&competing);
            assert_eq!(plan.len(), 1);
            assert!(matches!(
                restore_moves(&mut competing, &plan, move_file),
                Err(AppError::TooManyRequests(_))
            ));
            move_file(from, to)
        })
        .unwrap();
        let plan = fixture.restore_plan(&conn);
        assert_eq!(plan.len(), 1);
        restore_moves(&mut conn, &plan, move_file).unwrap();
        fixture.assert_originals(&conn);
    }

    // AC: @curation-game ac-4, ac-5, ac-10
    #[test]
    fn curation_verification_retains_interrupted_undo_and_deletes_real_missing_files() {
        let fixture = Fixture::new(2);
        let (lib, thumb) = fixture.library();
        let pool = lib.directory_db.get_pool(1).unwrap();
        let mut conn = pool.get().unwrap();
        for (id, source, dumped) in &fixture.moves {
            conn.execute(
                "UPDATE image_files SET original_path=?1,
                curation_original_path=?2, curation_discarded_at='interrupted' WHERE id=?3",
                params![dumped.to_string_lossy(), source.to_string_lossy(), id],
            )
            .unwrap();
        }
        assert_eq!(
            crate::services::file_tracker::clean_deleted_files(&lib, 1).unwrap(),
            0
        );
        assert_eq!(
            crate::services::file_tracker::verify_directory_files(&lib, 1)
                .unwrap()
                .deleted,
            0
        );
        assert!(thumb.exists());
        let plan = fixture.restore_plan(&conn);
        assert_eq!(plan.len(), 2);
        restore_moves(&mut conn, &plan, |_, _| {
            panic!("Originals must remain intact")
        })
        .unwrap();
        fixture.assert_originals(&conn);
        std::fs::remove_file(&fixture.moves[0].1).unwrap();
        assert_eq!(
            crate::services::file_tracker::verify_directory_files(&lib, 1)
                .unwrap()
                .deleted,
            1
        );
        assert!(thumb.exists());
        std::fs::remove_file(&fixture.moves[1].1).unwrap();
        assert_eq!(
            crate::services::file_tracker::clean_deleted_files(&lib, 1).unwrap(),
            1
        );
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM images", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert!(!thumb.exists());
    }

    // AC: @curation-game ac-5
    #[test]
    fn curation_concurrent_moves_never_overwrite_the_destination() {
        let fixture = Fixture::new(2);
        let destination = fixture.root.join("contended.png");
        let start = std::sync::Barrier::new(2);
        let results = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                start.wait();
                move_file(&fixture.moves[0].1, &destination)
            });
            let second = scope.spawn(|| {
                start.wait();
                move_file(&fixture.moves[1].1, &destination)
            });
            [first.join().unwrap(), second.join().unwrap()]
        });
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        for (index, result) in results.iter().enumerate() {
            let source = &fixture.moves[index].1;
            if result.is_ok() {
                assert!(!source.exists());
                assert_eq!(
                    std::fs::read(&destination).unwrap(),
                    format!("synthetic-media-{}", index + 1).as_bytes()
                );
            } else {
                assert_eq!(
                    std::fs::read(source).unwrap(),
                    format!("synthetic-media-{}", index + 1).as_bytes()
                );
            }
        }
    }

    // AC: @curation-game ac-4, ac-5
    #[test]
    fn move_file_refuses_to_overwrite_and_round_trips() {
        let root = std::env::temp_dir().join(format!("curation-move-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("source.png");
        let dumpster = root.join("dumpster/source.png");
        std::fs::write(&source, b"media").unwrap();
        move_file(&source, &dumpster).unwrap();
        assert!(!source.exists());
        assert_eq!(std::fs::read(&dumpster).unwrap(), b"media");

        std::fs::write(&source, b"collision").unwrap();
        assert!(move_file(&dumpster, &source).is_err());
        assert_eq!(std::fs::read(&source).unwrap(), b"collision");
        assert_eq!(std::fs::read(&dumpster).unwrap(), b"media");
        let _ = std::fs::remove_dir_all(root);
    }
}
