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
    match std::fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(source, destination)
                .map_err(|error| io_error("Failed to copy media to dumpster", error))?;
            if let Err(error) = std::fs::remove_file(source) {
                let _ = std::fs::remove_file(destination);
                return Err(io_error("Failed to remove original media", error));
            }
            Ok(())
        }
    }
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

        let mut completed: Vec<(PathBuf, PathBuf)> = Vec::new();
        for (_, source, destination) in &moves {
            if let Err(error) = move_file(source, destination) {
                for (original, dumped) in completed.iter().rev() {
                    let _ = move_file(dumped, original);
                }
                return Err(error);
            }
            completed.push((source.clone(), destination.clone()));
        }

        let tx = conn.transaction()?;
        for (file_id, source, destination) in &moves {
            tx.execute(
                "UPDATE image_files
                 SET original_path = ?1, curation_original_path = ?2,
                     curation_discarded_at = datetime('now')
                 WHERE id = ?3",
                params![
                    destination.to_string_lossy(),
                    source.to_string_lossy(),
                    file_id
                ],
            )?;
        }
        if let Err(error) = tx.commit() {
            for (original, dumped) in completed.iter().rev() {
                let _ = move_file(dumped, original);
            }
            return Err(error.into());
        }
        Ok(moves.len())
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
        if let Some((_, _, path)) = moves.iter().find(|(_, _, original)| original.exists()) {
            return Err(AppError::BadRequest(format!(
                "Cannot restore because the original path already exists: {}",
                path.display()
            )));
        }

        let mut completed: Vec<(PathBuf, PathBuf)> = Vec::new();
        for (_, dumped, original) in &moves {
            if let Err(error) = move_file(dumped, original) {
                for (dumped_path, original_path) in completed.iter().rev() {
                    let _ = move_file(original_path, dumped_path);
                }
                return Err(error);
            }
            completed.push((dumped.clone(), original.clone()));
        }

        let tx = conn.transaction()?;
        for (file_id, _, original) in &moves {
            tx.execute(
                "UPDATE image_files
                 SET original_path = ?1, curation_original_path = NULL,
                     curation_discarded_at = NULL
                 WHERE id = ?2",
                params![original.to_string_lossy(), file_id],
            )?;
        }
        if let Err(error) = tx.commit() {
            for (dumped, original) in completed.iter().rev() {
                let _ = move_file(original, dumped);
            }
            return Err(error.into());
        }
        Ok(moves.len())
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
