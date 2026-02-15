use axum::extract::{Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{delete, get};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", delete(clear_all))
        .route(
            "/{image_id}",
            get(get_position).post(save_position).delete(delete_history),
        )
        .route("/continue-watching", get(continue_watching))
}

#[derive(Deserialize)]
struct SavePositionBody {
    playback_position: f64,
    duration: f64,
}

#[derive(Deserialize)]
struct ContinueWatchingQuery {
    #[serde(default = "default_limit")]
    limit: i64,
}

fn default_limit() -> i64 {
    20
}

/// POST /api/watch-history/{image_id} — Save/update playback position.
///
/// Uses INSERT OR REPLACE with the UNIQUE(image_id) constraint to upsert.
/// Automatically marks as completed if playback_position / duration >= 0.9.
async fn save_position(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Json(body): Json<SavePositionBody>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let completed = if body.duration > 0.0 {
            body.playback_position / body.duration >= 0.9
        } else {
            false
        };

        conn.execute(
            "INSERT OR REPLACE INTO watch_history (image_id, playback_position, duration, completed, last_watched)
             VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)",
            params![image_id, body.playback_position, body.duration, completed],
        )?;

        let progress = if body.duration > 0.0 {
            body.playback_position / body.duration
        } else {
            0.0
        };

        Ok::<_, AppError>(json!({
            "image_id": image_id,
            "playback_position": body.playback_position,
            "duration": body.duration,
            "progress": progress,
            "completed": completed
        }))
    })
    .await??;

    Ok(Json(result))
}

/// GET /api/watch-history/continue-watching — List videos with partial progress.
///
/// Returns watch history entries that are not completed, ordered by most recently watched.
/// Only returns playback info (image_id, position, duration, progress); the frontend
/// hydrates image details separately since images live in per-directory databases.
async fn continue_watching(
    State(state): State<AppState>,
    Query(params): Query<ContinueWatchingQuery>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    let limit = params.limit.clamp(1, 100);

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let mut stmt = conn.prepare(
            "SELECT image_id, playback_position, duration, completed, last_watched
             FROM watch_history
             WHERE completed = 0 AND playback_position > 0
             ORDER BY last_watched DESC
             LIMIT ?1",
        )?;

        let items: Vec<Value> = stmt
            .query_map(params![limit], |row| {
                let position: f64 = row.get(1)?;
                let duration: f64 = row.get(2)?;
                let progress = if duration > 0.0 {
                    position / duration
                } else {
                    0.0
                };
                Ok(json!({
                    "image_id": row.get::<_, i64>(0)?,
                    "playback_position": position,
                    "duration": duration,
                    "progress": progress,
                    "completed": row.get::<_, bool>(3)?,
                    "last_watched": row.get::<_, Option<String>>(4)?
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok::<_, AppError>(json!({
            "items": items,
            "total": items.len()
        }))
    })
    .await??;

    Ok(Json(result))
}

/// GET /api/watch-history/{image_id} — Get playback position for a video.
async fn get_position(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        conn.query_row(
            "SELECT image_id, playback_position, duration, completed, last_watched
             FROM watch_history
             WHERE image_id = ?1",
            params![image_id],
            |row| {
                let position: f64 = row.get(1)?;
                let duration: f64 = row.get(2)?;
                let progress = if duration > 0.0 {
                    position / duration
                } else {
                    0.0
                };
                Ok(json!({
                    "image_id": row.get::<_, i64>(0)?,
                    "playback_position": position,
                    "duration": duration,
                    "progress": progress,
                    "completed": row.get::<_, bool>(3)?,
                    "last_watched": row.get::<_, Option<String>>(4)?
                }))
            },
        )
        .map_err(|_| AppError::NotFound(format!("No watch history for image {}", image_id)))
    })
    .await??;

    Ok(Json(result))
}

/// DELETE /api/watch-history/{image_id} — Remove watch history for a video.
async fn delete_history(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let deleted = conn.execute(
            "DELETE FROM watch_history WHERE image_id = ?1",
            params![image_id],
        )?;

        if deleted == 0 {
            return Err(AppError::NotFound(format!(
                "No watch history for image {}",
                image_id
            )));
        }

        Ok::<_, AppError>(json!({ "success": true }))
    })
    .await??;

    Ok(Json(result))
}

/// DELETE /api/watch-history — Clear all watch history.
async fn clear_all(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let deleted = conn.execute("DELETE FROM watch_history", [])?;

        Ok::<_, AppError>(json!({
            "success": true,
            "deleted": deleted
        }))
    })
    .await??;

    Ok(Json(result))
}
