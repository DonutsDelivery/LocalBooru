use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{delete, get};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::middleware::AccessTier;
use crate::server::state::AppState;
use crate::server::utils::get_visible_directory_ids;

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
    #[serde(alias = "position", default)]
    playback_position: f64,
    #[serde(default)]
    duration: f64,
    #[serde(default)]
    directory_id: Option<i64>,
    #[serde(default)]
    library_id: Option<String>,
}

#[derive(Default, Deserialize)]
struct ImageLocatorQuery {
    directory_id: Option<i64>,
    library_id: Option<String>,
}

fn image_exists(
    library: &crate::db::library::LibraryContext,
    directory_id: i64,
    image_id: i64,
) -> bool {
    library
        .directory_db
        .get_pool(directory_id)
        .ok()
        .and_then(|pool| pool.get().ok())
        .is_some_and(|conn| {
            conn.query_row(
                "SELECT 1 FROM images WHERE id = ?1",
                params![image_id],
                |_| Ok(()),
            )
            .is_ok()
        })
}

fn resolve_library(
    state: &AppState,
    library_id: &str,
) -> Option<std::sync::Arc<crate::db::library::LibraryContext>> {
    let primary = state.library_manager().primary();
    if library_id == "primary" || library_id == primary.uuid {
        Some(primary.clone())
    } else {
        state.resolve_library(Some(library_id)).ok()
    }
}

fn exact_or_unique_image_locator(
    state: &AppState,
    library_id: Option<&str>,
    directory_id: Option<i64>,
    image_id: i64,
) -> Option<(String, i64)> {
    let libraries = match library_id {
        Some(library_id) => vec![resolve_library(state, library_id)?],
        None => state.library_manager().all_mounted(),
    };
    let mut matches = Vec::new();
    for library in libraries {
        let directory_ids = directory_id
            .map(|id| vec![id])
            .unwrap_or_else(|| library.directory_db.get_all_directory_ids());
        for candidate_directory in directory_ids {
            if image_exists(&library, candidate_directory, image_id) {
                matches.push((library.uuid.clone(), candidate_directory));
            }
        }
    }
    (matches.len() == 1).then(|| matches.remove(0))
}

fn legacy_history_matches_locator(
    state: &AppState,
    image_id: i64,
    library_id: &str,
    directory_id: i64,
) -> (bool, bool) {
    let Ok(conn) = state.main_db().get() else {
        return (false, false);
    };
    let Ok(mut statement) = conn.prepare(
        "SELECT directory_id FROM watch_history WHERE library_id IS NULL AND image_id = ?1",
    ) else {
        return (false, false);
    };
    let Ok(rows) = statement.query_map(params![image_id], |row| row.get::<_, Option<i64>>(0))
    else {
        return (false, false);
    };
    let target = (library_id.to_string(), directory_id);
    let mut matches = (false, false);
    for stored_directory in rows.filter_map(Result::ok) {
        if exact_or_unique_image_locator(state, None, stored_directory, image_id)
            .is_some_and(|locator| locator == target)
        {
            if stored_directory.is_some() {
                matches.0 = true;
            } else {
                matches.1 = true;
            }
        }
    }
    matches
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
/// Upserts by exact library/directory/image identity.
/// Automatically marks as completed if playback_position / duration >= 0.9.
async fn save_position(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
    Json(body): Json<SavePositionBody>,
) -> Result<Json<Value>, AppError> {
    // Silently skip empty updates (no playback position or duration).
    // This handles cleanup-triggered saves when navigating away from videos
    // while the media element is already reset.
    if body.playback_position <= 0.0 && body.duration <= 0.0 {
        return Ok(Json(json!({
            "image_id": image_id,
            "playback_position": 0.0,
            "duration": 0.0,
            "progress": 0.0,
            "completed": false,
            "skipped": true
        })));
    }
    if locator.directory_id.is_some()
        && body.directory_id.is_some()
        && locator.directory_id != body.directory_id
    {
        return Err(AppError::BadRequest(
            "Watch history directory locator does not match request body".into(),
        ));
    }
    if locator.library_id.is_some()
        && body.library_id.is_some()
        && locator.library_id != body.library_id
    {
        return Err(AppError::BadRequest(
            "Watch history library locator does not match request body".into(),
        ));
    }
    let requested_directory = locator.directory_id.or(body.directory_id);
    let requested_library = locator.library_id.or(body.library_id);
    let (library_id, directory_id) = exact_or_unique_image_locator(
        &state,
        requested_library.as_deref(),
        requested_directory,
        image_id,
    )
    .ok_or_else(|| {
        AppError::NotFound(format!(
            "Image {} was not found at the requested watch-history locator",
            image_id
        ))
    })?;
    let (adopt_legacy_directory, adopt_legacy_without_directory) =
        legacy_history_matches_locator(&state, image_id, &library_id, directory_id);
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let completed = if body.duration > 0.0 {
            body.playback_position / body.duration >= 0.9
        } else {
            false
        };

        conn.execute(
            "DELETE FROM watch_history
             WHERE library_id IS NULL AND image_id = ?1 AND (
                 (?2 AND directory_id = ?4)
                 OR (?3 AND directory_id IS NULL)
             )",
            params![
                image_id,
                adopt_legacy_directory,
                adopt_legacy_without_directory,
                directory_id
            ],
        )?;
        conn.execute(
            "INSERT INTO watch_history
                 (library_id, directory_id, image_id, playback_position, duration, completed, last_watched)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, CURRENT_TIMESTAMP)
             ON CONFLICT(library_id, directory_id, image_id) DO UPDATE SET
                 playback_position = excluded.playback_position,
                 duration = excluded.duration,
                 completed = excluded.completed,
                 last_watched = CURRENT_TIMESTAMP",
            params![
                library_id,
                directory_id,
                image_id,
                body.playback_position,
                body.duration,
                completed
            ],
        )?;

        let progress = if body.duration > 0.0 {
            body.playback_position / body.duration
        } else {
            0.0
        };

        Ok::<_, AppError>(json!({
            "image_id": image_id,
            "library_id": library_id,
            "directory_id": directory_id,
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
async fn continue_watching(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(params): Query<ContinueWatchingQuery>,
) -> Result<Json<Value>, AppError> {
    let client_ip = addr.ip();
    let state_clone = state.clone();
    let limit = params.limit.clamp(1, 100);

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let mut stmt = conn.prepare(
            "SELECT image_id, playback_position, duration, completed, last_watched,
                    directory_id, library_id
             FROM watch_history
             WHERE completed = 0 AND playback_position > 0
             ORDER BY last_watched DESC",
        )?;
        let rows: Vec<(
            i64,
            f64,
            f64,
            bool,
            Option<String>,
            Option<i64>,
            Option<String>,
        )> = stmt
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
            .filter_map(Result::ok)
            .collect();
        drop(stmt);
        drop(conn);

        let tier = AccessTier::from_ip(&client_ip);
        let family_locked = state_clone.is_family_mode_locked();
        let mut items = Vec::new();
        for (
            image_id,
            position,
            duration,
            completed,
            last_watched,
            stored_directory,
            stored_library,
        ) in rows
        {
            let Some((library_id, directory_id)) = exact_or_unique_image_locator(
                &state_clone,
                stored_library.as_deref(),
                stored_directory,
                image_id,
            ) else {
                continue;
            };
            let primary = state_clone.library_manager().primary();
            let library = if library_id == primary.uuid {
                primary.clone()
            } else if let Ok(library) = state_clone.resolve_library(Some(&library_id)) {
                library
            } else {
                continue;
            };
            let visible = {
                let main_conn = library.main_pool.get()?;
                get_visible_directory_ids(&main_conn, tier, family_locked)?
            };
            if visible
                .as_ref()
                .is_some_and(|directories| !directories.contains(&directory_id))
            {
                continue;
            }
            let progress = if duration > 0.0 {
                position / duration
            } else {
                0.0
            };
            items.push(json!({
                "image_id": image_id,
                "library_id": library_id,
                "directory_id": directory_id,
                "playback_position": position,
                "duration": duration,
                "progress": progress,
                "completed": completed,
                "last_watched": last_watched,
            }));
            if items.len() >= limit as usize {
                break;
            }
        }

        Ok::<_, AppError>(json!({
            "total": items.len(),
            "items": items,
        }))
    })
    .await??;

    Ok(Json(result))
}

/// GET /api/watch-history/{image_id} — Get playback position for a video.
async fn get_position(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
) -> Result<Json<Value>, AppError> {
    let (library_id, directory_id) = exact_or_unique_image_locator(
        &state,
        locator.library_id.as_deref(),
        locator.directory_id,
        image_id,
    )
    .ok_or_else(|| AppError::NotFound(format!("Image {} was not found", image_id)))?;
    let (include_legacy_directory, include_legacy_without_directory) =
        legacy_history_matches_locator(&state, image_id, &library_id, directory_id);
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let result = conn.query_row(
            "SELECT image_id, playback_position, duration, completed, last_watched
             FROM watch_history
             WHERE image_id = ?3 AND (
                 (library_id = ?1 AND directory_id = ?2)
                 OR (?4 AND library_id IS NULL AND directory_id = ?2)
                 OR (?5 AND library_id IS NULL AND directory_id IS NULL)
             )
             ORDER BY library_id IS NULL ASC
             LIMIT 1",
            params![
                library_id,
                directory_id,
                image_id,
                include_legacy_directory,
                include_legacy_without_directory
            ],
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
                    "library_id": library_id,
                    "directory_id": directory_id,
                    "playback_position": position,
                    "duration": duration,
                    "progress": progress,
                    "completed": row.get::<_, bool>(3)?,
                    "last_watched": row.get::<_, Option<String>>(4)?
                }))
            },
        );

        match result {
            Ok(val) => Ok(val),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                // No history yet — return defaults
                Ok(json!({
                    "image_id": image_id,
                    "library_id": library_id,
                    "directory_id": directory_id,
                    "playback_position": 0.0,
                    "duration": 0.0,
                    "progress": 0.0,
                    "completed": false,
                    "last_watched": null
                }))
            }
            Err(e) => Err(AppError::Internal(e.to_string())),
        }
    })
    .await??;

    Ok(Json(result))
}

/// DELETE /api/watch-history/{image_id} — Remove watch history for a video.
async fn delete_history(
    State(state): State<AppState>,
    AxumPath(image_id): AxumPath<i64>,
    Query(locator): Query<ImageLocatorQuery>,
) -> Result<Json<Value>, AppError> {
    let (library_id, directory_id) = match (locator.library_id.as_deref(), locator.directory_id) {
        (Some(library_id), Some(directory_id)) => {
            let library = resolve_library(&state, library_id).ok_or_else(|| {
                AppError::NotFound(format!("Library {} was not found", library_id))
            })?;
            (library.uuid.clone(), directory_id)
        }
        _ => exact_or_unique_image_locator(
            &state,
            locator.library_id.as_deref(),
            locator.directory_id,
            image_id,
        )
        .ok_or_else(|| AppError::NotFound(format!("Image {} was not found", image_id)))?,
    };
    let (include_legacy_directory, include_legacy_without_directory) =
        legacy_history_matches_locator(&state, image_id, &library_id, directory_id);
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let deleted = conn.execute(
            "DELETE FROM watch_history
             WHERE image_id = ?3 AND (
                 (library_id = ?1 AND directory_id = ?2)
                 OR (?4 AND library_id IS NULL AND directory_id = ?2)
                 OR (?5 AND library_id IS NULL AND directory_id IS NULL)
             )",
            params![
                library_id,
                directory_id,
                image_id,
                include_legacy_directory,
                include_legacy_without_directory
            ],
        )?;

        if deleted == 0 {
            return Err(AppError::NotFound(format!(
                "No watch history for image {}",
                image_id
            )));
        }

        Ok::<_, AppError>(json!({
            "success": true,
            "library_id": library_id,
            "directory_id": directory_id,
            "image_id": image_id,
        }))
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

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use axum::body::{to_bytes, Body};
    use axum::extract::ConnectInfo;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt;

    use super::*;

    fn request(method: Method, uri: &str, body: serde_json::Value) -> Request<Body> {
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 50000))));
        request
    }

    fn insert_image(
        library: &crate::db::library::LibraryContext,
        directory_id: i64,
        image_id: i64,
    ) {
        let pool = library.directory_db.get_pool(directory_id).unwrap();
        pool.get()
            .unwrap()
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (?1, 'video.mp4', ?2)",
                rusqlite::params![image_id, format!("hash-{}", library.uuid)],
            )
            .unwrap();
    }

    // AC: @identity-safe-image-adjustments ac-canonical-entry
    #[tokio::test]
    async fn watch_history_keeps_colliding_library_images_independent() {
        let root =
            std::env::temp_dir().join(format!("localbooru-watch-history-{}", uuid::Uuid::new_v4()));
        let primary_root = root.join("primary");
        let secondary_root = root.join("secondary");
        std::fs::create_dir_all(&primary_root).unwrap();
        std::fs::create_dir_all(&secondary_root).unwrap();
        std::fs::write(
            secondary_root.join("settings.json"),
            r#"{"library_uuid":"secondary"}"#,
        )
        .unwrap();

        let state = AppState::new(&primary_root, 0).unwrap();
        let primary = state.library_manager().primary().clone();
        insert_image(&primary, 1, 12);
        let secondary =
            crate::db::library::LibraryContext::open(&secondary_root, "Secondary").unwrap();
        insert_image(&secondary, 1, 12);
        state.library_manager().mount(secondary);
        let primary_id = primary.uuid.clone();
        assert_eq!(
            exact_or_unique_image_locator(&state, Some("secondary"), None, 12),
            Some(("secondary".to_string(), 1))
        );
        let app = router().with_state(state.clone());

        for (library_id, position) in [(primary_id.as_str(), 4.0), ("secondary", 8.0)] {
            let response = app
                .clone()
                .oneshot(request(
                    Method::POST,
                    &format!("/12?directory_id=1&library_id={}", library_id),
                    json!({
                        "position": position,
                        "duration": 20.0,
                        "directory_id": 1,
                        "library_id": library_id,
                    }),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        state
            .main_db()
            .get()
            .unwrap()
            .execute(
                "INSERT INTO watch_history
                 (library_id, directory_id, image_id, playback_position, duration, completed, last_watched)
                 VALUES ('missing-library', 1, 999, 5, 20, 0, '9999-12-31 23:59:59')",
                [],
            )
            .unwrap();
        let response = app
            .clone()
            .oneshot(request(
                Method::GET,
                "/continue-watching?limit=1",
                serde_json::Value::Null,
            ))
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["items"].as_array().unwrap().len(), 1);
        assert_ne!(body["items"][0]["image_id"], 999);

        let response = app
            .clone()
            .oneshot(request(
                Method::GET,
                "/continue-watching",
                serde_json::Value::Null,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["items"].as_array().unwrap().len(), 2);
        let libraries: std::collections::HashSet<_> = body["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["library_id"].as_str().unwrap())
            .collect();
        assert!(libraries.contains(primary_id.as_str()));
        assert!(libraries.contains("secondary"));

        let response = app
            .clone()
            .oneshot(request(
                Method::DELETE,
                &format!("/12?directory_id=1&library_id={}", primary_id),
                serde_json::Value::Null,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .clone()
            .oneshot(request(
                Method::GET,
                "/continue-watching",
                serde_json::Value::Null,
            ))
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["items"].as_array().unwrap().len(), 1);
        assert_eq!(body["items"][0]["library_id"], "secondary");

        let conn = state.main_db().get().unwrap();
        conn.execute(
            "INSERT INTO watch_history
             (library_id, directory_id, image_id, playback_position, duration, completed)
             VALUES (NULL, 1, 12, 6, 20, 0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO watch_history
             (library_id, directory_id, image_id, playback_position, duration, completed)
             VALUES (NULL, NULL, 12, 7, 20, 0)",
            [],
        )
        .unwrap();
        drop(conn);
        let response = app
            .clone()
            .oneshot(request(
                Method::DELETE,
                "/12?directory_id=1&library_id=secondary",
                serde_json::Value::Null,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let legacy_count: i64 = state
            .main_db()
            .get()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM watch_history WHERE library_id IS NULL AND image_id = 12",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(legacy_count, 2);

        let response = app
            .oneshot(request(
                Method::GET,
                &format!("/12?directory_id=1&library_id={}", primary_id),
                serde_json::Value::Null,
            ))
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["playback_position"], 0.0);

        let _ = std::fs::remove_dir_all(root);
    }
}
