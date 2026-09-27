use std::net::SocketAddr;
use std::collections::HashSet;

use axum::extract::{ConnectInfo, Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{get, patch, post};
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
        .route("/", get(list_collections).post(create_collection))
        .route(
            "/{collection_id}",
            get(get_collection)
                .patch(update_collection)
                .delete(delete_collection),
        )
        .route(
            "/{collection_id}/items",
            post(add_items).delete(remove_items),
        )
        .route("/{collection_id}/items/reorder", patch(reorder_items))
}

#[derive(Deserialize)]
struct CollectionCreate {
    name: String,
    description: Option<String>,
    media_type: Option<String>,
}

#[derive(Deserialize)]
struct CollectionUpdate {
    name: Option<String>,
    description: Option<String>,
    cover_image_id: Option<i64>,
}

#[derive(Deserialize)]
struct CollectionItemsBody {
    image_ids: Vec<i64>,
}

#[derive(Deserialize)]
struct PaginationParams {
    page: Option<i64>,
    per_page: Option<i64>,
    media_type: Option<String>,
}

#[derive(Deserialize)]
struct CollectionListQuery {
    media_type: Option<String>,
}

fn validate_media_type(media_type: Option<&str>) -> Result<(), AppError> {
    if matches!(media_type, None | Some("image") | Some("video")) {
        Ok(())
    } else {
        Err(AppError::BadRequest("media_type must be image or video".into()))
    }
}

// Legacy collections have no type. Read their actual members so a mixed
// collection remains visible in both sections without changing membership.
fn matching_media_ids(state: &AppState, media_type: &str) -> Result<HashSet<i64>, AppError> {
    let extensions = match media_type {
        "video" => crate::routes::images::helpers::VIDEO_EXTENSIONS,
        _ => crate::routes::images::helpers::IMAGE_EXTENSIONS,
    };
    let quoted = extensions.iter().map(|ext| format!("'{}'", ext.trim_start_matches('.'))).collect::<Vec<_>>().join(",");
    let mut ids = HashSet::new();
    for directory_id in state.directory_db().get_all_directory_ids() {
        let pool = state.directory_db().get_pool(directory_id)?;
        let conn = pool.get()?;
        let sql = format!(
            "SELECT DISTINCT image_id FROM image_files WHERE file_extension IN ({}) AND file_status != 'missing' AND curation_discarded_at IS NULL",
            quoted
        );
        let mut stmt = conn.prepare(&sql)?;
        for id in stmt.query_map([], |row| row.get::<_, i64>(0))?.filter_map(Result::ok) {
            ids.insert(id);
        }
    }
    Ok(ids)
}

/// GET /api/collections
async fn list_collections(State(state): State<AppState>, Query(query): Query<CollectionListQuery>) -> Result<Json<Value>, AppError> {
    validate_media_type(query.media_type.as_deref())?;
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let matching_ids = query.media_type.as_deref().map(|media_type| matching_media_ids(&state_clone, media_type)).transpose()?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, cover_image_id, item_count, created_at, updated_at, media_type
             FROM collections ORDER BY COALESCE(updated_at, created_at) DESC",
        )?;

        let collections: Vec<Value> = stmt
            .query_map([], |row| {
                let collection_id: i64 = row.get(0)?;
                let declared_type: Option<String> = row.get(7)?;
                let cover_image_id: Option<i64> = row.get(3)?;
                let cover_thumbnail_url =
                    cover_image_id.map(|id| format!("/api/images/{}/thumbnail", id));
                Ok(json!({
                    "id": collection_id,
                    "name": row.get::<_, String>(1)?,
                    "description": row.get::<_, Option<String>>(2)?,
                    "cover_image_id": cover_image_id,
                    "cover_thumbnail_url": cover_thumbnail_url,
                    "item_count": row.get::<_, i64>(4)?,
                    "created_at": row.get::<_, Option<String>>(5)?,
                    "updated_at": row.get::<_, Option<String>>(6)?,
                    "media_type": declared_type,
                }))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|mut collection| {
                let Some(ref ids) = matching_ids else { return Some(collection) };
                let requested = query.media_type.as_deref().unwrap_or_default();
                if collection["media_type"].as_str().is_some_and(|kind| kind != requested) {
                    return None;
                }
                let collection_id = collection["id"].as_i64()?;
                let mut members = conn.prepare("SELECT image_id FROM collection_items WHERE collection_id = ?1").ok()?;
                let member_ids = members.query_map(params![collection_id], |row| row.get::<_, i64>(0)).ok()?;
                let matching = member_ids.filter_map(Result::ok).filter(|id| ids.contains(id)).collect::<Vec<_>>();
                if matching.is_empty() && collection["media_type"].is_null() {
                    return None;
                }
                collection["item_count"] = json!(matching.len());
                if !collection["cover_image_id"].as_i64().is_some_and(|id| ids.contains(&id)) {
                    collection["cover_image_id"] = json!(matching.first());
                    collection["cover_thumbnail_url"] = matching.first().map(|id| json!(format!("/api/images/{}/thumbnail", id))).unwrap_or(Value::Null);
                }
                Some(collection)
            })
            .collect();

        Ok::<_, AppError>(Json(json!({ "collections": collections })))
    })
    .await?
}

/// POST /api/collections
async fn create_collection(
    State(state): State<AppState>,
    Json(body): Json<CollectionCreate>,
) -> Result<Json<Value>, AppError> {
    validate_media_type(body.media_type.as_deref())?;
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO collections (name, description, media_type, item_count, created_at) VALUES (?1, ?2, ?3, 0, ?4)",
            params![&body.name, &body.description, &body.media_type, &now],
        )?;
        let id = conn.last_insert_rowid();
        Ok::<_, AppError>(Json(json!({
            "id": id,
            "name": body.name,
            "description": body.description,
            "media_type": body.media_type,
            "item_count": 0
        })))
    })
    .await?
}

/// GET /api/collections/:collection_id
async fn get_collection(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumPath(collection_id): AxumPath<i64>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Value>, AppError> {
    validate_media_type(params.media_type.as_deref())?;
    let page = params.page.unwrap_or(1).max(1);
    let per_page = params.per_page.unwrap_or(50).clamp(1, 200);
    let offset = (page - 1) * per_page;

    let client_ip = addr.ip();
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Build visible directory set for filtering
        let tier = AccessTier::from_ip(&client_ip);
        let family_locked = state_clone.is_family_mode_locked();
        let visible_dir_ids = get_visible_directory_ids(&conn, tier, family_locked)?;

        // Get collection info
        let collection = conn.query_row(
            "SELECT id, name, description, cover_image_id, item_count, created_at, updated_at, media_type FROM collections WHERE id = ?1",
            params![collection_id],
            |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "description": row.get::<_, Option<String>>(2)?,
                    "cover_image_id": row.get::<_, Option<i64>>(3)?,
                    "item_count": row.get::<_, i64>(4)?,
                    "created_at": row.get::<_, Option<String>>(5)?,
                    "updated_at": row.get::<_, Option<String>>(6)?,
                    "media_type": row.get::<_, Option<String>>(7)?,
                }))
            },
        ).map_err(|_| AppError::NotFound("Collection not found".into()))?;
        if let Some(ref requested) = params.media_type {
            if collection["media_type"].as_str().is_some_and(|kind| kind != requested) {
                return Err(AppError::NotFound("Collection not found in this library".into()));
            }
        }

        // Get item IDs
        let matching_ids = params.media_type.as_deref().map(|kind| matching_media_ids(&state_clone, kind)).transpose()?;
        let mut stmt = conn.prepare("SELECT ci.image_id FROM collection_items ci WHERE ci.collection_id = ?1 ORDER BY ci.sort_order")?;
        let all_ids: Vec<i64> = stmt
            .query_map(params![collection_id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        let visible_ids = all_ids.into_iter().filter(|id| matching_ids.as_ref().is_none_or(|matches| matches.contains(id))).collect::<Vec<_>>();
        let scoped_count = visible_ids.len();
        let image_ids = visible_ids.into_iter().skip(offset as usize).take(per_page as usize).collect::<Vec<_>>();

        // Hydrate image objects from directory DBs
        let library_id = state_clone.library_manager().primary().uuid.clone();
        let encoded_library_id =
            crate::routes::images::adjustments::encode_query_component(&library_id);
        let extension_clause = match params.media_type.as_deref() {
            Some("video") => Some(crate::routes::images::helpers::VIDEO_EXTENSIONS),
            Some("image") => Some(crate::routes::images::helpers::IMAGE_EXTENSIONS),
            _ => None,
        }.map(|extensions| extensions.iter().map(|ext| format!("'{}'", ext.trim_start_matches('.'))).collect::<Vec<_>>().join(","));
        let media_file_clause = extension_clause.as_ref().map(|extensions| format!("AND file_extension IN ({})", extensions)).unwrap_or_default();
        let image_sql = format!(
            "SELECT id, filename, file_hash, width, height, file_size, duration, rating, is_favorite, view_count, \
             (SELECT original_path FROM image_files WHERE image_id = images.id AND file_status != 'missing' {} LIMIT 1) \
             FROM images WHERE id = ?1 AND EXISTS (SELECT 1 FROM image_files WHERE image_id = images.id AND file_status != 'missing' {})",
            media_file_clause, media_file_clause
        );
        let mut images: Vec<Value> = Vec::new();
        for &img_id in &image_ids {
            let mut found = false;
            let all_dir_ids = state_clone.directory_db().get_all_directory_ids();
            for dir_id in &all_dir_ids {
                // Skip directories not visible to this client
                if let Some(ref visible_ids) = visible_dir_ids {
                    if !visible_ids.contains(dir_id) {
                        continue;
                    }
                }

                if !state_clone.directory_db().db_exists(*dir_id) {
                    continue;
                }
                let dir_pool = match state_clone.directory_db().get_pool(*dir_id) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let dir_conn = match dir_pool.get() {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if let Ok(img) = dir_conn.query_row(
                    &image_sql,
                    params![img_id],
                    |row| {
                        let hash: String = row.get(2)?;
                        let file_path: Option<String> = row.get(10)?;
                        let original_filename = file_path.as_deref().and_then(|path| std::path::Path::new(path).file_name()).and_then(|name| name.to_str());
                        Ok(json!({
                            "id": row.get::<_, i64>(0)?,
                            "filename": row.get::<_, String>(1)?,
                            "original_filename": original_filename,
                            "file_path": file_path,
                            "file_hash": hash.clone(),
                            "width": row.get::<_, Option<i32>>(3)?,
                            "height": row.get::<_, Option<i32>>(4)?,
                            "file_size": row.get::<_, Option<i64>>(5)?,
                            "duration": row.get::<_, Option<f64>>(6)?,
                            "rating": row.get::<_, String>(7)?,
                            "is_favorite": row.get::<_, bool>(8)?,
                            "view_count": row.get::<_, i32>(9)?,
                            "library_id": library_id.clone(),
                            "directory_id": dir_id,
                            "thumbnail_url": format!("/api/images/{}/thumbnail?directory_id={}&library_id={}&file_hash={}", img_id, dir_id, encoded_library_id, hash),
                            "url": format!("/api/images/{}/file?directory_id={}&library_id={}&file_hash={}", img_id, dir_id, encoded_library_id, hash),
                        }))
                    },
                ) {
                    images.push(img);
                    found = true;
                    break;
                }
            }
            if !found && params.media_type.is_none() {
                // Fallback: include stub with ID so frontend knows something exists
                images.push(json!({"id": img_id}));
            }
        }

        let mut result = collection;
        result["images"] = json!(images);
        result["item_count"] = json!(scoped_count);
        result["page"] = json!(page);
        result["per_page"] = json!(per_page);
        result["has_more"] = json!((offset as usize + image_ids.len()) < scoped_count);

        Ok::<_, AppError>(Json(result))
    })
    .await?
}

/// PATCH /api/collections/:collection_id
async fn update_collection(
    State(state): State<AppState>,
    AxumPath(collection_id): AxumPath<i64>,
    Json(body): Json<CollectionUpdate>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let now = chrono::Utc::now().to_rfc3339();

        let mut sets = Vec::new();
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        // Always update timestamp
        sql_params.push(Box::new(now));
        sets.push(format!("updated_at = ?{}", sql_params.len()));

        if let Some(name) = &body.name {
            sql_params.push(Box::new(name.clone()));
            sets.push(format!("name = ?{}", sql_params.len()));
        }
        if let Some(desc) = &body.description {
            sql_params.push(Box::new(desc.clone()));
            sets.push(format!("description = ?{}", sql_params.len()));
        }
        if let Some(cover) = body.cover_image_id {
            sql_params.push(Box::new(cover));
            sets.push(format!("cover_image_id = ?{}", sql_params.len()));
        }

        sql_params.push(Box::new(collection_id));
        let sql = format!(
            "UPDATE collections SET {} WHERE id = ?{}",
            sets.join(", "),
            sql_params.len()
        );

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        conn.execute(&sql, param_refs.as_slice())?;

        Ok::<_, AppError>(Json(json!({ "success": true })))
    })
    .await?
}

/// DELETE /api/collections/:collection_id
async fn delete_collection(
    State(state): State<AppState>,
    AxumPath(collection_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        conn.execute(
            "DELETE FROM collection_items WHERE collection_id = ?1",
            params![collection_id],
        )?;
        conn.execute(
            "DELETE FROM collections WHERE id = ?1",
            params![collection_id],
        )?;
        Ok::<_, AppError>(Json(json!({ "success": true })))
    })
    .await?
}

/// POST /api/collections/:collection_id/items
async fn add_items(
    State(state): State<AppState>,
    AxumPath(collection_id): AxumPath<i64>,
    Json(body): Json<CollectionItemsBody>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let collection_type: Option<String> = conn.query_row(
            "SELECT media_type FROM collections WHERE id = ?1",
            params![collection_id],
            |row| row.get(0),
        ).map_err(|_| AppError::NotFound("Collection not found".into()))?;
        if let Some(ref media_type) = collection_type {
            let matching = matching_media_ids(&state_clone, media_type)?;
            if body.image_ids.iter().any(|id| !matching.contains(id)) {
                return Err(AppError::BadRequest(format!("Collection accepts {} media only", media_type)));
            }
        }

        // Get current max sort order
        let max_order: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(sort_order), 0) FROM collection_items WHERE collection_id = ?1",
                params![collection_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let mut added = 0i64;
        for (i, image_id) in body.image_ids.iter().enumerate() {
            // Check if already in collection
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM collection_items WHERE collection_id = ?1 AND image_id = ?2",
                    params![collection_id, image_id],
                    |row| row.get::<_, i64>(0).map(|c| c > 0),
                )
                .unwrap_or(false);

            if exists {
                continue;
            }

            conn.execute(
                "INSERT INTO collection_items (collection_id, image_id, sort_order) VALUES (?1, ?2, ?3)",
                params![collection_id, image_id, max_order + i as i64 + 1],
            )?;
            added += 1;
        }

        if added > 0 {
            conn.execute(
                "UPDATE collections SET item_count = item_count + ?1, updated_at = ?2 WHERE id = ?3",
                params![added, chrono::Utc::now().to_rfc3339(), collection_id],
            )?;

            // Auto-set cover if none set
            if let Some(first_id) = body.image_ids.first() {
                conn.execute(
                    "UPDATE collections SET cover_image_id = ?1 WHERE id = ?2 AND cover_image_id IS NULL",
                    params![first_id, collection_id],
                )?;
            }
        }

        Ok::<_, AppError>(Json(json!({ "success": true, "added": added })))
    })
    .await?
}

/// DELETE /api/collections/:collection_id/items
async fn remove_items(
    State(state): State<AppState>,
    AxumPath(collection_id): AxumPath<i64>,
    Json(body): Json<CollectionItemsBody>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        for image_id in &body.image_ids {
            conn.execute(
                "DELETE FROM collection_items WHERE collection_id = ?1 AND image_id = ?2",
                params![collection_id, image_id],
            )?;
        }

        // Update item count
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM collection_items WHERE collection_id = ?1",
                params![collection_id],
                |row| row.get(0),
            )
            .unwrap_or(0);
        conn.execute(
            "UPDATE collections SET item_count = ?1, updated_at = ?2 WHERE id = ?3",
            params![count, chrono::Utc::now().to_rfc3339(), collection_id],
        )?;

        Ok::<_, AppError>(Json(json!({ "success": true })))
    })
    .await?
}

/// POST /api/collections/:collection_id/items/reorder
async fn reorder_items(
    State(state): State<AppState>,
    AxumPath(collection_id): AxumPath<i64>,
    Json(body): Json<CollectionItemsBody>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        for (i, image_id) in body.image_ids.iter().enumerate() {
            conn.execute(
                "UPDATE collection_items SET sort_order = ?1 WHERE collection_id = ?2 AND image_id = ?3",
                params![i as i64, collection_id, image_id],
            )?;
        }
        Ok::<_, AppError>(Json(json!({ "success": true })))
    })
    .await?
}
