use axum::extract::{Path as AxumPath, Query, State};
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_collections).post(create_collection))
        .route(
            "/{collection_id}",
            get(get_collection)
                .patch(update_collection)
                .delete(delete_collection),
        )
        .route("/{collection_id}/items", post(add_items).delete(remove_items))
        .route("/{collection_id}/items/reorder", post(reorder_items))
}

#[derive(Deserialize)]
struct CollectionCreate {
    name: String,
    description: Option<String>,
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
}

/// GET /api/collections
async fn list_collections(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let mut stmt = conn.prepare(
            "SELECT id, name, description, cover_image_id, item_count, created_at, updated_at
             FROM collections ORDER BY COALESCE(updated_at, created_at) DESC",
        )?;

        let collections: Vec<Value> = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "description": row.get::<_, Option<String>>(2)?,
                    "cover_image_id": row.get::<_, Option<i64>>(3)?,
                    "item_count": row.get::<_, i64>(4)?,
                    "created_at": row.get::<_, Option<String>>(5)?,
                    "updated_at": row.get::<_, Option<String>>(6)?
                }))
            })?
            .filter_map(|r| r.ok())
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
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO collections (name, description, item_count, created_at) VALUES (?1, ?2, 0, ?3)",
            params![&body.name, &body.description, &now],
        )?;
        let id = conn.last_insert_rowid();
        Ok::<_, AppError>(Json(json!({
            "id": id,
            "name": body.name,
            "description": body.description,
            "item_count": 0
        })))
    })
    .await?
}

/// GET /api/collections/:collection_id
async fn get_collection(
    State(state): State<AppState>,
    AxumPath(collection_id): AxumPath<i64>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Value>, AppError> {
    let page = params.page.unwrap_or(1).max(1);
    let per_page = params.per_page.unwrap_or(50).clamp(1, 200);
    let offset = (page - 1) * per_page;

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Get collection info
        let collection = conn.query_row(
            "SELECT id, name, description, cover_image_id, item_count, created_at, updated_at FROM collections WHERE id = ?1",
            params![collection_id],
            |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "description": row.get::<_, Option<String>>(2)?,
                    "cover_image_id": row.get::<_, Option<i64>>(3)?,
                    "item_count": row.get::<_, i64>(4)?,
                    "created_at": row.get::<_, Option<String>>(5)?,
                    "updated_at": row.get::<_, Option<String>>(6)?
                }))
            },
        ).map_err(|_| AppError::NotFound("Collection not found".into()))?;

        // Get items
        let mut stmt = conn.prepare(
            "SELECT ci.image_id FROM collection_items ci
             WHERE ci.collection_id = ?1
             ORDER BY ci.sort_order
             LIMIT ?2 OFFSET ?3",
        )?;
        let image_ids: Vec<i64> = stmt
            .query_map(params![collection_id, per_page, offset], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut result = collection;
        result["images"] = json!(image_ids);
        result["page"] = json!(page);
        result["per_page"] = json!(per_page);
        result["has_more"] = json!(image_ids.len() as i64 == per_page);

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

        let mut sets = vec!["updated_at = ?1"];
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(now)];

        if let Some(name) = &body.name {
            sets.push("name = ?");
            sql_params.push(Box::new(name.clone()));
        }
        if let Some(desc) = &body.description {
            sets.push("description = ?");
            sql_params.push(Box::new(desc.clone()));
        }
        if let Some(cover) = body.cover_image_id {
            sets.push("cover_image_id = ?");
            sql_params.push(Box::new(cover));
        }

        sql_params.push(Box::new(collection_id));

        // Rebuild with positional params
        let mut parts = Vec::new();
        for (i, set) in sets.iter().enumerate() {
            if set.contains('?') && !set.contains("?1") {
                parts.push(set.replace('?', &format!("?{}", i + 1)));
            } else {
                parts.push(set.to_string());
            }
        }
        let sql = format!(
            "UPDATE collections SET {} WHERE id = ?{}",
            parts.join(", "),
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
