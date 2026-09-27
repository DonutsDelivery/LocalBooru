use std::collections::HashSet;
use std::net::SocketAddr;

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

#[derive(Clone, Deserialize)]
struct CollectionItemLocator {
    image_id: i64,
    directory_id: i64,
    library_id: String,
}

#[derive(Deserialize)]
struct CollectionItemsBody {
    #[serde(default)]
    image_ids: Vec<i64>,
    #[serde(default)]
    items: Vec<CollectionItemLocator>,
}

#[derive(Clone)]
struct CollectionMember {
    image_id: i64,
    directory_id: Option<i64>,
    library_id: Option<String>,
}

impl CollectionMember {
    fn locator(&self) -> Option<CollectionItemLocator> {
        Some(CollectionItemLocator {
            image_id: self.image_id,
            directory_id: self.directory_id?,
            library_id: self.library_id.clone()?,
        })
    }
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
        Err(AppError::BadRequest(
            "media_type must be image or video".into(),
        ))
    }
}

// Legacy collections have no type. Read their actual members so a mixed
// collection remains visible in both sections without changing membership.
struct MediaMatches {
    exact: HashSet<(String, i64, i64)>,
    legacy_ids: HashSet<i64>,
}

impl MediaMatches {
    fn includes(&self, member: &CollectionMember) -> bool {
        if let Some(locator) = member.locator() {
            self.exact
                .contains(&(locator.library_id, locator.directory_id, locator.image_id))
        } else {
            self.legacy_ids.contains(&member.image_id)
        }
    }

    fn includes_locator(&self, locator: &CollectionItemLocator) -> bool {
        self.exact.contains(&(
            locator.library_id.clone(),
            locator.directory_id,
            locator.image_id,
        ))
    }
}

fn matching_media_ids(state: &AppState, media_type: &str) -> Result<MediaMatches, AppError> {
    let extensions = match media_type {
        "video" => crate::routes::images::helpers::VIDEO_EXTENSIONS,
        _ => crate::routes::images::helpers::IMAGE_EXTENSIONS,
    };
    let quoted = extensions
        .iter()
        .map(|ext| format!("'{}'", ext.trim_start_matches('.')))
        .collect::<Vec<_>>()
        .join(",");
    let mut matches = MediaMatches {
        exact: HashSet::new(),
        legacy_ids: HashSet::new(),
    };
    for library in state.library_manager().all_mounted() {
        for directory_id in library.directory_db.get_all_directory_ids() {
            let pool = library.directory_db.get_pool(directory_id)?;
            let conn = pool.get()?;
            let sql = format!(
                "SELECT DISTINCT image_id FROM image_files WHERE file_extension IN ({}) AND file_status != 'missing' AND curation_discarded_at IS NULL",
                quoted
            );
            let mut stmt = conn.prepare(&sql)?;
            for id in stmt
                .query_map([], |row| row.get::<_, i64>(0))?
                .filter_map(Result::ok)
            {
                matches
                    .exact
                    .insert((library.uuid.clone(), directory_id, id));
                matches.legacy_ids.insert(id);
            }
        }
    }
    Ok(matches)
}

fn collection_members(
    conn: &rusqlite::Connection,
    collection_id: i64,
) -> Result<Vec<CollectionMember>, AppError> {
    let mut stmt = conn.prepare("SELECT image_id, directory_id, library_id FROM collection_items WHERE collection_id = ?1 ORDER BY sort_order")?;
    let members = stmt
        .query_map(params![collection_id], |row| {
            Ok(CollectionMember {
                image_id: row.get(0)?,
                directory_id: row.get(1)?,
                library_id: row.get(2)?,
            })
        })?
        .filter_map(Result::ok)
        .collect();
    Ok(members)
}

fn member_thumbnail_url(member: &CollectionMember) -> String {
    if let Some(locator) = member.locator() {
        let library =
            crate::routes::images::adjustments::encode_query_component(&locator.library_id);
        format!(
            "/api/images/{}/thumbnail?directory_id={}&library_id={}",
            locator.image_id, locator.directory_id, library
        )
    } else {
        format!("/api/images/{}/thumbnail", member.image_id)
    }
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::extract::ConnectInfo;
    use axum::http::{Request, StatusCode};
    use std::net::SocketAddr;
    use tower::ServiceExt;

    use super::*;

    fn request(uri: &str) -> Request<Body> {
        let mut request = Request::builder().uri(uri).body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 50000))));
        request
    }

    #[tokio::test]
    async fn legacy_mixed_collection_keeps_members_in_both_galleries() {
        let root = std::env::temp_dir().join(format!(
            "dmc-mixed-collection-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let state = AppState::new(&root, 0).unwrap();
        let dir_pool = state.directory_db().get_pool(11).unwrap();
        let dir_conn = dir_pool.get().unwrap();
        for (id, extension) in [(1, "png"), (2, "mp4")] {
            dir_conn
                .execute(
                    "INSERT INTO images (id, filename, file_hash) VALUES (?1, ?2, ?3)",
                    params![id, format!("file.{extension}"), format!("hash-{id}")],
                )
                .unwrap();
            dir_conn.execute("INSERT INTO image_files (image_id, original_path, file_extension) VALUES (?1, ?2, ?3)", params![id, format!("/media/file.{extension}"), extension]).unwrap();
        }
        dir_conn.execute("INSERT INTO images (id, filename, file_hash) VALUES (3, 'image.png', 'image-three')", []).unwrap();
        dir_conn.execute("INSERT INTO image_files (image_id, original_path, file_extension) VALUES (3, '/media/image-three.png', 'png')", []).unwrap();
        drop(dir_conn);
        let other_pool = state.directory_db().get_pool(12).unwrap();
        let other_conn = other_pool.get().unwrap();
        other_conn.execute("INSERT INTO images (id, filename, file_hash) VALUES (3, 'video.mp4', 'video-three')", []).unwrap();
        other_conn.execute("INSERT INTO image_files (image_id, original_path, file_extension) VALUES (3, '/media/video-three.mp4', 'mp4')", []).unwrap();
        drop(other_conn);
        let main_conn = state.main_db().get().unwrap();
        main_conn.execute("INSERT INTO collections (id, name, item_count, created_at) VALUES (1, 'Mixed', 2, datetime('now'))", []).unwrap();
        for id in [1, 2] {
            main_conn.execute("INSERT INTO collection_items (collection_id, image_id, sort_order) VALUES (1, ?1, ?1)", params![id]).unwrap();
        }
        let library_id = state.library_manager().primary().uuid.clone();
        for (collection_id, media_type, directory_id) in [(2, "image", 11), (3, "video", 12)] {
            main_conn.execute("INSERT INTO collections (id, name, media_type, item_count, created_at) VALUES (?1, 'Exact', ?2, 1, datetime('now'))", params![collection_id, media_type]).unwrap();
            main_conn.execute("INSERT INTO collection_items (collection_id, image_id, directory_id, library_id, sort_order) VALUES (?1, 3, ?2, ?3, 1)", params![collection_id, directory_id, &library_id]).unwrap();
        }
        drop(main_conn);

        let app = Router::new()
            .nest("/api/collections", router())
            .with_state(state);
        for (media_type, expected_id) in [("image", 1), ("video", 2)] {
            let response = app
                .clone()
                .oneshot(request(&format!(
                    "/api/collections?media_type={media_type}"
                )))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let listing: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(listing["collections"][0]["item_count"], 1);

            let response = app
                .clone()
                .oneshot(request(&format!(
                    "/api/collections/1?media_type={media_type}"
                )))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let detail: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(detail["images"][0]["id"], expected_id);
            assert_eq!(detail["images"][0]["collection_legacy_member"], true);
            assert_eq!(detail["item_count"], 1);
        }
        for (collection_id, media_type, directory_id, hash) in [
            (2, "image", 11, "image-three"),
            (3, "video", 12, "video-three"),
        ] {
            let response = app
                .clone()
                .oneshot(request(&format!(
                    "/api/collections/{collection_id}?media_type={media_type}"
                )))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let detail: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(detail["images"][0]["directory_id"], directory_id);
            assert_eq!(detail["images"][0]["file_hash"], hash);
            assert_eq!(detail["images"][0]["collection_legacy_member"], false);
        }
        let add_request = Request::builder()
            .method("POST")
            .uri("/api/collections/1/items")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"items": [
                    {"image_id": 3, "directory_id": 11, "library_id": library_id},
                    {"image_id": 3, "directory_id": 12, "library_id": library_id},
                ]})
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(add_request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let added: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(added["added"], 2);
        for (media_type, expected_hash) in [("image", "image-three"), ("video", "video-three")] {
            let response = app
                .clone()
                .oneshot(request(&format!(
                    "/api/collections/1?media_type={media_type}"
                )))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let detail: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(detail["item_count"], 2);
            assert!(detail["images"]
                .as_array()
                .unwrap()
                .iter()
                .any(|image| image["file_hash"] == expected_hash));
        }
        let remove_request = Request::builder()
            .method("DELETE")
            .uri("/api/collections/1/items")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"items": [{"image_id": 3, "directory_id": 12, "library_id": library_id}], "image_ids": [1]})
                    .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(remove_request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        for (media_type, hash) in [("image", "image-three"), ("video", "hash-2")] {
            let response = app
                .clone()
                .oneshot(request(&format!(
                    "/api/collections/1?media_type={media_type}"
                )))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let detail: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(detail["item_count"], 1);
            assert_eq!(detail["images"][0]["file_hash"], hash);
        }
        let _ = std::fs::remove_dir_all(root);
    }
}

/// GET /api/collections
async fn list_collections(
    State(state): State<AppState>,
    Query(query): Query<CollectionListQuery>,
) -> Result<Json<Value>, AppError> {
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
                let matching = collection_members(&conn, collection_id).ok()?.into_iter().filter(|member| ids.includes(member)).collect::<Vec<_>>();
                if matching.is_empty() && collection["media_type"].is_null() && requested != "image" {
                    return None;
                }
                collection["item_count"] = json!(matching.len());
                // Cover IDs in old rows have no directory identity. Use the
                // first scoped member so a video collection never shows an image.
                collection["cover_image_id"] = matching.first().map(|member| json!(member.image_id)).unwrap_or(Value::Null);
                collection["cover_thumbnail_url"] = matching.first().map(|member| json!(member_thumbnail_url(member))).unwrap_or(Value::Null);
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

        // Filter exact memberships by their full locator. Old rows have no
        // locator, so their numeric ID can only be resolved best-effort.
        let matches = params.media_type.as_deref().map(|kind| matching_media_ids(&state_clone, kind)).transpose()?;
        let visible_members = collection_members(&conn, collection_id)?.into_iter()
            .filter(|member| matches.as_ref().is_none_or(|matches| matches.includes(member)))
            .collect::<Vec<_>>();
        let scoped_count = visible_members.len();
        let page_members = visible_members.into_iter().skip(offset as usize).take(per_page as usize).collect::<Vec<_>>();

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
        for member in &page_members {
            let locator = member.locator();
            let libraries = if let Some(ref locator) = locator {
                vec![state_clone.resolve_library(Some(&locator.library_id))?]
            } else {
                vec![state_clone.library_manager().primary().clone()]
            };
            let mut found = false;
            for library in libraries {
                let lib_main = library.main_pool.get()?;
                let visible_dir_ids = get_visible_directory_ids(&lib_main, tier, family_locked)?;
                let directory_ids = locator.as_ref().map(|locator| vec![locator.directory_id])
                    .unwrap_or_else(|| library.directory_db.get_all_directory_ids());
                for dir_id in directory_ids {
                    if visible_dir_ids.as_ref().is_some_and(|visible| !visible.contains(&dir_id)) { continue; }
                    if !library.directory_db.db_exists(dir_id) { continue; }
                    let dir_pool = match library.directory_db.get_pool(dir_id) { Ok(pool) => pool, Err(_) => continue };
                    let dir_conn = match dir_pool.get() { Ok(conn) => conn, Err(_) => continue };
                    let library_id = library.uuid.clone();
                    let encoded_library_id = crate::routes::images::adjustments::encode_query_component(&library_id);
                    if let Ok(img) = dir_conn.query_row(&image_sql, params![member.image_id], |row| {
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
                            "library_id": library_id,
                            "directory_id": dir_id,
                            "collection_legacy_member": locator.is_none(),
                            "thumbnail_url": format!("/api/images/{}/thumbnail?directory_id={}&library_id={}&file_hash={}", member.image_id, dir_id, encoded_library_id, hash),
                            "url": format!("/api/images/{}/file?directory_id={}&library_id={}&file_hash={}", member.image_id, dir_id, encoded_library_id, hash),
                        }))
                    }) {
                        images.push(img);
                        found = true;
                        break;
                    }
                }
                if found { break; }
            }
            if !found && params.media_type.is_none() && locator.is_none() {
                images.push(json!({"id": member.image_id, "collection_legacy_member": true}));
            }
        }

        let mut result = collection;
        result["images"] = json!(images);
        result["item_count"] = json!(scoped_count);
        result["page"] = json!(page);
        result["per_page"] = json!(per_page);
        result["has_more"] = json!((offset as usize + page_members.len()) < scoped_count);

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
            if body.items.iter().any(|item| !matching.includes_locator(item))
                || body.image_ids.iter().any(|id| !matching.legacy_ids.contains(id)) {
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
        for (i, item) in body.items.iter().enumerate() {
            if item.library_id.is_empty() {
                return Err(AppError::BadRequest("Collection item requires a library_id".into()));
            }
            added += conn.execute(
                "INSERT OR IGNORE INTO collection_items (collection_id, image_id, directory_id, library_id, sort_order) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![collection_id, item.image_id, item.directory_id, item.library_id, max_order + i as i64 + 1],
            )? as i64;
        }
        for (i, image_id) in body.image_ids.iter().enumerate() {
            // Check if already in collection
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM collection_items WHERE collection_id = ?1 AND image_id = ?2 AND directory_id IS NULL AND library_id IS NULL",
                    params![collection_id, image_id],
                    |row| row.get::<_, i64>(0).map(|c| c > 0),
                )
                .unwrap_or(false);

            if exists {
                continue;
            }

            conn.execute(
                "INSERT INTO collection_items (collection_id, image_id, sort_order) VALUES (?1, ?2, ?3)",
                params![collection_id, image_id, max_order + body.items.len() as i64 + i as i64 + 1],
            )?;
            added += 1;
        }

        if added > 0 {
            conn.execute(
                "UPDATE collections SET item_count = item_count + ?1, updated_at = ?2 WHERE id = ?3",
                params![added, chrono::Utc::now().to_rfc3339(), collection_id],
            )?;

            // Covers are derived from scoped members when listed. The legacy
            // cover_image_id column points at the main DB's images table, but
            // actual files live in per-directory databases.
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
        for item in &body.items {
            conn.execute(
                "DELETE FROM collection_items WHERE collection_id = ?1 AND image_id = ?2 AND directory_id = ?3 AND library_id = ?4",
                params![collection_id, item.image_id, item.directory_id, item.library_id],
            )?;
        }
        for image_id in &body.image_ids {
            conn.execute(
                "DELETE FROM collection_items WHERE collection_id = ?1 AND image_id = ?2 AND directory_id IS NULL AND library_id IS NULL",
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
