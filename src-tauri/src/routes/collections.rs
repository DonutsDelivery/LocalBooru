use std::collections::{HashMap, HashSet};
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

#[derive(Debug, Default, Deserialize)]
struct PaginationParams {
    page: Option<i64>,
    per_page: Option<i64>,
    media_type: Option<String>,
    tags: Option<String>,
    exclude_tags: Option<String>,
    rating: Option<String>,
    sort: Option<String>,
    timeframe: Option<String>,
    #[serde(default)]
    favorites_only: bool,
    #[serde(default)]
    exclude_favorites: bool,
    min_age: Option<i32>,
    max_age: Option<i32>,
    has_faces: Option<bool>,
    min_width: Option<i32>,
    min_height: Option<i32>,
    orientation: Option<String>,
    min_duration: Option<i32>,
    max_duration: Option<i32>,
    filename: Option<String>,
    import_source: Option<String>,
    watched_status: Option<String>,
    directory_id: Option<i64>,
    library_id: Option<String>,
}
impl PaginationParams {
    fn image_params(&self) -> Result<crate::routes::images::helpers::ImageQueryParams, AppError> {
        if self.favorites_only && self.exclude_favorites {
            return Err(AppError::BadRequest(
                "favorites_only and exclude_favorites cannot both be true".into(),
            ));
        }
        if self.rating.as_deref().is_some_and(|ratings| {
            ratings
                .split(',')
                .any(|rating| !["pg", "pg13", "r", "x", "xxx"].contains(&rating.trim()))
        }) {
            return Err(AppError::BadRequest("Invalid rating filter".into()));
        }
        if self.sort.as_deref().is_some_and(|sort| {
            ![
                "collection",
                "newest",
                "oldest",
                "filename_asc",
                "filename_desc",
                "filesize_largest",
                "filesize_smallest",
                "resolution_high",
                "resolution_low",
                "duration_longest",
                "duration_shortest",
                "folder_asc",
                "folder_desc",
                "random",
            ]
            .contains(&sort)
        }) {
            return Err(AppError::BadRequest("Unsupported collection sort".into()));
        }
        if self
            .timeframe
            .as_deref()
            .is_some_and(|value| !["all", "today", "week", "month", "year"].contains(&value))
        {
            return Err(AppError::BadRequest("Unsupported timeframe".into()));
        }
        if self
            .orientation
            .as_deref()
            .is_some_and(|value| !["landscape", "portrait", "square"].contains(&value))
        {
            return Err(AppError::BadRequest("Invalid orientation".into()));
        }
        if self
            .watched_status
            .as_deref()
            .is_some_and(|value| !["watched", "unwatched", "in_progress"].contains(&value))
        {
            return Err(AppError::BadRequest("Invalid watch filter".into()));
        }
        if self.min_age.is_some_and(|n| n < 0)
            || self.max_age.is_some_and(|n| n < 0)
            || self.min_width.is_some_and(|n| n < 1)
            || self.min_height.is_some_and(|n| n < 1)
            || self.min_duration.is_some_and(|n| n < 0)
            || self.max_duration.is_some_and(|n| n < 0)
            || self
                .min_age
                .zip(self.max_age)
                .is_some_and(|(min, max)| min > max)
            || self
                .min_duration
                .zip(self.max_duration)
                .is_some_and(|(min, max)| min > max)
        {
            return Err(AppError::BadRequest("Invalid filter range".into()));
        }
        let names = |value: Option<&str>| {
            value
                .unwrap_or("")
                .split(',')
                .map(|name| name.trim().to_lowercase().replace(' ', "_"))
                .filter(|name| !name.is_empty())
                .collect()
        };
        Ok(crate::routes::images::helpers::ImageQueryParams {
            tags: names(self.tags.as_deref()),
            exclude_tags: names(self.exclude_tags.as_deref()),
            rating: names(self.rating.as_deref()),
            favorites_only: self.favorites_only,
            exclude_favorites: self.exclude_favorites,
            min_age: self.min_age,
            max_age: self.max_age,
            has_faces: self.has_faces,
            timeframe: self.timeframe.clone(),
            min_width: self.min_width,
            min_height: self.min_height,
            orientation: self.orientation.clone(),
            min_duration: self.min_duration,
            max_duration: self.max_duration,
            filename: self.filename.clone(),
            import_source: self.import_source.clone(),
            watched_status: self.watched_status.clone(),
            directory_id: self.directory_id,
            library_id: self.library_id.clone(),
            sort: self.sort.clone().unwrap_or_else(|| "collection".into()),
            limit: i64::MAX,
            offset: 0,
            show_images: self.media_type.as_deref() != Some("video"),
            show_videos: self.media_type.as_deref() != Some("image"),
        })
    }
    fn has_filters(&self) -> bool {
        self.tags.is_some()
            || self.exclude_tags.is_some()
            || self.rating.is_some()
            || self.media_type.is_some()
            || self.favorites_only
            || self.exclude_favorites
            || self.min_age.is_some()
            || self.max_age.is_some()
            || self.has_faces.is_some()
            || self.timeframe.as_deref().is_some_and(|time| time != "all")
            || self.min_width.is_some()
            || self.min_height.is_some()
            || self.orientation.is_some()
            || self.min_duration.is_some()
            || self.max_duration.is_some()
            || self.filename.is_some()
            || self.import_source.is_some()
            || self.watched_status.is_some()
            || self.directory_id.is_some()
            || self.library_id.is_some()
    }
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

fn resolve_legacy_members(
    state: &AppState,
    members: &[CollectionMember],
) -> HashMap<i64, CollectionItemLocator> {
    let library = state.library_manager().primary();
    let mut unresolved = members
        .iter()
        .filter(|member| member.locator().is_none())
        .map(|member| member.image_id)
        .collect::<HashSet<_>>();
    let mut resolved = HashMap::new();
    let mut directories = library.directory_db.get_all_directory_ids();
    directories.sort_unstable();
    for directory_id in directories {
        if unresolved.is_empty() {
            break;
        }
        if !library.directory_db.db_exists(directory_id) {
            continue;
        }
        let Ok(pool) = library.directory_db.get_pool(directory_id) else {
            continue;
        };
        let Ok(conn) = pool.get() else {
            continue;
        };
        let ids = unresolved.iter().copied().collect::<Vec<_>>();
        for chunk in ids.chunks(500) {
            let sql = format!(
                "SELECT id FROM images WHERE id IN ({})",
                chunk
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let Ok(mut stmt) = conn.prepare(&sql) else {
                continue;
            };
            let Ok(rows) = stmt.query_map([], |row| row.get::<_, i64>(0)) else {
                continue;
            };
            for id in rows.filter_map(Result::ok) {
                resolved.insert(
                    id,
                    CollectionItemLocator {
                        image_id: id,
                        directory_id,
                        library_id: library.uuid.clone(),
                    },
                );
                unresolved.remove(&id);
            }
        }
    }
    resolved
}

fn sort_collection_images(images: &mut Vec<Value>, sort: &str, collection_id: i64) {
    if sort == "collection" {
        return;
    }
    if sort == "random" {
        use sha2::{Digest, Sha256};
        // Include the complete locator, not just a database-local numeric ID.
        // A stable hash keeps separate page and restoration requests aligned.
        images.sort_by_cached_key(|image| {
            let mut hash = Sha256::new();
            hash.update(b"dmc-collection-order-v1");
            hash.update(collection_id.to_be_bytes());
            hash.update(image["library_id"].as_str().unwrap_or_default().as_bytes());
            hash.update(
                image["directory_id"]
                    .as_i64()
                    .unwrap_or_default()
                    .to_be_bytes(),
            );
            hash.update(image["id"].as_i64().unwrap_or_default().to_be_bytes());
            hash.update([u8::from(
                image["collection_legacy_member"].as_bool().unwrap_or(false),
            )]);
            <[u8; 32]>::from(hash.finalize())
        });
    } else {
        images.sort_by(|a, b| {
            a["library_id"]
                .as_str()
                .cmp(&b["library_id"].as_str())
                .then_with(|| a["directory_id"].as_i64().cmp(&b["directory_id"].as_i64()))
        });
        crate::routes::images::list::sort_merged_images(images, sort);
    }
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

    async fn detail(app: &Router, query: &str) -> Value {
        let response = app.clone().oneshot(request(query)).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn collection_filters_sort_all_scoped_members_before_pagination() {
        let root = std::env::temp_dir().join(format!(
            "dmc-collection-sidebar-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let state = AppState::new(&root, 0).unwrap();
        let primary_id = state.library_manager().primary().uuid.clone();
        let directory = state.directory_db().get_pool(11).unwrap();
        {
            let conn = directory.get().unwrap();
            for (id, name, favorite, width, height, age, duration, extension) in [
                (1, "z.png", false, 400, 400, 10, 0, "png"),
                (2, "b.png", true, 600, 600, 20, 30, "png"),
                (3, "a.png", true, 1200, 800, 25, 60, "png"),
                (4, "discarded.png", false, 800, 800, 20, 0, "png"),
                (5, "v.mp4", true, 1600, 900, 30, 120, "mp4"),
            ] {
                conn.execute("INSERT INTO images (id,filename,original_filename,file_hash,is_favorite,width,height,min_detected_age,max_detected_age,num_faces,duration) VALUES (?1,?2,?2,?3,?4,?5,?6,?7,?7,1,?8)",params![id,name,format!("primary-{id}"),favorite,width,height,age,duration]).unwrap();
                conn.execute("INSERT INTO image_files (image_id,original_path,file_extension) VALUES (?1,?2,?3)",params![id,format!("/media/{name}"),extension]).unwrap();
            }
            conn.execute(
                "UPDATE images SET original_filename=NULL,view_count=7 WHERE id=3",
                [],
            )
            .unwrap();
            conn.execute("INSERT INTO image_files (image_id,original_path,file_extension,curation_discarded_at) VALUES (4,'/media/discarded.mp4','mp4',datetime('now'))",[]).unwrap();
            conn.execute(
                "INSERT INTO image_tags (image_id,tag_id) VALUES (2,1),(2,2),(3,1)",
                [],
            )
            .unwrap();
        }
        {
            let other = state.directory_db().get_pool(12).unwrap();
            let conn = other.get().unwrap();
            conn.execute("INSERT INTO images (id,filename,original_filename,file_hash,rating,is_favorite) VALUES (3,'other.mp4','other.mp4','other-dir','x',1)",[]).unwrap();
            conn.execute("INSERT INTO image_files (image_id,original_path,file_extension) VALUES (3,'/other/3.mp4','mp4')",[]).unwrap();
        }
        let secondary_root = root.join("secondary");
        std::fs::create_dir_all(&secondary_root).unwrap();
        let secondary =
            crate::db::library::LibraryContext::open(&secondary_root, "Secondary").unwrap();
        let secondary_id = secondary.uuid.clone();
        {
            let other = secondary.directory_db.get_pool(11).unwrap();
            let conn = other.get().unwrap();
            conn.execute("INSERT INTO images (id,filename,original_filename,file_hash,rating,is_favorite) VALUES (3,'secondary.mp4','secondary.mp4','other-library','pg',1)",[]).unwrap();
            conn.execute("INSERT INTO image_files (image_id,original_path,file_extension) VALUES (3,'/secondary/3.mp4','mp4')",[]).unwrap();
        }
        state.library_manager().mount(secondary);
        {
            let conn = state.main_db().get().unwrap();
            conn.execute(
                "INSERT INTO tags (id,name) VALUES (1,'sunrise'),(2,'indoor')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO collections (id,name,item_count) VALUES (1,'Scoped',7),(2,'Legacy',1)",
                [],
            )
            .unwrap();
            for id in 1..=5 {
                conn.execute("INSERT INTO collection_items (collection_id,image_id,directory_id,library_id,sort_order) VALUES (1,?1,11,?2,?1)",params![id,primary_id]).unwrap();
            }
            conn.execute("INSERT INTO collection_items (collection_id,image_id,directory_id,library_id,sort_order) VALUES (1,3,12,?1,6),(1,3,11,?2,7)",params![primary_id,secondary_id]).unwrap();
            conn.execute(
                "INSERT INTO collection_items (collection_id,image_id,sort_order) VALUES (2,3,1)",
                [],
            )
            .unwrap();
            conn.execute("INSERT INTO watch_history (image_id,directory_id,library_id,completed) VALUES (3,12,?1,1)",params![primary_id]).unwrap();
        }
        let app = Router::new()
            .nest("/api/collections", router())
            .with_state(state);
        for (page, expected_id, has_more) in [(1, 3, true), (2, 2, false)] {
            let data = detail(&app,&format!("/api/collections/1?media_type=image&favorites_only=true&rating=pg&sort=filename_asc&page={page}&per_page=1")).await;
            assert_eq!(data["item_count"], 2);
            assert_eq!(data["images"][0]["id"], expected_id);
            assert_eq!(data["has_more"], has_more);
            assert_eq!(data["images"][0]["directory_id"], 11);
            assert_eq!(data["images"][0]["library_id"], primary_id);
        }
        let data = detail(&app, "/api/collections/1?per_page=2").await;
        assert_eq!(data["item_count"], 7);
        assert_eq!(data["images"][0]["id"], 1);
        assert_eq!(data["images"][1]["id"], 2);
        let data = detail(&app,"/api/collections/1?tags=sunrise&exclude_tags=indoor&min_age=21&max_age=27&has_faces=true&min_width=1000&min_height=700&orientation=landscape&min_duration=40&max_duration=90").await;
        assert_eq!(data["item_count"], 1);
        assert_eq!(data["images"][0]["file_hash"], "primary-3");
        assert_eq!(data["images"][0]["original_filename"], "a.png");
        assert_eq!(data["images"][0]["view_count"], 7);
        let data = detail(
            &app,
            "/api/collections/1?media_type=video&watched_status=watched",
        )
        .await;
        assert_eq!(data["item_count"], 1);
        assert_eq!(data["images"][0]["file_hash"], "other-dir");
        assert_eq!(data["images"][0]["library_id"], primary_id);
        let data = detail(
            &app,
            &format!("/api/collections/1?directory_id=11&library_id={secondary_id}"),
        )
        .await;
        assert_eq!(data["item_count"], 1);
        assert_eq!(data["images"][0]["file_hash"], "other-library");
        assert_eq!(data["images"][0]["library_id"], secondary_id);
        let data = detail(&app, "/api/collections/2?media_type=video").await;
        assert_eq!(
            data["item_count"], 0,
            "legacy primary image must not retarget a colliding video ID"
        );
        let data = detail(&app,"/api/collections/1?media_type=video&directory_id=11&library_id=primary&filename=discarded").await;
        assert_eq!(
            data["item_count"], 0,
            "discarded video files cannot classify a surviving image as video"
        );
        let full_random = detail(&app, "/api/collections/1?sort=random&per_page=200").await;
        let expected = full_random["images"].as_array().unwrap();
        let mut paginated = Vec::new();
        for page in 1..=4 {
            let data = detail(
                &app,
                &format!("/api/collections/1?sort=random&per_page=2&page={page}"),
            )
            .await;
            assert_eq!(data["item_count"], 7);
            paginated.extend(data["images"].as_array().unwrap().iter().cloned());
        }
        assert_eq!(
            &paginated, expected,
            "Random collection pages must use the same full-locator order"
        );
        let restored = detail(&app, "/api/collections/1?sort=random&per_page=2").await;
        assert_eq!(
            restored["images"].as_array().unwrap().as_slice(),
            &expected[..2]
        );
        let locators = paginated
            .iter()
            .map(|image| {
                (
                    image["library_id"].as_str().unwrap(),
                    image["directory_id"].as_i64().unwrap(),
                    image["id"].as_i64().unwrap(),
                )
            })
            .collect::<HashSet<_>>();
        assert_eq!(locators.len(), 7);
        drop(app);
        drop(directory);
        let _ = std::fs::remove_dir_all(root);
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
    let image_params = params.image_params()?;
    let page = params.page.unwrap_or(1).clamp(1, 10000);
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

        // Resolve each membership before querying filters. A legacy numeric ID
        // keeps its primary-library best-effort identity even when a duplicate
        // in another directory would satisfy the selected filter.
        let members = collection_members(&conn, collection_id)?;
        let legacy_locators = resolve_legacy_members(&state_clone,&members);
        let requested_library = params.library_id.as_deref().map(|id| state_clone.resolve_library(Some(id))).transpose()?;
        let mut visibility: HashMap<String,Option<HashSet<i64>>> = HashMap::new();
        let mut resolved = Vec::new();
        let mut grouped: HashMap<(String,i64), Vec<i64>> = HashMap::new();
        for member in members {
            let is_legacy = member.locator().is_none();
            let locator = member.locator().or_else(|| legacy_locators.get(&member.image_id).cloned());
            let Some(locator) = locator else {
                if is_legacy && !params.has_filters() { resolved.push((member,None)); }
                continue;
            };
            if params.directory_id.is_some_and(|id| id != locator.directory_id) { continue; }
            let library = match state_clone.resolve_library(Some(&locator.library_id)) { Ok(library) => library, Err(_) => continue };
            if requested_library.as_ref().is_some_and(|requested| requested.uuid != library.uuid) { continue; }
            if !visibility.contains_key(&library.uuid) {
                let lib_main = library.main_pool.get()?;
                visibility.insert(library.uuid.clone(),get_visible_directory_ids(&lib_main,tier,family_locked)?);
            }
            if visibility.get(&library.uuid).and_then(Option::as_ref).is_some_and(|ids| !ids.contains(&locator.directory_id)) { continue; }
            if !library.directory_db.db_exists(locator.directory_id) { continue; }
            grouped.entry((library.uuid.clone(),locator.directory_id)).or_default().push(locator.image_id);
            resolved.push((member,Some(locator)));
        }
        let mut eligible: HashMap<(String,i64,i64),Value> = HashMap::new();
        for ((library_id,directory_id), ids) in grouped {
            let library = state_clone.resolve_library(Some(&library_id))?;
            let pool = library.directory_db.get_pool(directory_id)?;
            let mut directory_params = image_params.clone();
            let lib_main = library.main_pool.get()?;
            let (show_images,show_videos,dir_name) = lib_main.query_row("SELECT show_images,show_videos,name FROM watch_directories WHERE id=?1",params![directory_id],|row| Ok((row.get::<_,bool>(0)?,row.get::<_,bool>(1)?,row.get::<_,Option<String>>(2)?))).unwrap_or((true,true,None));
            directory_params.show_images &= show_images;
            directory_params.show_videos &= show_videos;
            for chunk in ids.chunks(500) {
                let (rows,_) = crate::routes::images::helpers::query_directory_images_scoped(&pool,directory_id,&library.main_pool,dir_name.as_deref(),&directory_params,Some(&library.uuid),Some(chunk))?;
                for mut image in rows {
                    // Legacy imported rows may have no original_filename; keep
                    // the collection's previous visible filename and sort key.
                    if image["original_filename"].is_null() {
                        let name = image["file_path"].as_str().and_then(|path| std::path::Path::new(path).file_name()).and_then(|name| name.to_str()).or_else(|| image["filename"].as_str()).map(str::to_string);
                        image["original_filename"] = json!(name);
                    }
                    if let Some(image_id) = image["id"].as_i64() { eligible.insert((library.uuid.clone(),directory_id,image_id),image); }
                }
            }
        }
        let mut images = Vec::new();
        for (member,locator) in resolved {
            if let Some(locator) = locator {
                if let Some(image) = eligible.get(&(locator.library_id,locator.directory_id,locator.image_id)) {
                    let mut image = image.clone();
                    image["collection_legacy_member"] = json!(member.locator().is_none());
                    images.push(image);
                }
            } else { images.push(json!({"id":member.image_id,"collection_legacy_member":true})); }
        }
        sort_collection_images(&mut images,&image_params.sort,collection_id);
        let scoped_count = images.len();
        let images = images.into_iter().skip(offset as usize).take(per_page as usize).collect::<Vec<_>>();
        let has_more = (offset as usize).saturating_add(images.len()) < scoped_count;

        let mut result = collection;
        result["images"] = json!(images);
        result["item_count"] = json!(scoped_count);
        result["page"] = json!(page);
        result["per_page"] = json!(per_page);
        result["has_more"] = json!(has_more);

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
