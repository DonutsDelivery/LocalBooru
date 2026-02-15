use axum::extract::{Query, State};
use axum::response::Json;
use serde::Deserialize;
use serde_json::json;

use crate::server::error::AppError;
use crate::server::state::AppState;

use super::helpers::{query_directory_images, ImageQueryParams};

#[derive(Debug, Deserialize)]
pub struct ListImagesQuery {
    #[serde(default = "default_page")]
    pub page: i64,
    #[serde(default = "default_per_page")]
    pub per_page: i64,
    pub tags: Option<String>,
    pub exclude_tags: Option<String>,
    pub rating: Option<String>,
    #[serde(default)]
    pub favorites_only: bool,
    pub directory_id: Option<i64>,
    pub min_age: Option<i32>,
    pub max_age: Option<i32>,
    pub has_faces: Option<bool>,
    pub timeframe: Option<String>,
    pub filename: Option<String>,
    pub min_width: Option<i32>,
    pub min_height: Option<i32>,
    pub orientation: Option<String>,
    pub min_duration: Option<i32>,
    pub max_duration: Option<i32>,
    pub import_source: Option<String>,
    #[serde(default = "default_sort")]
    pub sort: String,
}

fn default_page() -> i64 { 1 }
fn default_per_page() -> i64 { 50 }
fn default_sort() -> String { "newest".into() }

/// GET /api/images — List images with filtering and pagination.
pub async fn list_images(
    State(state): State<AppState>,
    Query(q): Query<ListImagesQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let tag_names: Vec<String> = q
        .tags
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|t| t.trim().to_lowercase().replace(' ', "_"))
        .filter(|t| !t.is_empty())
        .collect();

    let exclude_names: Vec<String> = q
        .exclude_tags
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|t| t.trim().to_lowercase().replace(' ', "_"))
        .filter(|t| !t.is_empty())
        .collect();

    let valid_ratings = ["pg", "pg13", "r", "x", "xxx"];
    let rating_list: Vec<String> = q
        .rating
        .as_deref()
        .unwrap_or("")
        .split(',')
        .filter(|r| valid_ratings.contains(r))
        .map(|r| r.to_string())
        .collect();

    let page = q.page.max(1);
    let per_page = q.per_page.clamp(1, 400);
    let offset = (page - 1) * per_page;

    let params = ImageQueryParams {
        tags: tag_names,
        exclude_tags: exclude_names,
        rating: rating_list,
        favorites_only: q.favorites_only,
        min_age: q.min_age,
        max_age: q.max_age,
        has_faces: q.has_faces,
        timeframe: q.timeframe.clone(),
        filename: q.filename.clone(),
        min_width: q.min_width,
        min_height: q.min_height,
        orientation: q.orientation.clone(),
        min_duration: q.min_duration,
        max_duration: q.max_duration,
        import_source: q.import_source.clone(),
        sort: q.sort.clone(),
        limit: per_page,
        offset,
        show_images: true,
        show_videos: true,
    };

    let directory_id = q.directory_id;
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        // If specific directory requested
        if let Some(dir_id) = directory_id {
            if state_clone.directory_db().db_exists(dir_id) {
                // Get directory's media type settings
                let main_conn = state_clone.main_db().get()?;
                let (show_images, show_videos) = main_conn
                    .query_row(
                        "SELECT show_images, show_videos FROM watch_directories WHERE id = ?1",
                        rusqlite::params![dir_id],
                        |row| Ok((row.get::<_, bool>(0)?, row.get::<_, bool>(1)?)),
                    )
                    .unwrap_or((true, true));

                let dir_name: Option<String> = main_conn
                    .query_row(
                        "SELECT name FROM watch_directories WHERE id = ?1",
                        rusqlite::params![dir_id],
                        |row| row.get(0),
                    )
                    .ok();

                let mut query_params = params;
                query_params.show_images = show_images;
                query_params.show_videos = show_videos;

                let dir_pool = state_clone.directory_db().get_pool(dir_id)?;
                return query_directory_images(
                    &dir_pool,
                    dir_id,
                    state_clone.main_db(),
                    dir_name.as_deref(),
                    &query_params,
                );
            }
        }

        // No specific directory — query all directory DBs
        let all_dir_ids = state_clone.directory_db().get_all_directory_ids();

        if !all_dir_ids.is_empty() && directory_id.is_none() {
            for dir_id in &all_dir_ids {
                if !state_clone.directory_db().db_exists(*dir_id) {
                    continue;
                }
                let dir_pool = match state_clone.directory_db().get_pool(*dir_id) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let main_conn = state_clone.main_db().get()?;
                let (show_images, show_videos) = main_conn
                    .query_row(
                        "SELECT show_images, show_videos FROM watch_directories WHERE id = ?1",
                        rusqlite::params![dir_id],
                        |row| Ok((row.get::<_, bool>(0)?, row.get::<_, bool>(1)?)),
                    )
                    .unwrap_or((true, true));

                let dir_name: Option<String> = main_conn
                    .query_row(
                        "SELECT name FROM watch_directories WHERE id = ?1",
                        rusqlite::params![dir_id],
                        |row| row.get(0),
                    )
                    .ok();

                let mut query_params = params.clone();
                query_params.show_images = show_images;
                query_params.show_videos = show_videos;

                match query_directory_images(
                    &dir_pool,
                    *dir_id,
                    state_clone.main_db(),
                    dir_name.as_deref(),
                    &query_params,
                ) {
                    Ok((images, total)) if !images.is_empty() => {
                        return Ok((images, total));
                    }
                    Ok(_) => continue,
                    Err(e) => {
                        log::warn!("[Images] Error querying directory {}: {}", dir_id, e);
                        continue;
                    }
                }
            }

            // No images found in any directory
            return Ok((vec![], 0));
        }

        // Fallback: query main/legacy DB
        let main_conn = state_clone.main_db().get()?;
        query_main_db_images(&main_conn, &params)
    })
    .await??;

    let (images_data, total) = result;
    let total_pages = if per_page > 0 {
        (total + per_page - 1) / per_page
    } else {
        0
    };

    Ok(Json(json!({
        "images": images_data,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })))
}

/// GET /api/images/folders — List folders grouped by import_source.
pub async fn list_folders(
    State(state): State<AppState>,
    Query(q): Query<ListFoldersQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let tag_names: Vec<String> = q
        .tags
        .as_deref()
        .unwrap_or("")
        .split(',')
        .map(|t| t.trim().to_lowercase().replace(' ', "_"))
        .filter(|t| !t.is_empty())
        .collect();

    let valid_ratings = ["pg", "pg13", "r", "x", "xxx"];
    let rating_list: Vec<String> = q
        .rating
        .as_deref()
        .unwrap_or("")
        .split(',')
        .filter(|r| valid_ratings.contains(r))
        .map(|r| r.to_string())
        .collect();

    let directory_id = q.directory_id;
    let favorites_only = q.favorites_only;

    let state_clone = state.clone();

    let folders = tokio::task::spawn_blocking(move || {
        let all_dir_ids = state_clone.directory_db().get_all_directory_ids();
        let dir_ids_to_query = if let Some(did) = directory_id {
            vec![did]
        } else {
            all_dir_ids
        };

        let main_conn = state_clone.main_db().get()?;

        // Resolve tag IDs from main DB
        let mut tag_ids: Vec<i64> = Vec::new();
        for tag_name in &tag_names {
            match main_conn.query_row(
                "SELECT id FROM tags WHERE name = ?1",
                rusqlite::params![tag_name],
                |row| row.get::<_, i64>(0),
            ) {
                Ok(id) => tag_ids.push(id),
                Err(_) => return Ok::<_, AppError>(vec![]), // Tag doesn't exist
            }
        }

        let mut folders_map: std::collections::HashMap<
            String,
            (i64, Option<String>, Option<i32>, Option<i32>, i64),
        > = std::collections::HashMap::new();
        // key -> (count, thumbnail_url, width, height, directory_id)

        for dir_id in &dir_ids_to_query {
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

            // Build WHERE clause
            let mut where_parts: Vec<String> = Vec::new();
            where_parts.push(
                "i.id IN (SELECT image_id FROM image_files WHERE file_status != 'missing')".into(),
            );

            if favorites_only {
                where_parts.push("i.is_favorite = 1".into());
            }

            if !rating_list.is_empty() {
                let quoted: Vec<String> = rating_list.iter().map(|r| format!("'{}'", r)).collect();
                where_parts.push(format!("i.rating IN ({})", quoted.join(",")));
            }

            // Tag filters
            for tag_id in &tag_ids {
                where_parts.push(format!(
                    "i.id IN (SELECT image_id FROM image_tags WHERE tag_id = {})",
                    tag_id
                ));
            }

            let where_sql = format!("WHERE {}", where_parts.join(" AND "));

            // Count by import_source
            let count_sql = format!(
                "SELECT i.import_source, COUNT(i.id) FROM images i {} GROUP BY i.import_source",
                where_sql
            );

            if let Ok(mut stmt) = dir_conn.prepare(&count_sql) {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?))
                }) {
                    for row in rows.flatten() {
                        let key = row.0.unwrap_or_default();
                        let entry = folders_map
                            .entry(key.clone())
                            .or_insert((0, None, None, None, *dir_id));
                        entry.0 += row.1;
                    }
                }
            }

            // Get representative thumbnail for each folder
            for (key, entry) in folders_map.iter_mut() {
                if entry.1.is_some() {
                    continue; // Already have thumbnail
                }

                let source_filter = if key.is_empty() {
                    "i.import_source IS NULL".to_string()
                } else {
                    format!("i.import_source = '{}'", key.replace('\'', "''"))
                };

                let thumb_sql = format!(
                    "SELECT i.id, i.width, i.height, i.created_at FROM images i {} AND {} \
                     ORDER BY COALESCE(i.file_modified_at, i.created_at) DESC LIMIT 1",
                    where_sql, source_filter
                );

                if let Ok(mut stmt) = dir_conn.prepare(&thumb_sql) {
                    if let Ok(thumb) = stmt.query_row([], |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, Option<i32>>(1)?,
                            row.get::<_, Option<i32>>(2)?,
                        ))
                    }) {
                        entry.1 = Some(format!(
                            "/api/images/{}/thumbnail?directory_id={}",
                            thumb.0, *dir_id
                        ));
                        entry.2 = thumb.1;
                        entry.3 = thumb.2;
                    }
                }
            }
        }

        // Build folder list (skip single-item folders)
        let mut folders: Vec<serde_json::Value> = folders_map
            .iter()
            .filter(|(_, data)| data.0 > 1)
            .map(|(key, data)| {
                let path = if key.is_empty() { None } else { Some(key.as_str()) };
                let name = if key.is_empty() {
                    "Unfiled".to_string()
                } else {
                    std::path::Path::new(key)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(key)
                        .to_string()
                };
                json!({
                    "path": path,
                    "name": name,
                    "count": data.0,
                    "thumbnail_url": data.1,
                    "width": data.2,
                    "height": data.3,
                })
            })
            .collect();

        folders.sort_by(|a, b| {
            let a_null = a["path"].is_null();
            let b_null = b["path"].is_null();
            if a_null != b_null {
                return a_null.cmp(&b_null);
            }
            let a_name = a["name"].as_str().unwrap_or("").to_lowercase();
            let b_name = b["name"].as_str().unwrap_or("").to_lowercase();
            a_name.cmp(&b_name)
        });

        Ok(folders)
    })
    .await??;

    let total = folders.len();
    Ok(Json(json!({
        "folders": folders,
        "total": total
    })))
}

#[derive(Debug, Deserialize)]
pub struct ListFoldersQuery {
    pub directory_id: Option<i64>,
    pub rating: Option<String>,
    #[serde(default)]
    pub favorites_only: bool,
    pub tags: Option<String>,
}

/// Fallback: query images from the main/legacy database.
fn query_main_db_images(
    conn: &rusqlite::Connection,
    params: &ImageQueryParams,
) -> Result<(Vec<serde_json::Value>, i64), AppError> {
    let mut where_clauses: Vec<String> = Vec::new();
    let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    // Exclude missing files
    where_clauses
        .push("i.id IN (SELECT image_id FROM image_files WHERE file_status != 'missing')".into());

    if params.favorites_only {
        where_clauses.push("i.is_favorite = 1".into());
    }

    if !params.rating.is_empty() {
        let quoted: Vec<String> = params.rating.iter().map(|r| format!("'{}'", r)).collect();
        where_clauses.push(format!("i.rating IN ({})", quoted.join(",")));
    }

    if let Some(min_age) = params.min_age {
        sql_params.push(Box::new(min_age));
        where_clauses.push(format!("i.max_detected_age >= ?{}", sql_params.len()));
    }
    if let Some(max_age) = params.max_age {
        sql_params.push(Box::new(max_age));
        where_clauses.push(format!("i.min_detected_age <= ?{}", sql_params.len()));
    }
    if let Some(has_faces) = params.has_faces {
        if has_faces {
            where_clauses.push("i.num_faces > 0".into());
        } else {
            where_clauses.push("(i.num_faces = 0 OR i.num_faces IS NULL)".into());
        }
    }

    if let Some(ref timeframe) = params.timeframe {
        let now = chrono::Local::now();
        let start = match timeframe.as_str() {
            "today" => now.date_naive().and_hms_opt(0, 0, 0).map(|dt| dt.to_string()),
            "week" => Some((now - chrono::Duration::days(7)).format("%Y-%m-%d %H:%M:%S").to_string()),
            "month" => Some((now - chrono::Duration::days(30)).format("%Y-%m-%d %H:%M:%S").to_string()),
            "year" => Some((now - chrono::Duration::days(365)).format("%Y-%m-%d %H:%M:%S").to_string()),
            _ => None,
        };
        if let Some(start_dt) = start {
            sql_params.push(Box::new(start_dt));
            where_clauses.push(format!("i.created_at >= ?{}", sql_params.len()));
        }
    }

    // Tag filters
    for tag_name in &params.tags {
        sql_params.push(Box::new(tag_name.clone()));
        where_clauses.push(format!(
            "i.id IN (SELECT it.image_id FROM image_tags it JOIN tags t ON t.id = it.tag_id WHERE t.name = ?{})",
            sql_params.len()
        ));
    }
    for tag_name in &params.exclude_tags {
        sql_params.push(Box::new(tag_name.clone()));
        where_clauses.push(format!(
            "i.id NOT IN (SELECT it.image_id FROM image_tags it JOIN tags t ON t.id = it.tag_id WHERE t.name = ?{})",
            sql_params.len()
        ));
    }

    if let Some(min_height) = params.min_height {
        sql_params.push(Box::new(min_height));
        where_clauses.push(format!("MIN(i.width, i.height) >= ?{}", sql_params.len()));
    }

    if let Some(ref orientation) = params.orientation {
        match orientation.as_str() {
            "landscape" => where_clauses.push("i.width > i.height".into()),
            "portrait" => where_clauses.push("i.height > i.width".into()),
            "square" => where_clauses.push("i.width = i.height".into()),
            _ => {}
        }
    }

    if let Some(min_dur) = params.min_duration {
        sql_params.push(Box::new(min_dur));
        where_clauses.push(format!("i.duration >= ?{}", sql_params.len()));
    }
    if let Some(max_dur) = params.max_duration {
        sql_params.push(Box::new(max_dur));
        where_clauses.push(format!("i.duration <= ?{}", sql_params.len()));
    }

    if let Some(ref import_source) = params.import_source {
        if import_source == "__unfiled__" {
            where_clauses.push("i.import_source IS NULL".into());
        } else {
            sql_params.push(Box::new(import_source.clone()));
            where_clauses.push(format!("i.import_source = ?{}", sql_params.len()));
        }
    }

    let where_sql = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    // Count
    let count_sql = format!("SELECT COUNT(*) FROM images i {}", where_sql);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        sql_params.iter().map(|p| p.as_ref()).collect();
    let total: i64 = conn.query_row(&count_sql, param_refs.as_slice(), |row| row.get(0))?;

    // Sort
    let order_by = match params.sort.as_str() {
        "newest" => "ORDER BY COALESCE(i.file_modified_at, i.created_at) DESC, i.id DESC",
        "oldest" => "ORDER BY COALESCE(i.file_modified_at, i.created_at) ASC, i.id ASC",
        "filename_asc" => "ORDER BY LOWER(i.original_filename) ASC, i.id ASC",
        "filename_desc" => "ORDER BY LOWER(i.original_filename) DESC, i.id DESC",
        "filesize_largest" => "ORDER BY COALESCE(i.file_size, 0) DESC, i.id DESC",
        "filesize_smallest" => "ORDER BY COALESCE(i.file_size, 0) ASC, i.id ASC",
        "resolution_high" => "ORDER BY COALESCE(i.width,0)*COALESCE(i.height,0) DESC, i.id DESC",
        "resolution_low" => "ORDER BY COALESCE(i.width,0)*COALESCE(i.height,0) ASC, i.id ASC",
        "duration_longest" => "ORDER BY COALESCE(i.duration, 0) DESC, i.id DESC",
        "duration_shortest" => "ORDER BY CASE WHEN i.duration IS NULL THEN 1 ELSE 0 END ASC, COALESCE(i.duration, 0) ASC, i.id ASC",
        "random" => "ORDER BY RANDOM()",
        _ => "ORDER BY COALESCE(i.file_modified_at, i.created_at) DESC, i.id DESC",
    };

    let select_sql = format!(
        "SELECT i.id, i.filename, i.original_filename, i.file_hash, i.width, i.height,
                i.file_size, i.duration, i.rating, i.is_favorite, i.prompt, i.negative_prompt,
                i.model_name, i.sampler, i.seed, i.steps, i.cfg_scale, i.num_faces,
                i.min_detected_age, i.max_detected_age, i.created_at, i.import_source
         FROM images i {} {} LIMIT ?{} OFFSET ?{}",
        where_sql,
        order_by,
        sql_params.len() + 1,
        sql_params.len() + 2
    );

    sql_params.push(Box::new(params.limit));
    sql_params.push(Box::new(params.offset));

    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        sql_params.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&select_sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok((
            row.get::<_, i64>(0)?,       // id
            row.get::<_, String>(1)?,     // filename
            row.get::<_, Option<String>>(2)?, // original_filename
            row.get::<_, String>(3)?,     // file_hash
            row.get::<_, Option<i32>>(4)?, // width
            row.get::<_, Option<i32>>(5)?, // height
            row.get::<_, Option<i64>>(6)?, // file_size
            row.get::<_, Option<f64>>(7)?, // duration
            row.get::<_, String>(8)?,     // rating
            row.get::<_, bool>(9)?,       // is_favorite
            row.get::<_, Option<String>>(10)?, // prompt
            row.get::<_, Option<String>>(11)?, // negative_prompt
            row.get::<_, Option<String>>(12)?, // model_name
            row.get::<_, Option<String>>(13)?, // sampler
            row.get::<_, Option<String>>(14)?, // seed
            row.get::<_, Option<i32>>(15)?, // steps
            row.get::<_, Option<f64>>(16)?, // cfg_scale
            row.get::<_, Option<i32>>(17)?, // num_faces
            row.get::<_, Option<i32>>(18)?, // min_detected_age
            row.get::<_, Option<i32>>(19)?, // max_detected_age
            row.get::<_, Option<String>>(20)?, // created_at
            row.get::<_, Option<String>>(21)?, // import_source
        ))
    })?;

    let images: Vec<serde_json::Value> = rows
        .filter_map(|r| r.ok())
        .map(|r| {
            json!({
                "id": r.0,
                "directory_id": serde_json::Value::Null,
                "filename": r.1,
                "original_filename": r.2,
                "width": r.4,
                "height": r.5,
                "rating": r.8,
                "is_favorite": r.9,
                "thumbnail_url": format!("/api/images/{}/thumbnail", r.0),
                "url": format!("/api/images/{}/file", r.0),
                "file_status": "available",
                "tags": [],
                "num_faces": r.17,
                "min_age": r.18,
                "max_age": r.19,
                "created_at": r.20,
                "file_size": r.6,
                "prompt": r.10,
                "negative_prompt": r.11,
                "model_name": r.12,
                "sampler": r.13,
                "seed": r.14,
                "steps": r.15,
                "cfg_scale": r.16,
                "duration": r.7
            })
        })
        .collect();

    Ok((images, total))
}
