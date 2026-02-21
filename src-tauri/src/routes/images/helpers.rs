use std::collections::HashSet;

use rusqlite::params;
use serde_json::json;

use crate::db::pool::DbPool;
use crate::db::DirectoryDbManager;
use crate::server::error::AppError;

/// File extensions for media type filtering.
pub const IMAGE_EXTENSIONS: &[&str] = &[".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"];
pub const VIDEO_EXTENSIONS: &[&str] = &[".webm", ".mp4", ".mov", ".avi", ".mkv"];

/// Query images from a single directory database with filters, sorting, and pagination.
/// Returns (images_data as JSON values, total_count).
pub fn query_directory_images(
    dir_pool: &DbPool,
    directory_id: i64,
    main_pool: &DbPool,
    dir_name: Option<&str>,
    params: &ImageQueryParams,
    library_id: Option<&str>,
) -> Result<(Vec<serde_json::Value>, i64), AppError> {
    let conn = dir_pool.get()?;
    let main_conn = main_pool.get()?;

    // Build WHERE clauses and params
    let mut where_clauses: Vec<String> = Vec::new();
    let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    // Exclude missing files
    where_clauses.push(
        "i.id IN (SELECT image_id FROM image_files WHERE file_status != 'missing')".into(),
    );

    // Media type filtering
    if !params.show_images && !params.show_videos {
        return Ok((vec![], 0));
    } else if !params.show_images {
        // Only videos
        let conditions: Vec<String> = VIDEO_EXTENSIONS
            .iter()
            .map(|ext| format!("original_path LIKE '%{}'", ext))
            .collect();
        where_clauses.push(format!(
            "i.id IN (SELECT image_id FROM image_files WHERE {})",
            conditions.join(" OR ")
        ));
    } else if !params.show_videos {
        // Only images
        let conditions: Vec<String> = IMAGE_EXTENSIONS
            .iter()
            .map(|ext| format!("original_path LIKE '%{}'", ext))
            .collect();
        where_clauses.push(format!(
            "i.id IN (SELECT image_id FROM image_files WHERE {})",
            conditions.join(" OR ")
        ));
    }

    // Favorites
    if params.favorites_only {
        where_clauses.push("i.is_favorite = 1".into());
    }

    // Rating filter — also include unrated (NULL) images so newly imported files aren't hidden
    if !params.rating.is_empty() {
        let quoted: Vec<String> = params.rating.iter().map(|r| format!("'{}'", r)).collect();
        where_clauses.push(format!(
            "(i.rating IN ({}) OR i.rating IS NULL)",
            quoted.join(",")
        ));
    }

    // Age filters
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

    // Timeframe filter
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

    // Filename search
    if let Some(ref filename) = params.filename {
        let pattern = format!("%{}%", filename);
        sql_params.push(Box::new(pattern));
        where_clauses.push(format!(
            "i.id IN (SELECT image_id FROM image_files WHERE original_path LIKE ?{})",
            sql_params.len()
        ));
    }

    // Tag filters (need to resolve tag names → IDs via main DB)
    if !params.tags.is_empty() {
        for tag_name in &params.tags {
            let tag_id: Option<i64> = main_conn
                .query_row("SELECT id FROM tags WHERE name = ?1", params![tag_name], |row| {
                    row.get(0)
                })
                .ok();

            match tag_id {
                Some(id) => {
                    sql_params.push(Box::new(id));
                    where_clauses.push(format!(
                        "i.id IN (SELECT image_id FROM image_tags WHERE tag_id = ?{})",
                        sql_params.len()
                    ));
                }
                None => {
                    // Tag doesn't exist — no images will match
                    return Ok((vec![], 0));
                }
            }
        }
    }

    // Exclude tags
    for tag_name in &params.exclude_tags {
        let tag_id: Option<i64> = main_conn
            .query_row("SELECT id FROM tags WHERE name = ?1", params![tag_name], |row| {
                row.get(0)
            })
            .ok();

        if let Some(id) = tag_id {
            sql_params.push(Box::new(id));
            where_clauses.push(format!(
                "i.id NOT IN (SELECT image_id FROM image_tags WHERE tag_id = ?{})",
                sql_params.len()
            ));
        }
    }

    // Resolution filter (shorter dimension >= min_height)
    if let Some(min_height) = params.min_height {
        sql_params.push(Box::new(min_height));
        where_clauses.push(format!("MIN(i.width, i.height) >= ?{}", sql_params.len()));
    }

    // Orientation
    if let Some(ref orientation) = params.orientation {
        match orientation.as_str() {
            "landscape" => where_clauses.push("i.width > i.height".into()),
            "portrait" => where_clauses.push("i.height > i.width".into()),
            "square" => where_clauses.push("i.width = i.height".into()),
            _ => {}
        }
    }

    // Duration filters
    if let Some(min_dur) = params.min_duration {
        sql_params.push(Box::new(min_dur));
        where_clauses.push(format!("i.duration >= ?{}", sql_params.len()));
    }
    if let Some(max_dur) = params.max_duration {
        sql_params.push(Box::new(max_dur));
        where_clauses.push(format!("i.duration <= ?{}", sql_params.len()));
    }

    // Import source filter
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

    // Count total
    let count_sql = format!("SELECT COUNT(*) FROM images i {}", where_sql);
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        sql_params.iter().map(|p| p.as_ref()).collect();
    let total: i64 = conn.query_row(&count_sql, param_refs.as_slice(), |row| row.get(0))?;

    // Sorting
    let order_by = match params.sort.as_str() {
        "newest" => "ORDER BY COALESCE(i.file_modified_at, i.created_at) DESC, i.id DESC",
        "oldest" => "ORDER BY COALESCE(i.file_modified_at, i.created_at) ASC, i.id ASC",
        "filename_asc" => "ORDER BY LOWER(i.original_filename) ASC, i.id ASC",
        "filename_desc" => "ORDER BY LOWER(i.original_filename) DESC, i.id DESC",
        "filesize_largest" => "ORDER BY COALESCE(i.file_size, 0) DESC, i.id DESC",
        "filesize_smallest" => "ORDER BY COALESCE(i.file_size, 0) ASC, i.id ASC",
        "resolution_high" => {
            "ORDER BY COALESCE(i.width, 0) * COALESCE(i.height, 0) DESC, i.id DESC"
        }
        "resolution_low" => {
            "ORDER BY COALESCE(i.width, 0) * COALESCE(i.height, 0) ASC, i.id ASC"
        }
        "duration_longest" => "ORDER BY COALESCE(i.duration, 0) DESC, i.id DESC",
        "duration_shortest" => {
            "ORDER BY CASE WHEN i.duration IS NULL THEN 1 ELSE 0 END ASC, COALESCE(i.duration, 0) ASC, i.id ASC"
        }
        "folder_asc" => "ORDER BY COALESCE(LOWER(i.import_source), '') ASC, i.id ASC",
        "folder_desc" => "ORDER BY COALESCE(LOWER(i.import_source), '') DESC, i.id DESC",
        "random" => "ORDER BY RANDOM()",
        _ => "ORDER BY COALESCE(i.file_modified_at, i.created_at) DESC, i.id DESC",
    };

    // Main query
    let select_sql = format!(
        "SELECT i.id, i.filename, i.original_filename, i.file_hash, i.width, i.height,
                i.file_size, i.duration, i.rating, i.is_favorite, i.prompt, i.negative_prompt,
                i.model_name, i.sampler, i.seed, i.steps, i.cfg_scale, i.num_faces,
                i.min_detected_age, i.max_detected_age, i.created_at, i.import_source,
                i.file_modified_at
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
        Ok(ImageRow {
            id: row.get(0)?,
            filename: row.get(1)?,
            original_filename: row.get(2)?,
            file_hash: row.get(3)?,
            width: row.get(4)?,
            height: row.get(5)?,
            file_size: row.get(6)?,
            duration: row.get(7)?,
            rating: row.get(8)?,
            is_favorite: row.get(9)?,
            prompt: row.get(10)?,
            negative_prompt: row.get(11)?,
            model_name: row.get(12)?,
            sampler: row.get(13)?,
            seed: row.get(14)?,
            steps: row.get(15)?,
            cfg_scale: row.get(16)?,
            num_faces: row.get(17)?,
            min_detected_age: row.get(18)?,
            max_detected_age: row.get(19)?,
            created_at: row.get(20)?,
            import_source: row.get(21)?,
            file_modified_at: row.get(22)?,
        })
    })?;

    let images: Vec<ImageRow> = rows.filter_map(|r| r.ok()).collect();
    let image_ids: Vec<i64> = images.iter().map(|img| img.id).collect();

    // Get file info for all images
    let file_info = get_file_info_batch(&conn, &image_ids)?;

    // Batch fetch tags
    let tags_by_image = get_tags_batch(&conn, &main_conn, &image_ids)?;

    // Build response
    let images_data: Vec<serde_json::Value> = images
        .iter()
        .map(|img| {
            let (file_path, file_status) = file_info
                .get(&img.id)
                .map(|(p, s)| (Some(p.as_str()), s.as_str()))
                .unwrap_or((None, "unknown"));

            let tags_list = tags_by_image.get(&img.id).cloned().unwrap_or_default();

            json!({
                "id": img.id,
                "directory_id": directory_id,
                "filename": img.filename,
                "original_filename": img.original_filename,
                "width": img.width,
                "height": img.height,
                "rating": img.rating,
                "is_favorite": img.is_favorite,
                "thumbnail_url": if let Some(lib_id) = library_id {
                    format!("/api/images/{}/thumbnail?directory_id={}&library_id={}", img.id, directory_id, lib_id)
                } else {
                    format!("/api/images/{}/thumbnail?directory_id={}", img.id, directory_id)
                },
                "url": if let Some(lib_id) = library_id {
                    format!("/api/images/{}/file?directory_id={}&library_id={}", img.id, directory_id, lib_id)
                } else {
                    format!("/api/images/{}/file?directory_id={}", img.id, directory_id)
                },
                "file_status": file_status,
                "tags": tags_list,
                "num_faces": img.num_faces,
                "min_age": img.min_detected_age,
                "max_age": img.max_detected_age,
                "created_at": img.created_at,
                "file_size": img.file_size,
                "file_path": file_path,
                "directory_name": dir_name,
                "prompt": img.prompt,
                "negative_prompt": img.negative_prompt,
                "model_name": img.model_name,
                "sampler": img.sampler,
                "seed": img.seed,
                "steps": img.steps,
                "cfg_scale": img.cfg_scale,
                "duration": img.duration,
                "file_hash": img.file_hash,
                "import_source": img.import_source,
                "file_modified_at": img.file_modified_at
            })
        })
        .collect();

    Ok((images_data, total))
}

/// Get file path and status for a batch of image IDs from the directory DB.
fn get_file_info_batch(
    conn: &rusqlite::Connection,
    image_ids: &[i64],
) -> Result<std::collections::HashMap<i64, (String, String)>, AppError> {
    let mut result = std::collections::HashMap::new();
    if image_ids.is_empty() {
        return Ok(result);
    }

    let placeholders: Vec<String> = (1..=image_ids.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT image_id, original_path, file_status FROM image_files WHERE image_id IN ({})",
        placeholders.join(",")
    );
    let params: Vec<&dyn rusqlite::types::ToSql> =
        image_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    for row in rows.flatten() {
        result.entry(row.0).or_insert((row.1, row.2));
    }

    Ok(result)
}

/// Batch fetch tags for image IDs.
/// Tag IDs come from directory DB's image_tags, tag details from main DB's tags.
fn get_tags_batch(
    dir_conn: &rusqlite::Connection,
    main_conn: &rusqlite::Connection,
    image_ids: &[i64],
) -> Result<std::collections::HashMap<i64, Vec<serde_json::Value>>, AppError> {
    let mut result: std::collections::HashMap<i64, Vec<serde_json::Value>> =
        std::collections::HashMap::new();

    if image_ids.is_empty() {
        return Ok(result);
    }

    // Get all tag associations
    let placeholders: Vec<String> = (1..=image_ids.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT image_id, tag_id FROM image_tags WHERE image_id IN ({})",
        placeholders.join(",")
    );
    let params: Vec<&dyn rusqlite::types::ToSql> =
        image_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

    let mut stmt = dir_conn.prepare(&sql)?;
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    })?;

    let mut all_tag_ids: HashSet<i64> = HashSet::new();
    let mut associations: Vec<(i64, i64)> = Vec::new();

    for row in rows.flatten() {
        all_tag_ids.insert(row.1);
        associations.push(row);
    }

    if all_tag_ids.is_empty() {
        return Ok(result);
    }

    // Fetch tag details from main DB
    let tag_ids: Vec<i64> = all_tag_ids.into_iter().collect();
    let placeholders: Vec<String> = (1..=tag_ids.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT id, name, category FROM tags WHERE id IN ({})",
        placeholders.join(",")
    );
    let params: Vec<&dyn rusqlite::types::ToSql> =
        tag_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

    let mut stmt = main_conn.prepare(&sql)?;
    let tag_rows = stmt.query_map(params.as_slice(), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;

    let tags_by_id: std::collections::HashMap<i64, (String, String)> = tag_rows
        .flatten()
        .map(|(id, name, cat)| (id, (name, cat)))
        .collect();

    // Build per-image tag lists
    for (image_id, tag_id) in associations {
        if let Some((name, category)) = tags_by_id.get(&tag_id) {
            result
                .entry(image_id)
                .or_default()
                .push(json!({"name": name, "category": category}));
        }
    }

    Ok(result)
}

/// Find which directory contains an image by checking all directory DBs.
pub fn find_image_directory(
    directory_db: &DirectoryDbManager,
    image_id: i64,
    file_hash: Option<&str>,
) -> Option<i64> {
    for dir_id in directory_db.get_all_directory_ids() {
        if let Ok(pool) = directory_db.get_pool(dir_id) {
            if let Ok(conn) = pool.get() {
                let found = if let Some(hash) = file_hash {
                    conn.query_row(
                        "SELECT id FROM images WHERE file_hash = ?1 LIMIT 1",
                        params![hash],
                        |_| Ok(()),
                    )
                    .is_ok()
                } else {
                    conn.query_row(
                        "SELECT id FROM images WHERE id = ?1 LIMIT 1",
                        params![image_id],
                        |_| Ok(()),
                    )
                    .is_ok()
                };
                if found {
                    return Some(dir_id);
                }
            }
        }
    }
    None
}

/// Get tags for a specific image from a directory DB.
pub fn get_image_tags_from_directory(
    dir_pool: &DbPool,
    main_pool: &DbPool,
    image_id: i64,
) -> Result<Vec<serde_json::Value>, AppError> {
    let dir_conn = dir_pool.get()?;
    let main_conn = main_pool.get()?;

    let mut stmt = dir_conn.prepare("SELECT tag_id FROM image_tags WHERE image_id = ?1")?;
    let tag_ids: Vec<i64> = stmt
        .query_map(params![image_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    if tag_ids.is_empty() {
        return Ok(vec![]);
    }

    let placeholders: Vec<String> = (1..=tag_ids.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT name, category FROM tags WHERE id IN ({}) ORDER BY category, name",
        placeholders.join(",")
    );
    let params: Vec<&dyn rusqlite::types::ToSql> =
        tag_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();

    let mut stmt = main_conn.prepare(&sql)?;
    let tags = stmt
        .query_map(params.as_slice(), |row| {
            Ok(json!({
                "name": row.get::<_, String>(0)?,
                "category": row.get::<_, String>(1)?
            }))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(tags)
}

/// Parameters for image queries (shared between list and helpers).
#[derive(Debug, Clone)]
pub struct ImageQueryParams {
    pub tags: Vec<String>,
    pub exclude_tags: Vec<String>,
    pub rating: Vec<String>,
    pub favorites_only: bool,
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
    pub sort: String,
    pub limit: i64,
    pub offset: i64,
    pub show_images: bool,
    pub show_videos: bool,
}

/// Internal struct for reading image rows from SQLite.
struct ImageRow {
    id: i64,
    filename: String,
    original_filename: Option<String>,
    file_hash: String,
    width: Option<i32>,
    height: Option<i32>,
    file_size: Option<i64>,
    duration: Option<f64>,
    rating: Option<String>,
    is_favorite: Option<bool>,
    prompt: Option<String>,
    negative_prompt: Option<String>,
    model_name: Option<String>,
    sampler: Option<String>,
    seed: Option<String>,
    steps: Option<i32>,
    cfg_scale: Option<f64>,
    num_faces: Option<i32>,
    min_detected_age: Option<i32>,
    max_detected_age: Option<i32>,
    created_at: Option<String>,
    import_source: Option<String>,
    file_modified_at: Option<String>,
}
