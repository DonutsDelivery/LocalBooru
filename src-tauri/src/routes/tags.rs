use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;

use axum::extract::{ConnectInfo, Path as AxumPath, Query, State};
use axum::response::Json;
use rusqlite::params;
use serde::Deserialize;
use serde_json::json;

use crate::db::directory_db::DirectoryDbManager;
use crate::server::error::AppError;
use crate::server::middleware::AccessTier;
use crate::server::state::AppState;
use crate::server::utils::get_visible_directory_ids;

#[derive(Debug, Deserialize)]
pub struct ListTagsQuery {
    pub q: Option<String>,
    pub category: Option<String>,
    #[serde(default = "default_page")]
    pub page: i64,
    #[serde(default = "default_per_page")]
    pub per_page: i64,
    #[serde(default = "default_sort")]
    pub sort: String,
    pub library_id: Option<String>,
}

fn default_page() -> i64 { 1 }
fn default_per_page() -> i64 { 50 }
fn default_sort() -> String { "count".into() }

/// Aggregate tag counts across per-directory databases, filtered by allowed
/// ratings and visible directories.
/// Returns a map of tag_id -> filtered_post_count.
fn aggregate_filtered_tag_counts(
    directory_db: &DirectoryDbManager,
    visible_dir_ids: &Option<HashSet<i64>>,
    allowed_ratings: &[&str],
) -> Result<HashMap<i64, i64>, AppError> {
    // Determine which directories to scan
    let dir_ids: Vec<i64> = if let Some(ref ids) = visible_dir_ids {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        ids.iter().copied().collect()
    } else {
        // All directories with databases
        directory_db.get_all_directory_ids()
    };

    // Build rating IN clause (ratings are validated upstream, safe to inline)
    let rating_in: String = allowed_ratings
        .iter()
        .map(|r| format!("'{}'", r))
        .collect::<Vec<_>>()
        .join(",");

    let mut tag_counts: HashMap<i64, i64> = HashMap::new();

    for dir_id in &dir_ids {
        let dir_pool = match directory_db.get_pool(*dir_id) {
            Ok(p) => p,
            Err(_) => continue, // DB doesn't exist yet, skip
        };
        let dir_conn = match dir_pool.get() {
            Ok(c) => c,
            Err(_) => continue,
        };

        let sql = format!(
            "SELECT it.tag_id, COUNT(DISTINCT it.image_id) FROM image_tags it
             INNER JOIN images i ON i.id = it.image_id
             WHERE i.rating IN ({})
             GROUP BY it.tag_id",
            rating_in
        );

        let mut stmt = match dir_conn.prepare(&sql) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let rows: Vec<(i64, i64)> = stmt
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))
            .into_iter()
            .flatten()
            .flatten()
            .collect();

        for (tag_id, count) in rows {
            *tag_counts.entry(tag_id).or_insert(0) += count;
        }
    }

    Ok(tag_counts)
}

/// GET /api/tags — List tags with search and filtering.
pub async fn list_tags(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(q): Query<ListTagsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let library_id = q.library_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let lib = state_clone.resolve_library(library_id.as_deref())?;
        let conn = lib.main_pool.get()?;

        // Check visibility (access tier + family mode)
        let client_ip = addr.ip();
        let tier = AccessTier::from_ip(&client_ip);
        let family_locked = state_clone.is_family_mode_locked();
        let visible_dir_ids = get_visible_directory_ids(&conn, tier, family_locked)?;

        // When family mode is locked, compute tag counts from only SFW images
        let filtered_tag_counts: Option<HashMap<i64, i64>> = if family_locked {
            let sfw_ratings: Vec<&str> = vec!["pg", "pg13"];
            Some(aggregate_filtered_tag_counts(&lib.directory_db, &visible_dir_ids, &sfw_ratings)?)
        } else if visible_dir_ids.is_some() {
            // Non-localhost access: filter to visible directories but all ratings
            let all_ratings: Vec<&str> = vec!["pg", "pg13", "r", "x", "xxx"];
            Some(aggregate_filtered_tag_counts(&lib.directory_db, &visible_dir_ids, &all_ratings)?)
        } else {
            None // Localhost + unlocked: no filtering, use stored post_count
        };

        let mut where_clauses: Vec<String> = Vec::new();
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        // Search filter
        if let Some(ref search) = q.q {
            let term = search.to_lowercase().replace(' ', "_");
            let pattern = format!("%{}%", term);
            sql_params.push(Box::new(pattern));
            where_clauses.push(format!("name LIKE ?{}", sql_params.len()));
        }

        // Category filter
        if let Some(ref category) = q.category {
            let valid = ["general", "character", "copyright", "artist", "meta"];
            if !valid.contains(&category.as_str()) {
                return Err(AppError::BadRequest(format!(
                    "Invalid category: {}",
                    category
                )));
            }
            sql_params.push(Box::new(category.clone()));
            where_clauses.push(format!("category = ?{}", sql_params.len()));
        }

        // If we have filtered counts, only show tags that appear in visible content
        if let Some(ref tag_counts) = filtered_tag_counts {
            if tag_counts.is_empty() {
                return Ok(json!({
                    "tags": [],
                    "total": 0,
                    "page": q.page.max(1),
                    "per_page": q.per_page.clamp(1, 200)
                }));
            }
            let visible_ids: Vec<String> = tag_counts.keys().map(|id| id.to_string()).collect();
            where_clauses.push(format!("id IN ({})", visible_ids.join(",")));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Sort — when filtered, use filtered counts for count-based sorting
        let order_by = match q.sort.as_str() {
            "count" => "ORDER BY post_count DESC",
            "name" => "ORDER BY name ASC",
            "newest" => "ORDER BY created_at DESC",
            _ => "ORDER BY post_count DESC",
        };

        // Count total
        let count_sql = format!("SELECT COUNT(*) FROM tags {}", where_sql);
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        let total: i64 = conn.query_row(&count_sql, param_refs.as_slice(), |row| row.get(0))?;

        // Query with pagination
        let page = q.page.max(1);
        let per_page = q.per_page.clamp(1, 200);
        let offset = (page - 1) * per_page;

        let query_sql = format!(
            "SELECT id, name, category, post_count FROM tags {} {} LIMIT ?{} OFFSET ?{}",
            where_sql,
            order_by,
            sql_params.len() + 1,
            sql_params.len() + 2
        );

        sql_params.push(Box::new(per_page));
        sql_params.push(Box::new(offset));

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&query_sql)?;
        let tags: Vec<serde_json::Value> = stmt
            .query_map(param_refs.as_slice(), |row| {
                let id = row.get::<_, i64>(0)?;
                let name = row.get::<_, String>(1)?;
                let category = row.get::<_, String>(2)?;
                let stored_count = row.get::<_, i32>(3)?;
                // Use filtered count if available
                let count = if let Some(ref tc) = filtered_tag_counts {
                    *tc.get(&id).unwrap_or(&0) as i32
                } else {
                    stored_count
                };
                Ok(json!({
                    "id": id,
                    "name": name,
                    "category": category,
                    "post_count": count
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(json!({
            "tags": tags,
            "total": total,
            "page": page,
            "per_page": per_page
        }))
    })
    .await??;

    Ok(Json(result))
}

#[derive(Debug, Deserialize)]
pub struct AutocompleteQuery {
    pub q: String,
    #[serde(default = "default_autocomplete_limit")]
    pub limit: i64,
    pub library_id: Option<String>,
}

fn default_autocomplete_limit() -> i64 { 10 }

/// GET /api/tags/autocomplete — Tag autocomplete with prefix priority.
pub async fn autocomplete_tags(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Query(q): Query<AutocompleteQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    if q.q.is_empty() {
        return Err(AppError::BadRequest("Query parameter 'q' is required".into()));
    }

    let state_clone = state.clone();
    let library_id = q.library_id.clone();

    let tags = tokio::task::spawn_blocking(move || {
        let lib = state_clone.resolve_library(library_id.as_deref())?;
        let conn = lib.main_pool.get()?;

        // Check visibility (access tier + family mode)
        let client_ip = addr.ip();
        let tier = AccessTier::from_ip(&client_ip);
        let family_locked = state_clone.is_family_mode_locked();
        let visible_dir_ids = get_visible_directory_ids(&conn, tier, family_locked)?;

        // When filtering is needed, get visible tag IDs
        let visible_tag_ids: Option<HashSet<i64>> = if family_locked {
            let sfw_ratings: Vec<&str> = vec!["pg", "pg13"];
            let counts = aggregate_filtered_tag_counts(&lib.directory_db, &visible_dir_ids, &sfw_ratings)?;
            Some(counts.keys().copied().collect())
        } else if visible_dir_ids.is_some() {
            let all_ratings: Vec<&str> = vec!["pg", "pg13", "r", "x", "xxx"];
            let counts = aggregate_filtered_tag_counts(&lib.directory_db, &visible_dir_ids, &all_ratings)?;
            Some(counts.keys().copied().collect())
        } else {
            None
        };

        let search_term = q.q.to_lowercase().replace(' ', "_");
        let prefix_pattern = format!("{}%", search_term);
        let contains_pattern = format!("%{}%", search_term);
        let limit = q.limit.clamp(1, 50);

        // If filtering, add ID restriction
        let id_filter = if let Some(ref ids) = visible_tag_ids {
            if ids.is_empty() {
                return Ok(vec![]);
            }
            let id_list: String = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
            format!(" AND id IN ({})", id_list)
        } else {
            String::new()
        };

        // UNION: prefix matches (priority 0) then contains (priority 1)
        let sql = format!(
            "SELECT name, category, post_count FROM (
                SELECT name, category, post_count, 0 AS priority
                FROM tags WHERE name LIKE ?1{id_filter}
                UNION ALL
                SELECT name, category, post_count, 1 AS priority
                FROM tags WHERE name LIKE ?2 AND name NOT LIKE ?1{id_filter}
            ) ORDER BY priority, post_count DESC LIMIT ?3"
        );

        let mut stmt = conn.prepare(&sql)?;
        let tags: Vec<serde_json::Value> = stmt
            .query_map(
                params![prefix_pattern, contains_pattern, limit],
                |row| {
                    Ok(json!({
                        "name": row.get::<_, String>(0)?,
                        "category": row.get::<_, String>(1)?,
                        "post_count": row.get::<_, i32>(2)?
                    }))
                },
            )?
            .filter_map(|r| r.ok())
            .collect();

        Ok::<_, AppError>(tags)
    })
    .await??;

    Ok(Json(json!(tags)))
}

#[derive(Debug, Deserialize)]
pub struct LibraryQuery {
    pub library_id: Option<String>,
}

/// GET /api/tags/stats/overview — Tag statistics.
pub async fn tag_stats(
    State(state): State<AppState>,
    Query(q): Query<LibraryQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let library_id = q.library_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let lib = state_clone.resolve_library(library_id.as_deref())?;
        let conn = lib.main_pool.get()?;

        // Total count
        let total: i64 =
            conn.query_row("SELECT COUNT(*) FROM tags", [], |row| row.get(0))?;

        // Count by category
        let mut stmt = conn.prepare("SELECT category, COUNT(*) FROM tags GROUP BY category")?;
        let by_category: std::collections::HashMap<String, i64> = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        // Top 10 tags
        let mut stmt =
            conn.prepare("SELECT name, post_count FROM tags ORDER BY post_count DESC LIMIT 10")?;
        let top_tags: Vec<serde_json::Value> = stmt
            .query_map([], |row| {
                Ok(json!({
                    "name": row.get::<_, String>(0)?,
                    "count": row.get::<_, i32>(1)?
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok::<_, AppError>(json!({
            "total": total,
            "by_category": by_category,
            "top_tags": top_tags
        }))
    })
    .await??;

    Ok(Json(result))
}

/// GET /api/tags/:tag_name — Get tag details.
pub async fn get_tag(
    State(state): State<AppState>,
    AxumPath(tag_name): AxumPath<String>,
    Query(q): Query<LibraryQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();
    let library_id = q.library_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let lib = state_clone.resolve_library(library_id.as_deref())?;
        let conn = lib.main_pool.get()?;
        let normalized = tag_name.to_lowercase().replace(' ', "_");

        conn.query_row(
            "SELECT id, name, category, post_count, created_at FROM tags WHERE name = ?1",
            params![normalized],
            |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "category": row.get::<_, String>(2)?,
                    "post_count": row.get::<_, i32>(3)?,
                    "created_at": row.get::<_, Option<String>>(4)?
                }))
            },
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                AppError::NotFound("Tag not found".into())
            }
            other => AppError::from(other),
        })
    })
    .await??;

    Ok(Json(result))
}

#[derive(Debug, Deserialize)]
pub struct UpdateCategoryQuery {
    pub category: String,
    pub library_id: Option<String>,
}

/// PATCH /api/tags/:tag_name/category — Update tag category.
pub async fn update_tag_category(
    State(state): State<AppState>,
    AxumPath(tag_name): AxumPath<String>,
    Query(params): Query<UpdateCategoryQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let valid = ["general", "character", "copyright", "artist", "meta"];
    if !valid.contains(&params.category.as_str()) {
        return Err(AppError::BadRequest(format!(
            "Invalid category: {}",
            params.category
        )));
    }

    let state_clone = state.clone();
    let category = params.category.clone();
    let library_id = params.library_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        let lib = state_clone.resolve_library(library_id.as_deref())?;
        let conn = lib.main_pool.get()?;
        let normalized = tag_name.to_lowercase().replace(' ', "_");

        let updated = conn.execute(
            "UPDATE tags SET category = ?1 WHERE name = ?2",
            params![category, normalized],
        )?;

        if updated == 0 {
            return Err(AppError::NotFound("Tag not found".into()));
        }

        Ok(json!({
            "name": normalized,
            "category": category
        }))
    })
    .await??;

    Ok(Json(result))
}

/// Build the /api/tags router.
pub fn router() -> axum::Router<AppState> {
    use axum::routing::{get, patch};

    axum::Router::new()
        .route("/", get(list_tags))
        .route("/autocomplete", get(autocomplete_tags))
        .route("/stats/overview", get(tag_stats))
        .route("/{tag_name}", get(get_tag))
        .route("/{tag_name}/category", patch(update_tag_category))
}
