use axum::extract::{Path as AxumPath, Query, State};
use axum::response::Json;
use rusqlite::params;
use serde::Deserialize;
use serde_json::json;

use crate::server::error::AppError;
use crate::server::state::AppState;

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
}

fn default_page() -> i64 { 1 }
fn default_per_page() -> i64 { 50 }
fn default_sort() -> String { "count".into() }

/// GET /api/tags — List tags with search and filtering.
pub async fn list_tags(
    State(state): State<AppState>,
    Query(q): Query<ListTagsQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

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

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Sort
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
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "name": row.get::<_, String>(1)?,
                    "category": row.get::<_, String>(2)?,
                    "post_count": row.get::<_, i32>(3)?
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
}

fn default_autocomplete_limit() -> i64 { 10 }

/// GET /api/tags/autocomplete — Tag autocomplete with prefix priority.
pub async fn autocomplete_tags(
    State(state): State<AppState>,
    Query(q): Query<AutocompleteQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    if q.q.is_empty() {
        return Err(AppError::BadRequest("Query parameter 'q' is required".into()));
    }

    let state_clone = state.clone();

    let tags = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let search_term = q.q.to_lowercase().replace(' ', "_");
        let prefix_pattern = format!("{}%", search_term);
        let contains_pattern = format!("%{}%", search_term);
        let limit = q.limit.clamp(1, 50);

        // UNION: prefix matches (priority 0) then contains (priority 1)
        let sql = "
            SELECT name, category, post_count FROM (
                SELECT name, category, post_count, 0 AS priority
                FROM tags WHERE name LIKE ?1
                UNION ALL
                SELECT name, category, post_count, 1 AS priority
                FROM tags WHERE name LIKE ?2 AND name NOT LIKE ?1
            ) ORDER BY priority, post_count DESC LIMIT ?3
        ";

        let mut stmt = conn.prepare(sql)?;
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

/// GET /api/tags/stats/overview — Tag statistics.
pub async fn tag_stats(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

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
) -> Result<Json<serde_json::Value>, AppError> {
    let state_clone = state.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
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
        .map_err(|_| AppError::NotFound("Tag not found".into()))
    })
    .await??;

    Ok(Json(result))
}

#[derive(Debug, Deserialize)]
pub struct UpdateCategoryBody {
    pub category: String,
}

/// PATCH /api/tags/:tag_name/category — Update tag category.
pub async fn update_tag_category(
    State(state): State<AppState>,
    AxumPath(tag_name): AxumPath<String>,
    Json(body): Json<UpdateCategoryBody>,
) -> Result<Json<serde_json::Value>, AppError> {
    let valid = ["general", "character", "copyright", "artist", "meta"];
    if !valid.contains(&body.category.as_str()) {
        return Err(AppError::BadRequest(format!(
            "Invalid category: {}",
            body.category
        )));
    }

    let state_clone = state.clone();
    let category = body.category.clone();

    let result = tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
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
