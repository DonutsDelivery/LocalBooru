use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use axum::extract::{Path, Query, Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use rusqlite::{params, params_from_iter, OptionalExtension, Row};
use serde::Deserialize;
use serde_json::{json, Value};
use tower::ServiceExt;
use tower_http::services::ServeFile;

use crate::db::library::LibraryContext;
use crate::server::error::AppError;
use crate::server::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/albums", get(list_albums))
        .route("/albums/{id}", get(album_detail))
        .route("/tracks", get(list_tracks))
        .route("/tracks/{id}", get(track_detail))
        .route("/tracks/{id}/related", get(related_tracks))
        .route("/tracks/{id}/file", get(track_file))
        .route("/tracks/{id}/artwork", get(track_artwork))
        .route("/tracks/{id}/favorite", patch(set_favorite))
        .route("/facets", get(facets))
        .route(
            "/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/collections/{id}",
            get(collection_detail)
                .patch(update_collection)
                .delete(delete_collection),
        )
        .route(
            "/collections/{id}/items",
            post(add_collection_item).delete(remove_collection_item),
        )
}

#[derive(Default, Deserialize)]
struct MusicQuery {
    library_id: Option<String>,
    q: Option<String>,
    artist: Option<String>,
    album: Option<String>,
    genre: Option<String>,
    year: Option<i64>,
    favorites_only: Option<bool>,
    directory_id: Option<i64>,
    collection_id: Option<i64>,
    page: Option<i64>,
    per_page: Option<i64>,
    limit: Option<usize>,
    exclude_ids: Option<String>,
}

fn library(state: &AppState, q: &MusicQuery) -> Result<std::sync::Arc<LibraryContext>, AppError> {
    state.resolve_library(q.library_id.as_deref())
}

fn paging(q: &MusicQuery) -> (i64, i64) {
    (
        q.page.unwrap_or(1).max(1),
        q.per_page.unwrap_or(48).clamp(1, 200),
    )
}

const TRACK_SELECT: &str = "SELECT t.id, t.title, t.artist, a.title, t.album_id, t.genre,
    t.year, t.disc_number, t.track_number, t.duration, t.is_favorite, t.directory_id,
    COALESCE(t.artwork_path, a.artwork_path), t.path
    FROM music_tracks t JOIN music_albums a ON a.id = t.album_id";

fn track_json(row: &Row<'_>, library_id: &str) -> rusqlite::Result<Value> {
    let id: i64 = row.get(0)?;
    let has_art = row.get::<_, Option<String>>(12)?.is_some();
    Ok(json!({
        "id": id,
        "title": row.get::<_, String>(1)?,
        "artist": row.get::<_, String>(2)?,
        "album": row.get::<_, String>(3)?,
        "album_id": row.get::<_, i64>(4)?,
        "genre": row.get::<_, Option<String>>(5)?,
        "year": row.get::<_, Option<i64>>(6)?,
        "disc_number": row.get::<_, i64>(7)?,
        "track_number": row.get::<_, i64>(8)?,
        "duration": row.get::<_, Option<f64>>(9)?,
        "is_favorite": row.get::<_, i64>(10)? != 0,
        "directory_id": row.get::<_, i64>(11)?,
        "artwork_url": if has_art {Some(format!("/api/music/tracks/{id}/artwork?library_id={library_id}"))} else {None},
        "stream_url": format!("/api/music/tracks/{id}/file?library_id={library_id}"),
        "library_id": library_id,
    }))
}

fn album_json(row: &Row<'_>, library_id: &str) -> rusqlite::Result<Value> {
    let id: i64 = row.get(0)?;
    let art_track: Option<i64> = row.get(6)?;
    Ok(json!({
        "id": id,
        "title": row.get::<_, String>(1)?,
        "artist": row.get::<_, String>(2)?,
        "genre": row.get::<_, Option<String>>(3)?,
        "year": row.get::<_, Option<i64>>(4)?,
        "track_count": row.get::<_, i64>(5)?,
        "artwork_url": art_track.map(|track_id| format!("/api/music/tracks/{track_id}/artwork?library_id={library_id}")),
        "library_id": library_id,
    }))
}

const ALBUM_SELECT: &str = "SELECT a.id, a.title, a.artist, a.genre, a.year,
    COUNT(t.id), MIN(CASE WHEN COALESCE(t.artwork_path,a.artwork_path) IS NOT NULL THEN t.id END)
    FROM music_albums a JOIN music_tracks t ON t.album_id = a.id AND t.is_available = 1";

fn filters(q: &MusicQuery, albums: bool) -> (String, Vec<rusqlite::types::Value>) {
    use rusqlite::types::Value as SqlValue;
    let mut clauses = vec!["t.is_available = 1".to_string()];
    let mut values = Vec::new();
    if let Some(search) = q.q.as_deref().filter(|s| !s.trim().is_empty()) {
        let pattern = format!("%{}%", search.trim());
        clauses.push(if albums {
            "(a.title LIKE ? OR a.artist LIKE ?)".to_string()
        } else {
            "(t.title LIKE ? OR t.artist LIKE ? OR a.title LIKE ?)".to_string()
        });
        values.push(SqlValue::Text(pattern.clone()));
        values.push(SqlValue::Text(pattern.clone()));
        if !albums {
            values.push(SqlValue::Text(pattern));
        }
    }
    if let Some(artist) = q.artist.as_deref().filter(|s| !s.is_empty()) {
        clauses.push("(t.artist = ? OR a.artist = ?)".into());
        values.push(SqlValue::Text(artist.into()));
        values.push(SqlValue::Text(artist.into()));
    }
    if let Some(album) = q.album.as_deref().filter(|s| !s.is_empty()) {
        clauses.push("a.title = ?".into());
        values.push(SqlValue::Text(album.into()));
    }
    if let Some(genre) = q.genre.as_deref().filter(|s| !s.is_empty()) {
        clauses.push("(t.genre = ? OR a.genre = ?)".into());
        values.push(SqlValue::Text(genre.into()));
        values.push(SqlValue::Text(genre.into()));
    }
    if let Some(year) = q.year {
        clauses.push("(t.year = ? OR a.year = ?)".into());
        values.push(SqlValue::Integer(year));
        values.push(SqlValue::Integer(year));
    }
    if q.favorites_only.unwrap_or(false) {
        clauses.push("t.is_favorite = 1".into());
    }
    if let Some(directory_id) = q.directory_id {
        clauses.push("t.directory_id = ?".into());
        values.push(SqlValue::Integer(directory_id));
    }
    if let Some(collection_id) = q.collection_id {
        clauses.push(
            "EXISTS (SELECT 1 FROM music_collection_items ci WHERE ci.collection_id = ?
             AND ((ci.item_type = 'track' AND ci.item_id = t.id)
               OR (ci.item_type = 'album' AND ci.item_id = a.id)))"
                .into(),
        );
        values.push(SqlValue::Integer(collection_id));
    }
    (format!(" WHERE {}", clauses.join(" AND ")), values)
}

async fn list_tracks(
    State(state): State<AppState>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let (page, per_page) = paging(&q);
    let (where_sql, values) = filters(&q, false);
    let conn = lib.main_pool.get()?;
    let total: i64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM music_tracks t JOIN music_albums a ON a.id=t.album_id {where_sql}"),
        params_from_iter(values.iter()), |row| row.get(0))?;
    let mut page_values = values;
    page_values.push((per_page).into());
    page_values.push(((page - 1) * per_page).into());
    let sql = format!("{TRACK_SELECT} {where_sql} ORDER BY t.artist COLLATE NOCASE, a.title COLLATE NOCASE, t.disc_number, t.track_number, t.id LIMIT ? OFFSET ?");
    let mut stmt = conn.prepare(&sql)?;
    let tracks = stmt
        .query_map(params_from_iter(page_values.iter()), |row| {
            track_json(row, &lib.uuid)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Json(
        json!({"tracks":tracks,"total":total,"page":page,"per_page":per_page}),
    ))
}

async fn list_albums(
    State(state): State<AppState>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let (page, per_page) = paging(&q);
    let (where_sql, values) = filters(&q, true);
    let conn = lib.main_pool.get()?;
    let total: i64 = conn.query_row(
        &format!("SELECT COUNT(DISTINCT a.id) FROM music_albums a JOIN music_tracks t ON t.album_id=a.id {where_sql}"),
        params_from_iter(values.iter()), |row| row.get(0))?;
    let mut page_values = values;
    page_values.push(per_page.into());
    page_values.push(((page - 1) * per_page).into());
    let sql = format!("{ALBUM_SELECT} {where_sql} GROUP BY a.id ORDER BY a.artist COLLATE NOCASE, a.title COLLATE NOCASE LIMIT ? OFFSET ?");
    let mut stmt = conn.prepare(&sql)?;
    let albums = stmt
        .query_map(params_from_iter(page_values.iter()), |row| {
            album_json(row, &lib.uuid)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Json(
        json!({"albums":albums,"total":total,"page":page,"per_page":per_page}),
    ))
}

fn get_track(lib: &LibraryContext, id: i64) -> Result<Value, AppError> {
    let conn = lib.main_pool.get()?;
    let sql = format!("{TRACK_SELECT} WHERE t.id = ?1 AND t.is_available = 1");
    conn.query_row(&sql, params![id], |row| track_json(row, &lib.uuid))
        .optional()?
        .ok_or_else(|| AppError::NotFound("Track not found".into()))
}

async fn track_detail(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    Ok(Json(json!({"track": get_track(&lib, id)?})))
}

async fn album_detail(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    let album_sql = format!("{ALBUM_SELECT} WHERE a.id = ?1 GROUP BY a.id");
    let album: Value = conn
        .query_row(&album_sql, params![id], |row| album_json(row, &lib.uuid))
        .optional()?
        .ok_or_else(|| AppError::NotFound("Album not found".into()))?;
    let sql = format!("{TRACK_SELECT} WHERE t.album_id = ?1 AND t.is_available = 1 ORDER BY t.disc_number, t.track_number, t.id");
    let mut stmt = conn.prepare(&sql)?;
    let tracks = stmt
        .query_map(params![id], |row| track_json(row, &lib.uuid))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Json(json!({"album":album,"tracks":tracks})))
}

async fn facets(
    State(state): State<AppState>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    let strings = |sql: &str| -> Result<Vec<String>, AppError> {
        let mut stmt = conn.prepare(sql)?;
        let values = stmt
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(values)
    };
    let artists = strings("SELECT DISTINCT artist FROM music_tracks WHERE is_available=1 ORDER BY artist COLLATE NOCASE")?;
    let albums = strings("SELECT DISTINCT a.title FROM music_albums a JOIN music_tracks t ON t.album_id=a.id WHERE t.is_available=1 ORDER BY a.title COLLATE NOCASE")?;
    let genres = strings("SELECT DISTINCT genre FROM music_tracks WHERE is_available=1 AND genre IS NOT NULL ORDER BY genre COLLATE NOCASE")?;
    let mut stmt = conn.prepare("SELECT DISTINCT year FROM music_tracks WHERE is_available=1 AND year IS NOT NULL ORDER BY year DESC")?;
    let years = stmt
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut stmt = conn.prepare("SELECT id,name,path FROM watch_directories WHERE id IN (SELECT DISTINCT directory_id FROM music_tracks WHERE is_available=1) ORDER BY COALESCE(name,path)")?;
    let folders = stmt
        .query_map([], |row| {
            let name: Option<String> = row.get(1)?;
            let path: String = row.get(2)?;
            Ok(json!({"id":row.get::<_,i64>(0)?,"name":name.unwrap_or(path)}))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(Json(
        json!({"artists":artists,"albums":albums,"genres":genres,"years":years,"folders":folders}),
    ))
}

fn genre_set(value: &str) -> HashSet<String> {
    value
        .split(|c: char| matches!(c, ',' | ';' | '/' | '|'))
        .map(|part| part.trim().to_lowercase())
        .filter(|part| !part.is_empty())
        .collect()
}

async fn related_tracks(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let seed = get_track(&lib, id)?;
    let excluded: HashSet<i64> = q
        .exclude_ids
        .as_deref()
        .unwrap_or("")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let conn = lib.main_pool.get()?;
    let sql = format!("{TRACK_SELECT} WHERE t.is_available = 1 AND t.id != ?1");
    let mut stmt = conn.prepare(&sql)?;
    let candidates = stmt
        .query_map(params![id], |row| track_json(row, &lib.uuid))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut used_artists: HashMap<String, usize> = HashMap::new();
    for candidate in &candidates {
        if candidate["id"]
            .as_i64()
            .is_some_and(|candidate_id| excluded.contains(&candidate_id))
        {
            if let Some(artist) = candidate["artist"].as_str() {
                *used_artists.entry(artist.to_lowercase()).or_default() += 1;
            }
        }
    }
    let seed_artist = seed["artist"].as_str().unwrap_or("").to_lowercase();
    let seed_genres = genre_set(seed["genre"].as_str().unwrap_or(""));
    let seed_year = seed["year"].as_i64();
    let mut scored: Vec<(i64, Value)> = candidates
        .into_iter()
        .filter_map(|candidate| {
            let candidate_id = candidate["id"].as_i64()?;
            if excluded.contains(&candidate_id) {
                return None;
            }
            let artist = candidate["artist"].as_str().unwrap_or("").to_lowercase();
            let same_artist = artist != "unknown artist" && artist == seed_artist;
            let genres = genre_set(candidate["genre"].as_str().unwrap_or(""));
            let genre_matches = genres.intersection(&seed_genres).count() as i64;
            // A match must share an actual musical descriptor. Year or album alone
            // never makes an otherwise unrelated file a recommendation.
            if !same_artist && genre_matches == 0 {
                return None;
            }
            let mut score = genre_matches * 10 + if same_artist { 7 } else { 0 };
            if let (Some(a), Some(b)) = (seed_year, candidate["year"].as_i64()) {
                if (a - b).abs() <= 5 {
                    score += 2;
                }
            }
            score -= used_artists.get(&artist).copied().unwrap_or(0) as i64 * 5;
            if same_artist {
                score -= 2;
            }
            Some((score, candidate))
        })
        .collect();
    scored.sort_by(|(a_score, a), (b_score, b)| {
        b_score
            .cmp(a_score)
            .then_with(|| a["id"].as_i64().cmp(&b["id"].as_i64()))
    });
    let limit = q.limit.unwrap_or(20).clamp(1, 100);
    let mut result = Vec::new();
    let mut recent_artists = Vec::<String>::new();
    while !scored.is_empty() && result.len() < limit {
        let position = scored
            .iter()
            .position(|(_, value)| {
                let artist = value["artist"].as_str().unwrap_or("").to_lowercase();
                !recent_artists.contains(&artist)
            })
            .unwrap_or(0);
        let (_, track) = scored.remove(position);
        let artist = track["artist"].as_str().unwrap_or("").to_lowercase();
        recent_artists.push(artist);
        if recent_artists.len() > 2 {
            recent_artists.remove(0);
        }
        result.push(track);
    }
    let reason = if result.is_empty() {
        Some("No further related tracks in this library")
    } else {
        None
    };
    Ok(Json(
        json!({"tracks":result,"reason":reason,"seed_track_id":id}),
    ))
}

#[derive(Deserialize)]
struct FavoriteBody {
    is_favorite: bool,
}

async fn set_favorite(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
    Json(body): Json<FavoriteBody>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    let changed = conn.execute(
        "UPDATE music_tracks SET is_favorite=?1 WHERE id=?2",
        params![body.is_favorite, id],
    )?;
    if changed == 0 {
        return Err(AppError::NotFound("Track not found".into()));
    }
    Ok(Json(json!({"id":id,"is_favorite":body.is_favorite})))
}

fn track_path(lib: &LibraryContext, id: i64, artwork: bool) -> Result<PathBuf, AppError> {
    let conn = lib.main_pool.get()?;
    let sql = if artwork {
        "SELECT COALESCE(t.artwork_path,a.artwork_path) FROM music_tracks t JOIN music_albums a ON a.id=t.album_id WHERE t.id=?1 AND t.is_available=1"
    } else {
        "SELECT t.path FROM music_tracks t WHERE t.id=?1 AND t.is_available=1"
    };
    let path: Option<String> = conn
        .query_row(sql, params![id], |row| row.get(0))
        .optional()?
        .flatten();
    let path = path.ok_or_else(|| {
        AppError::NotFound(
            if artwork {
                "Artwork unavailable"
            } else {
                "Track not found"
            }
            .into(),
        )
    })?;
    let path = PathBuf::from(path);
    if !path.is_file() {
        return Err(AppError::NotFound("File unavailable".into()));
    }
    Ok(path)
}

async fn track_file(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
    request: Request,
) -> Result<Response, AppError> {
    let lib = library(&state, &q)?;
    let path = track_path(&lib, id, false)?;
    Ok(ServeFile::new(path)
        .oneshot(request)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .into_response())
}

async fn track_artwork(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
    request: Request,
) -> Result<Response, AppError> {
    let lib = library(&state, &q)?;
    let path = track_path(&lib, id, true)?;
    Ok(ServeFile::new(path)
        .oneshot(request)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .into_response())
}

fn collection_json(
    conn: &rusqlite::Connection,
    id: i64,
    library_id: &str,
) -> Result<Value, AppError> {
    conn.query_row(
        "SELECT c.id,c.name,c.description,c.created_at,c.updated_at,
          (SELECT COUNT(*) FROM music_collection_items WHERE collection_id=c.id),
          (SELECT COALESCE(t.id,(SELECT MIN(t2.id) FROM music_tracks t2 WHERE t2.album_id=ci.item_id))
           FROM music_collection_items ci LEFT JOIN music_tracks t ON ci.item_type='track' AND t.id=ci.item_id
           WHERE ci.collection_id=c.id ORDER BY ci.sort_order LIMIT 1)
         FROM music_collections c WHERE c.id=?1",
        params![id], |row| {
            let artwork_track: Option<i64> = row.get(6)?;
            Ok(json!({
                "id":row.get::<_,i64>(0)?,"name":row.get::<_,String>(1)?,
                "description":row.get::<_,Option<String>>(2)?,
                "created_at":row.get::<_,String>(3)?,"updated_at":row.get::<_,Option<String>>(4)?,
                "item_count":row.get::<_,i64>(5)?,
                "artwork_url":artwork_track.map(|id| format!("/api/music/tracks/{id}/artwork?library_id={library_id}")),
                "library_id":library_id,
            }))
        }
    ).optional()?.ok_or_else(|| AppError::NotFound("Music collection not found".into()))
}

async fn list_collections(
    State(state): State<AppState>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    let mut stmt = conn.prepare(
        "SELECT id FROM music_collections ORDER BY COALESCE(updated_at,created_at) DESC, id DESC",
    )?;
    let ids = stmt
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let collections = ids
        .into_iter()
        .map(|id| collection_json(&conn, id, &lib.uuid))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Json(json!({"collections":collections})))
}

#[derive(Deserialize)]
struct CollectionBody {
    name: Option<String>,
    description: Option<String>,
}

async fn create_collection(
    State(state): State<AppState>,
    Query(q): Query<MusicQuery>,
    Json(body): Json<CollectionBody>,
) -> Result<Json<Value>, AppError> {
    let name = body.name.unwrap_or_default().trim().to_string();
    if name.is_empty() {
        return Err(AppError::BadRequest("Collection name is required".into()));
    }
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    conn.execute(
        "INSERT INTO music_collections (name,description) VALUES (?1,?2)",
        params![name, body.description],
    )?;
    Ok(Json(
        json!({"collection":collection_json(&conn,conn.last_insert_rowid(),&lib.uuid)?}),
    ))
}

async fn update_collection(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
    Json(body): Json<CollectionBody>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    if body
        .name
        .as_deref()
        .is_some_and(|name| name.trim().is_empty())
    {
        return Err(AppError::BadRequest(
            "Collection name cannot be empty".into(),
        ));
    }
    conn.execute("UPDATE music_collections SET name=COALESCE(?1,name),description=COALESCE(?2,description),updated_at=datetime('now') WHERE id=?3",params![body.name,body.description,id])?;
    Ok(Json(
        json!({"collection":collection_json(&conn,id,&lib.uuid)?}),
    ))
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    let changed = conn.execute("DELETE FROM music_collections WHERE id=?1", params![id])?;
    if changed == 0 {
        return Err(AppError::NotFound("Music collection not found".into()));
    }
    Ok(Json(json!({"deleted":true})))
}

#[derive(Deserialize)]
struct CollectionItemBody {
    item_type: String,
    item_id: i64,
}

async fn add_collection_item(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
    Json(body): Json<CollectionItemBody>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    collection_json(&conn, id, &lib.uuid)?;
    let table = match body.item_type.as_str() {
        "album" => "music_albums",
        "track" => "music_tracks",
        _ => {
            return Err(AppError::BadRequest(
                "item_type must be album or track".into(),
            ))
        }
    };
    let found: bool = conn.query_row(
        &format!("SELECT EXISTS(SELECT 1 FROM {table} WHERE id=?1)"),
        params![body.item_id],
        |row| row.get(0),
    )?;
    if !found {
        return Err(AppError::NotFound("Music item not found".into()));
    }
    conn.execute("INSERT OR IGNORE INTO music_collection_items (collection_id,item_type,item_id,sort_order) VALUES (?1,?2,?3,(SELECT COALESCE(MAX(sort_order),0)+1 FROM music_collection_items WHERE collection_id=?1))",params![id,body.item_type,body.item_id])?;
    conn.execute(
        "UPDATE music_collections SET updated_at=datetime('now') WHERE id=?1",
        params![id],
    )?;
    Ok(Json(
        json!({"collection":collection_json(&conn,id,&lib.uuid)?}),
    ))
}

async fn remove_collection_item(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
    Json(body): Json<CollectionItemBody>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    conn.execute(
        "DELETE FROM music_collection_items WHERE collection_id=?1 AND item_type=?2 AND item_id=?3",
        params![id, body.item_type, body.item_id],
    )?;
    conn.execute(
        "UPDATE music_collections SET updated_at=datetime('now') WHERE id=?1",
        params![id],
    )?;
    Ok(Json(
        json!({"collection":collection_json(&conn,id,&lib.uuid)?}),
    ))
}

async fn collection_detail(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Query(q): Query<MusicQuery>,
) -> Result<Json<Value>, AppError> {
    let lib = library(&state, &q)?;
    let conn = lib.main_pool.get()?;
    let collection = collection_json(&conn, id, &lib.uuid)?;
    let mut stmt = conn.prepare("SELECT item_type,item_id FROM music_collection_items WHERE collection_id=?1 ORDER BY sort_order,id")?;
    let items = stmt
        .query_map(params![id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut albums = Vec::new();
    let mut tracks = Vec::new();
    for (kind, item_id) in items {
        if kind == "album" {
            let sql = format!("{ALBUM_SELECT} WHERE a.id=?1 GROUP BY a.id");
            if let Some(album) = conn
                .query_row(&sql, params![item_id], |row| album_json(row, &lib.uuid))
                .optional()?
            {
                albums.push(album);
            }
        } else if let Ok(track) = get_track(&lib, item_id) {
            tracks.push(track);
        }
    }
    Ok(Json(
        json!({"collection":collection,"albums":albums,"tracks":tracks}),
    ))
}
