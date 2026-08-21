use std::path::Path;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use chrono::Utc;
use futures_util::StreamExt;
use reqwest::redirect::Policy;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::online::models::{
    PublicationSnapshot, RemoteItem, RemotePage, RemoteSource, PUBLICATION_STATES,
};
use crate::online::{provider, security};
use crate::server::error::AppError;
use crate::server::state::AppState;

const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_MEDIA_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/sources", get(list_sources).post(create_source))
        .route(
            "/sources/{source_id}",
            patch(update_source).delete(delete_source),
        )
        .route("/sources/{source_id}/probe", post(probe_source))
        .route(
            "/sources/{source_id}/connections",
            get(list_connections).post(create_connection),
        )
        .route(
            "/sources/{source_id}/connections/{connection_id}",
            axum::routing::delete(delete_connection),
        )
        .route("/sources/{source_id}/browse", get(browse))
        .route(
            "/sources/{source_id}/items/{post_id}/asset/{kind}",
            get(asset),
        )
        .route(
            "/sources/{source_id}/items/{post_id}/import",
            post(import_item),
        )
        .route("/publications/preflight", post(preflight_publication))
        .route(
            "/publications",
            get(list_publications).post(create_publication),
        )
        .route(
            "/publications/{publication_id}/simulate",
            post(simulate_delivery),
        )
        .route(
            "/publications/{publication_id}/deliver",
            post(deliver_publication),
        )
        .route(
            "/publications/{publication_id}/targets/{target_id}",
            patch(update_publication_target),
        )
}

fn open_db(data_dir: &Path) -> Result<Connection, AppError> {
    let conn = Connection::open(data_dir.join("online.db"))?;
    conn.execute_batch(
        "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
         CREATE TABLE IF NOT EXISTS remote_sources(
           source_id TEXT PRIMARY KEY, provider_family TEXT NOT NULL, display_name TEXT NOT NULL,
           normalized_base_url TEXT NOT NULL UNIQUE, policy_profile TEXT NOT NULL DEFAULT 'unrestricted',
           enabled INTEGER NOT NULL DEFAULT 1, allow_local_network INTEGER NOT NULL DEFAULT 0,
           capabilities_json TEXT NOT NULL, last_probe_at TEXT, created_at TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS remote_connections(
           connection_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
           account_display_name TEXT, credential_ref TEXT, trust_state TEXT NOT NULL DEFAULT 'unverified', last_probe_at TEXT);
         CREATE TABLE IF NOT EXISTS remote_cache(
           cache_key TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
           payload_json TEXT NOT NULL, etag TEXT, fetched_at TEXT NOT NULL, expires_at TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS remote_item_cache(
           source_id TEXT NOT NULL, remote_post_id TEXT NOT NULL, item_json TEXT NOT NULL, fetched_at TEXT NOT NULL,
           PRIMARY KEY(source_id,remote_post_id));
         CREATE TABLE IF NOT EXISTS import_provenance(
           source_id TEXT NOT NULL, remote_post_id TEXT NOT NULL, library_id TEXT NOT NULL,
           directory_id INTEGER NOT NULL, image_id INTEGER, content_sha256 TEXT NOT NULL,
           canonical_url TEXT NOT NULL, imported_at TEXT NOT NULL,
           UNIQUE(source_id,remote_post_id,library_id,directory_id,content_sha256));
         CREATE TABLE IF NOT EXISTS publication_intents(
           publication_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS publication_targets(
           target_id TEXT PRIMARY KEY, publication_id TEXT NOT NULL REFERENCES publication_intents(publication_id) ON DELETE CASCADE,
           connection_id TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
           attempt_count INTEGER NOT NULL DEFAULT 0, remote_post_id TEXT, remote_url TEXT,
           receipt_json TEXT, error_json TEXT, updated_at TEXT NOT NULL);"
    )?;
    Ok(conn)
}

fn source_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RemoteSource> {
    let caps: String = row.get(7)?;
    Ok(RemoteSource {
        source_id: row.get(0)?,
        provider_family: row.get(1)?,
        display_name: row.get(2)?,
        normalized_base_url: row.get(3)?,
        policy_profile: row.get(4)?,
        enabled: row.get::<_, i64>(5)? != 0,
        allow_local_network: row.get::<_, i64>(6)? != 0,
        capabilities: serde_json::from_str(&caps)
            .unwrap_or_else(|_| provider::capabilities("unknown")),
        last_probe_at: row.get(8)?,
    })
}

fn load_source(data_dir: &Path, source_id: &str) -> Result<RemoteSource, AppError> {
    open_db(data_dir)?.query_row(
        "SELECT source_id,provider_family,display_name,normalized_base_url,policy_profile,enabled,allow_local_network,capabilities_json,last_probe_at FROM remote_sources WHERE source_id=?1",
        params![source_id], source_from_row,
    ).optional()?.ok_or_else(|| AppError::NotFound("Remote source not found".into()))
}

#[derive(Debug, Deserialize, Serialize)]
struct StoredCredential {
    auth_kind: String,
    secret: String,
}

fn credential_dir(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join("online-credentials")
}

fn credential_path(data_dir: &Path, credential_ref: &str) -> Result<std::path::PathBuf, AppError> {
    if Path::new(credential_ref)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(credential_ref)
    {
        return Err(AppError::Internal("Invalid credential reference".into()));
    }
    Ok(credential_dir(data_dir).join(credential_ref))
}

fn write_credential(
    data_dir: &Path,
    credential_ref: &str,
    credential: &StoredCredential,
) -> Result<(), AppError> {
    let bytes =
        serde_json::to_vec(credential).map_err(|error| AppError::Internal(error.to_string()))?;
    crate::server::credentials::store_application_secret(
        &credential_path(data_dir, credential_ref)?,
        &bytes,
    )?;
    Ok(())
}

fn load_credential(data_dir: &Path, source_id: &str) -> Result<Option<StoredCredential>, AppError> {
    let credential_ref: Option<String> = open_db(data_dir)?
        .query_row(
            "SELECT credential_ref FROM remote_connections WHERE source_id=?1 AND credential_ref IS NOT NULL ORDER BY last_probe_at DESC, connection_id LIMIT 1",
            params![source_id],
            |row| row.get(0),
        )
        .optional()?;
    credential_ref
        .map(|credential_ref| {
            let bytes = crate::server::credentials::load_application_secret(&credential_path(
                data_dir,
                &credential_ref,
            )?)?;
            serde_json::from_slice(&bytes)
                .map_err(|_| AppError::Internal("Stored remote credential is corrupt".into()))
        })
        .transpose()
}

fn authenticated_request(
    request: reqwest::RequestBuilder,
    credential: Option<&StoredCredential>,
) -> Result<reqwest::RequestBuilder, AppError> {
    match credential {
        None => Ok(request),
        Some(credential) if credential.auth_kind == "bearer" => {
            Ok(request.bearer_auth(&credential.secret))
        }
        Some(credential) if credential.auth_kind == "api_key_header" => {
            let value =
                reqwest::header::HeaderValue::from_str(&credential.secret).map_err(|_| {
                    AppError::BadRequest("Credential contains invalid header bytes".into())
                })?;
            Ok(request.header("X-API-Key", value))
        }
        Some(_) => Err(AppError::BadRequest(
            "Unsupported remote authentication kind".into(),
        )),
    }
}

#[derive(Deserialize)]
struct ConnectionRequest {
    account_display_name: Option<String>,
    auth_kind: String,
    secret: String,
}

async fn list_connections(
    State(state): State<AppState>,
    AxumPath(source_id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    load_source(state.data_dir(), &source_id)?;
    let conn = open_db(state.data_dir())?;
    let mut stmt = conn.prepare("SELECT connection_id,account_display_name,trust_state,last_probe_at FROM remote_connections WHERE source_id=?1 ORDER BY account_display_name,connection_id")?;
    let connections = stmt
        .query_map(params![source_id], |row| {
            Ok(json!({
                "connection_id": row.get::<_, String>(0)?,
                "account_display_name": row.get::<_, Option<String>>(1)?,
                "trust_state": row.get::<_, String>(2)?,
                "last_probe_at": row.get::<_, Option<String>>(3)?,
                "has_credential": true,
            }))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Json(json!({"connections": connections})))
}

async fn create_connection(
    State(state): State<AppState>,
    AxumPath(source_id): AxumPath<String>,
    Json(body): Json<ConnectionRequest>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    load_source(state.data_dir(), &source_id)?;
    if !matches!(body.auth_kind.as_str(), "bearer" | "api_key_header") {
        return Err(AppError::BadRequest(
            "Authentication kind must be bearer or api_key_header".into(),
        ));
    }
    if body.secret.is_empty() || body.secret.len() > 16 * 1024 {
        return Err(AppError::BadRequest(
            "Credential is empty or too large".into(),
        ));
    }
    let connection_id = Uuid::new_v4().to_string();
    let credential_ref = format!("{}.json", Uuid::new_v4());
    write_credential(
        state.data_dir(),
        &credential_ref,
        &StoredCredential {
            auth_kind: body.auth_kind,
            secret: body.secret,
        },
    )?;
    let insert = open_db(state.data_dir())?.execute(
        "INSERT INTO remote_connections(connection_id,source_id,account_display_name,credential_ref,trust_state,last_probe_at) VALUES(?1,?2,?3,?4,'unverified',NULL)",
        params![connection_id, source_id, body.account_display_name, credential_ref],
    );
    if let Err(error) = insert {
        let _ = std::fs::remove_file(credential_path(state.data_dir(), &credential_ref)?);
        return Err(error.into());
    }
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "connection_id": connection_id,
            "source_id": source_id,
            "account_display_name": body.account_display_name,
            "trust_state": "unverified",
            "has_credential": true,
        })),
    ))
}

async fn delete_connection(
    State(state): State<AppState>,
    AxumPath((source_id, connection_id)): AxumPath<(String, String)>,
) -> Result<StatusCode, AppError> {
    let conn = open_db(state.data_dir())?;
    let credential_ref: Option<String> = conn
        .query_row(
            "SELECT credential_ref FROM remote_connections WHERE source_id=?1 AND connection_id=?2",
            params![source_id, connection_id],
            |row| row.get(0),
        )
        .optional()?;
    let Some(credential_ref) = credential_ref else {
        return Err(AppError::NotFound("Remote connection not found".into()));
    };
    conn.execute(
        "DELETE FROM remote_connections WHERE source_id=?1 AND connection_id=?2",
        params![source_id, connection_id],
    )?;
    let _ = std::fs::remove_file(credential_path(state.data_dir(), &credential_ref)?);
    Ok(StatusCode::NO_CONTENT)
}

async fn list_sources(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let conn = open_db(state.data_dir())?;
    let mut stmt = conn.prepare("SELECT source_id,provider_family,display_name,normalized_base_url,policy_profile,enabled,allow_local_network,capabilities_json,last_probe_at FROM remote_sources ORDER BY display_name")?;
    let sources: Vec<_> = stmt
        .query_map([], source_from_row)?
        .collect::<Result<_, _>>()?;
    Ok(Json(json!({"sources":sources})))
}

#[derive(Deserialize)]
struct SourceRequest {
    provider_family: String,
    display_name: String,
    base_url: String,
    policy_profile: Option<String>,
    allow_local_network: Option<bool>,
}
async fn create_source(
    State(state): State<AppState>,
    Json(body): Json<SourceRequest>,
) -> Result<(StatusCode, Json<RemoteSource>), AppError> {
    let supported = ["donutbooru", "danbooru"].contains(&body.provider_family.as_str())
        || (cfg!(debug_assertions) && body.provider_family == "fake");
    if !supported {
        return Err(AppError::BadRequest(
            "Supported provider families are donutbooru and danbooru".into(),
        ));
    }
    let allow_local = body.allow_local_network.unwrap_or(false);
    let base = security::normalize_base_url(&body.base_url, allow_local)?;
    let source = RemoteSource {
        source_id: Uuid::new_v4().to_string(),
        provider_family: body.provider_family.clone(),
        display_name: body.display_name.trim().to_string(),
        normalized_base_url: base,
        policy_profile: body.policy_profile.unwrap_or_else(|| "unrestricted".into()),
        enabled: true,
        allow_local_network: allow_local,
        capabilities: provider::capabilities(&body.provider_family),
        last_probe_at: None,
    };
    if source.display_name.is_empty() {
        return Err(AppError::BadRequest("Display name is required".into()));
    }
    open_db(state.data_dir())?
        .execute(
            "INSERT INTO remote_sources VALUES(?1,?2,?3,?4,?5,1,?6,?7,NULL,?8)",
            params![
                source.source_id,
                source.provider_family,
                source.display_name,
                source.normalized_base_url,
                source.policy_profile,
                source.allow_local_network as i64,
                serde_json::to_string(&source.capabilities).unwrap(),
                Utc::now().to_rfc3339()
            ],
        )
        .map_err(|e| {
            if e.to_string().contains("UNIQUE") {
                AppError::BadRequest("A source with this origin already exists".into())
            } else {
                e.into()
            }
        })?;
    Ok((StatusCode::CREATED, Json(source)))
}

#[derive(Deserialize)]
struct SourcePatch {
    display_name: Option<String>,
    enabled: Option<bool>,
    policy_profile: Option<String>,
}
async fn update_source(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<SourcePatch>,
) -> Result<Json<RemoteSource>, AppError> {
    let mut s = load_source(state.data_dir(), &id)?;
    if let Some(v) = body.display_name {
        s.display_name = v;
    }
    if let Some(v) = body.enabled {
        s.enabled = v;
    }
    if let Some(v) = body.policy_profile {
        s.policy_profile = v;
    }
    open_db(state.data_dir())?.execute(
        "UPDATE remote_sources SET display_name=?2,enabled=?3,policy_profile=?4 WHERE source_id=?1",
        params![id, s.display_name, s.enabled as i64, s.policy_profile],
    )?;
    Ok(Json(s))
}
async fn delete_source(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, AppError> {
    let conn = open_db(state.data_dir())?;
    let credential_refs = {
        let mut stmt = conn.prepare(
            "SELECT credential_ref FROM remote_connections WHERE source_id=?1 AND credential_ref IS NOT NULL",
        )?;
        let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    let changed = conn.execute("DELETE FROM remote_sources WHERE source_id=?1", params![id])?;
    if changed == 0 {
        return Err(AppError::NotFound("Remote source not found".into()));
    }
    for credential_ref in credential_refs {
        let _ = std::fs::remove_file(credential_path(state.data_dir(), &credential_ref)?);
    }
    Ok(StatusCode::NO_CONTENT)
}

fn client() -> Result<reqwest::Client, AppError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        .redirect(Policy::none())
        .user_agent(format!("LocalBooru/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| AppError::Internal(e.to_string()))
}
async fn checked_json(
    data_dir: &Path,
    source: &RemoteSource,
    url: reqwest::Url,
) -> Result<(Value, Option<String>), AppError> {
    security::validate_resolved_url(&url, source.allow_local_network).await?;
    let credential = load_credential(data_dir, &source.source_id)?;
    let mut current = url;
    for _ in 0..=3 {
        let response = authenticated_request(client()?.get(current.clone()), credential.as_ref())?
            .send()
            .await
            .map_err(|e| {
                AppError::ServiceUnavailable(format!("Remote source request failed: {e}"))
            })?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| {
                    AppError::ServiceUnavailable("Remote redirect omitted Location".into())
                })?;
            current = current
                .join(location)
                .map_err(|_| AppError::ServiceUnavailable("Remote redirect was invalid".into()))?;
            security::validate_resolved_url(&current, source.allow_local_network).await?;
            continue;
        }
        if !response.status().is_success() {
            return Err(AppError::ServiceUnavailable(format!(
                "Remote source returned {}",
                response.status()
            )));
        }
        let etag = response
            .headers()
            .get(header::ETAG)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AppError::ServiceUnavailable(e.to_string()))?;
            if bytes.len().saturating_add(chunk.len()) > MAX_JSON_BYTES {
                return Err(AppError::ServiceUnavailable(
                    "Remote response exceeded size limit".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        return Ok((
            serde_json::from_slice(&bytes).map_err(|_| {
                AppError::ServiceUnavailable("Remote source returned invalid JSON".into())
            })?,
            etag,
        ));
    }
    Err(AppError::ServiceUnavailable(
        "Remote source exceeded redirect limit".into(),
    ))
}

async fn probe_source(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    let path = provider::list_path(&source.provider_family, "", 1, 1)?;
    let url = reqwest::Url::parse(&(source.normalized_base_url.clone() + &path))
        .map_err(|_| AppError::BadRequest("Invalid source URL".into()))?;
    let (value, _) = checked_json(state.data_dir(), &source, url).await?;
    provider::normalize_page(
        &source.provider_family,
        &id,
        &source.normalized_base_url,
        value,
        1,
        1,
    )?;
    let now = Utc::now().to_rfc3339();
    open_db(state.data_dir())?.execute(
        "UPDATE remote_sources SET last_probe_at=?2 WHERE source_id=?1",
        params![id, now],
    )?;
    open_db(state.data_dir())?.execute(
        "UPDATE remote_connections SET trust_state='verified',last_probe_at=?2 WHERE source_id=?1",
        params![id, now],
    )?;
    Ok(Json(
        json!({"ok":true,"capabilities":source.capabilities,"last_probe_at":now}),
    ))
}

#[derive(Deserialize)]
struct BrowseQuery {
    q: Option<String>,
    page: Option<u32>,
    per_page: Option<u32>,
}
async fn browse(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<BrowseQuery>,
) -> Result<Json<RemotePage>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    if !source.enabled {
        return Err(AppError::Forbidden("Remote source is disabled".into()));
    }
    let page = query.page.unwrap_or(1).max(1);
    let per_page = query.per_page.unwrap_or(40).clamp(1, 100);
    let q = query.q.unwrap_or_default();
    let key = format!("{}:{}:{}:{}", id, page, per_page, q);
    let path = provider::list_path(&source.provider_family, &q, page, per_page)?;
    let url = reqwest::Url::parse(&(source.normalized_base_url.clone() + &path))
        .map_err(|_| AppError::BadRequest("Invalid source request URL".into()))?;
    match checked_json(state.data_dir(), &source, url).await {
        Ok((value, etag)) => {
            let mut remote_page = provider::normalize_page(
                &source.provider_family,
                &id,
                &source.normalized_base_url,
                value,
                page,
                per_page,
            )?;
            enforce_content_policy(
                &mut remote_page,
                &source.policy_profile,
                state.is_family_mode_locked(),
            );
            let now = Utc::now();
            let conn = open_db(state.data_dir())?;
            conn.execute(
                "INSERT OR REPLACE INTO remote_cache VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    key,
                    id,
                    serde_json::to_string(&remote_page).unwrap(),
                    etag,
                    now.to_rfc3339(),
                    (now + chrono::Duration::minutes(10)).to_rfc3339()
                ],
            )?;
            for item in &remote_page.items {
                conn.execute(
                    "INSERT OR REPLACE INTO remote_item_cache VALUES(?1,?2,?3,?4)",
                    params![
                        id,
                        item.remote_post_id,
                        serde_json::to_string(item).unwrap(),
                        now.to_rfc3339()
                    ],
                )?;
            }
            Ok(Json(remote_page))
        }
        Err(error) => {
            let conn = open_db(state.data_dir())?;
            let cached: Option<String> = conn
                .query_row(
                    "SELECT payload_json FROM remote_cache WHERE cache_key=?1",
                    params![key],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(payload) = cached {
                let mut p: RemotePage = serde_json::from_str(&payload).map_err(|_| error)?;
                enforce_content_policy(
                    &mut p,
                    &source.policy_profile,
                    state.is_family_mode_locked(),
                );
                p.stale = true;
                Ok(Json(p))
            } else {
                Err(error)
            }
        }
    }
}

fn enforce_content_policy(page: &mut RemotePage, profile: &str, family_locked: bool) {
    let safe_only = family_locked || profile == "safe";
    let hide_adult = safe_only || profile == "adult-hidden";
    page.items.retain(|item| {
        let rating = item.rating.as_deref().unwrap_or("").to_ascii_lowercase();
        if safe_only {
            matches!(rating.as_str(), "" | "g" | "general" | "s" | "safe")
        } else if hide_adult {
            !matches!(rating.as_str(), "e" | "explicit" | "adult")
        } else {
            true
        }
    });
}

fn cached_item(data_dir: &Path, source_id: &str, post_id: &str) -> Result<RemoteItem, AppError> {
    let text: Option<String> = open_db(data_dir)?
        .query_row(
            "SELECT item_json FROM remote_item_cache WHERE source_id=?1 AND remote_post_id=?2",
            params![source_id, post_id],
            |r| r.get(0),
        )
        .optional()?;
    serde_json::from_str(
        &text.ok_or_else(|| AppError::NotFound("Remote item is not in the browse cache".into()))?,
    )
    .map_err(|_| AppError::Internal("Remote item cache is corrupt".into()))
}
fn variant_url(item: &RemoteItem, kind: &str) -> Result<String, AppError> {
    item.media
        .iter()
        .find(|v| v.kind == kind)
        .map(|v| v.asset_url.clone())
        .ok_or_else(|| AppError::NotFound("Requested media variant is unavailable".into()))
}
async fn fetch_media(
    data_dir: &Path,
    source: &RemoteSource,
    url: &str,
    range: Option<&HeaderValue>,
) -> Result<reqwest::Response, AppError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| AppError::BadRequest("Invalid remote asset URL".into()))?;
    security::validate_resolved_url(&parsed, source.allow_local_network).await?;
    let mut request = client()?.get(parsed);
    if let Some(value) = range.and_then(|v| v.to_str().ok()) {
        if value.starts_with("bytes=") {
            request = request.header(reqwest::header::RANGE, value);
        }
    }
    let credential = load_credential(data_dir, &source.source_id)?;
    let response = authenticated_request(request, credential.as_ref())?
        .send()
        .await
        .map_err(|e| AppError::ServiceUnavailable(e.to_string()))?;
    if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT
    {
        return Err(AppError::ServiceUnavailable(format!(
            "Remote media returned {}",
            response.status()
        )));
    }
    if response
        .content_length()
        .is_some_and(|n| n > MAX_MEDIA_BYTES)
    {
        return Err(AppError::ServiceUnavailable(
            "Remote media exceeds size limit".into(),
        ));
    }
    Ok(response)
}
async fn asset(
    State(state): State<AppState>,
    AxumPath((sid, pid, kind)): AxumPath<(String, String, String)>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let source = load_source(state.data_dir(), &sid)?;
    let item = cached_item(state.data_dir(), &sid, &pid)?;
    let response = fetch_media(
        state.data_dir(),
        &source,
        &variant_url(&item, &kind)?,
        headers.get(header::RANGE),
    )
    .await?;
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out = HeaderMap::new();
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_LENGTH,
        header::CONTENT_RANGE,
        header::ACCEPT_RANGES,
    ] {
        if let Some(v) = response.headers().get(name.as_str()) {
            if let Ok(v) = HeaderValue::from_bytes(v.as_bytes()) {
                out.insert(name, v);
            }
        }
    }
    out.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("inline"),
    );
    Ok((status, out, Body::from_stream(response.bytes_stream())).into_response())
}

#[derive(Deserialize)]
struct ImportRequest {
    kind: Option<String>,
    directory_id: i64,
    expected_sha256: Option<String>,
    filename: Option<String>,
}
async fn import_item(
    State(state): State<AppState>,
    AxumPath((sid, pid)): AxumPath<(String, String)>,
    Json(body): Json<ImportRequest>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let source = load_source(state.data_dir(), &sid)?;
    let item = cached_item(state.data_dir(), &sid, &pid)?;
    let existing: Option<(i64, String)> = open_db(state.data_dir())?
        .query_row(
            "SELECT image_id,content_sha256 FROM import_provenance WHERE source_id=?1 AND remote_post_id=?2 AND directory_id=?3 ORDER BY imported_at DESC LIMIT 1",
            params![sid, pid, body.directory_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((image_id, sha256)) = existing {
        return Ok((
            StatusCode::OK,
            Json(
                json!({"status":"already_imported","image_id":image_id,"directory_id":body.directory_id,"sha256":sha256}),
            ),
        ));
    }
    let kind = body.kind.as_deref().unwrap_or("original");
    let asset_url = variant_url(&item, kind)?;
    let inferred_extension = reqwest::Url::parse(&asset_url)
        .ok()
        .and_then(|url| {
            Path::new(url.path())
                .extension()
                .and_then(|value| value.to_str())
                .map(str::to_ascii_lowercase)
        })
        .filter(|value| {
            matches!(
                value.as_str(),
                "jpg" | "jpeg" | "png" | "gif" | "webp" | "avif" | "mp4" | "webm" | "mov"
            )
        })
        .unwrap_or_else(|| "bin".into());
    let response = fetch_media(state.data_dir(), &source, &asset_url, None).await?;
    let directory_path: String = state
        .main_db()
        .get()?
        .query_row(
            "SELECT path FROM watch_directories WHERE id=?1",
            params![body.directory_id],
            |r| r.get(0),
        )
        .optional()?
        .ok_or_else(|| {
            AppError::BadRequest("Import destination is not a watched directory".into())
        })?;
    let root = std::fs::canonicalize(&directory_path)
        .map_err(|_| AppError::BadRequest("Import destination is unavailable".into()))?;
    let staging = state.data_dir().join("online-import-staging");
    tokio::fs::create_dir_all(&staging).await?;
    let temp = staging.join(format!("{}.part", Uuid::new_v4()));
    let mut file = tokio::fs::File::create(&temp).await?;
    let mut stream = response.bytes_stream();
    let mut hash = Sha256::new();
    let mut size = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AppError::ServiceUnavailable(e.to_string()))?;
        size += chunk.len() as u64;
        if size > MAX_MEDIA_BYTES {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(AppError::BadRequest(
                "Remote media exceeds import limit".into(),
            ));
        }
        hash.update(&chunk);
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    let digest = format!("{:x}", hash.finalize());
    if body
        .expected_sha256
        .as_deref()
        .is_some_and(|e| !e.eq_ignore_ascii_case(&digest))
    {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(AppError::BadRequest(
            "Downloaded media hash did not match".into(),
        ));
    }
    let proposed = body
        .filename
        .unwrap_or_else(|| format!("remote-{}-{}.{}", sid, pid, inferred_extension));
    let safe: String = proposed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let extension = inferred_extension;
    let mut destination = root.join(if Path::new(&safe).extension().is_some() {
        safe
    } else {
        format!("{safe}.{extension}")
    });
    let mut n = 1;
    while destination.exists() {
        destination = root.join(format!("remote-{pid}-{n}.{extension}"));
        n += 1;
    }
    tokio::fs::rename(&temp, &destination).await?;
    let library_id = state.library_manager().primary().uuid.clone();
    let result = crate::services::importer::import_image(
        &state,
        state.library_manager().primary(),
        destination.to_string_lossy().as_ref(),
        body.directory_id,
        false,
    );
    let result = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = tokio::fs::remove_file(&destination).await;
            return Err(e);
        }
    };
    let image_id = result.image_id;
    open_db(state.data_dir())?.execute(
        "INSERT OR IGNORE INTO import_provenance VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
        params![
            sid,
            pid,
            library_id,
            body.directory_id,
            image_id,
            digest,
            item.canonical_url,
            Utc::now().to_rfc3339()
        ],
    )?;
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"status":"imported","image_id":image_id,"directory_id":body.directory_id,"sha256":digest}),
        ),
    ))
}

#[derive(Deserialize)]
struct PreflightRequest {
    target_source_ids: Vec<String>,
}
async fn preflight_publication(
    State(state): State<AppState>,
    Json(body): Json<PreflightRequest>,
) -> Result<Json<Value>, AppError> {
    let mut targets = vec![];
    for id in body.target_source_ids {
        let s = load_source(state.data_dir(), &id)?;
        targets.push(json!({"source_id":id,"display_name":s.display_name,"capabilities":s.capabilities,"eligible":s.enabled&&s.capabilities.upload,"reason":if !s.enabled{"disabled"}else if !s.capabilities.upload{"read_only"}else{"ready"}}));
    }
    Ok(Json(json!({"targets":targets})))
}
#[derive(Deserialize)]
struct CreatePublication {
    snapshot: PublicationSnapshot,
    target_source_ids: Vec<String>,
}
async fn create_publication(
    State(state): State<AppState>,
    Json(body): Json<CreatePublication>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    if body.target_source_ids.is_empty() {
        return Err(AppError::BadRequest(
            "Select at least one publication target".into(),
        ));
    }
    let mut conn = open_db(state.data_dir())?;
    let tx = conn.transaction()?;
    let publication_id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    tx.execute(
        "INSERT INTO publication_intents VALUES(?1,?2,?3)",
        params![
            publication_id,
            serde_json::to_string(&body.snapshot).unwrap(),
            now
        ],
    )?;
    let mut targets = vec![];
    for source_id in body.target_source_ids {
        let source_row: Option<(i64, String)> = tx
            .query_row(
                "SELECT enabled,capabilities_json FROM remote_sources WHERE source_id=?1",
                params![source_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((enabled, capabilities_json)) = source_row else {
            return Err(AppError::BadRequest(
                "Publication target does not exist".into(),
            ));
        };
        let capabilities: crate::online::models::SourceCapabilities =
            serde_json::from_str(&capabilities_json).map_err(|_| {
                AppError::Internal("Publication target capabilities are corrupt".into())
            })?;
        if enabled == 0 || !capabilities.upload {
            return Err(AppError::BadRequest(
                "Publication target is disabled or read-only".into(),
            ));
        }
        let target_id = Uuid::new_v4().to_string();
        let key = format!("{}:{}:publish:1", publication_id, source_id);
        tx.execute(
            "INSERT INTO publication_targets VALUES(?1,?2,?3,?4,'queued',0,NULL,NULL,NULL,NULL,?5)",
            params![target_id, publication_id, source_id, key, now],
        )?;
        targets.push(json!({"target_id":target_id,"source_id":source_id,"state":"queued","idempotency_key":key}));
    }
    tx.commit()?;
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"publication_id":publication_id,"snapshot":body.snapshot,"targets":targets,"created_at":now}),
        ),
    ))
}
async fn list_publications(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let conn = open_db(state.data_dir())?;
    let mut stmt=conn.prepare("SELECT p.publication_id,p.snapshot_json,p.created_at,t.target_id,t.connection_id,t.state,t.attempt_count,t.remote_post_id,t.remote_url,t.receipt_json,t.error_json,t.updated_at FROM publication_intents p JOIN publication_targets t ON t.publication_id=p.publication_id ORDER BY p.created_at DESC,t.target_id")?;
    let rows=stmt.query_map([],|r|Ok(json!({"publication_id":r.get::<_,String>(0)?,"snapshot":serde_json::from_str::<Value>(&r.get::<_,String>(1)?).unwrap_or(Value::Null),"created_at":r.get::<_,String>(2)?,"target":{"target_id":r.get::<_,String>(3)?,"source_id":r.get::<_,String>(4)?,"state":r.get::<_,String>(5)?,"attempt_count":r.get::<_,i64>(6)?,"remote_post_id":r.get::<_,Option<String>>(7)?,"remote_url":r.get::<_,Option<String>>(8)?,"receipt":r.get::<_,Option<String>>(9)?.and_then(|v|serde_json::from_str::<Value>(&v).ok()),"error":r.get::<_,Option<String>>(10)?.and_then(|v|serde_json::from_str::<Value>(&v).ok()),"updated_at":r.get::<_,String>(11)?}})))?.collect::<Result<Vec<_>,_>>()?;
    Ok(Json(json!({"publications":rows})))
}
async fn simulate_delivery(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let conn = open_db(state.data_dir())?;
    let changed=conn.execute(r#"UPDATE publication_targets SET state='published',attempt_count=attempt_count+1,remote_post_id=COALESCE(remote_post_id,'fake-'||substr(target_id,1,8)),remote_url=COALESCE(remote_url,'https://fake.invalid/posts/'||substr(target_id,1,8)),receipt_json='{"provider":"fake","idempotent":true}',updated_at=?2 WHERE publication_id=?1 AND state IN ('queued','failed_retryable') AND connection_id IN (SELECT source_id FROM remote_sources WHERE provider_family='fake')"#,params![id,Utc::now().to_rfc3339()])?;
    if changed == 0 {
        return Err(AppError::BadRequest(
            "No queued fake deliveries were available".into(),
        ));
    }
    Ok(Json(json!({"updated_targets":changed})))
}
async fn deliver_publication(
    State(state): State<AppState>,
    AxumPath(publication_id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let (snapshot, file_path, targets) = {
        let conn = open_db(state.data_dir())?;
        let snapshot_json: String = conn
            .query_row(
                "SELECT snapshot_json FROM publication_intents WHERE publication_id=?1",
                params![publication_id],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| AppError::NotFound("Publication not found".into()))?;
        let snapshot: PublicationSnapshot = serde_json::from_str(&snapshot_json)
            .map_err(|_| AppError::Internal("Publication snapshot is corrupt".into()))?;
        let pool = state
            .directory_db()
            .get_pool(snapshot.directory_id)
            .map_err(|error| {
                AppError::BadRequest(format!("Publication directory is unavailable: {error}"))
            })?;
        let directory = pool.get()?;
        let file_path: String = directory.query_row(
            "SELECT original_path FROM image_files WHERE image_id=?1 AND file_exists=1 ORDER BY id LIMIT 1",
            params![snapshot.image_id],
            |row| row.get(0),
        ).optional()?.ok_or_else(|| AppError::BadRequest("Publication media is unavailable".into()))?;
        let mut stmt = conn.prepare(
            "SELECT t.target_id,t.connection_id,t.idempotency_key,s.normalized_base_url,s.allow_local_network \
             FROM publication_targets t JOIN remote_sources s ON s.source_id=t.connection_id \
             WHERE t.publication_id=?1 AND t.state IN ('queued','failed_retryable') AND s.enabled=1 AND s.provider_family='donutbooru'"
        )?;
        let targets = stmt
            .query_map(params![publication_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)? != 0,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        (snapshot, file_path, targets)
    };
    if targets.is_empty() {
        return Err(AppError::BadRequest(
            "No queued DonutBooru deliveries were available".into(),
        ));
    }
    let content = tokio::fs::read(&file_path)
        .await
        .map_err(|_| AppError::BadRequest("Publication media is unavailable".into()))?;
    let checksum = format!("{:x}", Sha256::digest(&content));
    if snapshot.content_sha256.as_deref() != Some(checksum.as_str()) {
        return Err(AppError::BadRequest(
            "Publication media changed after the immutable snapshot was created".into(),
        ));
    }
    let snapshot_json =
        serde_json::to_string(&snapshot).map_err(|error| AppError::Internal(error.to_string()))?;
    let mut outcomes = Vec::new();
    for (target_id, source_id, idempotency_key, base_url, allow_local_network) in targets {
        let endpoint = format!(
            "{}/api/client/v1/publications",
            base_url.trim_end_matches('/')
        );
        let result = async {
            let parsed_endpoint = reqwest::Url::parse(&endpoint)
                .map_err(|_| AppError::BadRequest("Publication endpoint is invalid".into()))?;
            security::validate_resolved_url(&parsed_endpoint, allow_local_network).await?;
            let credential = load_credential(state.data_dir(), &source_id)?.ok_or_else(|| {
                AppError::BadRequest("Publication target has no credential".into())
            })?;
            let filename = Path::new(&file_path)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("publication.bin");
            let part =
                reqwest::multipart::Part::bytes(content.clone()).file_name(filename.to_string());
            let form = reqwest::multipart::Form::new()
                .text("snapshot_json", snapshot_json.clone())
                .part("file", part);
            let request = state
                .http_client()
                .post(&endpoint)
                .header("Idempotency-Key", &idempotency_key)
                .multipart(form);
            let response = authenticated_request(request, Some(&credential))?
                .send()
                .await
                .map_err(|error| {
                    AppError::ServiceUnavailable(format!("Publication delivery failed: {error}"))
                })?;
            let status = response.status();
            let body: Value = response.json().await.unwrap_or(Value::Null);
            if !status.is_success() {
                return Err(AppError::BadRequest(format!(
                    "Target returned HTTP {}",
                    status.as_u16()
                )));
            }
            Ok::<Value, AppError>(body)
        }
        .await;
        let now = Utc::now().to_rfc3339();
        match result {
            Ok(receipt) => {
                let state_value = receipt
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("submitted");
                let remote_post_id = receipt.get("remote_post_id").and_then(Value::as_str);
                let remote_url = receipt
                    .pointer("/receipt/canonical_url")
                    .and_then(Value::as_str);
                open_db(state.data_dir())?.execute(
                    "UPDATE publication_targets SET state=?3,attempt_count=attempt_count+1,remote_post_id=?4,remote_url=?5,receipt_json=?6,error_json=NULL,updated_at=?7 WHERE publication_id=?1 AND target_id=?2",
                    params![publication_id,target_id,state_value,remote_post_id,remote_url,receipt.to_string(),now],
                )?;
                outcomes.push(json!({"target_id":target_id,"source_id":source_id,"state":state_value,"receipt":receipt}));
            }
            Err(error) => {
                let error_json = json!({"message":error.to_string(),"retryable":true});
                open_db(state.data_dir())?.execute(
                    "UPDATE publication_targets SET state='failed_retryable',attempt_count=attempt_count+1,error_json=?3,updated_at=?4 WHERE publication_id=?1 AND target_id=?2",
                    params![publication_id,target_id,error_json.to_string(),now],
                )?;
                outcomes.push(json!({"target_id":target_id,"source_id":source_id,"state":"failed_retryable","error":error_json}));
            }
        }
    }
    Ok(Json(
        json!({"publication_id":publication_id,"targets":outcomes}),
    ))
}

#[derive(Deserialize)]
struct TargetPatch {
    state: String,
    error: Option<Value>,
}
async fn update_publication_target(
    State(state): State<AppState>,
    AxumPath((publication_id, target_id)): AxumPath<(String, String)>,
    Json(body): Json<TargetPatch>,
) -> Result<Json<Value>, AppError> {
    if !PUBLICATION_STATES.contains(&body.state.as_str()) {
        return Err(AppError::BadRequest("Invalid publication state".into()));
    }
    let changed=open_db(state.data_dir())?.execute("UPDATE publication_targets SET state=?3,error_json=?4,updated_at=?5 WHERE publication_id=?1 AND target_id=?2",params![publication_id,target_id,body.state,body.error.map(|v|v.to_string()),Utc::now().to_rfc3339()])?;
    if changed == 0 {
        return Err(AppError::NotFound("Publication target not found".into()));
    }
    Ok(Json(json!({"target_id":target_id,"state":body.state})))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn online_schema_is_independent_and_transactional() {
        let dir = tempfile::tempdir().unwrap();
        let mut conn = open_db(dir.path()).unwrap();
        let tx = conn.transaction().unwrap();
        tx.execute("INSERT INTO remote_sources VALUES('s','fake','Fake','https://fake.invalid','safe',1,0,'{}',NULL,'now')",[]).unwrap();
        tx.rollback().unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM remote_sources", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
    #[test]
    fn publication_states_cover_partial_failure_lifecycle() {
        for required in [
            "published",
            "pending_moderation",
            "rejected",
            "failed_retryable",
            "retracted",
        ] {
            assert!(PUBLICATION_STATES.contains(&required));
        }
    }

    #[test]
    fn policy_filter_is_fail_closed_when_family_mode_is_locked() {
        let item = |id: &str, rating: &str| RemoteItem {
            source_id: "source".into(),
            remote_post_id: id.into(),
            canonical_url: format!("https://example.test/{id}"),
            remote_revision: None,
            title: None,
            tags: vec![],
            rating: Some(rating.into()),
            source_url: None,
            width: None,
            height: None,
            duration: None,
            media: vec![],
        };
        let mut page = RemotePage {
            items: vec![item("safe", "safe"), item("explicit", "explicit")],
            page: 1,
            per_page: 20,
            total: Some(2),
            stale: false,
            next_cursor: None,
        };

        enforce_content_policy(&mut page, "unrestricted", true);

        assert_eq!(page.items.len(), 1);
        assert_eq!(page.items[0].remote_post_id, "safe");
    }

    #[test]
    fn credentials_are_referenced_not_stored_in_online_db() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open_db(dir.path()).unwrap();
        conn.execute("INSERT INTO remote_sources VALUES('s','fake','Fake','https://fake.invalid','safe',1,0,'{}',NULL,'now')", []).unwrap();
        write_credential(
            dir.path(),
            "credential.json",
            &StoredCredential {
                auth_kind: "bearer".into(),
                secret: "fixture-secret-value".into(),
            },
        )
        .unwrap();
        conn.execute(
            "INSERT INTO remote_connections VALUES('c','s','Fixture account','credential.json','unverified',NULL)",
            [],
        ).unwrap();

        let database = std::fs::read(dir.path().join("online.db")).unwrap();
        assert!(!database
            .windows(b"fixture-secret-value".len())
            .any(|window| window == b"fixture-secret-value"));
        let loaded = load_credential(dir.path(), "s").unwrap().unwrap();
        assert_eq!(loaded.secret, "fixture-secret-value");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(dir.path().join("online-credentials/credential.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }
}
