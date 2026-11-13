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
use crate::online::{hub, offline, provider, security, shared_accounts, transport};
use crate::server::error::AppError;
use crate::server::state::AppState;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::{rngs::OsRng, RngCore};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock,
};
use tower::ServiceExt;

const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_MEDIA_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/hubs/preview", post(preview_hub))
        .route("/hubs", get(list_hubs).post(create_hub))
        .route("/hubs/callback", post(complete_hub_callback))
        .route("/sources/{source_id}/hub", get(get_hub))
        .route("/sources/{source_id}/authorize", post(authorize_hub))
        .route(
            "/sources/{source_id}/shared-account",
            get(shared_account_status).post(reuse_shared_account),
        )
        .route("/sources/{source_id}/hub/authorize", post(authorize_hub))
        .route(
            "/publications/{publication_id}/refresh",
            post(refresh_publication),
        )
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
        .route("/sources/{source_id}/tags", get(source_tags))
        .route("/sources/{source_id}/creators", get(source_creators))
        .route(
            "/sources/{source_id}/creators/{creator_id}/preferences",
            patch(creator_preferences),
        )
        .route(
            "/sources/{source_id}/offline",
            get(offline_settings).patch(update_offline_settings),
        )
        .route(
            "/sources/{source_id}/offline/remove",
            post(remove_offline_copies),
        )
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

pub(crate) fn open_db(data_dir: &Path) -> Result<Connection, AppError> {
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
           receipt_json TEXT, error_json TEXT, updated_at TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS remote_hub_transport(
           source_id TEXT PRIMARY KEY REFERENCES remote_sources(source_id) ON DELETE CASCADE,
           instance_id TEXT NOT NULL, peer_id TEXT NOT NULL, relay_url TEXT NOT NULL,
           transport_mode TEXT NOT NULL DEFAULT 'relayed', descriptor_json TEXT NOT NULL);"
    )?;
    shared_accounts::schema(&conn)?;
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

pub(crate) fn load_source(data_dir: &Path, source_id: &str) -> Result<RemoteSource, AppError> {
    open_db(data_dir)?.query_row(
        "SELECT source_id,provider_family,display_name,normalized_base_url,policy_profile,enabled,allow_local_network,capabilities_json,last_probe_at FROM remote_sources WHERE source_id=?1",
        params![source_id], source_from_row,
    ).optional()?.ok_or_else(|| AppError::NotFound("Remote source not found".into()))
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub(crate) struct StoredCredential {
    pub(crate) auth_kind: String,
    pub(crate) secret: String,
    #[serde(default)]
    pub(crate) expires_at: Option<i64>,
}

fn credential_dir(data_dir: &Path) -> std::path::PathBuf {
    data_dir.join("online-credentials")
}

pub(crate) fn credential_path(
    data_dir: &Path,
    credential_ref: &str,
) -> Result<std::path::PathBuf, AppError> {
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

pub(crate) fn read_credential(
    data_dir: &Path,
    credential_ref: &str,
) -> Result<StoredCredential, AppError> {
    let bytes = crate::server::credentials::load_application_secret(&credential_path(
        data_dir,
        credential_ref,
    )?)?;
    serde_json::from_slice(&bytes)
        .map_err(|_| AppError::Internal("Stored remote credential is corrupt".into()))
}
fn load_credential(data_dir: &Path, source_id: &str) -> Result<Option<StoredCredential>, AppError> {
    let stored: Option<(String,String)> = open_db(data_dir)?.query_row(
        "SELECT connection_id,credential_ref FROM remote_connections WHERE source_id=?1 AND credential_ref IS NOT NULL ORDER BY last_probe_at DESC, connection_id LIMIT 1",
        params![source_id], |row| Ok((row.get(0)?,row.get(1)?)),
    ).optional()?;
    stored
        .map(|(connection_id, reference)| {
            shared_accounts::validate_alias(data_dir, source_id, &connection_id, &reference)?;
            let credential = read_credential(data_dir, &reference)?;
            if credential
                .expires_at
                .is_some_and(|expiry| expiry <= Utc::now().timestamp())
            {
                return Err(AppError::Unauthorized(
                    "Hub credential expired; sign in again".into(),
                ));
            }
            Ok(credential)
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
    let mut stmt = conn.prepare("SELECT connection_id,account_display_name,trust_state,last_probe_at,credential_ref FROM remote_connections WHERE source_id=?1 ORDER BY account_display_name,connection_id")?;
    let stored = stmt
        .query_map(params![source_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let connections: Vec<_> = stored.into_iter().map(|(id, name, trust, probe, reference)| {
        // Return lifecycle metadata only. A missing/corrupt/expired secret must
        // not be presented as a connected account or enable submission controls.
        let is_shared = open_db(state.data_dir()).ok().and_then(|conn|conn.query_row("SELECT EXISTS(SELECT 1 FROM remote_credential_aliases WHERE connection_id=?1)",params![id],|r|r.get::<_,bool>(0)).ok()).unwrap_or(false);
        let credential = reference.and_then(|reference| {
            shared_accounts::validate_alias(state.data_dir(), &source_id, &id, &reference).ok()?;
            let path = credential_path(state.data_dir(), &reference).ok()?;
            let bytes = crate::server::credentials::load_application_secret(&path).ok()?;
            serde_json::from_slice::<StoredCredential>(&bytes).ok()
        });
        let expiry = credential.as_ref().and_then(|credential| credential.expires_at);
        let expired = expiry.is_some_and(|expiry| expiry <= Utc::now().timestamp());
        let available = credential.as_ref().is_some_and(|credential| !credential.secret.is_empty()) && !expired;
        json!({"connection_id":id,"account_display_name":name,
            "trust_state":if expired {"expired"} else if !available {"unavailable"} else {trust.as_str()},
            "last_probe_at":probe,"has_credential":available,"expires_at":expiry,"shared_account":is_shared})
    }).collect();
    Ok(Json(json!({"connections": connections})))
}

async fn create_connection(
    State(state): State<AppState>,
    AxumPath(source_id): AxumPath<String>,
    Json(body): Json<ConnectionRequest>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    require_enabled(&load_source(state.data_dir(), &source_id)?)?;
    if load_hub(state.data_dir(), &source_id)?.is_some() && body.auth_kind != "bearer" {
        return Err(AppError::BadRequest(
            "Native hubs require bearer credentials".into(),
        ));
    }
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
    HeaderValue::from_str(&body.secret)
        .map_err(|_| AppError::BadRequest("Invalid credential header bytes".into()))?;
    let connection_id = Uuid::new_v4().to_string();
    let credential_ref = format!("{}.json", Uuid::new_v4());
    write_credential(
        state.data_dir(),
        &credential_ref,
        &StoredCredential {
            auth_kind: body.auth_kind,
            secret: body.secret,
            expires_at: None,
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
    purge_source_cache(&open_db(state.data_dir())?, &source_id)?;
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
    shared_accounts::remove_unused_secret(state.data_dir(), &credential_ref)?;
    purge_source_cache(&conn, &source_id)?;
    remove_pending_authorizations(&source_id);
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
    if !s.enabled {
        purge_source_cache(&open_db(state.data_dir())?, &id)?;
        remove_pending_authorizations(&id);
        offline::stop_source(state.data_dir(), &id)?;
    } else {
        let _ = offline::kick(&state, &id);
    }
    Ok(Json(s))
}
async fn delete_source(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, AppError> {
    let retained: bool = offline::open(state.data_dir())?.query_row("SELECT EXISTS(SELECT 1 FROM offline_items WHERE source_id=?1) OR EXISTS(SELECT 1 FROM offline_variants WHERE source_id=?1)",params![id],|r|r.get(0))?;
    if retained {
        return Err(AppError::Upstream(
            StatusCode::CONFLICT,
            "Remove offline copies first, or disable the connection to keep them".into(),
        ));
    }
    offline::stop_source(state.data_dir(), &id)?;
    let mut conn = open_db(state.data_dir())?;
    let tx = conn.transaction()?;
    let credential_refs = {
        let mut stmt = tx.prepare(
            "SELECT credential_ref FROM remote_connections WHERE source_id=?1 AND credential_ref IS NOT NULL",
        )?;
        let rows = stmt.query_map(params![id], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    tx.execute(
        "DELETE FROM publication_targets WHERE connection_id=?1",
        params![id],
    )?;
    purge_source_cache(&tx, &id)?;
    remove_pending_authorizations(&id);
    let changed = tx.execute("DELETE FROM remote_sources WHERE source_id=?1", params![id])?;
    if changed == 0 {
        return Err(AppError::NotFound("Remote source not found".into()));
    }
    tx.commit()?;
    for credential_ref in credential_refs {
        shared_accounts::remove_unused_secret(state.data_dir(), &credential_ref)?;
    }
    offline::cleanup_source(state.data_dir(), &id)?;
    Ok(StatusCode::NO_CONTENT)
}

fn client() -> Result<reqwest::Client, AppError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        .redirect(Policy::none())
        .user_agent(format!("DonutMediaCenter/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| AppError::Internal(e.to_string()))
}

fn same_origin(left: &reqwest::Url, right: &reqwest::Url) -> bool {
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

async fn checked_json(
    data_dir: &Path,
    source: &RemoteSource,
    url: reqwest::Url,
) -> Result<(Value, Option<String>), AppError> {
    require_enabled(source)?;
    if let Some(config) = load_hub(data_dir, &source.source_id)? {
        let path = hub::path(&source.normalized_base_url, url.as_str())?;
        let response = hub_request(data_dir, source, &config, "GET", &path, vec![], vec![]).await?;
        let etag = response
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case("etag"))
            .map(|(_, value)| value.clone());
        return Ok((hub_response_json(response).await?, etag));
    }
    security::validate_resolved_url(&url, source.allow_local_network).await?;
    let credential = load_credential(data_dir, &source.source_id)?;
    let source_origin = reqwest::Url::parse(&source.normalized_base_url)
        .map_err(|_| AppError::BadRequest("Invalid source URL".into()))?;
    let mut current = url;
    for _ in 0..=3 {
        let applicable_credential = credential
            .as_ref()
            .filter(|_| same_origin(&current, &source_origin));
        let response =
            authenticated_request(client()?.get(current.clone()), applicable_credential)?
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
    let mut source = load_source(state.data_dir(), &id)?;
    require_enabled(&source)?;
    let hub_config = load_hub(state.data_dir(), &id)?;
    let path = if hub_config.is_some() {
        hub::catalog_path("", 1, 1)?
    } else {
        provider::list_path(&source.provider_family, "", 1, 1)?
    };
    let url = reqwest::Url::parse(&(source.normalized_base_url.clone() + &path))
        .map_err(|_| AppError::BadRequest("Invalid source URL".into()))?;
    let (value, _) = checked_json(state.data_dir(), &source, url).await?;
    if hub_config.is_some() {
        hub::catalog_page(&id, &source.normalized_base_url, value, 1, 1)?;
    } else {
        provider::normalize_page(
            &source.provider_family,
            &id,
            &source.normalized_base_url,
            value,
            1,
            1,
        )?;
    }
    if source.provider_family == "donutbooru" {
        let capability_url = reqwest::Url::parse(&format!(
            "{}/api/client/v1/capabilities",
            source.normalized_base_url
        ))
        .map_err(|_| AppError::BadRequest("Invalid source URL".into()))?;
        if let Some(config) = hub_config.as_ref() {
            let (descriptor, _) = checked_json(state.data_dir(), &source, capability_url).await?;
            if hub::validate_descriptor(&source.normalized_base_url, &descriptor)?
                != config.instance_id
            {
                return Err(AppError::Forbidden(
                    "Hub identity changed; reconnect and confirm it".into(),
                ));
            }
            validate_advertised_bootstrap(&descriptor, &config.bootstrap)?;
            source.capabilities = hub::capabilities(&descriptor);
            open_db(state.data_dir())?.execute(
                "UPDATE remote_hub_transport SET descriptor_json=?2 WHERE source_id=?1",
                params![id, descriptor.to_string()],
            )?;
        } else {
            source.capabilities.upload = checked_json(state.data_dir(), &source, capability_url)
                .await
                .ok()
                .and_then(|(body, _)| {
                    body.pointer("/capabilities/publish")
                        .and_then(Value::as_bool)
                })
                .unwrap_or(false);
        }
    }
    let now = Utc::now().to_rfc3339();
    open_db(state.data_dir())?.execute(
        "UPDATE remote_sources SET last_probe_at=?2,capabilities_json=?3 WHERE source_id=?1",
        params![
            id,
            now,
            serde_json::to_string(&source.capabilities).unwrap()
        ],
    )?;
    open_db(state.data_dir())?.execute(
        "UPDATE remote_connections SET trust_state='verified',last_probe_at=?2 WHERE source_id=?1",
        params![id, now],
    )?;
    if hub_config.is_some() {
        shared_accounts::reuse_after_signin(state.data_dir(), &id).await;
    }
    Ok(Json(
        json!({"ok":true,"capabilities":source.capabilities,"last_probe_at":now}),
    ))
}

type BrowseQuery = hub::CatalogQuery;
async fn browse(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<BrowseQuery>,
) -> Result<Json<RemotePage>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    let hub_config = load_hub(state.data_dir(), &id)?;
    let mut query = query.normalized()?;
    if query.offline_only {
        return Ok(Json(offline::local_page(
            state.data_dir(),
            &id,
            &query,
            &source,
            state.is_family_mode_locked(),
        )?));
    }
    require_enabled(&source)?;
    let requested_query =
        serde_json::to_string(&query).map_err(|error| AppError::Internal(error.to_string()))?;
    let empty_policy_intersection = hub_config.is_some()
        && apply_catalog_policy(
            &mut query,
            &source.policy_profile,
            state.is_family_mode_locked(),
        );
    let page = query.page.unwrap_or(1);
    let per_page = query.per_page.unwrap_or(40);
    let q = query.q.as_deref().unwrap_or_default();
    if hub_config.is_none() && query.uses_hub_filters() {
        return Err(AppError::BadRequest("This legacy provider cannot execute collection sidebar filters before pagination; use an enrolled native hub".into()));
    }
    let path = if hub_config.is_some() {
        hub::catalog_path_for(&query)?
    } else {
        provider::list_path(&source.provider_family, q, page, per_page)?
    };
    // Canonical route contains every normalized filter; cached pages must never
    // be reused for a different media/rating/favorite/dimension/sort context.
    let key = format!("{id}:{path}:{requested_query}");
    let url = reqwest::Url::parse(&(source.normalized_base_url.clone() + &path))
        .map_err(|_| AppError::BadRequest("Invalid source request URL".into()))?;
    match checked_json(state.data_dir(), &source, url).await {
        Ok((value, etag)) => {
            let mut remote_page = if hub_config.is_some() {
                hub::catalog_page(&id, &source.normalized_base_url, value, page, per_page)?
            } else {
                provider::normalize_page(
                    &source.provider_family,
                    &id,
                    &source.normalized_base_url,
                    value,
                    page,
                    per_page,
                )?
            };
            if empty_policy_intersection {
                remote_page.items.clear();
                remote_page.total = Some(0);
                remote_page.next_cursor = None;
            }
            enforce_content_policy(
                &mut remote_page,
                &source.policy_profile,
                state.is_family_mode_locked(),
            );
            if source.policy_profile == "safe"
                || source.policy_profile == "adult-hidden"
                || state.is_family_mode_locked()
            {
                remote_page
                    .available_ratings
                    .retain(|rating| matches!(rating.as_str(), "pg" | "pg13"));
            }
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
            // Protected hub data never falls back to a previously authorized page.
            if hub_config.is_some() {
                return Err(error);
            }
            let conn = open_db(state.data_dir())?;
            let has_account: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM remote_connections WHERE source_id=?1)",
                params![id],
                |row| row.get(0),
            )?;
            if has_account {
                return Err(error);
            }
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
            matches!(
                rating.as_str(),
                "" | "g" | "general" | "s" | "safe" | "pg" | "pg13"
            )
        } else if hide_adult {
            !matches!(
                rating.as_str(),
                "e" | "explicit" | "adult" | "r" | "x" | "xxx"
            )
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
) -> Result<MediaResponse, AppError> {
    require_enabled(source)?;
    if let Some(config) = load_hub(data_dir, &source.source_id)? {
        let path = hub::path(&source.normalized_base_url, url)?;
        if !path.starts_with("/api/client/v1/media/") {
            return Err(AppError::Forbidden(
                "Hub assets must use protected media routes".into(),
            ));
        }
        let mut headers = vec![];
        if let Some(value) = range
            .and_then(|v| v.to_str().ok())
            .filter(|v| v.starts_with("bytes="))
        {
            headers.push(("Range".into(), value.into()));
        }
        let response =
            hub_request(data_dir, source, &config, "GET", &path, headers, vec![]).await?;
        return MediaResponse::from_hub(response);
    }
    let mut current = reqwest::Url::parse(url)
        .map_err(|_| AppError::BadRequest("Invalid remote asset URL".into()))?;
    let source_origin = reqwest::Url::parse(&source.normalized_base_url)
        .map_err(|_| AppError::BadRequest("Invalid source URL".into()))?;
    let credential = load_credential(data_dir, &source.source_id)?;
    for _ in 0..=3 {
        security::validate_resolved_url(&current, source.allow_local_network).await?;
        let mut request = client()?.get(current.clone());
        if let Some(value) = range.and_then(|v| v.to_str().ok()) {
            if value.starts_with("bytes=") {
                request = request.header(reqwest::header::RANGE, value);
            }
        }
        let applicable_credential = credential
            .as_ref()
            .filter(|_| same_origin(&current, &source_origin));
        let response = authenticated_request(request, applicable_credential)?
            .send()
            .await
            .map_err(|e| AppError::ServiceUnavailable(e.to_string()))?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| {
                    AppError::ServiceUnavailable("Remote redirect omitted Location".into())
                })?;
            current = current
                .join(location)
                .map_err(|_| AppError::ServiceUnavailable("Remote redirect was invalid".into()))?;
            continue;
        }
        if !response.status().is_success() {
            return Err(AppError::ServiceUnavailable(format!(
                "Remote media returned {}",
                response.status()
            )));
        }
        if response
            .content_length()
            .is_some_and(|size| size > MAX_MEDIA_BYTES)
        {
            return Err(AppError::ServiceUnavailable(
                "Remote media exceeds size limit".into(),
            ));
        }
        return Ok(MediaResponse::from_http(response));
    }
    Err(AppError::ServiceUnavailable(
        "Remote media exceeded redirect limit".into(),
    ))
}
#[derive(Default, Deserialize)]
struct AssetQuery {
    #[serde(default)]
    offline_only: bool,
}
async fn asset(
    State(state): State<AppState>,
    AxumPath((sid, pid, kind)): AxumPath<(String, String, String)>,
    headers: HeaderMap,
    Query(query): Query<AssetQuery>,
) -> Result<Response, AppError> {
    let source = load_source(state.data_dir(), &sid)?;
    let current = if query.offline_only {
        None
    } else {
        require_enabled(&source)?;
        Some(load_current_item(state.data_dir(), &source, &pid).await?)
    };
    if let Some(item) = offline::personal_item(state.data_dir(), &sid, &pid)? {
        enforce_item_policy(&item, &source.policy_profile, state.is_family_mode_locked())?;
        if let Some(record) = offline::personal_variant(state.data_dir(), &sid, &pid, &kind)?
            .filter(|record| {
                query.offline_only
                    || current.as_ref().is_some_and(|item| {
                        item.remote_revision.as_deref() == Some(record.revision.as_str())
                            && item.media.iter().find(|v| v.kind == kind).is_some_and(|v| {
                                v.sha256.as_deref().is_none_or(|sha| sha == record.sha256)
                                    && v.byte_size.is_none_or(|bytes| bytes == record.bytes)
                            })
                    })
            })
        {
            let checked = record.clone();
            tokio::task::spawn_blocking(move || offline::verify_file(&checked))
                .await
                .map_err(|e| AppError::Internal(e.to_string()))?
                .map_err(AppError::ServiceUnavailable)?;
            let request = axum::http::Request::builder()
                .uri("/")
                .body(Body::empty())
                .map_err(|e| AppError::Internal(e.to_string()))?;
            let (mut parts, body) = request.into_parts();
            parts.headers = headers;
            let response = tower_http::services::ServeFile::new(&record.path)
                .oneshot(axum::http::Request::from_parts(parts, body))
                .await
                .map_err(|e| AppError::Internal(e.to_string()))?;
            let mut response = response.into_response();
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("private, no-store"),
            );
            if let Some(mime) = record
                .mime_type
                .and_then(|v| HeaderValue::from_str(&v).ok())
            {
                response.headers_mut().insert(header::CONTENT_TYPE, mime);
            }
            return Ok(response);
        }
    }
    let item = current.ok_or_else(|| {
        AppError::NotFound("Personal offline representation is unavailable".into())
    })?;
    enforce_item_policy(&item, &source.policy_profile, state.is_family_mode_locked())?;
    let central = fetch_media(
        state.data_dir(),
        &source,
        &variant_url(&item, &kind)?,
        headers.get(header::RANGE),
    )
    .await;
    asset_delivery(central, || {
        peer_view_asset(&state, &source, &item, &kind, &headers)
    })
    .await
}

async fn asset_delivery<F, Fut>(
    central: Result<MediaResponse, AppError>,
    peers: F,
) -> Result<Response, AppError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<Option<Response>, AppError>>,
{
    let response = match central {
        Ok(response) => response,
        Err(error) if authorization_error(&error) => return Err(error),
        Err(error) => match peers().await? {
            Some(response) => return Ok(response),
            None => return Err(error),
        },
    };
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
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    out.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("inline"),
    );
    Ok((status, out, Body::from_stream(response.bytes_stream())).into_response())
}

const PEER_VIEW_LIMIT: u64 = 256 * 1024 * 1024;
static PEER_VIEW_TRANSFERS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
static PEER_VIEW_BYTES: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();

/// Body owns the verified temporary file and both capacity reservations. Drop
/// the file stream first so Windows can remove its enclosing private directory.
struct PeerViewBody {
    stream: Option<
        std::pin::Pin<
            Box<dyn futures_util::Stream<Item = Result<axum::body::Bytes, axum::Error>> + Send>,
        >,
    >,
    directory: Option<tempfile::TempDir>,
    _transfer: tokio::sync::OwnedSemaphorePermit,
    _bytes: tokio::sync::OwnedSemaphorePermit,
}
impl futures_util::Stream for PeerViewBody {
    type Item = Result<axum::body::Bytes, axum::Error>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.stream
            .as_mut()
            .expect("peer response body exists until drop")
            .as_mut()
            .poll_next(cx)
    }
}
impl Drop for PeerViewBody {
    fn drop(&mut self) {
        drop(self.stream.take());
        drop(self.directory.take());
    }
}

async fn peer_view_asset(
    state: &AppState,
    source: &RemoteSource,
    item: &RemoteItem,
    kind: &str,
    headers: &HeaderMap,
) -> Result<Option<Response>, AppError> {
    let Some(config) = load_hub(state.data_dir(), &source.source_id)? else {
        return Ok(None);
    };
    let Some(variant) = item.media.iter().find(|variant| variant.kind == kind) else {
        return Ok(None);
    };
    peer_view_asset_with(
        state,
        source,
        item,
        variant,
        headers,
        |destination| async move {
            try_peer_download(state, source, &config, item, variant, &destination, None).await
        },
    )
    .await
}

// Private seam permits real protocol tests with an ephemeral broker/provider.
// Production obtains grants and identities exclusively from the pinned hub.
async fn peer_view_asset_with<F, Fut>(
    state: &AppState,
    source: &RemoteSource,
    item: &RemoteItem,
    variant: &crate::online::models::RemoteMediaVariant,
    headers: &HeaderMap,
    download: F,
) -> Result<Option<Response>, AppError>
where
    F: FnOnce(std::path::PathBuf) -> Fut,
    Fut: std::future::Future<Output = Result<bool, AppError>>,
{
    let (Some(digest), Some(size)) = (&variant.sha256, variant.byte_size) else {
        return Ok(None);
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || size == 0
        || item.remote_revision.is_none()
    {
        return Ok(None);
    }
    if size > PEER_VIEW_LIMIT {
        return Err(AppError::ServiceUnavailable("Community viewing requires full checksum verification and currently supports renditions up to 256 MiB".into()));
    }
    let current_source = load_source(state.data_dir(), &source.source_id)?;
    require_enabled(&current_source)?;
    enforce_item_policy(
        item,
        &current_source.policy_profile,
        state.is_family_mode_locked(),
    )?;
    let transfer = tokio::time::timeout(
        Duration::from_secs(30),
        PEER_VIEW_TRANSFERS
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)))
            .clone()
            .acquire_owned(),
    )
    .await
    .map_err(|_| AppError::ServiceUnavailable("Community viewing queue is busy".into()))?
    .map_err(|_| AppError::ServiceUnavailable("Community viewing stopped".into()))?;
    let units = size.div_ceil(1024 * 1024) as u32;
    let bytes = tokio::time::timeout(
        Duration::from_secs(30),
        PEER_VIEW_BYTES
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(256)))
            .clone()
            .acquire_many_owned(units),
    )
    .await
    .map_err(|_| {
        AppError::ServiceUnavailable("Community viewing temporary storage is busy".into())
    })?
    .map_err(|_| AppError::ServiceUnavailable("Community viewing stopped".into()))?;
    let directory = tempfile::Builder::new()
        .prefix("dmc-peer-view-")
        .tempdir_in(crate::platform_paths::temporary_dir().map_err(AppError::Internal)?)?;
    let destination = directory.path().join("verified-media");
    if !download(destination.clone()).await? {
        return Ok(None);
    }
    // Validate independently of the chooser before exposing even Range bytes.
    let checked = offline::RetainedVariant {
        source_id: source.source_id.clone(),
        remote_post_id: item.remote_post_id.clone(),
        kind: variant.kind.clone(),
        revision: item.remote_revision.clone().unwrap(),
        path: destination.clone(),
        sha256: digest.clone(),
        bytes: size,
        mime_type: variant.mime_type.clone(),
        community_hosting: false,
        authoritative_digest: true,
        canonical_origin: source.normalized_base_url.clone(),
        instance_id: String::new(),
    };
    tokio::task::spawn_blocking(move || offline::verify_file(&checked))
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .map_err(AppError::ServiceUnavailable)?;
    let current_source = load_source(state.data_dir(), &source.source_id)?;
    require_enabled(&current_source)?;
    enforce_item_policy(
        item,
        &current_source.policy_profile,
        state.is_family_mode_locked(),
    )?;
    let request = axum::http::Request::builder()
        .uri("/")
        .body(Body::empty())
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let (mut parts, body) = request.into_parts();
    parts.headers = headers.clone();
    let response = tower_http::services::ServeFile::new(&destination)
        .oneshot(axum::http::Request::from_parts(parts, body))
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .into_response();
    let (mut parts, body) = response.into_parts();
    parts.headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    parts.headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("inline"),
    );
    if let Some(mime) = variant
        .mime_type
        .as_deref()
        .and_then(|v| HeaderValue::from_str(v).ok())
    {
        parts.headers.insert(header::CONTENT_TYPE, mime);
    }
    Ok(Some(Response::from_parts(
        parts,
        Body::from_stream(PeerViewBody {
            stream: Some(Box::pin(body.into_data_stream())),
            directory: Some(directory),
            _transfer: transfer,
            _bytes: bytes,
        }),
    )))
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
    require_enabled(&source)?;
    let item = load_current_item(state.data_dir(), &source, &pid).await?;
    enforce_item_policy(&item, &source.policy_profile, state.is_family_mode_locked())?;
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
    let url_extension = reqwest::Url::parse(&asset_url)
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
                "jpg" | "jpeg" | "png" | "gif" | "webp" | "mp4" | "webm" | "mov"
            )
        });
    let response = fetch_media(state.data_dir(), &source, &asset_url, None).await?;
    let inferred_extension = url_extension
        .or_else(|| {
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| match value.split(';').next()?.trim() {
                    "image/jpeg" => Some("jpg"),
                    "image/png" => Some("png"),
                    "image/gif" => Some("gif"),
                    "image/webp" => Some("webp"),
                    "video/mp4" => Some("mp4"),
                    "video/webm" => Some("webm"),
                    "video/quicktime" => Some("mov"),
                    _ => None,
                })
                .map(str::to_string)
        })
        .ok_or_else(|| AppError::BadRequest("Remote media format is unsupported".into()))?;
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
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                drop(file);
                let _ = tokio::fs::remove_file(&temp).await;
                return Err(AppError::ServiceUnavailable(error.to_string()));
            }
        };
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
    if result.status == crate::services::importer::ImportStatus::Error {
        let _ = tokio::fs::remove_file(&destination).await;
        return Err(AppError::BadRequest(
            result
                .message
                .unwrap_or_else(|| "Could not import remote media".into()),
        ));
    }
    let image_id = result
        .image_id
        .ok_or_else(|| AppError::Internal("Importer returned no image ID".into()))?;
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
    let mut snapshot = body.snapshot;
    let directory_pool = state
        .directory_db()
        .get_pool(snapshot.directory_id)
        .map_err(|_| AppError::BadRequest("Publication directory is unavailable".into()))?;
    let directory = directory_pool.get()?;
    let (file_path, rating, prompt): (String, String, Option<String>) = directory.query_row(
        "SELECT f.original_path,i.rating,i.prompt FROM images i JOIN image_files f ON f.image_id=i.id WHERE i.id=?1 AND f.file_exists=1 ORDER BY f.id LIMIT 1",
        params![snapshot.image_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).optional()?.ok_or_else(|| AppError::BadRequest("Publication media is unavailable".into()))?;
    if snapshot.tags.is_empty() {
        let mut stmt =
            directory.prepare("SELECT tag_id FROM image_tags WHERE image_id=?1 ORDER BY tag_id")?;
        let tag_ids = stmt
            .query_map(params![snapshot.image_id], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let main_db = state.main_db().get()?;
        for tag_id in tag_ids {
            if let Some(tag) = main_db
                .query_row(
                    "SELECT name FROM tags WHERE id=?1",
                    params![tag_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
            {
                snapshot.tags.push(tag);
            }
        }
    }
    drop(directory);
    if tokio::fs::metadata(&file_path).await?.len() > 60 * 1024 * 1024 {
        return Err(AppError::BadRequest(
            "Publication media exceeds native upload limit".into(),
        ));
    }
    let content = read_publication_media(&file_path).await?;
    let checksum = format!("{:x}", Sha256::digest(&content));
    if snapshot
        .content_sha256
        .as_deref()
        .is_some_and(|value| value != checksum)
    {
        return Err(AppError::BadRequest("Publication media changed".into()));
    }
    snapshot.content_sha256 = Some(checksum);
    snapshot.library_id = state.library_manager().primary().uuid.clone();
    snapshot.rating = Some(rating);
    if snapshot.caption.is_none() {
        snapshot.caption = prompt;
    }
    let mut conn = open_db(state.data_dir())?;
    let tx = conn.transaction()?;
    let publication_id = Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    tx.execute(
        "INSERT INTO publication_intents VALUES(?1,?2,?3)",
        params![
            publication_id,
            serde_json::to_string(&snapshot).unwrap(),
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
            json!({"publication_id":publication_id,"snapshot":snapshot,"targets":targets,"created_at":now}),
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
    if tokio::fs::metadata(&file_path).await?.len() > 60 * 1024 * 1024 {
        return Err(AppError::BadRequest(
            "Publication media exceeds native upload limit".into(),
        ));
    }
    let content = read_publication_media(&file_path).await?;
    let checksum = format!("{:x}", Sha256::digest(&content));
    if snapshot.content_sha256.as_deref() != Some(checksum.as_str()) {
        return Err(AppError::BadRequest(
            "Publication media changed after the immutable snapshot was created".into(),
        ));
    }
    let snapshot_json =
        serde_json::to_string(&snapshot).map_err(|error| AppError::Internal(error.to_string()))?;
    if snapshot_json.len() > 128 * 1024 {
        return Err(AppError::BadRequest(
            "Publication metadata exceeds upload limit".into(),
        ));
    }
    let mut outcomes = Vec::new();
    for (target_id, source_id, idempotency_key, base_url, allow_local_network) in targets {
        let endpoint = format!(
            "{}/api/client/v1/publications",
            base_url.trim_end_matches('/')
        );
        let result = async {
            let source = load_source(state.data_dir(), &source_id)?;
            require_enabled(&source)?;
            if load_credential(state.data_dir(), &source_id)?.is_none() {
                return Err(AppError::BadRequest(
                    "Publication target has no credential".into(),
                ));
            }
            if let Some(config) = load_hub(state.data_dir(), &source_id)? {
                let (content_type, body) =
                    publication_multipart(&snapshot_json, &file_path, &content)?;
                let response = hub_request(
                    state.data_dir(),
                    &source,
                    &config,
                    "POST",
                    "/api/client/v1/publications",
                    vec![
                        ("Content-Type".into(), content_type),
                        ("Idempotency-Key".into(), idempotency_key.clone()),
                    ],
                    body,
                )
                .await?;
                let receipt = hub_response_json(response).await?;
                validate_hub_receipt(&source, &receipt)?;
                return Ok(receipt);
            }
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
            let request = client()?
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
                let state_value = receipt_local_state(&receipt);
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

// Hub configuration is additive: legacy RemoteSource records keep their shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HubConfig {
    pub(crate) instance_id: String,
    pub(crate) bootstrap: transport::HubBootstrap,
    pub(crate) transport_mode: transport::TransportMode,
    pub(crate) descriptor: Value,
}
pub(crate) fn load_hub(data_dir: &Path, source_id: &str) -> Result<Option<HubConfig>, AppError> {
    open_db(data_dir)?.query_row(
        "SELECT instance_id,peer_id,relay_url,transport_mode,descriptor_json FROM remote_hub_transport WHERE source_id=?1",
        params![source_id], |row| {
            let mode: String = row.get(3)?;
            let descriptor: String = row.get(4)?;
            Ok((row.get::<_,String>(0)?, row.get::<_,String>(1)?, row.get::<_,String>(2)?, mode, descriptor))
        },
    ).optional()?.map(|(instance_id, peer_id, relay_url, mode, descriptor)| {
        Ok(HubConfig { instance_id, bootstrap: transport::HubBootstrap { peer_id, relay_url },
            transport_mode: serde_json::from_value(Value::String(mode)).map_err(|_| AppError::Internal("Stored hub transport is invalid".into()))?,
            descriptor: serde_json::from_str(&descriptor).map_err(|_| AppError::Internal("Stored hub descriptor is invalid".into()))? })
    }).transpose()
}

/// Discover contract upgrades through the enrolled authenticated hub gateway.
/// Signing keys supplied by another provider are never a source of authority.
pub(crate) async fn refresh_hub_descriptor(
    data_dir: &Path,
    source: &RemoteSource,
    config: &HubConfig,
) -> Result<HubConfig, AppError> {
    let response = hub_request(
        data_dir,
        source,
        config,
        "GET",
        "/api/client/v1/capabilities",
        vec![("Accept".into(), "application/json".into())],
        vec![],
    )
    .await?;
    let descriptor = hub_response_json(response).await?;
    persist_refreshed_hub_descriptor(data_dir, source, config, descriptor)
}

fn persist_refreshed_hub_descriptor(
    data_dir: &Path,
    source: &RemoteSource,
    config: &HubConfig,
    descriptor: Value,
) -> Result<HubConfig, AppError> {
    if hub::validate_descriptor(&source.normalized_base_url, &descriptor)? != config.instance_id {
        return Err(AppError::Forbidden(
            "Hub identity changed; reconnect and confirm it".into(),
        ));
    }
    if !descriptor
        .pointer("/transports/iroh")
        .is_some_and(Value::is_object)
    {
        return Err(AppError::Forbidden(
            "Refreshed hub descriptor must retain its pinned bootstrap".into(),
        ));
    }
    validate_advertised_bootstrap(&descriptor, &config.bootstrap)?;
    if let Some(realm) = config
        .descriptor
        .pointer("/instance/realm")
        .and_then(Value::as_str)
    {
        if descriptor
            .pointer("/instance/realm")
            .and_then(Value::as_str)
            != Some(realm)
        {
            return Err(AppError::Forbidden(
                "Hub realm changed during descriptor refresh".into(),
            ));
        }
    }
    let mode = serde_json::to_value(config.transport_mode)
        .map_err(|e| AppError::Internal(e.to_string()))?;
    let mut conn = open_db(data_dir)?;
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    // Compare-and-update keeps an in-flight reply from overwriting an explicit
    // source transport, pin or origin change made while the request was running.
    let updated = tx.execute("UPDATE remote_hub_transport SET descriptor_json=?2 WHERE source_id=?1 AND instance_id=?3 AND peer_id=?4 AND relay_url=?5 AND transport_mode=?6", params![source.source_id, descriptor.to_string(), config.instance_id, config.bootstrap.peer_id, config.bootstrap.relay_url, mode.as_str()])?;
    if updated != 1 {
        return Err(AppError::Forbidden(
            "Hub enrollment changed during descriptor refresh".into(),
        ));
    }
    let updated = tx.execute("UPDATE remote_sources SET capabilities_json=?2,last_probe_at=?3 WHERE source_id=?1 AND normalized_base_url=?4 AND enabled=1", params![source.source_id, serde_json::to_string(&hub::capabilities(&descriptor)).map_err(|e| AppError::Internal(e.to_string()))?, Utc::now().to_rfc3339(), source.normalized_base_url])?;
    if updated != 1 {
        return Err(AppError::Forbidden(
            "Source changed during descriptor refresh".into(),
        ));
    }
    tx.commit()?;
    Ok(HubConfig {
        descriptor,
        ..config.clone()
    })
}

fn require_enabled(source: &RemoteSource) -> Result<(), AppError> {
    if !source.enabled {
        return Err(AppError::Forbidden("Remote source is disabled".into()));
    }
    Ok(())
}
fn purge_source_cache(conn: &Connection, source_id: &str) -> Result<(), AppError> {
    conn.execute(
        "DELETE FROM remote_cache WHERE source_id=?1",
        params![source_id],
    )?;
    conn.execute(
        "DELETE FROM remote_item_cache WHERE source_id=?1",
        params![source_id],
    )?;
    Ok(())
}
fn default_transport_mode() -> transport::TransportMode {
    transport::TransportMode::Relayed
}
#[derive(Deserialize)]
struct HubRequestBody {
    base_url: String,
    peer_id: String,
    relay_url: String,
    #[serde(default = "default_transport_mode")]
    transport_mode: transport::TransportMode,
    #[serde(default)]
    allow_local_network: bool,
    expected_instance_id: Option<String>,
    display_name: Option<String>,
    policy_profile: Option<String>,
}
async fn fetch_hub_preview(body: &HubRequestBody) -> Result<(String, HubConfig), AppError> {
    if matches!(body.transport_mode, transport::TransportMode::Anonymous) {
        return Err(AppError::BadRequest(
            "Anonymous hub routing is unsupported; choose Relayed or Standard explicitly".into(),
        ));
    }
    let base = hub::origin(&body.base_url, body.allow_local_network)?;
    let bootstrap = transport::HubBootstrap {
        peer_id: body.peer_id.clone(),
        relay_url: body.relay_url.clone(),
    };
    transport::validate_bootstrap(&bootstrap).map_err(AppError::BadRequest)?;
    validate_relay(&bootstrap, body.allow_local_network).await?;
    let response = transport::request(
        &bootstrap,
        body.transport_mode,
        transport::HubRequest {
            method: "GET".into(),
            path: "/api/client/v1/capabilities".into(),
            headers: vec![("Accept".into(), "application/json".into())],
            body: vec![],
        },
    )
    .await
    .map_err(AppError::ServiceUnavailable)?;
    let descriptor = hub_response_json(response).await?;
    let instance_id = hub::validate_descriptor(&base, &descriptor)?;
    validate_advertised_bootstrap(&descriptor, &bootstrap)?;
    Ok((
        base,
        HubConfig {
            instance_id,
            bootstrap,
            transport_mode: body.transport_mode,
            descriptor,
        },
    ))
}
async fn preview_hub(Json(body): Json<HubRequestBody>) -> Result<Json<Value>, AppError> {
    let (base, config) = fetch_hub_preview(&body).await?;
    Ok(Json(
        json!({"normalized_base_url":base,"descriptor":config.descriptor,"bootstrap":config.bootstrap,"transport_mode":config.transport_mode}),
    ))
}
async fn create_hub(
    State(state): State<AppState>,
    Json(body): Json<HubRequestBody>,
) -> Result<(StatusCode, Json<Value>), AppError> {
    let expected = body.expected_instance_id.as_deref().ok_or_else(|| {
        AppError::BadRequest("Confirm the previewed hub identity before adding it".into())
    })?;
    let (base, config) = fetch_hub_preview(&body).await?;
    if expected != config.instance_id {
        return Err(AppError::Forbidden(
            "Hub identity changed after preview; review it again".into(),
        ));
    }
    let display_name = body
        .display_name
        .as_deref()
        .or_else(|| {
            config
                .descriptor
                .pointer("/instance/name")
                .and_then(Value::as_str)
        })
        .unwrap_or("")
        .trim()
        .to_string();
    if display_name.is_empty() || display_name.len() > 256 {
        return Err(AppError::BadRequest(
            "Hub display name is required and must be at most 256 bytes".into(),
        ));
    }
    let source = RemoteSource {
        source_id: Uuid::new_v4().to_string(),
        provider_family: "donutbooru".into(),
        display_name,
        normalized_base_url: base,
        policy_profile: body.policy_profile.unwrap_or_else(|| "unrestricted".into()),
        enabled: true,
        allow_local_network: body.allow_local_network,
        capabilities: hub::capabilities(&config.descriptor),
        last_probe_at: Some(Utc::now().to_rfc3339()),
    };
    {
        let mut conn = open_db(state.data_dir())?;
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO remote_sources VALUES(?1,?2,?3,?4,?5,1,?6,?7,?8,?8)",
            params![
                source.source_id,
                source.provider_family,
                source.display_name,
                source.normalized_base_url,
                source.policy_profile,
                source.allow_local_network as i64,
                serde_json::to_string(&source.capabilities).unwrap(),
                source.last_probe_at
            ],
        )
        .map_err(|error| {
            if error.to_string().contains("UNIQUE") {
                AppError::BadRequest("A source with this origin already exists".into())
            } else {
                error.into()
            }
        })?;
        tx.execute(
            "INSERT INTO remote_hub_transport VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                source.source_id,
                config.instance_id,
                config.bootstrap.peer_id,
                config.bootstrap.relay_url,
                serde_json::to_value(config.transport_mode)
                    .unwrap()
                    .as_str()
                    .unwrap(),
                config.descriptor.to_string()
            ],
        )?;
        tx.commit()?;
    }
    let shared_account = shared_accounts::reuse(state.data_dir(), &source.source_id, None, true)
        .await
        .unwrap_or_else(|_| json!({"state":"unavailable"}));
    Ok((
        StatusCode::CREATED,
        Json(json!({"source":source,"hub":config,"shared_account":shared_account})),
    ))
}
async fn shared_account_status(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    Ok(Json(shared_accounts::status(state.data_dir(), &id).await?))
}
#[derive(Deserialize)]
struct SharedAccountRequest {
    donor_connection_id: Option<String>,
}
async fn reuse_shared_account(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<SharedAccountRequest>,
) -> Result<Json<Value>, AppError> {
    Ok(Json(
        shared_accounts::reuse(
            state.data_dir(),
            &id,
            body.donor_connection_id.as_deref(),
            false,
        )
        .await?,
    ))
}
async fn get_hub(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<HubConfig>, AppError> {
    load_source(state.data_dir(), &id)?;
    Ok(Json(load_hub(state.data_dir(), &id)?.ok_or_else(|| {
        AppError::NotFound("Source is not an enrolled hub".into())
    })?))
}
async fn list_hubs(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let ids = {
        let conn = open_db(state.data_dir())?;
        let mut stmt =
            conn.prepare("SELECT source_id FROM remote_hub_transport ORDER BY source_id")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    let mut hubs = vec![];
    for id in ids {
        let source = load_source(state.data_dir(), &id)?;
        if let Some(config) = load_hub(state.data_dir(), &id)? {
            hubs.push(json!({"source_id":id,"display_name":source.display_name,"enabled":source.enabled,
                "instance_id":config.instance_id,"bootstrap":config.bootstrap,"transport_mode":config.transport_mode,"descriptor":config.descriptor}));
        }
    }
    Ok(Json(json!({"hubs":hubs})))
}
pub(crate) async fn hub_request(
    data_dir: &Path,
    source: &RemoteSource,
    config: &HubConfig,
    method: &str,
    path: &str,
    mut headers: Vec<(String, String)>,
    body: Vec<u8>,
) -> Result<transport::HubResponse, AppError> {
    // Reload immediately before every request, including credential exchange.
    require_enabled(&load_source(data_dir, &source.source_id)?)?;
    let path = hub::path(&source.normalized_base_url, path)?;
    validate_relay(&config.bootstrap, source.allow_local_network).await?;
    if let Some(credential) = if path == "/api/client/v1/capabilities" {
        None
    } else {
        load_credential(data_dir, &source.source_id)?
    } {
        if credential.auth_kind != "bearer" {
            return Err(AppError::BadRequest(
                "Native hubs require a source-scoped bearer credential".into(),
            ));
        }
        HeaderValue::from_str(&credential.secret)
            .map_err(|_| AppError::BadRequest("Invalid credential header".into()))?;
        headers.push((
            "Authorization".into(),
            format!("Bearer {}", credential.secret),
        ));
    }
    transport::request(
        &config.bootstrap,
        config.transport_mode,
        transport::HubRequest {
            method: method.into(),
            path,
            headers,
            body,
        },
    )
    .await
    .map_err(AppError::ServiceUnavailable)
}
pub(crate) async fn hub_response_json(response: transport::HubResponse) -> Result<Value, AppError> {
    let status = response.status;
    if !(200..300).contains(&status) {
        return Err(AppError::Upstream(
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            format!("Hub returned HTTP {status}"),
        ));
    }
    let bytes = response
        .into_bytes(MAX_JSON_BYTES)
        .await
        .map_err(AppError::ServiceUnavailable)?;
    serde_json::from_slice(&bytes)
        .map_err(|_| AppError::ServiceUnavailable("Hub returned invalid JSON".into()))
}
async fn load_current_item(
    data_dir: &Path,
    source: &RemoteSource,
    post_id: &str,
) -> Result<RemoteItem, AppError> {
    require_enabled(source)?;
    if let Some(config) = load_hub(data_dir, &source.source_id)? {
        // Never authorize media access using a stale browse item.
        if post_id.is_empty()
            || post_id.len() > 20
            || !post_id.bytes().all(|ch| ch.is_ascii_digit())
        {
            return Err(AppError::BadRequest("Invalid hub catalog item ID".into()));
        }
        let response = hub_request(
            data_dir,
            source,
            &config,
            "GET",
            &format!("/api/client/v1/catalog/{post_id}"),
            vec![],
            vec![],
        )
        .await?;
        hub::catalog_item(
            &source.source_id,
            &source.normalized_base_url,
            &hub_response_json(response).await?,
        )
    } else {
        cached_item(data_dir, &source.source_id, post_id)
    }
}

struct MediaResponse {
    status: StatusCode,
    headers: HeaderMap,
    stream:
        std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<Vec<u8>, std::io::Error>> + Send>>,
}
impl MediaResponse {
    fn from_http(response: reqwest::Response) -> Self {
        let status = response.status();
        let headers = response.headers().clone();
        Self {
            status,
            headers,
            stream: Box::pin(
                response
                    .bytes_stream()
                    .map(|r| r.map(|bytes| bytes.to_vec()).map_err(std::io::Error::other)),
            ),
        }
    }
    fn from_hub(response: transport::HubResponse) -> Result<Self, AppError> {
        let status = StatusCode::from_u16(response.status)
            .map_err(|_| AppError::ServiceUnavailable("Invalid hub media status".into()))?;
        if !status.is_success() {
            return Err(AppError::Upstream(
                status,
                "Hub media request was denied".into(),
            ));
        }
        let mut headers = HeaderMap::new();
        for (name, value) in response.headers {
            if let (Ok(name), Ok(value)) = (
                axum::http::HeaderName::try_from(name),
                HeaderValue::try_from(value),
            ) {
                headers.insert(name, value);
            }
        }
        if headers
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .is_some_and(|n| n > MAX_MEDIA_BYTES)
        {
            return Err(AppError::ServiceUnavailable(
                "Hub media exceeds size limit".into(),
            ));
        }
        let mut size = 0u64;
        let stream = response.body.map(move |chunk| match chunk {
            Ok(bytes) => {
                size = size.saturating_add(bytes.len() as u64);
                if size > MAX_MEDIA_BYTES {
                    Err(std::io::Error::other("Hub media exceeds size limit"))
                } else {
                    Ok(bytes)
                }
            }
            Err(error) => Err(std::io::Error::other(error)),
        });
        Ok(Self {
            status,
            headers,
            stream: Box::pin(stream),
        })
    }
    fn status(&self) -> StatusCode {
        self.status
    }
    fn headers(&self) -> &HeaderMap {
        &self.headers
    }
    fn bytes_stream(self) -> impl futures_util::Stream<Item = Result<Vec<u8>, std::io::Error>> {
        self.stream
    }
}

struct PendingAuthorization {
    data_dir: std::path::PathBuf,
    source_id: String,
    instance_id: String,
    generation: i64,
    config_fingerprint: String,
    verifier: String,
    expires_at: std::time::Instant,
}
static PENDING_AUTHORIZATIONS: OnceLock<Mutex<HashMap<String, PendingAuthorization>>> =
    OnceLock::new();
fn pending_authorizations() -> &'static Mutex<HashMap<String, PendingAuthorization>> {
    PENDING_AUTHORIZATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}
fn remove_pending_authorizations(source_id: &str) {
    if let Ok(mut pending) = pending_authorizations().lock() {
        pending.retain(|_, auth| auth.source_id != source_id);
    }
}
fn random_secret() -> Result<String, AppError> {
    let mut bytes = [0u8; 32];
    OsRng
        .try_fill_bytes(&mut bytes)
        .map_err(|_| AppError::Internal("Secure randomness unavailable".into()))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}
#[derive(Deserialize)]
struct AuthorizeHubRequest {
    #[serde(default)]
    allow_browser_direct: bool,
}
async fn authorize_hub(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<AuthorizeHubRequest>,
) -> Result<Json<Value>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    require_enabled(&source)?;
    let config = load_hub(state.data_dir(), &id)?
        .ok_or_else(|| AppError::BadRequest("Source is not an enrolled hub".into()))?;
    let account_generation = shared_accounts::generation(&open_db(state.data_dir())?, &id)?;
    let config_fingerprint = serde_json::to_string(&config).unwrap();
    if !body.allow_browser_direct {
        return Err(AppError::Forbidden("Browser sign-in connects directly to the hub and identity provider outside Iroh. Explicit browser consent is required; use a manual bearer credential to keep all hub requests on the selected route".into()));
    }
    if matches!(config.transport_mode, transport::TransportMode::Anonymous) {
        return Err(AppError::BadRequest(
            "Anonymous authorization is unsupported".into(),
        ));
    }
    // Revalidate public identity using the saved peer, rather than trusting stale auth capability metadata.
    let response = hub_request(
        state.data_dir(),
        &source,
        &config,
        "GET",
        "/api/client/v1/capabilities",
        vec![],
        vec![],
    )
    .await?;
    let descriptor = hub_response_json(response).await?;
    validate_advertised_bootstrap(&descriptor, &config.bootstrap)?;
    if hub::validate_descriptor(&source.normalized_base_url, &descriptor)? != config.instance_id {
        return Err(AppError::Forbidden(
            "Hub identity changed; reconnect and confirm it".into(),
        ));
    }
    let supports_s256 = descriptor
        .pointer("/authentication/pkce_methods")
        .and_then(Value::as_array)
        .is_some_and(|methods| methods.iter().any(|m| m.as_str() == Some("S256")));
    if !hub::capabilities(&descriptor).authenticate || !supports_s256 {
        return Err(AppError::BadRequest(
            "Hub browser authorization is unavailable".into(),
        ));
    }
    if shared_accounts::generation(&open_db(state.data_dir())?, &id)? != account_generation
        || load_hub(state.data_dir(), &id)?
            .as_ref()
            .map(|c| serde_json::to_string(c).unwrap())
            .as_deref()
            != Some(config_fingerprint.as_str())
    {
        return Err(AppError::Upstream(
            StatusCode::CONFLICT,
            "Community account changed while starting sign-in; try again".into(),
        ));
    }
    let oauth_state = random_secret()?;
    let verifier = random_secret()?;
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let mut url = reqwest::Url::parse(&format!(
        "{}/api/client/v1/authorize",
        source.normalized_base_url
    ))
    .map_err(|_| AppError::BadRequest("Invalid hub authorization origin".into()))?;
    url.query_pairs_mut().extend_pairs([
        ("response_type", "code"),
        ("client_id", "donutmediacenter"),
        ("redirect_uri", "donutmediacenter://auth/callback"),
        ("state", &oauth_state),
        ("code_challenge", &challenge),
        ("code_challenge_method", "S256"),
        ("scope", "catalog:read publish:create publish:read"),
    ]);
    {
        let mut pending = pending_authorizations()
            .lock()
            .map_err(|_| AppError::Internal("Authorization state unavailable".into()))?;
        pending
            .retain(|_, auth| auth.expires_at > std::time::Instant::now() && auth.source_id != id);
        if pending.len() >= 64 {
            return Err(AppError::TooManyRequests(
                "Too many pending hub authorizations".into(),
            ));
        }
        pending.insert(
            oauth_state,
            PendingAuthorization {
                data_dir: state.data_dir().to_path_buf(),
                source_id: id,
                instance_id: config.instance_id,
                generation: account_generation,
                config_fingerprint,
                verifier,
                expires_at: std::time::Instant::now() + Duration::from_secs(300),
            },
        );
    }
    Ok(Json(json!({"authorization_url":url.to_string()})))
}
#[derive(Deserialize)]
struct HubCallbackRequest {
    callback_url: String,
}
async fn complete_hub_callback(
    State(state): State<AppState>,
    Json(body): Json<HubCallbackRequest>,
) -> Result<Json<Value>, AppError> {
    if body.callback_url.len() > 16 * 1024 {
        return Err(AppError::BadRequest(
            "Authorization callback exceeds size limit".into(),
        ));
    }
    let url = reqwest::Url::parse(&body.callback_url)
        .map_err(|_| AppError::BadRequest("Invalid authorization callback".into()))?;
    if url.scheme() != "donutmediacenter"
        || url.host_str() != Some("auth")
        || url.path() != "/callback"
        || url.port().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(AppError::BadRequest(
            "Authorization callback must exactly match donutmediacenter://auth/callback".into(),
        ));
    }
    let mut values = HashMap::new();
    for (key, value) in url.query_pairs() {
        if !matches!(
            key.as_ref(),
            "state" | "code" | "error" | "error_description"
        ) || values
            .insert(key.into_owned(), value.into_owned())
            .is_some()
        {
            return Err(AppError::BadRequest(
                "Invalid or repeated callback parameter".into(),
            ));
        }
    }
    let oauth_state = values
        .get("state")
        .ok_or_else(|| AppError::BadRequest("Authorization callback has no state".into()))?;
    // Remove atomically before any network request: replay and concurrent callbacks fail closed.
    let pending = pending_authorizations()
        .lock()
        .map_err(|_| AppError::Internal("Authorization state unavailable".into()))?
        .remove(oauth_state)
        .ok_or_else(|| {
            AppError::BadRequest("Authorization state is unknown or already consumed".into())
        })?;
    if pending.data_dir != state.data_dir() || pending.expires_at <= std::time::Instant::now() {
        return Err(AppError::BadRequest(
            "Authorization state expired or belongs to another client".into(),
        ));
    }
    if values.contains_key("error") {
        return Err(AppError::BadRequest(
            "Hub authorization was denied; no account was connected".into(),
        ));
    }
    let code = values
        .get("code")
        .filter(|code| !code.is_empty() && code.len() <= 4096)
        .ok_or_else(|| AppError::BadRequest("Authorization callback has no valid code".into()))?;
    let source = load_source(state.data_dir(), &pending.source_id)?;
    require_enabled(&source)?;
    let config = load_hub(state.data_dir(), &pending.source_id)?
        .ok_or_else(|| AppError::BadRequest("Hub was disconnected during authorization".into()))?;
    if config.instance_id != pending.instance_id
        || shared_accounts::generation(&open_db(state.data_dir())?, &pending.source_id)?
            != pending.generation
        || serde_json::to_string(&config).unwrap() != pending.config_fingerprint
    {
        return Err(AppError::Forbidden(
            "Authorization hub identity changed".into(),
        ));
    }
    let form = reqwest::Url::parse_with_params(
        "https://form.invalid/",
        &[
            ("grant_type", "authorization_code"),
            ("code", code.as_str()),
            ("client_id", "donutmediacenter"),
            ("redirect_uri", "donutmediacenter://auth/callback"),
            ("code_verifier", &pending.verifier),
        ],
    )
    .map_err(|_| AppError::BadRequest("Invalid authorization exchange".into()))?
    .query()
    .unwrap_or_default()
    .as_bytes()
    .to_vec();
    // A token exchange never uses or replaces the selected Iroh routing policy.
    validate_relay(&config.bootstrap, source.allow_local_network).await?;
    let response = transport::request(
        &config.bootstrap,
        config.transport_mode,
        transport::HubRequest {
            method: "POST".into(),
            path: "/api/client/v1/token".into(),
            headers: vec![(
                "Content-Type".into(),
                "application/x-www-form-urlencoded".into(),
            )],
            body: form,
        },
    )
    .await
    .map_err(AppError::ServiceUnavailable)?;
    let token = hub_response_json(response).await?;
    if token.get("token_type").and_then(Value::as_str) != Some("Bearer") {
        return Err(AppError::ServiceUnavailable(
            "Hub returned an unsupported credential type".into(),
        ));
    }
    let secret = token
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty() && token.len() <= 16 * 1024)
        .ok_or_else(|| {
            AppError::ServiceUnavailable("Hub token response has no valid credential".into())
        })?;
    HeaderValue::from_str(secret)
        .map_err(|_| AppError::ServiceUnavailable("Hub returned an invalid credential".into()))?;
    require_enabled(&load_source(state.data_dir(), &pending.source_id)?)?;
    let account = token
        .pointer("/account/name")
        .or_else(|| token.pointer("/account/display_name"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let connection_id = Uuid::new_v4().to_string();
    let credential_ref = format!("{}.json", Uuid::new_v4());
    let lifetime = token
        .get("expires_in")
        .and_then(Value::as_i64)
        .filter(|seconds| *seconds > 0 && *seconds <= 366 * 24 * 3600)
        .ok_or_else(|| {
            AppError::ServiceUnavailable("Hub credential must have a finite lifetime".into())
        })?;
    write_credential(
        state.data_dir(),
        &credential_ref,
        &StoredCredential {
            auth_kind: "bearer".into(),
            secret: secret.into(),
            expires_at: Some(Utc::now().timestamp() + lifetime),
        },
    )?;
    let insert = (|| -> Result<(), AppError> {
        let mut conn = open_db(state.data_dir())?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let enabled: bool = tx.query_row(
            "SELECT enabled FROM remote_sources WHERE source_id=?1",
            params![pending.source_id],
            |r| r.get(0),
        )?;
        if !enabled || shared_accounts::generation(&tx, &pending.source_id)? != pending.generation {
            return Err(AppError::Upstream(
                StatusCode::CONFLICT,
                "Community account changed during sign-in; start again".into(),
            ));
        }
        tx.execute(
            "INSERT INTO remote_connections VALUES(?1,?2,?3,?4,'verified',?5)",
            params![
                connection_id,
                pending.source_id,
                account,
                credential_ref,
                Utc::now().to_rfc3339()
            ],
        )?;
        tx.commit()?;
        Ok(())
    })();
    if let Err(error) = insert {
        let _ = std::fs::remove_file(credential_path(state.data_dir(), &credential_ref)?);
        return Err(error);
    }
    purge_source_cache(&open_db(state.data_dir())?, &pending.source_id)?;
    shared_accounts::reuse_after_signin(state.data_dir(), &pending.source_id).await;
    Ok(Json(
        json!({"connection_id":connection_id,"source_id":pending.source_id,"account_display_name":account,"trust_state":"verified","has_credential":true}),
    ))
}

pub(crate) async fn validate_relay(
    bootstrap: &transport::HubBootstrap,
    allow_local_network: bool,
) -> Result<(), AppError> {
    let relay = reqwest::Url::parse(&bootstrap.relay_url)
        .map_err(|_| AppError::BadRequest("Invalid hub relay URL".into()))?;
    // Resolving the relay is allowed in Relayed mode; the hub origin is never resolved.
    security::validate_resolved_url(&relay, allow_local_network).await
}
fn publication_multipart(
    snapshot_json: &str,
    file_path: &str,
    content: &[u8],
) -> Result<(String, Vec<u8>), AppError> {
    let boundary = format!("dmc-{}", Uuid::new_v4());
    let filename: String = Path::new(file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("publication.bin")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || "._-".contains(ch) {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"snapshot_json\"\r\n\r\n{snapshot_json}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").into_bytes();
    body.extend_from_slice(content);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    if body.len() > 64 * 1024 * 1024 {
        return Err(AppError::BadRequest(
            "Publication exceeds native request limit".into(),
        ));
    }
    Ok((format!("multipart/form-data; boundary={boundary}"), body))
}
fn receipt_local_state(receipt: &Value) -> &str {
    match receipt
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("submitted")
    {
        "accepted" => "published",
        "tombstoned" => "retracted",
        "processing" => "submitted",
        "pending_moderation" => "pending_moderation",
        "rejected" => "rejected",
        _ => "submitted",
    }
}
async fn refresh_publication(
    State(state): State<AppState>,
    AxumPath(publication_id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let targets = {
        let conn = open_db(state.data_dir())?;
        let mut stmt = conn.prepare("SELECT target_id,connection_id,receipt_json FROM publication_targets WHERE publication_id=?1 AND receipt_json IS NOT NULL")?;
        let rows = stmt.query_map(params![publication_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    let mut outcomes = vec![];
    for (target_id, source_id, stored) in targets {
        let source = load_source(state.data_dir(), &source_id)?;
        require_enabled(&source)?;
        let config = load_hub(state.data_dir(), &source_id)?.ok_or_else(|| {
            AppError::BadRequest("Receipt refresh requires an enrolled native hub".into())
        })?;
        let stored: Value = serde_json::from_str(&stored)
            .map_err(|_| AppError::Internal("Stored publication receipt is invalid".into()))?;
        let remote_id = stored
            .get("publication_id")
            .and_then(Value::as_str)
            .filter(|id| Uuid::parse_str(id).is_ok())
            .ok_or_else(|| {
                AppError::BadRequest("Receipt has no native publication identity".into())
            })?;
        let response = hub_request(
            state.data_dir(),
            &source,
            &config,
            "GET",
            &format!("/api/client/v1/publications/{remote_id}"),
            vec![],
            vec![],
        )
        .await?;
        let receipt = hub_response_json(response).await?;
        validate_hub_receipt(&source, &receipt)?;
        let local_state = receipt_local_state(&receipt);
        let remote_url = receipt
            .pointer("/receipt/canonical_url")
            .and_then(Value::as_str);
        if let Some(url) = remote_url {
            let base = reqwest::Url::parse(&source.normalized_base_url)
                .map_err(|_| AppError::BadRequest("Invalid hub origin".into()))?;
            let url = reqwest::Url::parse(url)
                .map_err(|_| AppError::ServiceUnavailable("Invalid receipt URL".into()))?;
            if base.origin() != url.origin() {
                return Err(AppError::Forbidden(
                    "Receipt URL differs from enrolled hub origin".into(),
                ));
            }
        }
        open_db(state.data_dir())?.execute("UPDATE publication_targets SET state=?3,remote_post_id=?4,remote_url=?5,receipt_json=?6,error_json=NULL,updated_at=?7 WHERE publication_id=?1 AND target_id=?2",
            params![publication_id,target_id,local_state,receipt.get("remote_post_id").and_then(Value::as_str),remote_url,receipt.to_string(),Utc::now().to_rfc3339()])?;
        outcomes.push(json!({"target_id":target_id,"source_id":source_id,"state":local_state,"receipt":receipt}));
    }
    Ok(Json(
        json!({"publication_id":publication_id,"targets":outcomes}),
    ))
}

fn validate_advertised_bootstrap(
    descriptor: &Value,
    bootstrap: &transport::HubBootstrap,
) -> Result<(), AppError> {
    if let Some(advertised) = descriptor.pointer("/transports/iroh") {
        if advertised.get("peer_id").and_then(Value::as_str) != Some(bootstrap.peer_id.as_str())
            || advertised
                .get("relay_url")
                .and_then(Value::as_str)
                .map(|s| s.trim_end_matches('/'))
                != Some(bootstrap.relay_url.trim_end_matches('/'))
        {
            return Err(AppError::Forbidden(
                "Hub descriptor bootstrap differs from the pinned enrollment peer and relay".into(),
            ));
        }
    }
    Ok(())
}
fn enforce_item_policy(
    item: &RemoteItem,
    profile: &str,
    family_locked: bool,
) -> Result<(), AppError> {
    let mut page = RemotePage {
        items: vec![item.clone()],
        page: 1,
        per_page: 1,
        total: None,
        available_ratings: vec![],
        stale: false,
        offline: false,
        partial: false,
        next_cursor: None,
    };
    enforce_content_policy(&mut page, profile, family_locked);
    if page.items.is_empty() {
        return Err(AppError::Forbidden(
            "Remote item is hidden by the current content policy".into(),
        ));
    }
    Ok(())
}
async fn read_publication_media(file_path: &str) -> Result<Vec<u8>, AppError> {
    use tokio::io::AsyncReadExt;
    let file = tokio::fs::File::open(file_path)
        .await
        .map_err(|_| AppError::BadRequest("Publication media is unavailable".into()))?;
    let mut bytes = Vec::new();
    file.take(60 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .await?;
    if bytes.len() > 60 * 1024 * 1024 {
        return Err(AppError::BadRequest(
            "Publication media exceeds native upload limit".into(),
        ));
    }
    Ok(bytes)
}

fn validate_hub_receipt(source: &RemoteSource, receipt: &Value) -> Result<(), AppError> {
    if !receipt
        .get("publication_id")
        .and_then(Value::as_str)
        .is_some_and(|id| Uuid::parse_str(id).is_ok())
        || !matches!(
            receipt.get("state").and_then(Value::as_str),
            Some("processing" | "pending_moderation" | "accepted" | "rejected" | "tombstoned")
        )
    {
        return Err(AppError::ServiceUnavailable(
            "Hub returned an invalid authoritative publication receipt".into(),
        ));
    }
    if let Some(status_url) = receipt
        .pointer("/receipt/status_url")
        .and_then(Value::as_str)
    {
        hub::path(&source.normalized_base_url, status_url)?;
    }
    if let Some(raw_url) = receipt
        .pointer("/receipt/canonical_url")
        .and_then(Value::as_str)
    {
        let origin = reqwest::Url::parse(&source.normalized_base_url)
            .map_err(|_| AppError::BadRequest("Invalid hub origin".into()))?;
        let url = reqwest::Url::parse(raw_url).map_err(|_| {
            AppError::ServiceUnavailable("Invalid hub receipt canonical URL".into())
        })?;
        if receipt.get("state").and_then(Value::as_str) != Some("accepted")
            || origin.origin() != url.origin()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(AppError::Forbidden(
                "Hub receipt must expose a same-origin canonical URL only after acceptance".into(),
            ));
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct SourceTagsQuery {
    query: Option<String>,
    limit: Option<u32>,
}
async fn source_tags(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<SourceTagsQuery>,
    // Independent extractors keep the URL contract flat while sharing the
    // browse parser (serde flatten cannot parse URL-encoded numeric fields).
    Query(filters): Query<hub::CatalogQuery>,
) -> Result<Json<Value>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    let mut filters = filters.normalized()?;
    if filters.offline_only {
        return Ok(Json(offline::local_tags(
            state.data_dir(),
            &id,
            &filters,
            &source,
            state.is_family_mode_locked(),
            query.query.as_deref().unwrap_or_default(),
            query.limit.unwrap_or(20) as usize,
        )?));
    }
    require_enabled(&source)?;
    let config = load_hub(state.data_dir(), &id)?.ok_or_else(|| {
        AppError::BadRequest("Visibility-safe tag facets require an enrolled native hub".into())
    })?;
    let empty_policy_intersection = apply_catalog_policy(
        &mut filters,
        &source.policy_profile,
        state.is_family_mode_locked(),
    );
    let path = hub::catalog_tags_path(
        &filters,
        &query.query.unwrap_or_default(),
        query.limit.unwrap_or(20),
    )?;
    let response = hub_request(
        state.data_dir(),
        &source,
        &config,
        "GET",
        &path,
        vec![],
        vec![],
    )
    .await?;
    let mut body = hub_response_json(response).await?;
    if !body.get("tags").is_some_and(Value::is_array)
        || !body.get("available_ratings").is_some_and(Value::is_array)
    {
        return Err(AppError::ServiceUnavailable(
            "Hub returned invalid tag facets".into(),
        ));
    }
    if empty_policy_intersection {
        // Preserve the protected read and its errors before hiding a disjoint
        // user/policy context, exactly as the catalog page does.
        body["tags"] = json!([]);
    }
    if source.policy_profile == "safe"
        || source.policy_profile == "adult-hidden"
        || state.is_family_mode_locked()
    {
        if let Some(ratings) = body
            .get_mut("available_ratings")
            .and_then(Value::as_array_mut)
        {
            ratings.retain(|rating| matches!(rating.as_str(), Some("pg" | "pg13")));
        }
    }
    Ok(Json(body))
}

// Send the effective policy to the hub as part of its query so counts and
// pagination use the same constraints as the displayed media.
fn apply_catalog_policy(query: &mut hub::CatalogQuery, profile: &str, family_locked: bool) -> bool {
    if !family_locked && !matches!(profile, "safe" | "adult-hidden") {
        return false;
    }
    let allowed = ["pg", "pg13"];
    let requested = query
        .rating
        .as_deref()
        .unwrap_or("pg,pg13,r,x,xxx")
        .split(',')
        .collect::<Vec<_>>();
    let effective = allowed
        .into_iter()
        .filter(|rating| requested.contains(rating))
        .collect::<Vec<_>>();
    let empty = effective.is_empty();
    // A disjoint user filter still makes an authorized read before returning
    // its constant empty result; it never exposes private cached counts.
    query.rating = Some(if empty {
        "pg,pg13".into()
    } else {
        effective.join(",")
    });
    empty
}

#[cfg(test)]
mod catalog_policy_tests {
    use super::*;

    #[test]
    fn facet_request_shares_flat_catalog_controls_and_prefix_parsing() {
        let uri: axum::http::Uri = "/online/sources/hub/tags?query=cat&limit=12&q=sunrise+-indoor&media_type=image&rating=pg13,xxx&favorites_only=true&min_width=1000&min_height=700&orientation=landscape&min_duration=0.5&max_duration=60&timeframe=today".parse().unwrap();
        let Query(prefix) = Query::<SourceTagsQuery>::try_from_uri(&uri).unwrap();
        let Query(filters) = Query::<hub::CatalogQuery>::try_from_uri(&uri).unwrap();
        let mut filters = filters.normalized().unwrap();
        assert!(!apply_catalog_policy(&mut filters, "safe", false));
        let path = hub::catalog_tags_path(
            &filters,
            prefix.query.as_deref().unwrap(),
            prefix.limit.unwrap(),
        )
        .unwrap();
        let url = reqwest::Url::parse(&format!("https://hub.invalid{path}")).unwrap();
        assert_eq!(url.path(), "/api/client/v1/catalog/tags");
        let pairs = url
            .query_pairs()
            .into_owned()
            .collect::<std::collections::HashMap<_, _>>();
        for (name, value) in [
            ("query", "cat"),
            ("limit", "12"),
            ("tags", "sunrise -indoor"),
            ("media_type", "image"),
            ("rating", "pg13"),
            ("favorites_only", "true"),
            ("min_width", "1000"),
            ("min_height", "700"),
            ("orientation", "landscape"),
            ("min_duration", "0.5"),
            ("max_duration", "60"),
            ("timeframe", "day"),
        ] {
            assert_eq!(pairs.get(name).map(String::as_str), Some(value), "{name}");
        }
        for field in ["page", "per_page", "collection", "sort"] {
            assert!(!pairs.contains_key(field));
        }
    }

    #[test]
    fn facets_validate_age_and_ranges_before_request_and_signal_disjoint_policy() {
        let uri: axum::http::Uri = "/online/sources/hub/tags?query=cat&min_age=0"
            .parse()
            .unwrap();
        let Query(filters) = Query::<hub::CatalogQuery>::try_from_uri(&uri).unwrap();
        assert!(matches!(filters.normalized(), Err(AppError::Forbidden(_))));
        let mut filters = hub::CatalogQuery {
            rating: Some("xxx".into()),
            media_type: Some("video".into()),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert!(apply_catalog_policy(&mut filters, "safe", false));
        assert!(hub::catalog_tags_path(&filters, "", 100)
            .unwrap()
            .contains("limit=50"));
        assert!(matches!(
            hub::catalog_tags_path(&filters, &"x".repeat(101), 20),
            Err(AppError::BadRequest(_))
        ));
        assert!(matches!(
            hub::CatalogQuery {
                min_width: Some(0),
                ..Default::default()
            }
            .normalized(),
            Err(AppError::BadRequest(_))
        ));
    }

    #[test]
    fn safe_catalog_policy_reaches_upstream_before_counting() {
        let mut query = hub::CatalogQuery {
            rating: Some("pg13,r,xxx".into()),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert!(!apply_catalog_policy(&mut query, "safe", false));
        assert_eq!(query.rating.as_deref(), Some("pg13"));
        assert!(hub::catalog_path_for(&query)
            .unwrap()
            .contains("rating=pg13"));

        let mut disjoint = hub::CatalogQuery {
            rating: Some("xxx".into()),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert!(apply_catalog_policy(&mut disjoint, "adult-hidden", false));
        assert_eq!(disjoint.rating.as_deref(), Some("pg,pg13"));
    }

    #[test]
    fn family_policy_overrides_unrestricted_source_without_changing_query_context() {
        let mut query = hub::CatalogQuery {
            q: Some("cat -dog".into()),
            media_type: Some("video".into()),
            favorites_only: true,
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert!(!apply_catalog_policy(&mut query, "unrestricted", true));
        assert_eq!(query.rating.as_deref(), Some("pg,pg13"));
        assert_eq!(query.q.as_deref(), Some("cat -dog"));
        assert_eq!(query.media_type.as_deref(), Some("video"));
        assert!(query.favorites_only);

        let mut unrestricted = hub::CatalogQuery::default().normalized().unwrap();
        assert!(!apply_catalog_policy(
            &mut unrestricted,
            "unrestricted",
            false
        ));
        assert_eq!(unrestricted.rating, None);
    }
}

pub(crate) fn initialize_offline_database(data_dir: &Path) -> Result<(), String> {
    open_db(data_dir).map_err(|e| e.to_string())?;
    offline::open(data_dir).map_err(|e| e.to_string())?;
    Ok(())
}

#[derive(Default, Deserialize)]
struct CreatorsQuery {
    query: Option<String>,
    media_type: Option<String>,
    view: Option<String>,
    page: Option<u32>,
    per_page: Option<u32>,
}
async fn source_creators(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(query): Query<CreatorsQuery>,
) -> Result<Json<Value>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    let view = query.view.as_deref().unwrap_or("available");
    if !matches!(view, "available" | "following" | "offline" | "all") {
        return Err(AppError::BadRequest(
            "Invalid creator directory view".into(),
        ));
    }
    let filters = hub::CatalogQuery {
        media_type: query.media_type.clone(),
        page: query.page,
        per_page: query.per_page,
        ..Default::default()
    }
    .normalized()?;
    let page = filters.page.unwrap_or(1);
    let per = filters.per_page.unwrap_or(30).min(100);
    let search = query.query.as_deref().unwrap_or_default().trim();
    if search.len() > 100 {
        return Err(AppError::BadRequest(
            "Creator search exceeds 100 bytes".into(),
        ));
    }
    if matches!(view, "following" | "offline") {
        let mut rows = offline::creator_rows(
            state.data_dir(),
            &id,
            query.media_type.as_deref(),
            state.is_family_mode_locked()
                || matches!(source.policy_profile.as_str(), "safe" | "adult-hidden"),
        )?;
        rows.retain(|row| {
            row["name"]
                .as_str()
                .unwrap_or_default()
                .to_lowercase()
                .contains(&search.to_lowercase())
                && if view == "following" {
                    row["following"] == true
                } else {
                    row.pointer("/offline/downloaded_items")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        > 0
                        || row.pointer("/offline/enabled") == Some(&Value::Bool(true))
                }
        });
        rows.sort_by(|a, b| {
            a["name"]
                .as_str()
                .cmp(&b["name"].as_str())
                .then(a["id"].as_str().cmp(&b["id"].as_str()))
        });
        let start = ((page as usize - 1) * per as usize).min(rows.len());
        let more = rows.len() > start + per as usize;
        return Ok(Json(
            json!({"creators":rows.into_iter().skip(start).take(per as usize).collect::<Vec<_>>(),"page":page,"has_more":more,"offline":true}),
        ));
    }
    require_enabled(&source)?;
    let config = load_hub(state.data_dir(), &id)?.ok_or_else(|| {
        AppError::BadRequest("Creator collections require an enrolled native hub".into())
    })?;
    let mut url = reqwest::Url::parse(&format!(
        "{}/api/client/v1/catalog/creators",
        source.normalized_base_url
    ))
    .map_err(|e| AppError::Internal(e.to_string()))?;
    url.query_pairs_mut()
        .append_pair("query", search)
        .append_pair("view", view)
        .append_pair("page", &page.to_string())
        .append_pair("per_page", &per.to_string());
    if let Some(media) = query.media_type.as_deref().filter(|m| *m != "all") {
        url.query_pairs_mut().append_pair("media_type", media);
    }
    if state.is_family_mode_locked()
        || matches!(source.policy_profile.as_str(), "safe" | "adult-hidden")
    {
        url.query_pairs_mut().append_pair("rating", "pg,pg13");
    }
    let body = hub_response_json(
        hub_request(
            state.data_dir(),
            &source,
            &config,
            "GET",
            &hub::path(&source.normalized_base_url, url.as_str())?,
            vec![],
            vec![],
        )
        .await?,
    )
    .await?;
    let remote = body["items"]
        .as_array()
        .ok_or_else(|| AppError::ServiceUnavailable("Invalid hub creator directory".into()))?;
    if remote.len() > per as usize {
        return Err(AppError::ServiceUnavailable(
            "Hub creator page exceeds limit".into(),
        ));
    }
    let mut ids = Vec::new();
    for row in remote {
        offline::remember_creator_directory(
            state.data_dir(),
            &id,
            row,
            query.media_type.as_deref(),
            state.is_family_mode_locked()
                || matches!(source.policy_profile.as_str(), "safe" | "adult-hidden"),
        )?;
        ids.push(row["id"].as_str().unwrap().to_owned());
    }
    let mut rows = offline::creator_rows(
        state.data_dir(),
        &id,
        query.media_type.as_deref(),
        state.is_family_mode_locked()
            || matches!(source.policy_profile.as_str(), "safe" | "adult-hidden"),
    )?;
    rows.retain(|r| ids.iter().any(|id| r["id"].as_str() == Some(id)));
    if view == "available" {
        rows.retain(offline::creator_potentially_available);
    }
    rows.sort_by_key(|r| {
        ids.iter()
            .position(|id| r["id"].as_str() == Some(id))
            .unwrap_or(usize::MAX)
    });
    Ok(Json(
        json!({"creators":rows,"page":page,"has_more":body["has_more"].as_bool().unwrap_or(false)}),
    ))
}
async fn offline_settings(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    load_source(state.data_dir(), &id)?;
    Ok(Json(offline::settings(state.data_dir(), &id)?))
}
async fn update_offline_settings(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<offline::SettingsPatch>,
) -> Result<Json<Value>, AppError> {
    require_enabled(&load_source(state.data_dir(), &id)?)?;
    if load_hub(state.data_dir(), &id)?.is_none() {
        return Err(AppError::BadRequest(
            "Offline creator sync requires an enrolled native hub".into(),
        ));
    }
    offline::patch_settings(state.data_dir(), &id, &body)?;
    if offline::active(state.data_dir(), &id)? {
        offline::kick(&state, &id)?;
    }
    Ok(Json(offline::settings(state.data_dir(), &id)?))
}
async fn creator_preferences(
    State(state): State<AppState>,
    AxumPath((id, creator)): AxumPath<(String, String)>,
    Json(body): Json<offline::CreatorPatch>,
) -> Result<Json<Value>, AppError> {
    let source = load_source(state.data_dir(), &id)?;
    require_enabled(&source)?;
    offline::patch_creator(state.data_dir(), &id, &creator, &body)?;
    if body.offline_enabled.is_some() && offline::active(state.data_dir(), &id)? {
        offline::kick(&state, &id)?;
    }
    let row = offline::creator_rows(
        state.data_dir(),
        &id,
        body.media_type.as_deref(),
        state.is_family_mode_locked()
            || matches!(source.policy_profile.as_str(), "safe" | "adult-hidden"),
    )?
    .into_iter()
    .find(|r| r["id"] == creator)
    .ok_or_else(|| AppError::NotFound("Creator not found".into()))?;
    Ok(Json(
        json!({"creator":row,"offline":offline::settings(state.data_dir(),&id)?}),
    ))
}

static OFFLINE_PERMITS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
fn sync_continues(state: &AppState, sid: &str, cancel: &AtomicBool) -> Result<bool, AppError> {
    Ok(!cancel.load(Ordering::Acquire)
        && load_source(state.data_dir(), sid)?.enabled
        && offline::active(state.data_dir(), sid)?)
}
/// One worker per source; at most two media workers across the profile. Cursors
/// and the bounded queue survive process restart, while files commit atomically.
pub(crate) async fn sync_offline_source(
    state: &AppState,
    sid: &str,
    cancel: &AtomicBool,
) -> Result<(), AppError> {
    let _permit = OFFLINE_PERMITS
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)))
        .clone()
        .acquire_owned()
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?;
    if !sync_continues(state, sid, cancel)? {
        return Ok(());
    }
    let source = load_source(state.data_dir(), sid)?;
    let config = load_hub(state.data_dir(), sid)?
        .ok_or_else(|| AppError::BadRequest("Offline sync requires a native hub".into()))?;
    offline::state(state.data_dir(), sid, "syncing", None)?;
    // Drain existing work first, preserving its cursor during interrupted cycles.
    drain_offline_queue(state, &source, &config, cancel).await?;
    let mut complete = true;
    for (creator, media) in offline::scopes(state.data_dir(), sid)? {
        let mut scanned = 0;
        loop {
            if !sync_continues(state, sid, cancel)? {
                return Ok(());
            }
            let page = offline::scan_cursor(state.data_dir(), sid, creator.as_deref(), &media)?;
            let mut query = hub::CatalogQuery {
                creator_id: creator.clone(),
                media_type: Some(media.clone()),
                page: Some(page),
                per_page: Some(100),
                sort: Some("oldest".into()),
                ..Default::default()
            }
            .normalized()?;
            apply_catalog_policy(
                &mut query,
                &source.policy_profile,
                state.is_family_mode_locked(),
            );
            let body = hub_response_json(
                hub_request(
                    state.data_dir(),
                    &source,
                    &config,
                    "GET",
                    &hub::catalog_path_for(&query)?,
                    vec![],
                    vec![],
                )
                .await?,
            )
            .await?;
            let rows = body["items"].as_array().ok_or_else(|| {
                AppError::ServiceUnavailable("Invalid hub offline catalog".into())
            })?;
            if rows.len() > 100 {
                return Err(AppError::ServiceUnavailable(
                    "Hub offline page exceeds limit".into(),
                ));
            }
            for row in rows {
                let item = hub::catalog_item(sid, &source.normalized_base_url, row)?;
                if !offline::selected(
                    state.data_dir(),
                    sid,
                    item.creator.as_ref().map(|c| c.id.as_str()),
                    offline::item_media(&item),
                )? {
                    continue;
                }
                if let Some(creator) = &item.creator {
                    offline::remember_creator(
                        state.data_dir(),
                        sid,
                        &json!({"id":creator.id,"name":creator.name}),
                    )?;
                }
                offline::enqueue(state.data_dir(), sid, &item, row)?;
            }
            complete &= body["total"].as_u64().is_some();
            let total = body["total"].as_u64().unwrap_or(0);
            let more = body["has_more"].as_bool().unwrap_or(false)
                || body["next_cursor"].as_str().is_some()
                || (rows.len() == 100 && (total == 0 || page as u64 * 100 < total));
            offline::scan_advance(
                state.data_dir(),
                sid,
                creator.as_deref(),
                &media,
                if more { page + 1 } else { 1 },
                total,
            )?;
            drain_offline_queue(state, &source, &config, cancel).await?;
            scanned += rows.len();
            if more && scanned >= offline::MAX_PENDING_ITEMS {
                complete = false;
                break;
            }
            if !more {
                break;
            }
        }
    }
    if sync_continues(state, sid, cancel)? {
        offline::open(state.data_dir())?.execute(
            "UPDATE offline_sources SET state=?3,error=NULL,last_sync_at=?2 WHERE source_id=?1",
            params![
                sid,
                Utc::now().to_rfc3339(),
                if complete && offline::pending_count(state.data_dir(), sid)? == 0 {
                    "ready"
                } else {
                    "queued"
                }
            ],
        )?;
    }
    Ok(())
}
async fn drain_offline_queue(
    state: &AppState,
    source: &RemoteSource,
    config: &HubConfig,
    cancel: &AtomicBool,
) -> Result<(), AppError> {
    for queued in offline::queue(
        state.data_dir(),
        &source.source_id,
        offline::MAX_PENDING_ITEMS,
    )? {
        if !sync_continues(state, &source.source_id, cancel)? {
            return Ok(());
        }
        if !offline::selected(
            state.data_dir(),
            &source.source_id,
            queued.item.creator.as_ref().map(|c| c.id.as_str()),
            offline::item_media(&queued.item),
        )? {
            offline::complete_queue(
                state.data_dir(),
                &source.source_id,
                &queued.item.remote_post_id,
            )?;
            continue;
        }
        let result = download_offline_item(state, source, config, &queued, cancel).await;
        match result {
            Ok(true) => offline::complete_queue(
                state.data_dir(),
                &source.source_id,
                &queued.item.remote_post_id,
            )?,
            Ok(false) => return Ok(()),
            Err(AppError::Upstream(status, _))
                if status == StatusCode::NOT_FOUND || status == StatusCode::GONE =>
            {
                offline::withdraw(
                    state.data_dir(),
                    &source.source_id,
                    &queued.item.remote_post_id,
                )?;
            }
            Err(error) => {
                offline::fail_queue(
                    state.data_dir(),
                    &source.source_id,
                    &queued.item.remote_post_id,
                    &error.to_string(),
                )?;
                return Err(error);
            }
        }
    }
    Ok(())
}
async fn download_offline_item(
    state: &AppState,
    source: &RemoteSource,
    config: &HubConfig,
    queued: &offline::QueuedItem,
    cancel: &AtomicBool,
) -> Result<bool, AppError> {
    // Reauthorize current metadata before acquiring each item's bytes. Withdrawn
    // or changed publications do not become new community copies.
    let row = hub_response_json(
        hub_request(
            state.data_dir(),
            source,
            config,
            "GET",
            &format!("/api/client/v1/catalog/{}", queued.item.remote_post_id),
            vec![],
            vec![],
        )
        .await?,
    )
    .await?;
    let current = hub::catalog_item(&source.source_id, &source.normalized_base_url, &row)?;
    enforce_item_policy(
        &current,
        &source.policy_profile,
        state.is_family_mode_locked(),
    )?;
    if current.remote_revision != queued.item.remote_revision || current.media != queued.item.media
    {
        offline::enqueue(state.data_dir(), &source.source_id, &current, &row)?;
        return Ok(false);
    }
    for variant in &current.media {
        if !sync_continues(state, &source.source_id, cancel)? {
            return Ok(false);
        }
        if offline::is_ready(
            state.data_dir(),
            &source.source_id,
            &current.remote_post_id,
            &queued.revision,
            &variant.kind,
        )? {
            if !offline::can_reuse_variant(
                state.data_dir(),
                &source.source_id,
                &current.remote_post_id,
                &queued.revision,
                variant,
            )? {
                return Err(AppError::ServiceUnavailable("Canonical rendition changed without a new revision; existing personal copy retained".into()));
            }
            continue;
        }
        let response = match fetch_media(state.data_dir(), source, &variant.asset_url, None).await {
            Ok(response) => response,
            Err(error) if authorization_error(&error) => return Err(error),
            Err(error) => {
                if try_peer_offline_variant(state, source, config, queued, variant, cancel).await? {
                    continue;
                }
                return Err(error);
            }
        };
        let content_length = response
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let expected = variant.byte_size.or(content_length);
        if let (Some(a), Some(b)) = (variant.byte_size, content_length) {
            if a != b {
                return Err(AppError::ServiceUnavailable(
                    "Media size differs from catalog".into(),
                ));
            }
        }
        // Unknown size starts at zero and reserves before every received chunk;
        // never buffers a full original or writes beyond the chosen budget.
        offline::reserve(
            state.data_dir(),
            &source.source_id,
            &current.remote_post_id,
            &queued.revision,
            &variant.kind,
            expected.unwrap_or(0),
        )?;
        let root = offline::cache_root(state.data_dir(), &source.source_id);
        tokio::fs::create_dir_all(&root).await?;
        let staging = root.join(format!("{}.part", Uuid::new_v4()));
        let result = write_offline_response(
            state, source, queued, variant, response, expected, &staging, cancel,
        )
        .await;
        let _ = tokio::fs::remove_file(&staging).await;
        if result.is_err() || matches!(result, Ok(false)) {
            offline::release(
                state.data_dir(),
                &source.source_id,
                &current.remote_post_id,
                &queued.revision,
                &variant.kind,
            )?;
        }
        if !result? {
            return Ok(false);
        }
    }
    // Legacy central storage computes authoritative digests during a complete
    // normal read. Refresh its catalog metadata before advertising our replica.
    let refreshed = hub_response_json(
        hub_request(
            state.data_dir(),
            source,
            config,
            "GET",
            &format!("/api/client/v1/catalog/{}", current.remote_post_id),
            vec![],
            vec![],
        )
        .await?,
    )
    .await?;
    let refreshed_item =
        hub::catalog_item(&source.source_id, &source.normalized_base_url, &refreshed)?;
    offline::refresh_metadata(
        state.data_dir(),
        &source.source_id,
        &refreshed_item,
        &refreshed,
    )?;
    Ok(true)
}
#[allow(clippy::too_many_arguments)]
async fn write_offline_response(
    state: &AppState,
    source: &RemoteSource,
    queued: &offline::QueuedItem,
    variant: &crate::online::models::RemoteMediaVariant,
    response: MediaResponse,
    expected: Option<u64>,
    staging: &Path,
    cancel: &AtomicBool,
) -> Result<bool, AppError> {
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging)
        .await?;
    let mut stream = response.bytes_stream();
    let mut size = 0u64;
    let mut hash = Sha256::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.next())
        .await
        .map_err(|_| AppError::ServiceUnavailable("Offline media read timed out".into()))?
    {
        if !sync_continues(state, &source.source_id, cancel)?
            || !offline::selected(
                state.data_dir(),
                &source.source_id,
                queued.item.creator.as_ref().map(|c| c.id.as_str()),
                offline::item_media(&queued.item),
            )?
        {
            return Ok(false);
        }
        let chunk = chunk.map_err(|e| AppError::ServiceUnavailable(e.to_string()))?;
        size = size
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| AppError::ServiceUnavailable("Media size overflow".into()))?;
        if size > offline::MAX_VARIANT_BYTES || expected.is_some_and(|n| size > n) {
            return Err(AppError::ServiceUnavailable(
                "Offline media exceeds expected size".into(),
            ));
        }
        if expected.is_none() {
            offline::reserve(
                state.data_dir(),
                &source.source_id,
                &queued.item.remote_post_id,
                &queued.revision,
                &variant.kind,
                size,
            )?;
        }
        hash.update(&chunk);
        file.write_all(&chunk).await?;
    }
    if size == 0 || expected.is_some_and(|n| size != n) {
        return Err(AppError::ServiceUnavailable(
            "Incomplete offline media".into(),
        ));
    }
    let digest = format!("{:x}", hash.finalize());
    if variant
        .sha256
        .as_deref()
        .is_some_and(|expected| expected != digest)
    {
        return Err(AppError::ServiceUnavailable(
            "Offline media digest mismatch".into(),
        ));
    }
    file.sync_all().await?;
    drop(file);
    if !sync_continues(state, &source.source_id, cancel)? {
        return Ok(false);
    }
    let destination = offline::cache_root(state.data_dir(), &source.source_id).join(format!(
        "{}-{}",
        digest,
        Uuid::new_v4()
    ));
    tokio::fs::rename(staging, &destination).await?;
    if let Err(error) = offline::finish(
        state.data_dir(),
        &source.source_id,
        queued,
        &variant.kind,
        &digest,
        size,
        &destination,
        variant.sha256.is_some(),
        variant.mime_type.as_deref(),
    ) {
        let _ = tokio::fs::remove_file(destination).await;
        return Err(error);
    }
    Ok(true)
}

#[derive(Deserialize)]
struct RemoveOfflineCopies {
    #[serde(default)]
    confirmed_remove: bool,
}
async fn remove_offline_copies(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<RemoveOfflineCopies>,
) -> Result<StatusCode, AppError> {
    load_source(state.data_dir(), &id)?;
    if !body.confirmed_remove {
        return Err(AppError::BadRequest(
            "Confirm permanently removing personal offline copies".into(),
        ));
    }
    offline::remove_copies(state.data_dir(), &id)?;
    Ok(StatusCode::NO_CONTENT)
}

fn authorization_error(error: &AppError) -> bool {
    matches!(error, AppError::Unauthorized(_) | AppError::Forbidden(_))
        || matches!(error,AppError::Upstream(status,_) if matches!(status.as_u16(),401|403))
}
async fn try_peer_offline_variant(
    state: &AppState,
    source: &RemoteSource,
    config: &HubConfig,
    queued: &offline::QueuedItem,
    variant: &crate::online::models::RemoteMediaVariant,
    cancel: &AtomicBool,
) -> Result<bool, AppError> {
    let (Some(digest), Some(size)) = (&variant.sha256, variant.byte_size) else {
        return Ok(false);
    };
    let root = offline::cache_root(state.data_dir(), &source.source_id);
    tokio::fs::create_dir_all(&root).await?;
    offline::reserve(
        state.data_dir(),
        &source.source_id,
        &queued.item.remote_post_id,
        &queued.revision,
        &variant.kind,
        size,
    )?;
    let destination = root.join(format!("{}-{}", digest, Uuid::new_v4()));
    let result = try_peer_download(
        state,
        source,
        config,
        &queued.item,
        variant,
        &destination,
        Some(cancel),
    )
    .await;
    let result = match result {
        Ok(true) => sync_continues(state, &source.source_id, cancel),
        other => other,
    };
    if matches!(&result, Ok(true)) {
        if let Err(error) = offline::finish(
            state.data_dir(),
            &source.source_id,
            queued,
            &variant.kind,
            digest,
            size,
            &destination,
            true,
            variant.mime_type.as_deref(),
        ) {
            let _ = tokio::fs::remove_file(&destination).await;
            offline::release(
                state.data_dir(),
                &source.source_id,
                &queued.item.remote_post_id,
                &queued.revision,
                &variant.kind,
            )?;
            return Err(error);
        }
        return Ok(true);
    }
    let _ = tokio::fs::remove_file(&destination).await;
    offline::release(
        state.data_dir(),
        &source.source_id,
        &queued.item.remote_post_id,
        &queued.revision,
        &variant.kind,
    )?;
    result.map(|_| false)
}

async fn try_peer_download(
    state: &AppState,
    source: &RemoteSource,
    config: &HubConfig,
    item: &RemoteItem,
    variant: &crate::online::models::RemoteMediaVariant,
    destination: &Path,
    cancel: Option<&AtomicBool>,
) -> Result<bool, AppError> {
    let (Some(digest), Some(size)) = (&variant.sha256, variant.byte_size) else {
        return Ok(false);
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || size == 0
        || item.remote_revision.is_none()
    {
        return Ok(false);
    }
    if config.transport_mode == transport::TransportMode::Anonymous {
        return Ok(false);
    }
    // Existing enrollments may predate peer_read; refresh through their existing
    // pinned gateway before looking up providers or trusting a signing key.
    let refreshed = refresh_hub_descriptor(state.data_dir(), source, config).await?;
    let config = &refreshed;
    if !config
        .descriptor
        .get("peer_read")
        .is_some_and(Value::is_object)
    {
        return Ok(false);
    }
    let mut url = reqwest::Url::parse(&format!(
        "{}/api/client/v1/catalog/{}/providers",
        source.normalized_base_url, item.remote_post_id
    ))
    .map_err(|e| AppError::Internal(e.to_string()))?;
    url.query_pairs_mut().append_pair("variant", &variant.kind);
    let lookup = hub_response_json(
        hub_request(
            state.data_dir(),
            source,
            config,
            "GET",
            &hub::path(&source.normalized_base_url, url.as_str())?,
            vec![],
            vec![],
        )
        .await?,
    )
    .await?;
    if lookup["peer_eligible"] != true {
        return Ok(false);
    }
    let Some(providers) = lookup["providers"]
        .as_array()
        .filter(|providers| !providers.is_empty())
    else {
        return Ok(false);
    };
    if !config
        .descriptor
        .get("peer_read")
        .is_some_and(Value::is_object)
    {
        return Ok(false);
    }
    crate::online::peer::enroll_authority(state.data_dir(), &source.source_id, source, config)
        .map_err(AppError::ServiceUnavailable)?;
    let identity = crate::online::peer::ensure_provider(
        state.data_dir(),
        &config.bootstrap,
        config.transport_mode,
    )
    .await
    .map_err(AppError::ServiceUnavailable)?;
    for provider in providers.iter().take(8) {
        if !peer_download_continues(state, source, item, cancel)? {
            return Ok(false);
        }
        let (Some(endpoint), Some(relay)) = (
            provider["endpoint_id"].as_str(),
            provider["relay_url"].as_str(),
        ) else {
            continue;
        };
        if provider["sha256"].as_str() != Some(digest)
            || provider["byte_size"].as_u64() != Some(size)
        {
            continue;
        }
        let body=serde_json::to_vec(&json!({"remote_post_id":item.remote_post_id,"variant":variant.kind,"provider_id":endpoint,"requester_id":identity.endpoint_id})).map_err(|e|AppError::Internal(e.to_string()))?;
        let grant = match hub_response_json(
            hub_request(
                state.data_dir(),
                source,
                config,
                "POST",
                "/api/client/v1/peer/grants",
                vec![("Content-Type".into(), "application/json".into())],
                body,
            )
            .await?,
        )
        .await
        {
            Ok(grant) => grant,
            Err(error) if authorization_error(&error) => return Err(error),
            Err(_) => continue,
        };
        let grant: crate::online::peer::Grant = serde_json::from_value(grant)
            .map_err(|_| AppError::ServiceUnavailable("Invalid peer grant".into()))?;
        let result = crate::online::peer::download(
            state.data_dir(),
            &source.source_id,
            &config.bootstrap,
            config.transport_mode,
            endpoint,
            relay,
            &grant,
            destination,
            digest,
            size,
            item.remote_revision.as_deref().ok_or_else(|| {
                AppError::ServiceUnavailable("Peer viewing requires a canonical revision".into())
            })?,
        )
        .await;
        if result.is_ok() && peer_download_continues(state, source, item, cancel)? {
            return Ok(true);
        }
        let _ = tokio::fs::remove_file(destination).await;
    }
    Ok(false)
}

fn peer_download_continues(
    state: &AppState,
    source: &RemoteSource,
    item: &RemoteItem,
    cancel: Option<&AtomicBool>,
) -> Result<bool, AppError> {
    if let Some(cancel) = cancel {
        return sync_continues(state, &source.source_id, cancel);
    }
    let current_source = load_source(state.data_dir(), &source.source_id)?;
    require_enabled(&current_source)?;
    enforce_item_policy(
        item,
        &current_source.policy_profile,
        state.is_family_mode_locked(),
    )?;
    Ok(true)
}

#[cfg(test)]
mod creator_offline_tests {
    use super::*;
    use crate::online::models::{RemoteCreator, RemoteMediaVariant};

    fn setup(
        cap: u64,
    ) -> (
        tempfile::TempDir,
        AppState,
        RemoteSource,
        offline::QueuedItem,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let state = AppState::new(dir.path(), 0).unwrap();
        initialize_offline_database(dir.path()).unwrap();
        let source = RemoteSource {
            source_id: "hub-test".into(),
            provider_family: "native-hub".into(),
            display_name: "Hub".into(),
            normalized_base_url: "https://hub.invalid".into(),
            policy_profile: "unrestricted".into(),
            enabled: true,
            allow_local_network: false,
            capabilities: provider::capabilities("native-hub"),
            last_probe_at: None,
        };
        open_db(dir.path()).unwrap().execute("INSERT INTO remote_sources(source_id,provider_family,display_name,normalized_base_url,capabilities_json,created_at) VALUES(?1,?2,?3,?4,?5,'now')",params![source.source_id,source.provider_family,source.display_name,source.normalized_base_url,serde_json::to_string(&source.capabilities).unwrap()]).unwrap();
        open_db(dir.path()).unwrap().execute("INSERT INTO remote_hub_transport VALUES('hub-test','realm-instance','p','https://relay.invalid','relayed','{}')",[]).unwrap();
        offline::remember_creator(
            dir.path(),
            &source.source_id,
            &json!({"id":"creator-1","name":"Creator"}),
        )
        .unwrap();
        offline::patch_settings(
            dir.path(),
            &source.source_id,
            &offline::SettingsPatch {
                sync_all_creators: Some(true),
                storage_limit_bytes: Some(cap),
                confirmed_download: true,
                ..Default::default()
            },
        )
        .unwrap();
        let item = RemoteItem {
            source_id: source.source_id.clone(),
            remote_post_id: "42".into(),
            canonical_url: "https://hub.invalid/post/42".into(),
            remote_revision: Some("revision-1".into()),
            title: Some("Sunrise".into()),
            caption: None,
            alt_text: None,
            content_warning: None,
            creator: Some(RemoteCreator {
                id: "creator-1".into(),
                name: "Creator".into(),
            }),
            tags: vec!["sunrise".into()],
            rating: Some("pg".into()),
            source_url: None,
            width: Some(100),
            height: Some(80),
            duration: None,
            media: vec![RemoteMediaVariant {
                kind: "original".into(),
                asset_url: "https://hub.invalid/api/client/v1/media/42/original".into(),
                mime_type: Some("image/jpeg".into()),
                width: Some(100),
                height: Some(80),
                sha256: None,
                byte_size: None,
            }],
        };
        let queued = offline::QueuedItem {
            item,
            metadata: json!({"created_at":"2026-10-01T00:00:00Z","filename":"sunrise.jpg"}),
            revision: "revision-1".into(),
        };
        (dir, state, source, queued)
    }
    fn retain(dir: &Path, source: &RemoteSource, queued: &offline::QueuedItem, bytes: &[u8]) {
        let root = offline::cache_root(dir, &source.source_id);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join(format!("retained-{}-{}", queued.revision, Uuid::new_v4()));
        std::fs::write(&path, bytes).unwrap();
        offline::reserve(
            dir,
            &source.source_id,
            &queued.item.remote_post_id,
            &queued.revision,
            "original",
            bytes.len() as u64,
        )
        .unwrap();
        offline::finish(
            dir,
            &source.source_id,
            queued,
            "original",
            &format!("{:x}", Sha256::digest(bytes)),
            bytes.len() as u64,
            &path,
            false,
            Some("image/jpeg"),
        )
        .unwrap();
    }

    #[test]
    fn following_does_not_queue_download_and_storage_reservations_resume_within_cap() {
        let (dir, _, source, queued) = setup(8);
        offline::patch_settings(
            dir.path(),
            &source.source_id,
            &offline::SettingsPatch {
                sync_all_creators: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
        offline::patch_creator(
            dir.path(),
            &source.source_id,
            "creator-1",
            &offline::CreatorPatch {
                following: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!offline::active(dir.path(), &source.source_id).unwrap());
        assert_eq!(
            offline::pending_count(dir.path(), &source.source_id).unwrap(),
            0
        );
        assert!(offline::patch_creator(
            dir.path(),
            &source.source_id,
            "creator-1",
            &offline::CreatorPatch {
                offline_enabled: Some(true),
                ..Default::default()
            }
        )
        .is_err());
        offline::patch_creator(
            dir.path(),
            &source.source_id,
            "creator-1",
            &offline::CreatorPatch {
                offline_enabled: Some(true),
                confirmed_download: true,
                ..Default::default()
            },
        )
        .unwrap();
        offline::enqueue(
            dir.path(),
            &source.source_id,
            &queued.item,
            &queued.metadata,
        )
        .unwrap();
        offline::scan_cursor(dir.path(), &source.source_id, Some("creator-1"), "image").unwrap();
        offline::scan_advance(
            dir.path(),
            &source.source_id,
            Some("creator-1"),
            "image",
            4,
            500,
        )
        .unwrap();
        offline::reserve(
            dir.path(),
            &source.source_id,
            "42",
            "revision-1",
            "original",
            5,
        )
        .unwrap();
        assert!(offline::reserve(dir.path(), &source.source_id, "43", "r", "original", 4).is_err());
        assert_eq!(
            offline::settings(dir.path(), &source.source_id).unwrap()["reserved_bytes"],
            5
        );
        assert_eq!(
            offline::scan_cursor(dir.path(), &source.source_id, Some("creator-1"), "image")
                .unwrap(),
            4
        );
        assert_eq!(
            offline::queue(dir.path(), &source.source_id, 1).unwrap()[0]
                .item
                .remote_post_id,
            "42"
        );
    }

    #[tokio::test]
    async fn creator_preferences_returns_creator_and_complete_source_settings() {
        let (_dir, state, source, _) = setup(64);
        let Json(result) = creator_preferences(
            State(state),
            AxumPath((source.source_id, "creator-1".into())),
            Json(offline::CreatorPatch {
                following: Some(true),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(result["creator"]["following"], true);
        assert_eq!(result["offline"]["storage_limit_bytes"], 64);
        assert_eq!(result["offline"]["sync_all_creators"], true);
        assert_eq!(result["offline"]["paused"], false);
        assert_eq!(result["offline"]["peer_hosting_enabled"], false);
    }

    #[test]
    fn paused_downloads_keep_hosting_selected_copies_and_require_observed_report() {
        let (dir, _, source, queued) = setup(64);
        retain(dir.path(), &source, &queued, b"copy");
        let mut canonical = queued.item.clone();
        canonical.media[0].sha256 = Some(format!("{:x}", Sha256::digest(b"copy")));
        canonical.media[0].byte_size = Some(4);
        offline::refresh_metadata(dir.path(), &source.source_id, &canonical, &queued.metadata)
            .unwrap();
        offline::patch_settings(
            dir.path(),
            &source.source_id,
            &offline::SettingsPatch {
                paused: Some(true),
                peer_hosting_enabled: Some(true),
                confirmed_hosting: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!offline::active(dir.path(), &source.source_id).unwrap());
        assert!(
            offline::retained_variant(dir.path(), &source.source_id, "42", "original", false)
                .unwrap()
                .is_some()
        );
        assert_eq!(
            offline::settings(dir.path(), &source.source_id).unwrap()["peer_hosting_state"],
            "idle"
        );
        offline::open(dir.path())
            .unwrap()
            .execute(
                "UPDATE offline_sources SET peer_reported_at=?1,peer_reported_items=1",
                params![Utc::now().timestamp()],
            )
            .unwrap();
        assert_eq!(
            offline::settings(dir.path(), &source.source_id).unwrap()["peer_hosting_state"],
            "hosting"
        );
        offline::patch_creator(
            dir.path(),
            &source.source_id,
            "creator-1",
            &offline::CreatorPatch {
                offline_enabled: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            offline::retained_variant(dir.path(), &source.source_id, "42", "original", false)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn disabled_or_withdrawn_source_preserves_personal_ranges_and_family_policy() {
        let (dir, state, source, queued) = setup(64);
        retain(dir.path(), &source, &queued, b"hello retained media");
        offline::withdraw(dir.path(), &source.source_id, "42").unwrap();
        update_source(
            State(state.clone()),
            AxumPath(source.source_id.clone()),
            Json(SourcePatch {
                display_name: None,
                enabled: Some(false),
                policy_profile: None,
            }),
        )
        .await
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::RANGE, HeaderValue::from_static("bytes=0-4"));
        let response = asset(
            State(state.clone()),
            AxumPath((source.source_id.clone(), "42".into(), "original".into())),
            headers,
            Query(AssetQuery { offline_only: true }),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap()
                .as_ref(),
            b"hello"
        );
        assert!(asset(
            State(state.clone()),
            AxumPath((source.source_id.clone(), "42".into(), "original".into())),
            HeaderMap::new(),
            Query(AssetQuery::default())
        )
        .await
        .is_err());
        let mut adult = queued.item.clone();
        adult.rating = Some("xxx".into());
        offline::open(dir.path())
            .unwrap()
            .execute(
                "UPDATE offline_items SET item_json=?1 WHERE post_id='42'",
                params![serde_json::to_string(&adult).unwrap()],
            )
            .unwrap();
        assert!(enforce_item_policy(&adult, "unrestricted", true).is_err());
        let page = offline::local_page(
            dir.path(),
            &source.source_id,
            &hub::CatalogQuery::default().normalized().unwrap(),
            &source,
            true,
        )
        .unwrap();
        assert!(page.items.is_empty());
        assert!(
            delete_source(State(state.clone()), AxumPath(source.source_id.clone()))
                .await
                .is_err()
        );
        remove_offline_copies(
            State(state.clone()),
            AxumPath(source.source_id.clone()),
            Json(RemoveOfflineCopies {
                confirmed_remove: true,
            }),
        )
        .await
        .unwrap();
        assert!(!offline::cache_root(dir.path(), &source.source_id).exists());
        assert!(offline::personal_item(dir.path(), &source.source_id, "42")
            .unwrap()
            .is_none());
        assert!(delete_source(State(state), AxumPath(source.source_id))
            .await
            .is_ok());
    }

    #[test]
    fn directory_counts_are_media_policy_scoped_and_expired_claims_are_unknown() {
        let (dir, _, source, _) = setup(128);
        offline::remember_creator_directory(dir.path(),&source.source_id,&json!({"id":"creator-1","name":"Creator","post_count":5,"availability":{"hosted_posts":5,"peer_status":"unknown"}}),Some("image"),false).unwrap();
        offline::remember_creator_directory(dir.path(),&source.source_id,&json!({"id":"creator-1","name":"Creator","post_count":2,"availability":{"state":"offline","hosted_posts":0,"peer_status":"recently_reported"}}),Some("video"),false).unwrap();
        assert_eq!(
            offline::creator_rows(dir.path(), &source.source_id, Some("image"), false).unwrap()[0]
                ["item_count"],
            5
        );
        assert_eq!(
            offline::creator_rows(dir.path(), &source.source_id, Some("video"), false).unwrap()[0]
                ["availability"],
            "offline"
        );
        assert_eq!(
            offline::creator_rows(dir.path(), &source.source_id, Some("image"), true).unwrap()[0]
                ["item_count"],
            0
        );
        offline::open(dir.path()).unwrap().execute("UPDATE offline_creators SET directory_json=json_set(directory_json,'$.snapshots.\"image:all\".directory_verified_at',0)",[]).unwrap();
        assert_eq!(
            offline::creator_rows(dir.path(), &source.source_id, Some("image"), false).unwrap()[0]
                ["availability"],
            "unknown"
        );
    }

    #[test]
    fn available_discovery_includes_recent_peer_claim_without_asserting_reachability() {
        let (dir, _, source, _) = setup(64);
        offline::remember_creator_directory(dir.path(),&source.source_id,&json!({"id":"creator-1","name":"Creator","post_count":2,"availability":{"state":"unknown","hosted_posts":0,"peer_reported_posts":1,"reachability":"unknown","peer_status":"recently_reported"}}),Some("image"),false).unwrap();
        let rows =
            offline::creator_rows(dir.path(), &source.source_id, Some("image"), false).unwrap();
        assert!(offline::creator_potentially_available(&rows[0]));
        assert_eq!(rows[0]["availability"], "unknown");
        assert_eq!(rows[0]["central_readiness"], "unavailable");
        assert_eq!(rows[0]["peer_availability"], "recently_reported");
        conn_expire_creator_snapshot(dir.path(), "image:all");
        let rows =
            offline::creator_rows(dir.path(), &source.source_id, Some("image"), false).unwrap();
        assert!(!offline::creator_potentially_available(&rows[0]));
        assert_eq!(rows[0]["peer_availability"], "unknown");
        assert!(rows[0]["provider_availability"].is_null());
    }
    fn conn_expire_creator_snapshot(data_dir: &Path, key: &str) {
        offline::open(data_dir)
            .unwrap()
            .execute(
                "UPDATE offline_creators SET directory_json=json_set(directory_json,?1,0)",
                params![format!("$.snapshots.\"{key}\".directory_verified_at")],
            )
            .unwrap();
    }

    #[test]
    fn revision_promotion_waits_for_all_variants_and_preserves_personal_policy() {
        let (dir, _, source, mut old) = setup(128);
        old.item.rating = Some("xxx".into());
        retain(dir.path(), &source, &old, b"adult original");
        let mut new = old;
        new.item.rating = Some("pg".into());
        new.revision = "revision-2".into();
        new.item.remote_revision = Some(new.revision.clone());
        let mut thumb = new.item.media[0].clone();
        thumb.kind = "thumbnail".into();
        new.item.media.push(thumb);
        let root = offline::cache_root(dir.path(), &source.source_id);
        let thumb_path = root.join("new-thumbnail");
        std::fs::write(&thumb_path, b"safe thumb").unwrap();
        offline::reserve(
            dir.path(),
            &source.source_id,
            "42",
            &new.revision,
            "thumbnail",
            10,
        )
        .unwrap();
        offline::finish(
            dir.path(),
            &source.source_id,
            &new,
            "thumbnail",
            &format!("{:x}", Sha256::digest(b"safe thumb")),
            10,
            &thumb_path,
            false,
            Some("image/jpeg"),
        )
        .unwrap();
        assert_eq!(
            offline::personal_item(dir.path(), &source.source_id, "42")
                .unwrap()
                .unwrap()
                .rating
                .as_deref(),
            Some("xxx")
        );
        assert_eq!(
            offline::personal_variant(dir.path(), &source.source_id, "42", "original")
                .unwrap()
                .unwrap()
                .revision,
            "revision-1"
        );
        let query = hub::CatalogQuery::default().normalized().unwrap();
        assert!(
            offline::local_page(dir.path(), &source.source_id, &query, &source, true)
                .unwrap()
                .items
                .is_empty()
        );
        retain(dir.path(), &source, &new, b"safe original");
        assert_eq!(
            offline::personal_item(dir.path(), &source.source_id, "42")
                .unwrap()
                .unwrap()
                .rating
                .as_deref(),
            Some("pg")
        );
        let page =
            offline::local_page(dir.path(), &source.source_id, &query, &source, true).unwrap();
        assert_eq!(page.items.len(), 1);
        assert_eq!(page.items[0].media.len(), 2);
    }

    #[test]
    fn learned_hash_promotes_matching_copy_and_changed_hash_queues_atomic_replacement() {
        let (dir, _, source, queued) = setup(128);
        retain(dir.path(), &source, &queued, b"original copy");
        let mut authoritative = queued.item.clone();
        authoritative.media[0].sha256 = Some(format!("{:x}", Sha256::digest(b"original copy")));
        authoritative.media[0].byte_size = Some(13);
        assert!(!offline::enqueue(
            dir.path(),
            &source.source_id,
            &authoritative,
            &queued.metadata
        )
        .unwrap());
        assert!(
            offline::personal_variant(dir.path(), &source.source_id, "42", "original")
                .unwrap()
                .unwrap()
                .authoritative_digest
        );
        authoritative.media[0].sha256 = Some("f".repeat(64));
        let mut corrected = authoritative.clone();
        corrected.rating = Some("xxx".into());
        assert!(offline::refresh_metadata(
            dir.path(),
            &source.source_id,
            &corrected,
            &queued.metadata
        )
        .is_err());
        assert_eq!(
            offline::personal_item(dir.path(), &source.source_id, "42")
                .unwrap()
                .unwrap()
                .rating
                .as_deref(),
            Some("pg")
        );
        assert!(offline::enqueue(
            dir.path(),
            &source.source_id,
            &authoritative,
            &queued.metadata
        )
        .unwrap());
        assert_eq!(
            offline::pending_count(dir.path(), &source.source_id).unwrap(),
            1
        );
        let retained = offline::personal_variant(dir.path(), &source.source_id, "42", "original")
            .unwrap()
            .unwrap();
        assert!(retained.revision.starts_with("revision-1-retained-"));
        offline::verify_file(&retained).unwrap();
        assert!(!offline::is_ready(
            dir.path(),
            &source.source_id,
            "42",
            "revision-1",
            "original"
        )
        .unwrap());
    }

    #[test]
    fn offline_sql_filters_and_facets_precede_bounded_pagination() {
        let (dir, _, source, queued) = setup(4096);
        retain(dir.path(), &source, &queued, b"retained");
        let conn = offline::open(dir.path()).unwrap();
        for id in 43..193 {
            let mut item = queued.item.clone();
            item.remote_post_id = id.to_string();
            item.rating = Some(if id % 2 == 0 { "pg" } else { "xxx" }.into());
            item.tags = vec![if id % 3 == 0 { "sunrise" } else { "night" }.into()];
            conn.execute("INSERT INTO offline_items SELECT source_id,?1,creator_id,media_type,revision,?2,metadata_json,withdrawn,last_verified_at FROM offline_items WHERE post_id='42'",params![item.remote_post_id,serde_json::to_string(&item).unwrap()]).unwrap();
            conn.execute("INSERT INTO offline_variants SELECT source_id,?1,revision,kind,sha256,authoritative_digest,bytes,reserved_bytes,relative_path,state,mime_type FROM offline_variants WHERE post_id='42'",params![item.remote_post_id]).unwrap();
        }
        let query = hub::CatalogQuery {
            q: Some("sunrise".into()),
            creator_id: Some("creator-1".into()),
            page: Some(2),
            per_page: Some(7),
            min_width: Some(100),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        let page =
            offline::local_page(dir.path(), &source.source_id, &query, &source, true).unwrap();
        assert_eq!(page.total, Some(26));
        assert_eq!(page.items.len(), 7);
        assert!(page
            .items
            .iter()
            .all(|item| item.rating.as_deref() == Some("pg") && item.tags == vec!["sunrise"]));
        let facets = offline::local_tags(
            dir.path(),
            &source.source_id,
            &query,
            &source,
            true,
            "sun",
            20,
        )
        .unwrap();
        assert_eq!(facets["tags"][0]["count"], 26);
        assert_eq!(facets["available_ratings"], json!(["pg"]));
    }

    #[tokio::test]
    async fn real_http_stream_commits_verified_bytes_and_rejects_budget_and_digest_mismatch() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route(
                    "/media",
                    get(|| async {
                        Body::from_stream(futures_util::stream::iter(vec![
                            Ok::<_, std::io::Error>(b"hello ".to_vec()),
                            Ok(b"stream".to_vec()),
                        ]))
                    }),
                ),
            )
            .await
            .unwrap()
        });
        let (dir, state, source, mut queued) = setup(12);
        let digest = format!("{:x}", Sha256::digest(b"hello stream"));
        queued.item.media[0].sha256 = Some(digest.clone());
        let root = offline::cache_root(dir.path(), &source.source_id);
        std::fs::create_dir_all(&root).unwrap();
        offline::reserve(
            dir.path(),
            &source.source_id,
            "42",
            "revision-1",
            "original",
            0,
        )
        .unwrap();
        let response = MediaResponse::from_http(
            reqwest::get(format!("http://{address}/media"))
                .await
                .unwrap(),
        );
        assert!(write_offline_response(
            &state,
            &source,
            &queued,
            &queued.item.media[0],
            response,
            None,
            &root.join("staging.part"),
            &AtomicBool::new(false)
        )
        .await
        .unwrap());
        let record = offline::personal_variant(dir.path(), &source.source_id, "42", "original")
            .unwrap()
            .unwrap();
        assert_eq!(record.sha256, digest);
        offline::verify_file(&record).unwrap();
        assert_eq!(
            offline::settings(dir.path(), &source.source_id).unwrap()["downloaded_bytes"],
            12
        );
        let mut second = queued;
        second.item.remote_post_id = "43".into();
        second.item.media[0].sha256 = Some("0".repeat(64));
        offline::reserve(
            dir.path(),
            &source.source_id,
            "43",
            "revision-1",
            "original",
            0,
        )
        .unwrap();
        let response = MediaResponse::from_http(
            reqwest::get(format!("http://{address}/media"))
                .await
                .unwrap(),
        );
        let error = write_offline_response(
            &state,
            &source,
            &second,
            &second.item.media[0],
            response,
            None,
            &root.join("cap.part"),
            &AtomicBool::new(false),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("storage limit"));
        assert!(offline::personal_item(dir.path(), &source.source_id, "43")
            .unwrap()
            .is_none());
        offline::release(
            dir.path(),
            &source.source_id,
            "43",
            "revision-1",
            "original",
        )
        .unwrap();
        offline::patch_settings(
            dir.path(),
            &source.source_id,
            &offline::SettingsPatch {
                storage_limit_bytes: Some(64),
                ..Default::default()
            },
        )
        .unwrap();
        offline::reserve(
            dir.path(),
            &source.source_id,
            "43",
            "revision-1",
            "original",
            12,
        )
        .unwrap();
        let response = MediaResponse::from_http(
            reqwest::get(format!("http://{address}/media"))
                .await
                .unwrap(),
        );
        let error = write_offline_response(
            &state,
            &source,
            &second,
            &second.item.media[0],
            response,
            Some(12),
            &root.join("digest.part"),
            &AtomicBool::new(false),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("digest mismatch"));
        assert!(offline::personal_item(dir.path(), &source.source_id, "43")
            .unwrap()
            .is_none());
        server.abort();
    }
}

#[cfg(test)]
mod descriptor_refresh_tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, RemoteSource, HubConfig) {
        let directory = tempfile::tempdir().unwrap();
        initialize_offline_database(directory.path()).unwrap();
        let source = RemoteSource {
            source_id: "old-hub".into(),
            provider_family: "donutbooru".into(),
            display_name: "Existing hub".into(),
            normalized_base_url: "https://hub.example".into(),
            policy_profile: "unrestricted".into(),
            enabled: true,
            allow_local_network: false,
            capabilities: crate::online::models::SourceCapabilities::read_only(),
            last_probe_at: None,
        };
        let bootstrap = transport::HubBootstrap {
            peer_id: iroh::SecretKey::from_bytes(&[42; 32]).public().to_string(),
            relay_url: "https://relay.example".into(),
        };
        let descriptor = json!({"protocol":"booru-node-client","version":"1.1","instance":{"id":"00000000-0000-0000-0000-000000000001","name":"Existing hub","origin":"https://hub.example","realm":"main"},"access":{"catalog":"public","registration":"closed","submission":"members","moderation":"required","retention":"hosted"},"authentication":{"methods":[],"pkce_methods":[],"authorization_endpoint":"/api/client/v1/authorize","token_endpoint":"/api/client/v1/token"},"capabilities":{"catalog":true,"publish":false,"idempotency":true,"revisions":true,"tombstones":true,"byte_ranges":true,"native_authorization":false},"transports":{"iroh":{"peer_id":bootstrap.peer_id,"relay_url":bootstrap.relay_url}}});
        let config = HubConfig {
            instance_id: "00000000-0000-0000-0000-000000000001".into(),
            bootstrap,
            transport_mode: transport::TransportMode::Relayed,
            descriptor,
        };
        let conn = open_db(directory.path()).unwrap();
        conn.execute("INSERT INTO remote_sources(source_id,provider_family,display_name,normalized_base_url,capabilities_json,created_at) VALUES(?1,?2,?3,?4,?5,'now')",params![source.source_id,source.provider_family,source.display_name,source.normalized_base_url,serde_json::to_string(&source.capabilities).unwrap()]).unwrap();
        conn.execute(
            "INSERT INTO remote_hub_transport VALUES(?1,?2,?3,?4,'relayed',?5)",
            params![
                source.source_id,
                config.instance_id,
                config.bootstrap.peer_id,
                config.bootstrap.relay_url,
                config.descriptor.to_string()
            ],
        )
        .unwrap();
        (directory, source, config)
    }
    fn upgraded(config: &HubConfig) -> Value {
        let mut descriptor = config.descriptor.clone();
        descriptor["peer_read"] = json!({"version":1,"signing_key":iroh::SecretKey::from_bytes(&[43;32]).public().to_string(),"requires_live_grant_check":true,"grant_ttl_seconds":60,"provider_ttl_seconds":120});
        descriptor["capabilities"]["publish"] = json!(true);
        descriptor
    }

    // AC: @creator-community-replication ac-5
    #[test]
    fn existing_enrollment_upgrades_cached_descriptor_without_changing_pins() {
        let (directory, source, config) = setup();
        assert!(load_hub(directory.path(), &source.source_id)
            .unwrap()
            .unwrap()
            .descriptor
            .get("peer_read")
            .is_none());
        let refreshed =
            persist_refreshed_hub_descriptor(directory.path(), &source, &config, upgraded(&config))
                .unwrap();
        crate::online::peer::enroll_authority(
            directory.path(),
            &source.source_id,
            &source,
            &refreshed,
        )
        .unwrap();
        let stored = load_hub(directory.path(), &source.source_id)
            .unwrap()
            .unwrap();
        assert!(stored.descriptor["peer_read"].is_object());
        assert_eq!(stored.bootstrap.peer_id, config.bootstrap.peer_id);
        assert_eq!(stored.bootstrap.relay_url, config.bootstrap.relay_url);
        assert_eq!(stored.transport_mode, config.transport_mode);
        assert!(
            load_source(directory.path(), &source.source_id)
                .unwrap()
                .capabilities
                .upload
        );
    }

    #[test]
    fn descriptor_upgrade_rejects_changed_origin_identity_bootstrap_or_inflight_mode() {
        let (directory, source, config) = setup();
        for (pointer, value) in [
            (
                "/instance/id",
                json!("00000000-0000-0000-0000-000000000002"),
            ),
            ("/instance/origin", json!("https://other.example")),
            ("/instance/realm", json!("adult")),
            (
                "/transports/iroh/peer_id",
                json!(iroh::SecretKey::from_bytes(&[44; 32]).public().to_string()),
            ),
            (
                "/transports/iroh/relay_url",
                json!("https://other-relay.example"),
            ),
            ("/transports/iroh", Value::Null),
        ] {
            let mut descriptor = upgraded(&config);
            *descriptor.pointer_mut(pointer).unwrap() = value;
            assert!(
                persist_refreshed_hub_descriptor(directory.path(), &source, &config, descriptor)
                    .is_err(),
                "{pointer}"
            );
            assert_eq!(
                load_hub(directory.path(), &source.source_id)
                    .unwrap()
                    .unwrap()
                    .descriptor,
                config.descriptor
            );
        }
        open_db(directory.path())
            .unwrap()
            .execute(
                "UPDATE remote_hub_transport SET transport_mode='standard' WHERE source_id=?1",
                params![source.source_id],
            )
            .unwrap();
        assert!(persist_refreshed_hub_descriptor(
            directory.path(),
            &source,
            &config,
            upgraded(&config)
        )
        .is_err());
        let stored = load_hub(directory.path(), &source.source_id)
            .unwrap()
            .unwrap();
        assert_eq!(stored.transport_mode, transport::TransportMode::Standard);
        assert_eq!(stored.descriptor, config.descriptor);
    }
}

#[cfg(test)]
mod peer_view_tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, AppState, RemoteSource, RemoteItem) {
        let directory = tempfile::tempdir().unwrap();
        let state = AppState::new(directory.path(), 0).unwrap();
        initialize_offline_database(directory.path()).unwrap();
        let source = RemoteSource {
            source_id: "viewer".into(),
            provider_family: "donutbooru".into(),
            display_name: "Community".into(),
            normalized_base_url: "https://hub.example".into(),
            policy_profile: "unrestricted".into(),
            enabled: true,
            allow_local_network: false,
            capabilities: crate::online::models::SourceCapabilities::read_only(),
            last_probe_at: None,
        };
        open_db(directory.path()).unwrap().execute("INSERT INTO remote_sources(source_id,provider_family,display_name,normalized_base_url,capabilities_json,created_at) VALUES(?1,?2,?3,?4,?5,'now')",params![source.source_id,source.provider_family,source.display_name,source.normalized_base_url,serde_json::to_string(&source.capabilities).unwrap()]).unwrap();
        let digest = format!("{:x}", Sha256::digest(vec![79u8; 180_000]));
        let item = hub::catalog_item("viewer", "https://hub.example", &json!({"remote_post_id":"12","canonical_url":"https://hub.example/image/12","remote_revision":"revision-1","tags":[],"rating":"pg","variants":[{"kind":"original","asset_url":"https://hub.example/api/client/v1/media/12/original","mime_type":"video/mp4","sha256":digest,"byte_size":180_000}]})).unwrap();
        (directory, state, source, item)
    }

    // AC: @creator-community-replication ac-5
    #[tokio::test]
    async fn normal_peer_view_after_central_loss_verifies_then_ranges_and_cleans_without_subscription(
    ) {
        let (directory, state, source, item) = setup();
        for ranged in [false, true] {
            let mut headers = HeaderMap::new();
            if ranged {
                headers.insert(header::RANGE, HeaderValue::from_static("bytes=100-199"));
            }
            let staged = Arc::new(Mutex::new(None::<std::path::PathBuf>));
            let observed = staged.clone();
            let response = asset_delivery(
                Err(AppError::Upstream(
                    StatusCode::NOT_FOUND,
                    "Central rendition is unavailable".into(),
                )),
                || {
                    peer_view_asset_with(
                        &state,
                        &source,
                        &item,
                        &item.media[0],
                        &headers,
                        |destination| async move {
                            *observed.lock().unwrap() = Some(destination.clone());
                            // A fake broker supplies the trusted grant/provider. Actual
                            // authenticated Iroh serving and verified receiving run here.
                            crate::online::peer::test_view_download(&destination).await;
                            Ok(true)
                        },
                    )
                },
            )
            .await
            .unwrap();
            assert_eq!(
                response.status(),
                if ranged {
                    StatusCode::PARTIAL_CONTENT
                } else {
                    StatusCode::OK
                }
            );
            assert_eq!(response.headers()[header::CONTENT_TYPE], "video/mp4");
            assert_eq!(
                response.headers()[header::CACHE_CONTROL],
                "private, no-store"
            );
            if ranged {
                assert_eq!(
                    response.headers()[header::CONTENT_RANGE],
                    "bytes 100-199/180000"
                );
            }
            let path = staged.lock().unwrap().clone().unwrap();
            assert!(path.exists());
            let bytes = axum::body::to_bytes(response.into_body(), 200_000)
                .await
                .unwrap();
            assert_eq!(
                bytes.as_ref(),
                vec![79u8; if ranged { 100 } else { 180_000 }]
            );
            assert!(
                !path.parent().unwrap().exists(),
                "the temporary directory disappears with its response body"
            );
        }
        let conn = open_db(directory.path()).unwrap();
        for table in [
            "offline_sources",
            "offline_items",
            "offline_variants",
            "offline_selections",
        ] {
            assert_eq!(
                conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r
                    .get::<_, i64>(0))
                    .unwrap(),
                0,
                "viewing must not subscribe or retain personal media: {table}"
            );
        }
    }

    #[tokio::test]
    async fn peer_view_body_drop_and_bad_digest_remove_ephemeral_files() {
        let (_directory, state, source, item) = setup();
        let staged = Arc::new(Mutex::new(None::<std::path::PathBuf>));
        let observed = staged.clone();
        let response = peer_view_asset_with(
            &state,
            &source,
            &item,
            &item.media[0],
            &HeaderMap::new(),
            |destination| async move {
                *observed.lock().unwrap() = Some(destination.clone());
                tokio::fs::write(&destination, vec![79u8; 180_000]).await?;
                Ok(true)
            },
        )
        .await
        .unwrap()
        .unwrap();
        let path = staged.lock().unwrap().clone().unwrap();
        assert!(path.exists());
        drop(response);
        assert!(!path.parent().unwrap().exists());
        let observed = staged.clone();
        let error = peer_view_asset_with(
            &state,
            &source,
            &item,
            &item.media[0],
            &HeaderMap::new(),
            |destination| async move {
                *observed.lock().unwrap() = Some(destination.clone());
                tokio::fs::write(&destination, vec![80u8; 180_000]).await?;
                Ok(true)
            },
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("digest"));
        assert!(!staged
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .parent()
            .unwrap()
            .exists());
    }

    #[tokio::test]
    async fn peer_view_never_bypasses_hub_denial_or_unknown_digest_or_size_limit() {
        let (_directory, state, source, mut item) = setup();
        let called = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
            let called = called.clone();
            let result = asset_delivery(
                Err(AppError::Upstream(status, "Denied".into())),
                || async move {
                    called.fetch_add(1, Ordering::SeqCst);
                    Ok(None)
                },
            )
            .await;
            assert!(matches!(result,Err(AppError::Upstream(code,_)) if code==status));
        }
        assert_eq!(called.load(Ordering::SeqCst), 0);
        item.media[0].sha256 = None;
        assert!(peer_view_asset_with(
            &state,
            &source,
            &item,
            &item.media[0],
            &HeaderMap::new(),
            |_| async {
                panic!("unknown digests must remain central-only");
                #[allow(unreachable_code)]
                Ok(false)
            }
        )
        .await
        .unwrap()
        .is_none());
        item.media[0].sha256 = Some("ab".repeat(32));
        item.media[0].byte_size = Some(PEER_VIEW_LIMIT + 1);
        assert!(peer_view_asset_with(
            &state,
            &source,
            &item,
            &item.media[0],
            &HeaderMap::new(),
            |_| async {
                panic!("oversized views must not reserve/download");
                #[allow(unreachable_code)]
                Ok(false)
            }
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("256 MiB"));
    }
}
