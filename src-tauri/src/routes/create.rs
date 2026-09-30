//! Donut Create's scoped studio transport and qualified library import.
use std::{net::SocketAddr, path::PathBuf, time::Duration};

use axum::{
    body::Body,
    extract::{
        ws::{rejection::WebSocketUpgradeRejection, Message, WebSocket, WebSocketUpgrade},
        ConnectInfo, FromRequestParts, Path, Query, Request, State,
    },
    http::{header, request::Parts, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{any, get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Message as UpstreamMessage},
};

use crate::{
    db::library::LibraryContext,
    server::{error::AppError, middleware::auth::AuthUser, state::AppState},
    services::{importer, metadata},
};

const MAX_OUTPUT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_STUDIO_UPLOAD_BYTES: usize = 32 * 1024 * 1024;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/studio/{session_id}/", any(studio_root))
        .route("/studio/{session_id}/{*rest}", any(studio))
        .route("/output/{session_id}/{output_id}", get(output))
        .route("/import", post(import_output))
        .route("/output-directory", post(create_output_directory))
        .route("/workflow", get(image_workflow))
}

/// Management/import routes require an actual write credential or the local desktop.
/// The studio URL carries a separate, limited capability checked by the sidecar.
pub(crate) async fn require_write(state: &AppState, parts: &mut Parts) -> Result<(), AppError> {
    if parts.headers.contains_key(header::AUTHORIZATION) {
        let user = AuthUser::from_request_parts(parts, state)
            .await
            .map_err(|_| AppError::Unauthorized("Invalid or revoked creation credential".into()))?;
        if !user.can_write {
            return Err(AppError::Forbidden("Creation requires write access".into()));
        }
        return Ok(());
    }
    if parts
        .extensions
        .get::<ConnectInfo<SocketAddr>>()
        .is_some_and(|info| info.0.ip().is_loopback())
    {
        Ok(())
    } else {
        Err(AppError::Unauthorized(
            "Creation requires write access".into(),
        ))
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn addon_url(state: &AppState) -> Result<String, AppError> {
    state
        .addon_manager()
        .addon_url("donut-create")
        .ok_or_else(|| AppError::ServiceUnavailable("Donut Create is not running".into()))
}

fn studio_prefix(session: &str, headers: &HeaderMap) -> String {
    let local = format!("/api/create/studio/{session}/");
    let remote = format!("/remote{local}");
    if headers
        .get("x-dmc-studio-prefix")
        .and_then(|h| h.to_str().ok())
        == Some(remote.as_str())
    {
        remote
    } else {
        local
    }
}

pub(crate) fn remote_studio_prefix(path: &str) -> Option<String> {
    let session = path
        .strip_prefix("/api/create/studio/")?
        .split('/')
        .next()?;
    valid_id(session).then(|| format!("/remote/api/create/studio/{session}/"))
}

async fn studio_root(
    State(state): State<AppState>,
    Path(session): Path<String>,
    req: Request,
) -> Result<Response, AppError> {
    studio_http(state, session, String::new(), req).await
}

async fn studio(
    State(state): State<AppState>,
    Path((session, rest)): Path<(String, String)>,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
    req: Request,
) -> Result<Response, AppError> {
    if !valid_id(&session) {
        return Err(AppError::BadRequest("Invalid studio capability".into()));
    }
    if rest == "ws" {
        let upgrade =
            upgrade.map_err(|_| AppError::BadRequest("Expected a WebSocket upgrade".into()))?;
        let prefix = studio_prefix(&session, req.headers());
        let url = format!(
            "{}/studio/{session}/ws{}",
            addon_url(&state)?,
            req.uri()
                .query()
                .map(|q| format!("?{q}"))
                .unwrap_or_default()
        );
        return Ok(proxy_websocket(upgrade, &[url], Some(&prefix), None).await);
    }
    studio_http(state, session, rest, req).await
}

async fn studio_http(
    state: AppState,
    session: String,
    rest: String,
    req: Request,
) -> Result<Response, AppError> {
    if !valid_id(&session) {
        return Err(AppError::BadRequest("Invalid studio capability".into()));
    }
    let url = format!(
        "{}/studio/{session}/{rest}{}",
        addon_url(&state)?,
        req.uri()
            .query()
            .map(|q| format!("?{q}"))
            .unwrap_or_default()
    );
    let mut outgoing = state
        .http_client()
        .request(req.method().clone(), url)
        .header(
            "x-dmc-studio-prefix",
            studio_prefix(&session, req.headers()),
        );
    for name in [header::CONTENT_TYPE, header::ACCEPT, header::RANGE] {
        if let Some(value) = req.headers().get(&name) {
            outgoing = outgoing.header(name, value);
        }
    }
    // No desktop Authorization/Cookie headers are ever passed to ComfyUI.
    let body = axum::body::to_bytes(req.into_body(), MAX_STUDIO_UPLOAD_BYTES)
        .await
        .map_err(|_| AppError::BadRequest("Studio upload is too large".into()))?;
    streamed_response(
        outgoing
            .body(body)
            .send()
            .await
            .map_err(|_| AppError::ServiceUnavailable("Could not reach Donut Create".into()))?,
    )
}

async fn output(
    State(state): State<AppState>,
    Path((session, output_id)): Path<(String, String)>,
    req: Request,
) -> Result<Response, AppError> {
    if !valid_id(&session) || !valid_id(&output_id) {
        return Err(AppError::BadRequest("Invalid generation output".into()));
    }
    let url = format!(
        "{}/create/sessions/{session}/outputs/{output_id}",
        addon_url(&state)?
    );
    let mut request = state.http_client().request(req.method().clone(), url);
    if let Some(range) = req.headers().get(header::RANGE) {
        request = request.header(header::RANGE, range);
    }
    streamed_response(
        request
            .send()
            .await
            .map_err(|_| AppError::ServiceUnavailable("Could not load generation output".into()))?,
    )
}

fn streamed_response(response: reqwest::Response) -> Result<Response, AppError> {
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut headers = HeaderMap::new();
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_DISPOSITION,
        header::CONTENT_LENGTH,
        header::CONTENT_RANGE,
        header::ACCEPT_RANGES,
    ] {
        if let Some(value) = response.headers().get(&name) {
            headers.insert(name, value.clone());
        }
    }
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    Ok((status, headers, Body::from_stream(response.bytes_stream())).into_response())
}

/// Also used by the native client's /remote proxy. Connect before acknowledging
/// the browser upgrade so an invalid/expired capability keeps its HTTP error.
pub(crate) async fn proxy_websocket(
    upgrade: WebSocketUpgrade,
    urls: &[String],
    prefix: Option<&str>,
    token: Option<&str>,
) -> Response {
    for (index, url) in urls.iter().enumerate() {
        let url = url
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1);
        let Ok(mut request) = url.into_client_request() else {
            return StatusCode::BAD_GATEWAY.into_response();
        };
        if let Some(prefix) = prefix {
            let Ok(value) = prefix.parse() else {
                return StatusCode::BAD_REQUEST.into_response();
            };
            request.headers_mut().insert("x-dmc-studio-prefix", value);
        }
        if let Some(token) = token {
            let Ok(value) = format!("Bearer {token}").parse() else {
                return StatusCode::BAD_GATEWAY.into_response();
            };
            request.headers_mut().insert("authorization", value);
        }
        match tokio::time::timeout(Duration::from_secs(15), connect_async(request)).await {
            Ok(Ok((upstream, _))) => {
                return upgrade
                    .max_message_size(16 * 1024 * 1024)
                    .on_upgrade(move |socket| relay(socket, upstream))
                    .into_response()
            }
            Ok(Err(tokio_tungstenite::tungstenite::Error::Http(response))) => {
                return StatusCode::from_u16(response.status().as_u16())
                    .unwrap_or(StatusCode::BAD_GATEWAY)
                    .into_response()
            }
            _ if index + 1 < urls.len() => continue,
            _ => {
                return (StatusCode::BAD_GATEWAY, "Creation connection unavailable").into_response()
            }
        }
    }
    StatusCode::BAD_GATEWAY.into_response()
}

async fn relay(
    mut browser: WebSocket,
    mut upstream: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) {
    loop {
        tokio::select! {
            message = browser.next() => {
                let Some(Ok(message)) = message else { break; };
                let message = match message {
                    Message::Text(text) => UpstreamMessage::Text(text.as_str().into()),
                    Message::Binary(bytes) => UpstreamMessage::Binary(bytes),
                    Message::Ping(bytes) => UpstreamMessage::Ping(bytes),
                    Message::Pong(bytes) => UpstreamMessage::Pong(bytes),
                    Message::Close(_) => break,
                };
                if upstream.send(message).await.is_err() { break; }
            }
            message = upstream.next() => {
                let Some(Ok(message)) = message else { break; };
                let message = match message {
                    UpstreamMessage::Text(text) => Message::Text(text.as_str().into()),
                    UpstreamMessage::Binary(bytes) => Message::Binary(bytes),
                    UpstreamMessage::Ping(bytes) => Message::Ping(bytes),
                    UpstreamMessage::Pong(bytes) => Message::Pong(bytes),
                    UpstreamMessage::Close(_) => break,
                    UpstreamMessage::Frame(_) => continue,
                };
                if browser.send(message).await.is_err() { break; }
            }
        }
    }
    let _ = upstream.close(None).await;
    let _ = browser.close().await;
}

#[derive(Deserialize)]
struct ImportRequest {
    session_id: String,
    output_id: String,
    library_id: String,
    directory_id: i64,
    #[serde(skip)]
    execution_metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ImportedOutput {
    library_id: String,
    directory_id: i64,
    image_id: i64,
    sha256: String,
    filename: String,
    status: &'static str,
}

#[derive(Deserialize)]
struct OutputDirectoryRequest {
    library_id: Option<String>,
}

async fn create_output_directory(
    State(state): State<AppState>,
    req: Request,
) -> Result<Json<serde_json::Value>, AppError> {
    let (mut parts, body) = req.into_parts();
    require_write(&state, &mut parts).await?;
    let bytes = axum::body::to_bytes(body, 4096)
        .await
        .map_err(|_| AppError::BadRequest("Invalid directory request".into()))?;
    let request: OutputDirectoryRequest = serde_json::from_slice(&bytes)
        .map_err(|_| AppError::BadRequest("Invalid directory request".into()))?;
    let lib = state.resolve_library(request.library_id.as_deref())?;
    let base = std::fs::canonicalize(&lib.data_dir)?;
    let root = base.join("Created Images");
    match std::fs::create_dir(&root) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    if !std::fs::symlink_metadata(&root)?.is_dir() {
        return Err(AppError::BadRequest(
            "Created Images must be a real directory".into(),
        ));
    }
    let path = root.to_string_lossy().into_owned();
    let existing = || -> Result<Option<i64>, AppError> {
        let row: Option<(i64, bool, bool, bool, bool)> = lib.main_pool.get()?.query_row(
            "SELECT id,enabled,show_images,show_videos,show_music FROM watch_directories WHERE path=?1",
            params![path], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?)),
        ).optional()?;
        match row {
            Some((id, true, true, false, false)) => Ok(Some(id)),
            Some(_) => Err(AppError::BadRequest("Created Images already exists with different settings. Enable Images only in Settings > Directories.".into())),
            None => Ok(None),
        }
    };
    let id = if let Some(id) = existing()? {
        // Repair an interrupted earlier registration before claiming success.
        // get_pool initializes the directory schema and scans are deduplicated.
        lib.directory_db.get_pool(id)?;
        let recursive: bool = lib.main_pool.get()?.query_row(
            "SELECT recursive FROM watch_directories WHERE id=?1",
            params![id],
            |row| row.get(0),
        )?;
        crate::services::task_queue::enqueue_task(
            &state,
            crate::services::task_queue::TASK_SCAN_DIRECTORY,
            &serde_json::json!({"directory_id":id,"directory_path":path,"library_id":lib.uuid,"recursive":recursive,"fast_import":true}),
            crate::services::task_queue::PRIORITY_INDEX,
            None,
        )?;
        if let Some(watcher) = state.directory_watcher() {
            watcher.add_directory_for_library(id, &path, recursive, lib.clone());
        }
        state.allow_asset_dir(&path);
        id
    } else {
        let added = super::directories::add_directory(
            State(state.clone()),
            Json(super::directories::DirectoryCreate {
                path: path.clone(),
                name: Some("Created Images".into()),
                recursive: true,
                auto_tag: false,
                auto_age_detect: false,
                show_images: true,
                show_videos: false,
                show_music: false,
                library_id: Some(lib.uuid.clone()),
            }),
        )
        .await?;
        added.0["id"]
            .as_i64()
            .ok_or_else(|| AppError::Internal("Directory was not registered".into()))?
    };
    Ok(Json(
        serde_json::json!({"library_id":lib.uuid,"directory_id":id,"name":"Created Images","path":path}),
    ))
}

#[derive(Deserialize)]
struct WorkflowQuery {
    library_id: String,
    directory_id: i64,
    image_id: i64,
    file_hash: Option<String>,
    #[serde(default)]
    summary: bool,
}

async fn image_workflow(
    State(state): State<AppState>,
    Query(query): Query<WorkflowQuery>,
    req: Request,
) -> Result<Json<serde_json::Value>, AppError> {
    let (mut parts, _) = req.into_parts();
    require_write(&state, &mut parts).await?;
    let lib = state.resolve_library(Some(&query.library_id))?;
    tokio::task::spawn_blocking(move || {
        let root = destination(&lib, query.directory_id)?;
        let (path, hash) =
            super::images::single::resolve_exact_media(&lib, query.directory_id, query.image_id)?;
        if query
            .file_hash
            .as_ref()
            .is_some_and(|expected| expected != &hash)
        {
            return Err(AppError::BadRequest(
                "The gallery image changed. Open its current version before loading the workflow."
                    .into(),
            ));
        }
        let path = std::fs::canonicalize(path)
            .map_err(|_| AppError::NotFound("Image file is unavailable".into()))?;
        if !path.starts_with(root) || !super::images::single::media_path_matches_hash(&path, &hash)
        {
            return Err(AppError::NotFound(
                "Image file does not match the gallery item".into(),
            ));
        }
        let workflow =
            match metadata::read_generation_sidecar(&path).map_err(AppError::BadRequest)? {
                Some(sidecar) => sidecar.workflow,
                None => metadata::extract_png_text_chunks(&path.to_string_lossy())
                    .ok()
                    .and_then(|chunks| {
                        chunks
                            .get("workflow")
                            .and_then(|value| serde_json::from_str(value).ok())
                    }),
            };
        let workflow = workflow.filter(|value| {
            value
                .get("nodes")
                .and_then(|v| v.as_array())
                .is_some_and(|nodes| !nodes.is_empty())
                && serde_json::to_vec(value).is_ok_and(|bytes| bytes.len() <= 4 * 1024 * 1024)
        });
        Ok(Json(if query.summary {
            serde_json::json!({"available":workflow.is_some()})
        } else {
            serde_json::json!({"available":workflow.is_some(),"workflow":workflow})
        }))
    })
    .await?
}

fn destination(lib: &LibraryContext, directory_id: i64) -> Result<PathBuf, AppError> {
    let (path, enabled, images): (String, bool, bool) = lib
        .main_pool
        .get()?
        .query_row(
            "SELECT path, enabled, show_images FROM watch_directories WHERE id=?1",
            params![directory_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or_else(|| {
            AppError::BadRequest("Choose an image directory in the selected library".into())
        })?;
    if !enabled || !images {
        return Err(AppError::BadRequest(
            "Destination is not an enabled image directory".into(),
        ));
    }
    let root = std::fs::canonicalize(path)
        .map_err(|_| AppError::BadRequest("Destination is unavailable".into()))?;
    if !root.is_dir() {
        return Err(AppError::BadRequest("Destination is unavailable".into()));
    }
    Ok(root)
}

async fn import_output(
    State(state): State<AppState>,
    req: Request,
) -> Result<Json<ImportedOutput>, AppError> {
    let (mut parts, body) = req.into_parts();
    require_write(&state, &mut parts).await?;
    let bytes = axum::body::to_bytes(body, 64 * 1024)
        .await
        .map_err(|_| AppError::BadRequest("Invalid import request".into()))?;
    let mut request: ImportRequest = serde_json::from_slice(&bytes)
        .map_err(|_| AppError::BadRequest("Invalid import request".into()))?;
    if !valid_id(&request.session_id) || !valid_id(&request.output_id) {
        return Err(AppError::BadRequest("Invalid generation output".into()));
    }
    let lib = state.resolve_library(Some(&request.library_id))?;
    let root = destination(&lib, request.directory_id)?;
    let url = format!(
        "{}/create/sessions/{}/outputs/{}",
        addon_url(&state)?,
        request.session_id,
        request.output_id
    );
    let mut metadata_response = state
        .http_client()
        .get(format!("{url}/provenance"))
        .timeout(Duration::from_secs(70))
        .send()
        .await
        .map_err(|_| {
            AppError::ServiceUnavailable("Could not load executed generation metadata".into())
        })?;
    // An already running older add-on may not expose full provenance yet.
    // Its embedded PNG graph remains a valid fallback until the add-on restarts.
    if metadata_response.status() == StatusCode::NOT_FOUND {
        metadata_response = state
            .http_client()
            .get(format!("{url}/metadata"))
            .timeout(Duration::from_secs(70))
            .send()
            .await
            .map_err(|_| {
                AppError::ServiceUnavailable("Could not load executed generation metadata".into())
            })?;
    }
    if !metadata_response.status().is_success() {
        return Err(AppError::BadRequest(
            "Generation output is unavailable or the studio has expired".into(),
        ));
    }
    request.execution_metadata = Some(
        metadata_response
            .json()
            .await
            .map_err(|_| AppError::ServiceUnavailable("Invalid generation metadata".into()))?,
    );
    let response = state
        .http_client()
        .get(url)
        .send()
        .await
        .map_err(|_| AppError::ServiceUnavailable("Could not load generation output".into()))?;
    if !response.status().is_success() {
        return Err(AppError::BadRequest(
            "Generation output is unavailable or the studio has expired".into(),
        ));
    }
    // Staging on the destination filesystem permits atomic no-clobber install.
    let temp = tempfile::Builder::new()
        .prefix(".donut-create-")
        .suffix(".part")
        .tempfile_in(&root)?;
    let mut file = tokio::fs::File::from_std(temp.reopen()?);
    let mut stream = response.bytes_stream();
    let mut hash = Sha256::new();
    let mut size = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|_| AppError::ServiceUnavailable("Output download was interrupted".into()))?;
        size += chunk.len() as u64;
        if size > MAX_OUTPUT_BYTES {
            return Err(AppError::BadRequest(
                "Generation output exceeds the import limit".into(),
            ));
        }
        hash.update(&chunk);
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    let sha256 = format!("{:x}", hash.finalize());
    let result = tokio::task::spawn_blocking(move || {
        save_output(&state, &lib, &request, root, temp, sha256)
    })
    .await??;
    Ok(Json(result))
}

fn file_digest(path: &std::path::Path) -> Result<String, AppError> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn save_output(
    state: &AppState,
    lib: &LibraryContext,
    request: &ImportRequest,
    root: PathBuf,
    temp: tempfile::NamedTempFile,
    sha256: String,
) -> Result<ImportedOutput, AppError> {
    if !state.library_manager().is_mounted(&lib.uuid) {
        return Err(AppError::BadRequest(
            "Selected library was unmounted while saving".into(),
        ));
    }
    if destination(lib, request.directory_id)? != root {
        return Err(AppError::BadRequest(
            "Destination changed while saving".into(),
        ));
    }
    let mut reader = image::ImageReader::open(temp.path())?.with_guessed_format()?;
    if reader.format() != Some(image::ImageFormat::Png) {
        return Err(AppError::BadRequest(
            "Only generated PNG images can be imported".into(),
        ));
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    reader.limits(limits);
    reader.decode().map_err(|_| {
        AppError::BadRequest("Generation output is not a valid supported PNG".into())
    })?;

    let mut provenance = rusqlite::Connection::open(lib.data_dir.join("donut-create.sqlite"))?;
    provenance.busy_timeout(Duration::from_secs(30))?;
    provenance.execute_batch("CREATE TABLE IF NOT EXISTS imports (session_id TEXT NOT NULL, output_id TEXT NOT NULL, directory_id INTEGER NOT NULL, image_id INTEGER NOT NULL, sha256 TEXT NOT NULL, filename TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(session_id,output_id,directory_id));")?;
    let tx = provenance.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let existing: Option<(i64, String, String)> = tx.query_row(
        "SELECT image_id,sha256,filename FROM imports WHERE session_id=?1 AND output_id=?2 AND directory_id=?3",
        params![request.session_id,request.output_id,request.directory_id], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
    ).optional()?;
    if let Some((image_id, digest, filename)) = existing {
        let path = root.join(&filename);
        let pool = lib.directory_db.get_pool(request.directory_id)?;
        let exists: bool = pool.get()?.query_row(
            "SELECT EXISTS(SELECT 1 FROM image_files WHERE image_id=?1 AND original_path=?2)",
            params![image_id, path.to_string_lossy()],
            |row| row.get(0),
        )?;
        if exists && path.is_file() && file_digest(&path)? == digest && digest == sha256 {
            persist_output_metadata(&path, &sha256, request.execution_metadata.as_ref())?;
            return Ok(ImportedOutput {
                library_id: lib.uuid.clone(),
                directory_id: request.directory_id,
                image_id,
                sha256,
                filename,
                status: "existing",
            });
        }
    }

    let mut temp = temp;
    let mut suffix = 0u32;
    let (path, created) = loop {
        let filename = if suffix == 0 {
            format!("donut-{sha256}.png")
        } else {
            format!("donut-{sha256}-{suffix}.png")
        };
        let path = root.join(filename);
        match temp.persist_noclobber(&path) {
            Ok(_) => break (path, true),
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                temp = error.file;
                if std::fs::symlink_metadata(&path)?.file_type().is_file()
                    && file_digest(&path)? == sha256
                {
                    break (path, false);
                }
                suffix += 1;
            }
            Err(error) => return Err(error.error.into()),
        }
    };
    // Write the portable recipe before indexing so ordinary metadata extraction
    // and later reindexing use the same executed settings as this import.
    if let Err(error) = persist_output_metadata(&path, &sha256, request.execution_metadata.as_ref())
    {
        if created {
            let _ = std::fs::remove_file(&path);
        }
        return Err(error);
    }
    let imported = importer::import_image(
        state,
        lib,
        &path.to_string_lossy(),
        request.directory_id,
        false,
    );
    let imported = match imported {
        Ok(result) if result.status != importer::ImportStatus::Error => result,
        result => {
            if created {
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(root.join(".donut-create").join(format!(
                    "{}.json",
                    path.file_name().unwrap().to_string_lossy()
                )));
            }
            return Err(match result {
                Err(error) => error,
                Ok(result) => AppError::BadRequest(
                    result
                        .message
                        .unwrap_or_else(|| "Could not import output".into()),
                ),
            });
        }
    };
    let image_id = imported
        .image_id
        .ok_or_else(|| AppError::Internal("Import returned no image identity".into()))?;
    let mut chunks = metadata::extract_png_text_chunks(&path.to_string_lossy()).unwrap_or_default();
    let pool = lib.directory_db.get_pool(request.directory_id)?;
    let conn = pool.get()?;
    save_generation_metadata(
        &conn,
        image_id,
        chunks.get("prompt"),
        request.execution_metadata.as_ref(),
    )?;
    if let Some(metadata) = &request.execution_metadata {
        chunks.insert("donut_create_execution".into(), metadata.to_string());
    }
    let filename = path.file_name().unwrap().to_string_lossy().into_owned();
    tx.execute("INSERT OR REPLACE INTO imports (session_id,output_id,directory_id,image_id,sha256,filename,metadata) VALUES(?1,?2,?3,?4,?5,?6,?7)", params![request.session_id,request.output_id,request.directory_id,image_id,sha256,filename,serde_json::to_string(&chunks).unwrap_or_default()])?;
    tx.commit()?;
    Ok(ImportedOutput {
        library_id: lib.uuid.clone(),
        directory_id: request.directory_id,
        image_id,
        sha256,
        filename,
        status: if created { "imported" } else { "existing" },
    })
}

fn text_input(
    graph: &serde_json::Value,
    value: &serde_json::Value,
    depth: usize,
) -> Option<String> {
    if depth > 16 {
        return None;
    }
    if let Some(text) = value.as_str() {
        return Some(text.to_owned());
    }
    let id = value.as_array()?.first()?.as_str()?;
    let inputs = graph.get(id)?.get("inputs")?;
    let text = ["text", "Text", "string", "prompt"]
        .iter()
        .find_map(|key| {
            inputs
                .get(key)
                .and_then(|v| text_input(graph, v, depth + 1))
        })?;
    let prefix = inputs
        .get("prefix")
        .and_then(|v| text_input(graph, v, depth + 1));
    let suffix = inputs
        .get("suffix")
        .and_then(|v| text_input(graph, v, depth + 1));
    let separator = inputs
        .get("separator")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    Some(
        prefix
            .into_iter()
            .chain(std::iter::once(text))
            .chain(suffix)
            .collect::<Vec<_>>()
            .join(separator),
    )
}

fn save_generation_metadata(
    conn: &rusqlite::Connection,
    image_id: i64,
    prompt: Option<&String>,
    executed: Option<&serde_json::Value>,
) -> Result<(), AppError> {
    let parsed = generation_metadata(prompt, executed);
    conn.execute("UPDATE images SET prompt=COALESCE(?1,prompt),negative_prompt=COALESCE(?2,negative_prompt),model_name=COALESCE(?3,model_name),sampler=COALESCE(?4,sampler),seed=COALESCE(?5,seed),steps=COALESCE(?6,steps),cfg_scale=COALESCE(?7,cfg_scale) WHERE id=?8", params![parsed.prompt,parsed.negative_prompt,parsed.model_name,parsed.sampler,parsed.seed,parsed.steps,parsed.cfg_scale,image_id])?;
    Ok(())
}

fn persist_output_metadata(
    path: &std::path::Path,
    sha256: &str,
    executed: Option<&serde_json::Value>,
) -> Result<(), AppError> {
    let chunks = metadata::extract_png_text_chunks(&path.to_string_lossy()).unwrap_or_default();
    let graph = executed
        .and_then(|value| value.get("execution_prompt"))
        .filter(|v| v.is_object())
        .cloned()
        .or_else(|| {
            chunks
                .get("prompt")
                .and_then(|value| serde_json::from_str(value).ok())
        });
    let prompt_text = graph.as_ref().map(serde_json::Value::to_string);
    let sidecar = metadata::GenerationSidecar {
        schema: "donut-create-output-v1".into(),
        image_sha256: sha256.into(),
        generation: generation_metadata(prompt_text.as_ref(), executed),
        workflow: executed
            .and_then(|value| value.get("workflow"))
            .filter(|v| v.is_object())
            .cloned()
            .or_else(|| {
                chunks
                    .get("workflow")
                    .and_then(|value| serde_json::from_str(value).ok())
            }),
        execution_prompt: graph,
        execution: executed.and_then(|value| value.get("execution")).cloned(),
        png_text: chunks,
    };
    metadata::write_generation_sidecar(path, &sidecar).map_err(|error| {
        AppError::BadRequest(format!("Could not save generation metadata: {error}"))
    })
}

fn generation_metadata(
    prompt: Option<&String>,
    executed: Option<&serde_json::Value>,
) -> metadata::GenerationMetadata {
    let graph = prompt
        .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .unwrap_or_default();
    let mut parsed = prompt
        .map(|p| metadata::parse_comfyui_metadata(p, &[], &[]).0)
        .unwrap_or_default();
    if let Some(nodes) = graph.as_object() {
        for node in nodes.values() {
            let class = node
                .get("class_type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let Some(inputs) = node.get("inputs") else {
                continue;
            };
            if class == "DonutPromptConditioning" {
                let variant = inputs
                    .get("prompt_sets_json")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                    .and_then(|sets| {
                        sets.as_array().and_then(|sets| {
                            inputs
                                .get("prompt_set_index")
                                .and_then(|v| v.as_u64())
                                .and_then(|index| index.checked_sub(1))
                                .and_then(|index| sets.get(index as usize))
                                .cloned()
                        })
                    });
                let input_text = |key: &str| {
                    variant
                        .as_ref()
                        .and_then(|variant| variant.get(key))
                        .or_else(|| inputs.get(key))
                        .and_then(|v| text_input(&graph, v, 0))
                };
                let face = input_text("face").unwrap_or_default();
                let scene = input_text("scene").unwrap_or_default();
                let separator = inputs
                    .get("separator")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                parsed.prompt = Some(format!("{face}{separator}{scene}"));
                parsed.negative_prompt = input_text("negative");
            }
            if class == "UNETLoader" {
                parsed.model_name = inputs
                    .get("unet_name")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
            }
        }
    }
    if let Some(executed) = executed {
        if executed.get("prompt_source").and_then(|v| v.as_str()) == Some("executed_output") {
            parsed.prompt = executed
                .get("prompt")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
        }
        if executed
            .get("negative_prompt_source")
            .and_then(|v| v.as_str())
            == Some("executed_output")
        {
            parsed.negative_prompt = executed
                .get("negative_prompt")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
        }
        if let Some(seed) = executed.get("seed").and_then(|v| v.as_u64()) {
            parsed.seed = Some(seed.to_string());
        }
        if let Some(steps) = executed
            .get("steps")
            .and_then(|v| v.as_i64())
            .and_then(|v| i32::try_from(v).ok())
        {
            parsed.steps = Some(steps);
        }
        if let Some(cfg) = executed.get("cfg").and_then(|v| v.as_f64()) {
            parsed.cfg_scale = Some(cfg);
        }
        if let Some(sampler) = executed.get("sampler").and_then(|v| v.as_str()) {
            parsed.sampler = Some(sampler.to_owned());
        }
    }
    parsed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture() -> (
        tempfile::TempDir,
        AppState,
        std::sync::Arc<LibraryContext>,
        PathBuf,
    ) {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::new(&temp.path().join("primary"), 0).unwrap();
        let lib =
            LibraryContext::create(&temp.path().join("secondary"), "Synthetic secondary").unwrap();
        let uuid = state.library_manager().mount(lib);
        let lib = state.resolve_library(Some(&uuid)).unwrap();
        let root = temp.path().join("images");
        std::fs::create_dir(&root).unwrap();
        for library in [state.library_manager().primary(), &lib] {
            library.main_pool.get().unwrap().execute("INSERT INTO watch_directories(id,path,name,show_images) VALUES(7,?1,'Synthetic images',1)", params![root.to_string_lossy()]).unwrap();
        }
        (temp, state, lib, root)
    }

    fn staged_png(root: &std::path::Path) -> (tempfile::NamedTempFile, String, Vec<u8>) {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(2, 2)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        let bytes = bytes.into_inner();
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let mut file = tempfile::NamedTempFile::new_in(root).unwrap();
        file.write_all(&bytes).unwrap();
        (file, sha256, bytes)
    }

    // AC: @donut-create-plugin ac-save-gallery
    #[test]
    fn save_is_qualified_idempotent_and_does_not_overwrite() {
        let (_temp, state, lib, root) = fixture();
        let request = ImportRequest {
            session_id: "synthetic-session".into(),
            output_id: "synthetic-output".into(),
            library_id: lib.uuid.clone(),
            directory_id: 7,
            execution_metadata: None,
        };
        let (file, digest, bytes) = staged_png(&root);
        let collision = root.join(format!("donut-{digest}.png"));
        std::fs::write(&collision, b"preserve unrelated file").unwrap();
        let saved =
            save_output(&state, &lib, &request, root.clone(), file, digest.clone()).unwrap();
        assert_eq!(saved.library_id, lib.uuid);
        assert_eq!(
            std::fs::read(&collision).unwrap(),
            b"preserve unrelated file"
        );
        assert_eq!(std::fs::read(root.join(&saved.filename)).unwrap(), bytes);
        let count: i64 = state
            .library_manager()
            .primary()
            .directory_db
            .get_pool(7)
            .unwrap()
            .get()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let (file, _, _) = staged_png(&root);
        let repeated = save_output(&state, &lib, &request, root.clone(), file, digest).unwrap();
        assert_eq!(repeated.image_id, saved.image_id);
        assert_eq!(repeated.filename, saved.filename);
        assert_eq!(repeated.status, "existing");
        assert_eq!(std::fs::read_dir(root).unwrap().count(), 3);
    }

    // AC: @donut-create-plugin ac-save-gallery
    #[test]
    fn non_image_or_unavailable_destinations_are_rejected() {
        let (_temp, state, lib, root) = fixture();
        for column in ["show_images", "enabled"] {
            lib.main_pool
                .get()
                .unwrap()
                .execute(
                    &format!("UPDATE watch_directories SET {column}=0 WHERE id=7"),
                    [],
                )
                .unwrap();
            assert!(matches!(destination(&lib, 7), Err(AppError::BadRequest(_))));
            lib.main_pool
                .get()
                .unwrap()
                .execute(
                    &format!("UPDATE watch_directories SET {column}=1 WHERE id=7"),
                    [],
                )
                .unwrap();
        }
        std::fs::remove_dir(root).unwrap();
        assert!(destination(&lib, 7).is_err());
        state.library_manager().unmount(&lib.uuid);
        assert!(state.resolve_library(Some(&lib.uuid)).is_err());
    }

    // AC: @donut-create-plugin ac-workflow-state
    #[test]
    fn v5_metadata_resolves_nested_prompt_links() {
        let (_temp, state, lib, root) = fixture();
        let request = ImportRequest {
            session_id: "metadata-session".into(),
            output_id: "metadata-output".into(),
            library_id: lib.uuid.clone(),
            directory_id: 7,
            execution_metadata: None,
        };
        let (file, digest, _) = staged_png(&root);
        let saved = save_output(&state, &lib, &request, root, file, digest).unwrap();
        let graph = serde_json::json!({
            "input": {"class_type":"DF_Text_Box","inputs":{"Text":"distinctive portrait"}},
            "group:face": {"class_type":"DonutText","inputs":{"text":["input",0]}},
            "group:scene": {"class_type":"DonutText","inputs":{"text":"at the synthetic harbor"}},
            "group:negative": {"class_type":"DonutText","inputs":{"text":"synthetic negative"}},
            "group:conditioning": {"class_type":"DonutPromptConditioning","inputs":{"face":["group:face",0],"scene":["group:scene",0],"negative":["group:negative",0],"separator":" "}},
            "sampler": {"class_type":"DonutSampler","inputs":{"seed":90210,"steps":12,"cfg":2.5,"sampler_name":"euler"}}
        }).to_string();
        let pool = lib.directory_db.get_pool(7).unwrap();
        let conn = pool.get().unwrap();
        save_generation_metadata(&conn, saved.image_id, Some(&graph), None).unwrap();
        let (prompt, negative, seed): (String, String, String) = conn
            .query_row(
                "SELECT prompt,negative_prompt,seed FROM images WHERE id=?1",
                params![saved.image_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(prompt, "distinctive portrait at the synthetic harbor");
        assert_eq!(negative, "synthetic negative");
        assert_eq!(seed, "90210");
        let executed = serde_json::json!({"prompt":"resolved selected variant","prompt_source":"executed_output","seed":12345,"cfg":1.75});
        save_generation_metadata(&conn, saved.image_id, Some(&graph), Some(&executed)).unwrap();
        let (prompt, seed, cfg): (String, String, f64) = conn
            .query_row(
                "SELECT prompt,seed,cfg_scale FROM images WHERE id=?1",
                params![saved.image_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(prompt, "resolved selected variant");
        assert_eq!(seed, "12345");
        assert_eq!(cfg, 1.75);
    }

    // AC: @donut-create-plugin ac-access-boundary
    #[tokio::test]
    async fn creation_requires_current_write_identity() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::new(temp.path(), 0).unwrap();
        let mut parts = Request::new(Body::empty()).into_parts().0;
        assert!(require_write(&state, &mut parts).await.is_err());
        parts
            .extensions
            .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 50000))));
        assert!(require_write(&state, &mut parts).await.is_ok());
        // A supplied invalid credential cannot inherit localhost access.
        parts.headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer invalid"),
        );
        assert!(require_write(&state, &mut parts).await.is_err());
        state.main_db().get().unwrap().execute("INSERT INTO users(username,password_hash,can_write,is_active,access_level) VALUES('synthetic-reader','unused',0,1,'local_network')", []).unwrap();
        let id = state
            .main_db()
            .get()
            .unwrap()
            .query_row(
                "SELECT id FROM users WHERE username='synthetic-reader'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        // The database revokes a write claim issued before this permission change.
        let token = crate::server::middleware::auth::create_jwt(
            id,
            "synthetic-reader",
            "local_network",
            true,
            state.jwt_secret(),
        )
        .unwrap();
        parts.headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
        assert!(matches!(
            require_write(&state, &mut parts).await,
            Err(AppError::Forbidden(_))
        ));
    }

    // AC: @donut-create-plugin ac-access-boundary
    #[tokio::test]
    async fn routed_management_and_import_reject_read_only_and_invalid_credentials() {
        use tower::ServiceExt;
        let temporary = tempfile::tempdir().unwrap();
        let state = AppState::new(temporary.path(), 0).unwrap();
        state.main_db().get().unwrap().execute(
            "INSERT INTO users(username,password_hash,can_write,is_active,access_level) VALUES('creation-reader','unused',0,1,'public')",
            [],
        ).unwrap();
        let id = state
            .main_db()
            .get()
            .unwrap()
            .query_row(
                "SELECT id FROM users WHERE username='creation-reader'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let token = crate::server::middleware::auth::create_jwt(
            id,
            "creation-reader",
            "public",
            false,
            state.jwt_secret(),
        )
        .unwrap();
        let app = crate::server::build_router(state, None);
        for (method, uri) in [
            ("GET", "/api/addons/donut-create/api/create/status"),
            ("POST", "/api/create/import"),
        ] {
            for (authorization, expected) in [
                (format!("Bearer {token}"), StatusCode::FORBIDDEN),
                ("Bearer invalid".into(), StatusCode::UNAUTHORIZED),
            ] {
                let mut request = Request::builder()
                    .method(method)
                    .uri(uri)
                    .header(header::AUTHORIZATION, authorization)
                    .body(Body::empty())
                    .unwrap();
                request
                    .extensions_mut()
                    .insert(ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 50000))));
                let response = app.clone().oneshot(request).await.unwrap();
                assert_eq!(response.status(), expected, "{method} {uri}");
            }
        }
    }
}
