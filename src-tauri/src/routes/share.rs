//! Media sharing/streaming subsystem.
//!
//! A share session allows streaming media to other devices on the network
//! with synchronized playback. Sessions are stored in-memory (DashMap) and
//! provide SSE-based real-time sync of playback state.

use std::convert::Infallible;

use axum::extract::{Path as AxumPath, State};
use axum::response::sse::{Event, Sse};
use axum::response::Json;
use axum::routing::{delete, get, post};
use axum::Router;
use chrono::Utc;
use dashmap::DashMap;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::broadcast;

use crate::routes::images::helpers::find_image_directory;
use crate::server::error::AppError;
use crate::server::state::AppState;

// ---- Data model ----

/// Playback state for a share session, updated via the /sync endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlaybackState {
    pub is_playing: bool,
    pub current_time: f64,
    pub updated_at: String,
}

/// A single share session binding a media item to a token with sync state.
#[derive(Clone, Debug)]
pub struct ShareSession {
    pub token: String,
    pub image_id: i64,
    pub directory_id: Option<i64>,
    pub created_at: String,
    pub created_by_ip: String,
    pub playback_state: PlaybackState,
    /// Broadcast channel for SSE sync events (per-session).
    pub event_tx: broadcast::Sender<String>,
}

/// Thread-safe map of token -> ShareSession.
pub type ShareSessions = DashMap<String, ShareSession>;

/// Create a new empty ShareSessions map.
pub fn create_share_sessions() -> ShareSessions {
    DashMap::new()
}

// ---- Broadcast capacity ----

const SESSION_CHANNEL_CAPACITY: usize = 64;

// ---- Router ----

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/create", post(create_session))
        .route("/network-info", get(network_info))
        .route("/{token}", delete(delete_session))
        .route("/{token}/sync", post(sync_playback))
        .route("/{token}/info", get(session_info))
        .route("/{token}/events", get(session_events))
        .route("/{token}/hls/playlist.m3u8", get(hls_playlist))
}

// ---- Request / response models ----

#[derive(Deserialize)]
struct CreateSessionBody {
    image_id: i64,
    directory_id: Option<i64>,
}

#[derive(Deserialize)]
struct SyncPlaybackBody {
    is_playing: bool,
    current_time: f64,
}

// ---- Handlers ----

/// POST /api/share/create -- Create a new share session.
///
/// Generates a random token, stores the session, and returns URLs for
/// accessing the shared media.
async fn create_session(
    State(state): State<AppState>,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    Json(body): Json<CreateSessionBody>,
) -> Result<Json<Value>, AppError> {
    let token = uuid::Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();

    let (event_tx, _) = broadcast::channel(SESSION_CHANNEL_CAPACITY);

    let session = ShareSession {
        token: token.clone(),
        image_id: body.image_id,
        directory_id: body.directory_id,
        created_at: now.clone(),
        created_by_ip: addr.ip().to_string(),
        playback_state: PlaybackState {
            is_playing: false,
            current_time: 0.0,
            updated_at: now,
        },
        event_tx,
    };

    state.share_sessions().insert(token.clone(), session);

    log::info!(
        "[Share] Created session {} for image {} (from {})",
        token,
        body.image_id,
        addr.ip()
    );

    Ok(Json(json!({
        "token": token,
        "share_url": format!("/share/{}", token),
        "stream_url": format!("/api/share/{}/hls/playlist.m3u8", token),
    })))
}

/// DELETE /api/share/{token} -- Delete a share session.
async fn delete_session(
    State(state): State<AppState>,
    AxumPath(token): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    if state.share_sessions().remove(&token).is_none() {
        return Err(AppError::NotFound(format!(
            "Share session '{}' not found",
            token
        )));
    }

    log::info!("[Share] Deleted session {}", token);
    Ok(Json(json!({ "success": true })))
}

/// POST /api/share/{token}/sync -- Update playback state and broadcast to SSE listeners.
async fn sync_playback(
    State(state): State<AppState>,
    AxumPath(token): AxumPath<String>,
    Json(body): Json<SyncPlaybackBody>,
) -> Result<Json<Value>, AppError> {
    let now = Utc::now().to_rfc3339();

    let event_tx = {
        let mut session = state
            .share_sessions()
            .get_mut(&token)
            .ok_or_else(|| AppError::NotFound(format!("Share session '{}' not found", token)))?;

        session.playback_state = PlaybackState {
            is_playing: body.is_playing,
            current_time: body.current_time,
            updated_at: now.clone(),
        };

        session.event_tx.clone()
    };

    // Broadcast sync event to all SSE listeners
    let event_data = json!({
        "type": "playback_sync",
        "data": {
            "is_playing": body.is_playing,
            "current_time": body.current_time,
        },
        "timestamp": now,
    });
    // Ignore send error if no subscribers
    let _ = event_tx.send(event_data.to_string());

    Ok(Json(json!({ "success": true })))
}

/// GET /api/share/{token}/info -- Return session info including playback state and image metadata.
async fn session_info(
    State(state): State<AppState>,
    AxumPath(token): AxumPath<String>,
) -> Result<Json<Value>, AppError> {
    let session = state
        .share_sessions()
        .get(&token)
        .ok_or_else(|| AppError::NotFound(format!("Share session '{}' not found", token)))?
        .clone();

    let image_id = session.image_id;
    let directory_id = session.directory_id;
    let state_clone = state.clone();

    // Look up image metadata from directory DB
    let image_data = tokio::task::spawn_blocking(move || {
        // Resolve directory_id: use provided one, or search all directories
        let dir_id = directory_id.or_else(|| {
            find_image_directory(state_clone.directory_db(), image_id, None)
        });

        let dir_id = match dir_id {
            Some(id) => id,
            None => return Ok::<Option<Value>, AppError>(None),
        };

        if !state_clone.directory_db().db_exists(dir_id) {
            return Ok(None);
        }

        let dir_pool = state_clone.directory_db().get_pool(dir_id)
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let dir_conn = dir_pool.get()?;

        let image = dir_conn.query_row(
            "SELECT id, filename, original_filename, file_hash, width, height,
                    file_size, duration, rating, is_favorite, created_at
             FROM images WHERE id = ?1",
            params![image_id],
            |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "filename": row.get::<_, String>(1)?,
                    "original_filename": row.get::<_, Option<String>>(2)?,
                    "file_hash": row.get::<_, String>(3)?,
                    "width": row.get::<_, Option<i32>>(4)?,
                    "height": row.get::<_, Option<i32>>(5)?,
                    "file_size": row.get::<_, Option<i64>>(6)?,
                    "duration": row.get::<_, Option<f64>>(7)?,
                    "rating": row.get::<_, String>(8)?,
                    "is_favorite": row.get::<_, bool>(9)?,
                    "created_at": row.get::<_, Option<String>>(10)?,
                    "directory_id": dir_id,
                    "thumbnail_url": format!("/api/images/{}/thumbnail?directory_id={}", image_id, dir_id),
                    "url": format!("/api/images/{}/file?directory_id={}", image_id, dir_id),
                }))
            },
        );

        match image {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AppError::Internal(format!("Database error: {}", e))),
        }
    })
    .await??;

    Ok(Json(json!({
        "token": session.token,
        "image_id": session.image_id,
        "directory_id": session.directory_id,
        "created_at": session.created_at,
        "created_by_ip": session.created_by_ip,
        "playback_state": {
            "is_playing": session.playback_state.is_playing,
            "current_time": session.playback_state.current_time,
            "updated_at": session.playback_state.updated_at,
        },
        "image": image_data,
    })))
}

/// GET /api/share/network-info -- Return network info for sharing.
///
/// Detects the local IP so remote devices know where to connect.
async fn network_info(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let port = state.port();

    let result = tokio::task::spawn_blocking(move || {
        let local_ip = get_local_ip();
        let hostname = std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());

        json!({
            "local_ip": local_ip,
            "port": port,
            "hostname": hostname,
        })
    })
    .await?;

    Ok(Json(result))
}

/// GET /api/share/{token}/events -- SSE stream for real-time playback sync.
///
/// Each connected client receives playback state changes as they happen.
/// Uses a per-session tokio broadcast channel (same pattern as EventBroadcaster).
async fn session_events(
    State(state): State<AppState>,
    AxumPath(token): AxumPath<String>,
) -> Result<Sse<impl futures_core::Stream<Item = Result<Event, Infallible>>>, AppError> {
    let session = state
        .share_sessions()
        .get(&token)
        .ok_or_else(|| AppError::NotFound(format!("Share session '{}' not found", token)))?;

    let mut rx = session.event_tx.subscribe();
    let current_state = session.playback_state.clone();
    drop(session); // release DashMap ref before entering the stream

    let stream = async_stream::stream! {
        // Send current state on connect
        let connected = json!({
            "type": "connected",
            "data": {
                "is_playing": current_state.is_playing,
                "current_time": current_state.current_time,
            },
            "timestamp": Utc::now().to_rfc3339(),
        });
        yield Ok(Event::default().data(connected.to_string()));

        loop {
            match rx.recv().await {
                Ok(msg) => {
                    yield Ok(Event::default().data(msg));
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    log::warn!("[Share SSE] Subscriber lagged by {} events", n);
                    // Continue -- subscriber just missed some events
                }
                Err(broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    };

    Ok(Sse::new(stream))
}

/// GET /api/share/{token}/hls/playlist.m3u8 -- Serve HLS playlist for the shared media.
///
/// For now, returns a simple single-segment HLS playlist that points to the
/// source file served via the existing `/api/images/{id}/file` endpoint.
/// This allows HLS-compatible players to stream the media without a full
/// transcode pipeline.
async fn hls_playlist(
    State(state): State<AppState>,
    AxumPath(token): AxumPath<String>,
) -> Result<axum::response::Response, AppError> {
    let session = state
        .share_sessions()
        .get(&token)
        .ok_or_else(|| AppError::NotFound(format!("Share session '{}' not found", token)))?
        .clone();

    let image_id = session.image_id;
    let directory_id = session.directory_id;
    let state_clone = state.clone();

    // Look up the duration from the directory DB
    let (duration, dir_id) = tokio::task::spawn_blocking(move || {
        let dir_id = directory_id.or_else(|| {
            find_image_directory(state_clone.directory_db(), image_id, None)
        });

        let dir_id = match dir_id {
            Some(id) => id,
            None => return Ok::<(f64, Option<i64>), AppError>((0.0, None)),
        };

        if !state_clone.directory_db().db_exists(dir_id) {
            return Ok((0.0, Some(dir_id)));
        }

        let dir_pool = state_clone.directory_db().get_pool(dir_id)
            .map_err(|e| AppError::Internal(e.to_string()))?;
        let dir_conn = dir_pool.get()?;

        let duration: f64 = dir_conn
            .query_row(
                "SELECT COALESCE(duration, 0) FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get(0),
            )
            .unwrap_or(0.0);

        Ok((duration, Some(dir_id)))
    })
    .await??;

    let resolved_dir_id = dir_id.unwrap_or(0);
    let file_url = format!(
        "/api/images/{}/file?directory_id={}",
        image_id, resolved_dir_id
    );

    // Generate a simple HLS VOD playlist pointing to the source file.
    // If the duration is known, use it; otherwise default to a large value
    // so the player treats it as a long segment.
    let target_duration = if duration > 0.0 {
        duration.ceil() as u64
    } else {
        // Unknown duration -- use a generous default
        7200
    };

    let playlist = format!(
        "#EXTM3U\n\
         #EXT-X-VERSION:3\n\
         #EXT-X-TARGETDURATION:{target_duration}\n\
         #EXT-X-MEDIA-SEQUENCE:0\n\
         #EXT-X-PLAYLIST-TYPE:VOD\n\
         #EXTINF:{duration:.3},\n\
         {file_url}\n\
         #EXT-X-ENDLIST\n",
        target_duration = target_duration,
        duration = if duration > 0.0 { duration } else { target_duration as f64 },
        file_url = file_url,
    );

    Ok(axum::response::Response::builder()
        .header("Content-Type", "application/vnd.apple.mpegurl")
        .header("Cache-Control", "no-cache")
        .body(axum::body::Body::from(playlist))
        .unwrap())
}

// ---- Helpers ----

/// Detect the primary local (non-loopback) IPv4 address.
///
/// Connects a UDP socket to a public IP (no data sent) to determine which
/// local interface the OS would route through.
fn get_local_ip() -> Option<String> {
    use std::net::UdpSocket;
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}
