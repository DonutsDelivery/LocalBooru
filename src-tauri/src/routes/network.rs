use std::path::Path;

use axum::extract::State;
use axum::response::Json;
use axum::routing::{delete, get, post};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::error::AppError;
use crate::server::state::AppState;

// ─── Router ──────────────────────────────────────────────────────────────────

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(get_network_config).post(update_network_config))
        .route("/test-port", post(test_port))
        .route("/qr-data", get(get_qr_data))
        // UPnP stubs
        .route("/upnp/discover", post(upnp_discover))
        .route("/upnp/open-port", post(upnp_open_port))
        .route("/upnp/close-port/{external_port}", delete(upnp_close_port))
        .route("/upnp/mappings", get(upnp_mappings))
        .route("/upnp/external-ip", get(upnp_external_ip))
        // Handshake stub
        .route("/verify-handshake", post(verify_handshake))
}

// ─── Default network settings ────────────────────────────────────────────────

const DEFAULT_LOCAL_PORT: u16 = 8790;
const DEFAULT_PUBLIC_PORT: u16 = 8791;

fn default_network_settings() -> Value {
    json!({
        "local_network_enabled": false,
        "public_network_enabled": false,
        "local_port": DEFAULT_LOCAL_PORT,
        "public_port": DEFAULT_PUBLIC_PORT,
        "auth_required_level": "local_network",
        "upnp_enabled": false,
        "allow_settings_local_network": false
    })
}

// ─── Settings file helpers ───────────────────────────────────────────────────

/// Load the entire settings.json, returning an empty object if missing or invalid.
fn load_settings(data_dir: &Path) -> Value {
    let path = data_dir.join("settings.json");
    match std::fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|_| json!({})),
        Err(_) => json!({}),
    }
}

/// Save settings to settings.json with pretty printing.
fn save_settings(data_dir: &Path, settings: &Value) -> Result<(), AppError> {
    let path = data_dir.join("settings.json");
    let content = serde_json::to_string_pretty(settings)
        .map_err(|e| AppError::Internal(format!("Failed to serialize settings: {}", e)))?;
    std::fs::write(&path, content)?;
    Ok(())
}

/// Get the network section from settings, merged with defaults so all keys are present.
fn get_network_settings_from_file(data_dir: &Path) -> Value {
    let settings = load_settings(data_dir);
    let defaults = default_network_settings();
    let stored = settings
        .get("network")
        .cloned()
        .unwrap_or_else(|| json!({}));

    // Merge: defaults first, then stored values on top
    let defaults_obj = defaults.as_object().unwrap();
    let stored_obj = stored.as_object().cloned().unwrap_or_default();

    let mut merged = defaults_obj.clone();
    for (key, value) in stored_obj {
        merged.insert(key, value);
    }

    Value::Object(merged)
}

/// Save the network section back into settings.json, preserving other top-level keys.
fn save_network_settings_to_file(data_dir: &Path, network: &Value) -> Result<(), AppError> {
    let mut settings = load_settings(data_dir);
    let obj = settings
        .as_object_mut()
        .ok_or_else(|| AppError::Internal("Settings file is not a JSON object".into()))?;
    obj.insert("network".into(), network.clone());
    save_settings(data_dir, &settings)
}

// ─── Network helpers ─────────────────────────────────────────────────────────

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

/// Get all local IPv4 addresses by scanning common private ranges.
///
/// Falls back to just the primary IP if interface enumeration is unavailable.
fn get_all_local_ips() -> Vec<String> {
    let mut ips = Vec::new();
    if let Some(primary) = get_local_ip() {
        ips.push(primary);
    }
    // On Linux we can try reading /proc/net/fib_trie or similar, but for
    // simplicity (and cross-platform parity) we just return the primary IP.
    // The Python version uses netifaces which isn't available here yet.
    ips
}

/// Test whether a TCP port is available for binding.
fn test_port_available(port: u16) -> (bool, Option<String>) {
    match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    }
}

// ─── Request models ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct NetworkConfigUpdate {
    local_network_enabled: Option<bool>,
    public_network_enabled: Option<bool>,
    local_port: Option<u16>,
    public_port: Option<u16>,
    auth_required_level: Option<String>,
    upnp_enabled: Option<bool>,
    allow_settings_local_network: Option<bool>,
}

#[derive(Deserialize)]
struct PortTestRequest {
    port: u16,
}

#[derive(Deserialize)]
struct UPnPPortRequest {
    external_port: u16,
    internal_port: Option<u16>,
    protocol: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct HandshakeVerifyRequest {
    nonce: String,
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/network — Get current network configuration and status.
async fn get_network_config(
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let port = state.port();

    let result = tokio::task::spawn_blocking(move || {
        let settings = get_network_settings_from_file(&data_dir);
        let local_ip = get_local_ip();
        let all_ips = get_all_local_ips();

        // Build access URLs
        let local_url = if let Some(ref ip) = local_ip {
            if settings
                .get("local_network_enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                let lp = settings
                    .get("local_port")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(port as u64);
                Some(format!("http://{}:{}", ip, lp))
            } else {
                None
            }
        } else {
            None
        };

        // UPnP not yet implemented — no public URL or UPnP status
        let public_url: Option<String> = None;
        let upnp_status: Option<Value> = if settings
            .get("upnp_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            Some(json!({
                "external_ip": null,
                "gateway_found": false
            }))
        } else {
            None
        };

        Ok::<_, AppError>(json!({
            "settings": settings,
            "local_ip": local_ip,
            "all_local_ips": all_ips,
            "local_url": local_url,
            "public_url": public_url,
            "upnp_status": upnp_status
        }))
    })
    .await??;

    Ok(Json(result))
}

/// POST /api/network — Update network configuration.
async fn update_network_config(
    State(state): State<AppState>,
    Json(body): Json<NetworkConfigUpdate>,
) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        let mut current = get_network_settings_from_file(&data_dir);
        let obj = current
            .as_object_mut()
            .ok_or_else(|| AppError::Internal("Network settings is not a JSON object".into()))?;

        // Update only provided fields
        if let Some(v) = body.local_network_enabled {
            obj.insert("local_network_enabled".into(), json!(v));
        }
        if let Some(v) = body.public_network_enabled {
            obj.insert("public_network_enabled".into(), json!(v));
        }
        if let Some(v) = body.local_port {
            obj.insert("local_port".into(), json!(v));
        }
        if let Some(v) = body.public_port {
            obj.insert("public_port".into(), json!(v));
        }
        if let Some(ref v) = body.auth_required_level {
            let valid = ["none", "public", "local_network", "always"];
            if !valid.contains(&v.as_str()) {
                return Err(AppError::BadRequest(format!(
                    "Invalid auth_required_level: {}. Must be one of: {}",
                    v,
                    valid.join(", ")
                )));
            }
            obj.insert("auth_required_level".into(), json!(v));
        }
        if let Some(v) = body.upnp_enabled {
            obj.insert("upnp_enabled".into(), json!(v));
        }
        if let Some(v) = body.allow_settings_local_network {
            obj.insert("allow_settings_local_network".into(), json!(v));
        }

        save_network_settings_to_file(&data_dir, &current)?;

        Ok::<_, AppError>(json!({
            "success": true,
            "settings": current,
            "restart_required": true
        }))
    })
    .await??;

    Ok(Json(result))
}

/// POST /api/network/test-port — Test if a port is available for binding.
async fn test_port(Json(body): Json<PortTestRequest>) -> Result<Json<Value>, AppError> {
    let port = body.port;

    let result = tokio::task::spawn_blocking(move || {
        let (available, error) = test_port_available(port);
        json!({
            "port": port,
            "available": available,
            "error": error
        })
    })
    .await?;

    Ok(Json(result))
}

/// GET /api/network/qr-data — Get QR code data for mobile app connection.
async fn get_qr_data(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let data_dir = state.data_dir().to_path_buf();
    let port = state.port();

    let result = tokio::task::spawn_blocking(move || {
        let settings = get_network_settings_from_file(&data_dir);
        let local_ip = get_local_ip();

        // Build local URL (always include if we have an IP)
        let local_url = local_ip.as_ref().map(|ip| {
            let lp = settings
                .get("local_port")
                .and_then(|v| v.as_u64())
                .unwrap_or(port as u64);
            format!("http://{}:{}", ip, lp)
        });

        // Public URL requires UPnP — not yet implemented
        let public_url: Option<String> = None;

        // Check if auth is required
        let auth_level = settings
            .get("auth_required_level")
            .and_then(|v| v.as_str())
            .unwrap_or("none");
        let auth_required = auth_level == "local_network" || auth_level == "always";

        // Nonce and fingerprint are stubs until auth service is ported
        json!({
            "type": "localbooru",
            "version": 2,
            "name": "LocalBooru",
            "local": local_url,
            "public": public_url,
            "auth": auth_required,
            "fingerprint": null,
            "cert_fingerprint": null,
            "nonce": null,
            "nonce_expires": null
        })
    })
    .await?;

    Ok(Json(result))
}

// ─── UPnP stubs ──────────────────────────────────────────────────────────────
//
// UPnP support depends on a UPnP service that hasn't been ported yet.
// These stubs return sensible defaults so the frontend doesn't get 404s.

/// POST /api/network/upnp/discover — Stub: discover UPnP gateway.
async fn upnp_discover() -> Result<Json<Value>, AppError> {
    Ok(Json(json!({
        "success": false,
        "gateway_found": false,
        "message": "UPnP service not yet available in v2"
    })))
}

/// POST /api/network/upnp/open-port — Stub: open a port via UPnP.
async fn upnp_open_port(Json(body): Json<UPnPPortRequest>) -> Result<Json<Value>, AppError> {
    let internal = body.internal_port.unwrap_or(body.external_port);
    let protocol = body.protocol.as_deref().unwrap_or("TCP");
    let description = body
        .description
        .as_deref()
        .unwrap_or("LocalBooru");

    Ok(Json(json!({
        "success": false,
        "external_port": body.external_port,
        "internal_port": internal,
        "protocol": protocol,
        "description": description,
        "error": "UPnP service not yet available in v2"
    })))
}

/// DELETE /api/network/upnp/close-port/:external_port — Stub: close a UPnP port mapping.
async fn upnp_close_port(
    axum::extract::Path(external_port): axum::extract::Path<u16>,
) -> Result<Json<Value>, AppError> {
    Ok(Json(json!({
        "success": false,
        "external_port": external_port,
        "error": "UPnP service not yet available in v2"
    })))
}

/// GET /api/network/upnp/mappings — Stub: list UPnP port mappings.
async fn upnp_mappings() -> Result<Json<Value>, AppError> {
    Ok(Json(json!({
        "mappings": []
    })))
}

/// GET /api/network/upnp/external-ip — Stub: get external IP via UPnP.
async fn upnp_external_ip() -> Result<Json<Value>, AppError> {
    Ok(Json(json!({
        "external_ip": null
    })))
}

// ─── Handshake stub ──────────────────────────────────────────────────────────

/// POST /api/network/verify-handshake — Stub: verify a handshake nonce.
///
/// This will be implemented once the auth/certificate services are ported.
async fn verify_handshake(
    Json(_body): Json<HandshakeVerifyRequest>,
) -> Result<Json<Value>, AppError> {
    // Always fail verification until the auth service is ported
    Err(AppError::BadRequest(
        "Handshake verification not yet available in v2".into(),
    ))
}
