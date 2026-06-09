use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, Response, StatusCode},
    response::IntoResponse,
};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::task::{Context, Poll};
use tower::{Layer, Service};
use std::future::Future;
use std::pin::Pin;

use super::auth::decode_jwt;

/// Endpoints that are always localhost-only (sensitive settings).
const LOCALHOST_ONLY_PREFIXES: &[&str] = &[
    "/api/settings",
    "/api/network",
    "/api/users",
];

/// Endpoints exempt from access control (prefix match).
const EXEMPT_PREFIXES: &[&str] = &[
    "/health",
    "/docs",
    "/assets",
    "/thumbnails",
    "/icon.png",
    "/api/share/",
    "/api/cast-media/",
    "/watch/",
];

/// Endpoints exempt from access control (exact match).
const EXEMPT_EXACT: &[&str] = &[
    "/api",
];

/// Endpoints under localhost-only prefixes that should still be accessible from network.
const LOCALHOST_EXEMPTIONS: &[&str] = &[
    "/api/network/verify-handshake",
    "/api/network/qr-data",
    "/api/settings/saved-searches",
    "/api/settings/family-mode",
    "/api/settings/video-playback",
    "/api/settings/optical-flow",
    "/api/settings/svp",
    "/api/settings/whisper",
    "/api/settings/cast",
    "/api/settings/transcode",
    "/api/settings/video-info",
    "/api/settings/util",
    "/api/users/login",
    "/api/users/verify",
    "/api/users/media-token",
];

/// Test whether an IPv4 address falls in the RFC 6598 carrier-grade NAT range
/// (100.64.0.0/10), which Tailscale uses for its internal "100.x.x.x" addresses.
/// std's `is_private()` does not include this range.
fn is_cgnat(v4: &std::net::Ipv4Addr) -> bool {
    let octets = v4.octets();
    octets[0] == 100 && (octets[1] & 0b1100_0000) == 64
}

/// Test whether an IPv6 address is in Tailscale's ULA range (fd7a:115c:a1e0::/48).
fn is_tailscale_v6(v6: &std::net::Ipv6Addr) -> bool {
    let segs = v6.segments();
    segs[0] == 0xfd7a && segs[1] == 0x115c && segs[2] == 0xa1e0
}

/// Classify an IP address into an access level.
pub fn classify_ip(ip: &std::net::IpAddr) -> &'static str {
    match ip {
        std::net::IpAddr::V4(v4) => {
            if v4.is_loopback() {
                "localhost"
            } else if v4.is_private() || v4.is_link_local() || is_cgnat(v4) {
                "local_network"
            } else {
                "public"
            }
        }
        std::net::IpAddr::V6(v6) => {
            if v6.is_loopback() {
                "localhost"
            } else if is_tailscale_v6(v6) {
                "local_network"
            } else {
                // Check for IPv4-mapped IPv6 (::ffff:127.0.0.1, etc.)
                if let Some(v4) = v6.to_ipv4_mapped() {
                    if v4.is_loopback() {
                        return "localhost";
                    }
                    if v4.is_private() || v4.is_link_local() || is_cgnat(&v4) {
                        return "local_network";
                    }
                }
                "public"
            }
        }
    }
}

/// Read the `network.allow_settings_local_network` opt-in flag from `settings.json`.
///
/// When `true`, the owner has explicitly allowed local-network devices (with a valid
/// JWT) to reach the otherwise localhost-only settings/network/user endpoints. Returns
/// `false` (the secure default) if the file or key is missing or unreadable, so a
/// missing/corrupt settings file fails closed.
fn lan_settings_opt_in(data_dir: &Path) -> bool {
    let path = data_dir.join("settings.json");
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("network")?
                .get("allow_settings_local_network")?
                .as_bool()
        })
        .unwrap_or(false)
}

// ─── Access tier ────────────────────────────────────────────────────────────

/// Typed access tier derived from a client IP address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessTier {
    Localhost,
    LocalNetwork,
    Public,
}

impl AccessTier {
    /// Classify a client IP into an access tier.
    pub fn from_ip(ip: &std::net::IpAddr) -> Self {
        match classify_ip(ip) {
            "localhost" => Self::Localhost,
            "local_network" => Self::LocalNetwork,
            _ => Self::Public,
        }
    }
}

// ─── Layer ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AccessControlLayer {
    pub jwt_secret: String,
    /// Data directory, used to read the `allow_settings_local_network` opt-in flag.
    pub data_dir: PathBuf,
}

impl<S> Layer<S> for AccessControlLayer {
    type Service = AccessControlService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AccessControlService {
            inner,
            jwt_secret: self.jwt_secret.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

// ─── Service ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AccessControlService<S> {
    inner: S,
    jwt_secret: String,
    data_dir: PathBuf,
}

impl<S> Service<Request<Body>> for AccessControlService<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let mut inner = self.inner.clone();
        // Swap so the clone is the "not ready" one
        std::mem::swap(&mut self.inner, &mut inner);

        let jwt_secret = self.jwt_secret.clone();
        let data_dir = self.data_dir.clone();

        Box::pin(async move {
            let path = req.uri().path().to_string();
            let method = req.method().as_str().to_string();

            // Always allow OPTIONS (CORS preflight)
            if method == "OPTIONS" {
                return inner.call(req).await;
            }

            // Skip access control for exempt endpoints (prefix match)
            for prefix in EXEMPT_PREFIXES {
                if path.starts_with(prefix) {
                    return inner.call(req).await;
                }
            }

            // Skip access control for exempt endpoints (exact match)
            if EXEMPT_EXACT.contains(&path.as_str()) {
                return inner.call(req).await;
            }

            // Extract client IP from ConnectInfo extension
            let client_ip = req
                .extensions()
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ci| ci.0.ip())
                .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));

            let access_level = classify_ip(&client_ip);

            // Localhost has full access
            if access_level == "localhost" {
                return inner.call(req).await;
            }

            // Decide whether a `?token=` query token grants access to THIS request.
            // A full session token (scope=None) grants access everywhere (still
            // subject to the localhost-only checks below). A media-scoped token is
            // read-only and only honored for GET requests on `/api/images/...` —
            // the one media path that requires a token (streams, cast-media, share,
            // /thumbnails and /watch are already exempt). This lets the frontend put
            // a short-lived media token in <img>/<video> URLs instead of the 30-day
            // session JWT, without breaking any media that needs auth.
            let query_token_grants = |token: &str| -> bool {
                match decode_jwt(token, &jwt_secret) {
                    Ok(claims) if claims.is_media_scoped() => {
                        method == "GET" && path.starts_with("/api/images/")
                    }
                    Ok(_) => true,
                    Err(_) => false,
                }
            };

            // Bearer header: accept full session tokens only. A media token is
            // rejected here so it can never authenticate a real API call.
            let has_valid_jwt = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|auth| {
                    auth.strip_prefix("Bearer ")
                        .or_else(|| auth.strip_prefix("bearer "))
                })
                .map(|token| {
                    matches!(decode_jwt(token, &jwt_secret), Ok(c) if !c.is_media_scoped())
                })
                .unwrap_or(false);

            // Query parameter (for <img>/<video> src URLs): full token, or a media
            // token limited to GET image routes.
            let has_valid_jwt = has_valid_jwt || req
                .uri()
                .query()
                .and_then(|q| {
                    q.split('&')
                        .find_map(|pair| pair.strip_prefix("token="))
                })
                .map(|token| query_token_grants(token))
                .unwrap_or(false);

            // Paths explicitly allowed from the network even though they sit under a
            // localhost-only prefix: pairing handshake, login/verify, and user-preference
            // settings the LAN/Android client legitimately needs.
            let is_exempt_path = LOCALHOST_EXEMPTIONS
                .iter()
                .any(|exempt| path == *exempt || path.starts_with(&format!("{}/", exempt)));

            // Localhost-only endpoints (settings / network / user management) stay
            // restricted even WITH a valid JWT. Localhost already returned above, so
            // reaching here means a non-localhost caller. A paired device's token must
            // NOT be able to change network exposure, rewrite global settings, or manage
            // users — UNLESS the owner has explicitly opted in from the host machine via
            // `network.allow_settings_local_network`, and only from the local network
            // (never from the public internet).
            let is_localhost_only = LOCALHOST_ONLY_PREFIXES
                .iter()
                .any(|prefix| path.starts_with(prefix))
                && !is_exempt_path;

            if is_localhost_only {
                let opt_in_allowed = has_valid_jwt
                    && access_level == "local_network"
                    && lan_settings_opt_in(&data_dir);

                if !opt_in_allowed {
                    let response = (
                        StatusCode::FORBIDDEN,
                        axum::Json(serde_json::json!({
                            "error": "This endpoint is only accessible from localhost",
                            "detail": "Settings, network, and user management are restricted to the host machine. Enable 'Allow settings changes over local network' on the host to manage them from a LAN device."
                        })),
                    )
                        .into_response();
                    return Ok(response);
                }

                // Opted in: a local-network device with a valid JWT may proceed.
                return inner.call(req).await;
            }

            // Authenticated requests (valid JWT) may access everything that is not
            // localhost-only (handled above): images, tags, media, casting, user prefs, etc.
            if has_valid_jwt {
                return inner.call(req).await;
            }

            // No valid JWT and not localhost → block with 401.
            // Localhost-exempted paths (e.g. verify-handshake, login) pass through so a
            // client can obtain a token in the first place.
            if !is_exempt_path {
                let response = (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({
                        "error": "Authentication required",
                        "detail": "Non-localhost requests require a valid JWT token. Pair via QR code to obtain one."
                    })),
                )
                    .into_response();
                return Ok(response);
            }

            inner.call(req).await
        })
    }
}
