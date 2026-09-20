use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, Response, StatusCode},
    response::IntoResponse,
};
use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::{Context, Poll};
use tower::{Layer, Service};

use super::auth::{decode_jwt, refresh_claim_identity, Claims};
use crate::db::pool::DbPool;

fn claim_allows_source(claims: &Claims, source: &str) -> bool {
    match claims.access_level.as_str() {
        "public" => true,
        "local_network" => source != "public",
        "localhost" => source == "localhost",
        _ => false,
    }
}

/// Endpoints that are always localhost-only (sensitive settings).
const LOCALHOST_ONLY_PREFIXES: &[&str] = &[
    "/api/settings",
    "/api/network",
    "/api/users",
    "/api/direct-files",
    "/api/device-pairing/desktop-sessions",
    "/api/device-pairing/devices",
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
const EXEMPT_EXACT: &[&str] = &["/api"];

/// Endpoints under localhost-only prefixes that should still be accessible from network.
const LOCALHOST_EXEMPTIONS: &[&str] = &[
    "/api/network/verify-handshake",
    "/api/settings/saved-searches",
    "/api/settings/family-mode",
    "/api/settings/video-playback",
    "/api/settings/optical-flow/stop",
    "/api/settings/svp",
    "/api/settings/whisper",
    "/api/settings/cast",
    "/api/settings/transcode",
    "/api/settings/video-info",
    "/api/settings/util",
    "/api/users/login",
    "/api/users/verify",
    "/api/users/media-token",
    "/api/device-pairing/grants",
    "/api/device-pairing/exchange",
];

/// POST endpoints that inspect or verify without mutating server state.
const READ_ONLY_POST_EXACT: &[&str] = &[
    "/api/network/verify-handshake",
    "/api/users/verify",
    "/api/settings/video-info",
    "/api/settings/audio-gain",
];

fn is_device_pairing_delivery(path: &str) -> bool {
    path.starts_with("/api/device-pairing/desktop-sessions/")
        && (path.ends_with("/deliver") || path.ends_with("/verify"))
}

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
    pub db: DbPool,
}

impl<S> Layer<S> for AccessControlLayer {
    type Service = AccessControlService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AccessControlService {
            inner,
            jwt_secret: self.jwt_secret.clone(),
            data_dir: self.data_dir.clone(),
            db: self.db.clone(),
        }
    }
}

// ─── Service ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AccessControlService<S> {
    inner: S,
    jwt_secret: String,
    data_dir: PathBuf,
    db: DbPool,
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
        let db = self.db.clone();

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

            // Decide whether a media-scoped `?token=` grants access to THIS request.
            // Full session tokens are never accepted from URLs. A media token is
            // read-only and only honored for GET requests on `/api/images/...` —
            // the one media path that requires a token (streams, cast-media, share,
            // /thumbnails and /watch are already exempt). This lets the frontend put
            // a short-lived media token in <img>/<video> URLs instead of the 30-day
            // session JWT, without breaking any media that needs auth.
            let query_token_grants = |token: &str| -> bool {
                match decode_jwt(token, &jwt_secret) {
                    Ok(claims) if claims.is_media_scoped() => {
                        method == "GET"
                            && path.starts_with("/api/images/")
                            && refresh_claim_identity(claims, &db)
                                .is_some_and(|claims| claim_allows_source(&claims, access_level))
                    }
                    Ok(_) => false,
                    Err(_) => false,
                }
            };

            // Bearer header: accept full session tokens only. A media token is
            // rejected here so it can never authenticate a real API call.
            let bearer_claims = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|auth| {
                    auth.strip_prefix("Bearer ")
                        .or_else(|| auth.strip_prefix("bearer "))
                })
                .and_then(|token| decode_jwt(token, &jwt_secret).ok())
                .filter(|claims| claims.scope.is_none())
                .and_then(|claims| refresh_claim_identity(claims, &db))
                .filter(|claims| claim_allows_source(claims, access_level));

            // Query parameters are accepted only for short-lived media tokens.
            let has_valid_query_jwt = req
                .uri()
                .query()
                .and_then(|q| q.split('&').find_map(|pair| pair.strip_prefix("token=")))
                .map(query_token_grants)
                .unwrap_or(false);
            let has_valid_jwt = bearer_claims.is_some() || has_valid_query_jwt;

            // Paths explicitly allowed from the network even though they sit under a
            // localhost-only prefix: pairing handshake, login/verify, and user-preference
            // settings the LAN/Android client legitimately needs.
            let is_exempt_path = LOCALHOST_EXEMPTIONS
                .iter()
                .any(|exempt| path == *exempt || path.starts_with(&format!("{}/", exempt)))
                || (method == "POST" && is_device_pairing_delivery(&path));

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

            let is_mutating = !matches!(method.as_str(), "GET" | "HEAD")
                && !(method == "POST" && READ_ONLY_POST_EXACT.contains(&path.as_str()));
            if is_mutating
                && bearer_claims
                    .as_ref()
                    .is_some_and(|claims| !claims.can_write)
            {
                let response = (
                    StatusCode::FORBIDDEN,
                    axum::Json(serde_json::json!({
                        "error": "Write access required",
                        "detail": "This credential is read-only."
                    })),
                )
                    .into_response();
                return Ok(response);
            }

            if is_localhost_only {
                let opt_in_allowed = !path.starts_with("/api/direct-files")
                    && !path.starts_with("/api/device-pairing/desktop-sessions")
                    && !path.starts_with("/api/device-pairing/devices")
                    && has_valid_jwt
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::middleware::auth::{create_jwt, create_media_jwt};
    use tower::{service_fn, ServiceExt};

    #[tokio::test]
    async fn paired_lan_client_can_stop_retired_optical_flow_playback() {
        let secret = "optical-flow-stop-test-secret";
        let token = create_jwt(0, "paired-device", "local_network", true, secret).unwrap();
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-access-control-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        let db = crate::db::pool::create_main_pool(&data_dir).unwrap();
        crate::db::schema::init_main_db(&db.get().unwrap()).unwrap();

        let inner = service_fn(|_request: Request<Body>| async move {
            Ok::<_, std::convert::Infallible>(Response::new(Body::empty()))
        });
        let service = AccessControlLayer {
            jwt_secret: secret.into(),
            data_dir: data_dir.clone(),
            db: db.clone(),
        }
        .layer(inner);

        let mut request = Request::builder()
            .method("POST")
            .uri("/api/settings/optical-flow/stop")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));

        let response = service.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test]
    async fn direct_file_capabilities_are_never_available_to_lan_clients() {
        let secret = "direct-file-localhost-test-secret";
        let token = create_jwt(0, "paired-device", "local_network", true, secret).unwrap();
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-direct-file-access-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        let db = crate::db::pool::create_main_pool(&data_dir).unwrap();
        crate::db::schema::init_main_db(&db.get().unwrap()).unwrap();
        std::fs::write(
            data_dir.join("settings.json"),
            r#"{"network":{"allow_settings_local_network":true}}"#,
        )
        .unwrap();

        let inner = service_fn(|_request: Request<Body>| async move {
            Ok::<_, std::convert::Infallible>(Response::new(Body::empty()))
        });
        let service = AccessControlLayer {
            jwt_secret: secret.into(),
            data_dir: data_dir.clone(),
            db: db.clone(),
        }
        .layer(inner);
        let mut request = Request::builder()
            .method("GET")
            .uri("/api/direct-files/unguessable-token")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));

        let response = service.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test]
    async fn lan_clients_cannot_mint_phone_pairing_offers() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-qr-offer-access-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        let db = crate::db::pool::create_main_pool(&data_dir).unwrap();
        crate::db::schema::init_main_db(&db.get().unwrap()).unwrap();
        let inner = service_fn(|_request: Request<Body>| async move {
            Ok::<_, std::convert::Infallible>(Response::new(Body::empty()))
        });
        let service = AccessControlLayer {
            jwt_secret: "qr-offer-test-secret".into(),
            data_dir: data_dir.clone(),
            db,
        }
        .layer(inner);
        let mut request = Request::builder()
            .method("GET")
            .uri("/api/network/qr-data")
            .body(Body::empty())
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));

        let response = service.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test]
    async fn full_query_tokens_and_read_only_mutations_are_rejected() {
        let secret = "query-token-and-write-test-secret";
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-query-token-access-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data_dir).unwrap();
        let db = crate::db::pool::create_main_pool(&data_dir).unwrap();
        crate::db::schema::init_main_db(&db.get().unwrap()).unwrap();
        db.get()
            .unwrap()
            .execute(
                "INSERT INTO users (id, username, password_hash, is_active, access_level, can_write) VALUES (5, 'reader', 'unused', 1, 'local_network', 0)",
                [],
            )
            .unwrap();
        let full_token = create_jwt(5, "reader", "local_network", false, secret).unwrap();
        let media_token = create_media_jwt(5, "reader", "local_network", None, secret).unwrap();
        let inner = service_fn(|_request: Request<Body>| async move {
            Ok::<_, std::convert::Infallible>(Response::new(Body::empty()))
        });
        let service = AccessControlLayer {
            jwt_secret: secret.into(),
            data_dir: data_dir.clone(),
            db,
        }
        .layer(inner);
        let mut full_query = Request::builder()
            .method("GET")
            .uri(format!("/api/images/1?token={full_token}"))
            .body(Body::empty())
            .unwrap();
        full_query
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));
        assert_eq!(
            service.clone().oneshot(full_query).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );

        let mut media_query = Request::builder()
            .method("GET")
            .uri(format!("/api/images/1?token={media_token}"))
            .body(Body::empty())
            .unwrap();
        media_query
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));
        assert_eq!(
            service.clone().oneshot(media_query).await.unwrap().status(),
            StatusCode::OK
        );

        let mut read_only_post = Request::builder()
            .method("POST")
            .uri("/api/users/verify")
            .header("authorization", format!("Bearer {full_token}"))
            .body(Body::empty())
            .unwrap();
        read_only_post
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));
        assert_eq!(
            service
                .clone()
                .oneshot(read_only_post)
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        let mut mutation = Request::builder()
            .method("POST")
            .uri("/api/images/1/favorite")
            .header("authorization", format!("Bearer {full_token}"))
            .body(Body::empty())
            .unwrap();
        mutation
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 168, 1, 25], 50000))));
        assert_eq!(
            service.oneshot(mutation).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        let _ = std::fs::remove_dir_all(data_dir);
    }
}
