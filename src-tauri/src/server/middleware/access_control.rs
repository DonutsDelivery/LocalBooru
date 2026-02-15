use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, Response, StatusCode},
    response::IntoResponse,
};
use std::net::SocketAddr;
use std::task::{Context, Poll};
use tower::{Layer, Service};
use std::future::Future;
use std::pin::Pin;

/// Endpoints that are always localhost-only (sensitive settings).
const LOCALHOST_ONLY_PREFIXES: &[&str] = &[
    "/api/settings",
    "/api/network",
    "/api/users",
];

/// Endpoints exempt from access control.
const EXEMPT_PREFIXES: &[&str] = &[
    "/health",
    "/docs",
    "/assets",
    "/icon.png",
    "/api/share/",
    "/api/cast-media/",
    "/watch/",
];

/// Endpoints under localhost-only prefixes that should still be accessible from network.
const LOCALHOST_EXEMPTIONS: &[&str] = &[
    "/api/network/verify-handshake",
    "/api/settings/svp",
];

/// Write HTTP methods that require elevated access.
const WRITE_METHODS: &[&str] = &["POST", "PUT", "PATCH", "DELETE"];

/// Classify an IP address into an access level.
pub fn classify_ip(ip: &std::net::IpAddr) -> &'static str {
    match ip {
        std::net::IpAddr::V4(v4) => {
            if v4.is_loopback() {
                "localhost"
            } else if v4.is_private() || v4.is_link_local() {
                "local_network"
            } else {
                "public"
            }
        }
        std::net::IpAddr::V6(v6) => {
            if v6.is_loopback() {
                "localhost"
            } else {
                // Check for IPv4-mapped IPv6 (::ffff:127.0.0.1, etc.)
                if let Some(v4) = v6.to_ipv4_mapped() {
                    if v4.is_loopback() {
                        return "localhost";
                    }
                    if v4.is_private() || v4.is_link_local() {
                        return "local_network";
                    }
                }
                "public"
            }
        }
    }
}

// ─── Layer ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AccessControlLayer;

impl<S> Layer<S> for AccessControlLayer {
    type Service = AccessControlService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AccessControlService { inner }
    }
}

// ─── Service ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AccessControlService<S> {
    inner: S,
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

        Box::pin(async move {
            let path = req.uri().path().to_string();
            let method = req.method().as_str().to_string();

            // Always allow OPTIONS (CORS preflight)
            if method == "OPTIONS" {
                return inner.call(req).await;
            }

            // Skip access control for exempt endpoints
            for prefix in EXEMPT_PREFIXES {
                if path.starts_with(prefix) {
                    return inner.call(req).await;
                }
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

            // Block localhost-only endpoints for non-localhost access
            for prefix in LOCALHOST_ONLY_PREFIXES {
                if path.starts_with(prefix) {
                    let is_exempted = LOCALHOST_EXEMPTIONS
                        .iter()
                        .any(|exempt| path == *exempt || path.starts_with(&format!("{}/", exempt)));

                    if !is_exempted {
                        let response = (
                            StatusCode::FORBIDDEN,
                            axum::Json(serde_json::json!({
                                "error": "This endpoint is only accessible from localhost",
                                "detail": format!(
                                    "Access to {} requires direct access to the machine running LocalBooru",
                                    prefix
                                )
                            })),
                        )
                            .into_response();
                        return Ok(response);
                    }
                }
            }

            // Block write operations for public internet
            if WRITE_METHODS.contains(&method.as_str()) && access_level == "public" {
                let response = (
                    StatusCode::FORBIDDEN,
                    axum::Json(serde_json::json!({
                        "error": "Write operations require local access",
                        "detail": "Public internet access is read-only."
                    })),
                )
                    .into_response();
                return Ok(response);
            }

            inner.call(req).await
        })
    }
}
