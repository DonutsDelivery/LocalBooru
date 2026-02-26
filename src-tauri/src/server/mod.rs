pub mod state;
pub mod middleware;
pub mod error;
pub mod utils;

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{Method, StatusCode, header, HeaderMap, HeaderValue},
    response::{Json, IntoResponse, Response},
    routing::get,
};
use tower_http::cors::{CorsLayer, AllowOrigin, Any};
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::services::{ServeDir, ServeFile};

use self::middleware::AccessControlLayer;
use self::state::AppState;

/// Build the full axum router with all routes, middleware, and static file serving.
pub fn build_router(state: AppState, frontend_dir: Option<PathBuf>) -> Router {
    // CORS — restrict to known origins (Tauri, Capacitor, dev servers)
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _parts| {
            let origin_str = match origin.to_str() {
                Ok(s) => s,
                Err(_) => return false,
            };
            origin_str == "tauri://localhost"
                || origin_str == "https://tauri.localhost"
                || origin_str == "http://tauri.localhost"
                || origin_str == "http://localhost"
                || origin_str == "http://127.0.0.1"
                || origin_str.starts_with("http://localhost:")
                || origin_str.starts_with("http://127.0.0.1:")
        }))
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(Any)
        .expose_headers([
            "Content-Range".parse().unwrap(),
            "Content-Length".parse().unwrap(),
            "Accept-Ranges".parse().unwrap(),
        ]);

    // API routes
    let api = Router::new()
        .route("/api", get(api_root))
        .route("/health", get(health))
        .nest("/api/images", crate::routes::images::router())
        .nest("/api/tags", crate::routes::tags::router())
        .nest("/api/directories", crate::routes::directories::router())
        .nest("/api/library", crate::routes::library::router())
        .nest("/api/collections", crate::routes::collections::router())
        .nest("/api/users", crate::routes::users::router())
        .nest("/api/settings", crate::routes::settings::router())
        .nest("/api/settings/migration", crate::routes::migration::router())
        .nest("/api/settings/svp/web", crate::routes::svp_web::router())
        .nest("/api/settings/models", crate::routes::models::router())
        .nest("/api/network", crate::routes::network::router())
        .nest("/api/watch-history", crate::routes::watch_history::router())
        .nest("/api/app-update", crate::routes::app_update::router())
        .nest("/api/addons", crate::routes::addons::router())
        .nest("/api/cast", crate::routes::cast::router())
        .nest("/api/share", crate::routes::share::router())
        .nest("/api/libraries", crate::routes::libraries::router());

    // Serve thumbnails as static files
    let thumbnails_dir = state.thumbnails_dir();
    let thumbnails_service = ServeDir::new(&thumbnails_dir);

    // Prevent browsers from caching API responses (stale data causes ghost images)
    let no_cache = SetResponseHeaderLayer::if_not_present(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );

    // Remote proxy route — forwards requests to a remote server (mobile mode).
    // This bypasses access control since it's only called from the local WebView.
    let proxy_router = Router::new()
        .fallback(remote_proxy_handler)
        .with_state(state.clone());

    let mut app = api
        .layer(no_cache)
        .nest_service("/thumbnails", thumbnails_service)
        .layer(AccessControlLayer {
            jwt_secret: state.jwt_secret().to_string(),
        })
        .with_state(state)
        .nest("/remote", proxy_router)
        // CORS must be outermost so it covers both API and /remote routes
        .layer(cors);

    // Serve frontend SPA if dist directory exists
    if let Some(ref frontend) = frontend_dir {
        if frontend.exists() {
            let assets_dir = frontend.join("assets");
            if assets_dir.exists() {
                app = app.nest_service("/assets", ServeDir::new(&assets_dir));
            }

            let icon_path = frontend.join("icon.png");
            if icon_path.exists() {
                app = app.route_service("/icon.png", ServeFile::new(&icon_path));
            }

            // SPA catch-all: serve index.html for any unmatched route
            let index_path = frontend.join("index.html");
            if index_path.exists() {
                app = app.fallback_service(
                    ServeDir::new(frontend).fallback(ServeFile::new(&index_path)),
                );
            }
        }
    }

    app
}

/// Start the axum HTTP server.
pub async fn start_server(
    state: AppState,
    frontend_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let port = state.port();
    // Bind to 0.0.0.0 if local network access is enabled, otherwise localhost only
    let bind_addr = if state.is_lan_enabled() {
        [0, 0, 0, 0]
    } else {
        [127, 0, 0, 1]
    };
    let addr = SocketAddr::from((bind_addr, port));
    log::info!("[Server] Starting axum server on {}", addr);

    let app = build_router(state.clone(), frontend_dir);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("[Server] Listening on {}", addr);
    state.set_server_ready(true);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;

    Ok(())
}

// ─── Route handlers ─────────────────────────────────────────────────────────

async fn api_root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "LocalBooru",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running"
    }))
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy"
    }))
}

/// Reverse proxy handler: forwards requests to the configured remote server.
/// Used on mobile to avoid mixed-content blocks in the WebView (https://tauri.localhost -> http://...).
async fn remote_proxy_handler(
    State(state): State<AppState>,
    req: Request,
) -> Response {
    let proxy = state.get_remote_proxy().await;
    let (base_url, token) = match proxy {
        Some(p) => p,
        None => {
            return (StatusCode::BAD_GATEWAY, "No remote server configured").into_response();
        }
    };

    // Build target URL: /remote/api/images -> http://remote:8790/api/images
    let path = req.uri().path();
    let query = req.uri().query().map(|q| format!("?{}", q)).unwrap_or_default();
    let target_url = format!("{}{}{}", base_url, path, query);

    // Forward the request
    let client = state.http_client();
    let method = req.method().clone();
    let mut builder = client.request(method, &target_url);

    // Forward relevant headers
    for (name, value) in req.headers() {
        // Skip host and connection headers
        if name == header::HOST || name == header::CONNECTION {
            continue;
        }
        if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
            if let Ok(n) = reqwest::header::HeaderName::from_bytes(name.as_ref()) {
                builder = builder.header(n, v);
            }
        }
    }

    // Add auth token if configured
    if let Some(ref tok) = token {
        builder = builder.header("Authorization", format!("Bearer {}", tok));
    }

    // Forward request body
    let body_bytes = match axum::body::to_bytes(req.into_body(), 100 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("Failed to read request body: {}", e)).into_response();
        }
    };
    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }

    // Send the proxied request
    match builder.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut headers = HeaderMap::new();
            for (name, value) in resp.headers() {
                if let Ok(v) = HeaderValue::from_bytes(value.as_bytes()) {
                    if let Ok(n) = axum::http::header::HeaderName::from_bytes(name.as_ref()) {
                        headers.insert(n, v);
                    }
                }
            }
            let body = resp.bytes().await.unwrap_or_default();
            (status, headers, Body::from(body)).into_response()
        }
        Err(e) => {
            log::error!("[Proxy] Request to {} failed: {}", target_url, e);
            (StatusCode::BAD_GATEWAY, format!("Proxy error: {}", e)).into_response()
        }
    }
}
