pub mod state;
pub mod middleware;
pub mod error;
pub mod utils;

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::{
    Router,
    http::{Method, header},
    response::Json,
    routing::get,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::services::{ServeDir, ServeFile};

use self::middleware::AccessControlLayer;
use self::state::AppState;

/// Build the full axum router with all routes, middleware, and static file serving.
pub fn build_router(state: AppState, frontend_dir: Option<PathBuf>) -> Router {
    // CORS — allow all origins (access control handled by middleware)
    let cors = CorsLayer::new()
        .allow_origin(Any)
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

    let mut app = api
        .layer(no_cache)
        .nest_service("/thumbnails", thumbnails_service)
        .layer(AccessControlLayer)
        .layer(cors)
        .with_state(state);

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

    let app = build_router(state, frontend_dir);
    let listener = tokio::net::TcpListener::bind(addr).await?;
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
