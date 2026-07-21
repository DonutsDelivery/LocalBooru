use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::wd14_sidecar::{run_operation, Wd14Operation, Wd14Request, Wd14Response};

const ADDON_ID: &str = "wd14-sidecar";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/import", post(import_sidecars))
        .route("/absorb", post(absorb_sidecars))
        .route("/export", post(export_sidecars))
}

async fn import_sidecars(
    State(state): State<AppState>,
    Json(request): Json<Wd14Request>,
) -> Result<Json<Wd14Response>, AppError> {
    execute(state, Wd14Operation::Import, request).await
}

async fn absorb_sidecars(
    State(state): State<AppState>,
    Json(request): Json<Wd14Request>,
) -> Result<Json<Wd14Response>, AppError> {
    execute(state, Wd14Operation::Absorb, request).await
}

async fn export_sidecars(
    State(state): State<AppState>,
    Json(request): Json<Wd14Request>,
) -> Result<Json<Wd14Response>, AppError> {
    execute(state, Wd14Operation::Export, request).await
}

async fn execute(
    state: AppState,
    operation: Wd14Operation,
    request: Wd14Request,
) -> Result<Json<Wd14Response>, AppError> {
    ensure_installed(&state)?;
    let response =
        tokio::task::spawn_blocking(move || run_operation(&state, operation, request)).await??;
    Ok(Json(response))
}

fn ensure_installed(state: &AppState) -> Result<(), AppError> {
    let installed = state
        .addon_manager()
        .get_addon(ADDON_ID)
        .is_some_and(|addon| addon.installed);
    if installed {
        Ok(())
    } else {
        Err(AppError::ServiceUnavailable(
            "Install WD14 Text Sidecars before running sidecar operations".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    use axum::body::Body;
    use axum::extract::ConnectInfo;
    use axum::http::{Request, StatusCode};
    use rusqlite::params;
    use tower::ServiceExt;

    use super::*;
    use crate::server::build_router;

    fn request(uri: &str, body: String, ip: Ipv4Addr) -> Request<Body> {
        let mut request = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::new(IpAddr::V4(ip), 12345)));
        request
    }

    fn empty_body() -> String {
        r#"{"directories":[],"overwrite":false}"#.to_string()
    }

    // AC: @wd14-text-sidecars ac-installed-gate
    #[tokio::test]
    async fn uninstalled_http_route_rejects_without_reading_or_mutating_sidecars() {
        let temp = tempfile::tempdir().unwrap();
        let media = temp.path().join("media");
        std::fs::create_dir(&media).unwrap();
        let state = AppState::new(&temp.path().join("data"), 0).unwrap();
        let library = state.resolve_library(None).unwrap();
        let library_id = library.uuid.clone();
        library
            .main_pool
            .get()
            .unwrap()
            .execute(
                "INSERT INTO watch_directories (id, path) VALUES (1, ?1)",
                params![media.to_string_lossy()],
            )
            .unwrap();
        let media_path = media.join("sample.jpg");
        let sidecar_path = media.join("sample.txt");
        std::fs::write(&media_path, b"media").unwrap();
        std::fs::write(&sidecar_path, "should_not_import").unwrap();
        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (1, 'sample.jpg', 'hash')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO image_files (image_id, original_path) VALUES (1, ?1)",
                params![media_path.to_string_lossy()],
            )
            .unwrap();
        drop(connection);
        drop(library);

        let body = serde_json::json!({
            "directories": [{"library_id": library_id, "directory_id": 1}],
            "overwrite": false
        })
        .to_string();
        let response = build_router(state.clone(), None)
            .oneshot(request(
                "/api/settings/wd14-sidecar/import",
                body,
                Ipv4Addr::LOCALHOST,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            std::fs::read_to_string(&sidecar_path).unwrap(),
            "should_not_import"
        );
        let library = state.resolve_library(None).unwrap();
        let count: i64 = library
            .directory_db
            .get_pool(1)
            .unwrap()
            .get()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM image_tags", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    // AC: @wd14-text-sidecars ac-builtin-install
    #[tokio::test]
    async fn installed_builtin_mounts_all_http_routes_without_a_process() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::new(temp.path(), 0).unwrap();
        state.addon_manager().install_addon(ADDON_ID).unwrap();
        let addon = state.addon_manager().get_addon(ADDON_ID).unwrap();
        assert!(addon.installed);
        assert!(addon.port.is_none());
        assert!(!addon.requires_start);
        assert!(!temp.path().join("addons/wd14-sidecar/venv").exists());
        assert!(state.addon_manager().start_addon(ADDON_ID).await.is_err());

        for operation in ["import", "absorb", "export"] {
            let response = build_router(state.clone(), None)
                .oneshot(request(
                    &format!("/api/settings/wd14-sidecar/{operation}"),
                    empty_body(),
                    Ipv4Addr::LOCALHOST,
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn settings_prefix_keeps_wd14_routes_localhost_only() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::new(temp.path(), 0).unwrap();
        state.addon_manager().install_addon(ADDON_ID).unwrap();
        let response = build_router(state, None)
            .oneshot(request(
                "/api/settings/wd14-sidecar/import",
                empty_body(),
                Ipv4Addr::new(192, 168, 1, 20),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }
}
