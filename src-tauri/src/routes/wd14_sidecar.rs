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
    use super::*;

    fn empty_request() -> Wd14Request {
        Wd14Request {
            directories: Vec::new(),
            overwrite: false,
        }
    }

    // AC: @wd14-text-sidecars ac-installed-gate
    #[tokio::test]
    async fn uninstalled_addon_rejects_before_request_processing() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::new(temp.path(), 0).unwrap();

        let error = execute(state, Wd14Operation::Import, empty_request())
            .await
            .unwrap_err();
        assert!(matches!(error, AppError::ServiceUnavailable(_)));
    }

    // AC: @wd14-text-sidecars ac-builtin-install
    #[tokio::test]
    async fn installed_builtin_exposes_all_operations_without_a_process() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::new(temp.path(), 0).unwrap();
        state.addon_manager().install_addon(ADDON_ID).unwrap();
        let addon = state.addon_manager().get_addon(ADDON_ID).unwrap();
        assert!(addon.installed);
        assert!(addon.port.is_none());

        for operation in [
            Wd14Operation::Import,
            Wd14Operation::Absorb,
            Wd14Operation::Export,
        ] {
            let error = execute(state.clone(), operation, empty_request())
                .await
                .unwrap_err();
            assert!(matches!(error, AppError::BadRequest(_)));
        }
    }
}
