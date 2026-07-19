use axum::extract::{Path as AxumPath, State};
use axum::response::Json;
use axum::routing::{any, get, post};
use axum::Router;
use serde_json::json;

use crate::addons::proxy::proxy_to_addon;
use crate::server::error::AppError;
use crate::server::state::AppState;

/// Build the /api/addons router with management and proxy endpoints.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_addons))
        .route("/{addon_id}", get(get_addon))
        .route("/{addon_id}/install", post(install_addon))
        .route("/{addon_id}/update", post(repair_addon))
        .route("/{addon_id}/uninstall", post(uninstall_addon))
        .route("/{addon_id}/start", post(start_addon))
        .route("/{addon_id}/stop", post(stop_addon))
        // Wildcard proxy: forward everything under /{addon_id}/api/* to the sidecar
        .route("/{addon_id}/api/{*rest}", any(proxy_to_addon))
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/addons — List all addons with their current status.
async fn list_addons(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    let addons = state.addon_manager().list_addons();
    Ok(Json(json!({ "addons": addons })))
}

/// GET /api/addons/{addon_id} — Get info for a single addon.
async fn get_addon(
    State(state): State<AppState>,
    AxumPath(addon_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let addon = state
        .addon_manager()
        .get_addon(&addon_id)
        .ok_or_else(|| AppError::NotFound(format!("Addon '{}' not found", addon_id)))?;

    let status = state.addon_manager().get_addon_status(&addon_id);

    Ok(Json(json!({
        "addon": addon,
        "status": status,
    })))
}

/// POST /api/addons/{addon_id}/install — Install an addon (create venv, install deps).
///
/// This is a blocking operation (runs pip install) so we use spawn_blocking.
async fn install_addon(
    State(state): State<AppState>,
    AxumPath(addon_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let requires_stop = state
        .addon_manager()
        .get_addon(&addon_id)
        .ok_or_else(|| AppError::NotFound(format!("Addon '{}' not found", addon_id)))?
        .requires_start;
    let mut restart_after_install = false;
    if requires_stop {
        restart_after_install = state
            .addon_manager()
            .begin_repair(&addon_id)
            .map_err(AppError::Internal)?;
        if restart_after_install {
            if let Err(error) = state.addon_manager().stop_addon_for_repair(&addon_id).await {
                state
                    .addon_manager()
                    .finish_failed_repair(&addon_id, error.clone());
                return Err(AppError::Internal(error));
            }
        }
    }

    let state_clone = state.clone();
    let id = addon_id.clone();

    let install_result = tokio::task::spawn_blocking(move || {
        state_clone
            .addon_manager()
            .install_addon(&id)
            .map_err(|e| AppError::Internal(format!("Failed to install addon '{}': {}", id, e)))
    })
    .await?;
    if let Err(error) = install_result {
        if requires_stop {
            state
                .addon_manager()
                .finish_failed_repair(&addon_id, error.to_string());
        }
        if restart_after_install {
            if let Err(restart_error) = state.addon_manager().start_addon(&addon_id).await {
                return Err(AppError::Internal(format!(
                    "{}; restoring the running add-on also failed: {}",
                    error, restart_error
                )));
            }
        }
        return Err(error);
    }

    if restart_after_install {
        state
            .addon_manager()
            .start_addon(&addon_id)
            .await
            .map_err(AppError::Internal)?;
    }

    Ok(Json(json!({
        "status": "installed",
        "addon_id": addon_id,
    })))
}

/// POST /api/addons/{addon_id}/update — Repair an add-on after dependency changes.
async fn repair_addon(
    State(state): State<AppState>,
    AxumPath(addon_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let was_running = state
        .addon_manager()
        .begin_repair(&addon_id)
        .map_err(AppError::Internal)?;
    if was_running {
        if let Err(error) = state.addon_manager().stop_addon_for_repair(&addon_id).await {
            state
                .addon_manager()
                .finish_failed_repair(&addon_id, error.clone());
            return Err(AppError::Internal(error));
        }
    }

    let state_clone = state.clone();
    let id = addon_id.clone();
    let repair_result = tokio::task::spawn_blocking(move || {
        state_clone
            .addon_manager()
            .repair_addon(&id)
            .map_err(|e| AppError::Internal(format!("Failed to repair addon '{}': {}", id, e)))
    })
    .await?;

    if let Err(error) = repair_result {
        state
            .addon_manager()
            .finish_failed_repair(&addon_id, error.to_string());
        if was_running {
            if let Err(restart_error) = state.addon_manager().start_addon(&addon_id).await {
                return Err(AppError::Internal(format!(
                    "{}; restoring the running add-on also failed: {}",
                    error, restart_error
                )));
            }
        }
        return Err(error);
    }
    if was_running {
        state
            .addon_manager()
            .start_addon(&addon_id)
            .await
            .map_err(AppError::Internal)?;
    }

    Ok(Json(json!({ "status": "updated", "addon_id": addon_id })))
}

/// POST /api/addons/{addon_id}/uninstall — Remove an addon.
///
/// Stops the addon first if running, then removes its directory.
async fn uninstall_addon(
    State(state): State<AppState>,
    AxumPath(addon_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let requires_stop = state
        .addon_manager()
        .get_addon(&addon_id)
        .ok_or_else(|| AppError::NotFound(format!("Addon '{}' not found", addon_id)))?
        .requires_start;
    state
        .addon_manager()
        .begin_repair(&addon_id)
        .map_err(AppError::Internal)?;
    if requires_stop {
        if let Err(error) = state.addon_manager().stop_addon_for_repair(&addon_id).await {
            state
                .addon_manager()
                .finish_failed_repair(&addon_id, error.clone());
            return Err(AppError::Internal(error));
        }
    }

    // Uninstall (blocking: removes venv directory)
    let state_clone = state.clone();
    let id = addon_id.clone();

    let uninstall_result = tokio::task::spawn_blocking(move || {
        state_clone
            .addon_manager()
            .uninstall_addon(&id)
            .map_err(|e| AppError::Internal(format!("Failed to uninstall addon '{}': {}", id, e)))
    })
    .await?;
    if let Err(error) = uninstall_result {
        state
            .addon_manager()
            .finish_failed_repair(&addon_id, error.to_string());
        return Err(error);
    }

    Ok(Json(json!({
        "status": "uninstalled",
        "addon_id": addon_id,
    })))
}

/// POST /api/addons/{addon_id}/start — Start an addon's sidecar process.
async fn start_addon(
    State(state): State<AppState>,
    AxumPath(addon_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .addon_manager()
        .start_addon(&addon_id)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to start addon '{}': {}", addon_id, e)))?;

    Ok(Json(json!({
        "status": "started",
        "addon_id": addon_id,
    })))
}

/// POST /api/addons/{addon_id}/stop — Stop an addon's sidecar process.
async fn stop_addon(
    State(state): State<AppState>,
    AxumPath(addon_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .addon_manager()
        .stop_addon(&addon_id)
        .await
        .map_err(|e| AppError::Internal(format!("Failed to stop addon '{}': {}", addon_id, e)))?;

    Ok(Json(json!({
        "status": "stopped",
        "addon_id": addon_id,
    })))
}
