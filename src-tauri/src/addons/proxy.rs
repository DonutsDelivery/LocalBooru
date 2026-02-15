use axum::body::Body;
use axum::extract::{Path as AxumPath, State, Request};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::server::error::AppError;
use crate::server::state::AppState;

/// Proxy handler that forwards requests to an addon's sidecar process.
///
/// Mounted as: /api/addons/:addon_id/api/*rest
///
/// This takes the incoming request and forwards it to the addon's local HTTP
/// server running on 127.0.0.1:{port}/{rest}?{query}, preserving method,
/// headers, and body.
pub async fn proxy_to_addon(
    State(state): State<AppState>,
    AxumPath((addon_id, rest)): AxumPath<(String, String)>,
    request: Request,
) -> Result<Response, AppError> {
    // 1. Get the addon's base URL (checks addon is running and has a port)
    let base_url = state
        .addon_manager()
        .addon_url(&addon_id)
        .ok_or_else(|| {
            AppError::ServiceUnavailable(format!(
                "Addon '{}' is not running or not found",
                addon_id
            ))
        })?;

    // 2. Build target URL: http://127.0.0.1:{port}/{rest}?{original_query}
    let uri = request.uri().clone();
    let query_string = uri.query().unwrap_or("");
    let target_url = if query_string.is_empty() {
        format!("{}/{}", base_url.trim_end_matches('/'), rest)
    } else {
        format!("{}/{}?{}", base_url.trim_end_matches('/'), rest, query_string)
    };

    // 3. Extract pieces from the incoming request
    let method = request.method().clone();
    let headers = request.headers().clone();
    let body_bytes = axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024)
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to read request body: {}", e)))?;

    // 4. Build the outgoing reqwest request
    let client = reqwest::Client::new();
    let mut outgoing = client.request(method, &target_url);

    // Forward Content-Type header
    if let Some(ct) = headers.get("content-type") {
        if let Ok(ct_str) = ct.to_str() {
            outgoing = outgoing.header("content-type", ct_str);
        }
    }

    // Forward Authorization header (if present)
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            outgoing = outgoing.header("authorization", auth_str);
        }
    }

    // Forward Accept header (if present)
    if let Some(accept) = headers.get("accept") {
        if let Ok(accept_str) = accept.to_str() {
            outgoing = outgoing.header("accept", accept_str);
        }
    }

    // Attach body (for POST, PUT, PATCH, etc.)
    if !body_bytes.is_empty() {
        outgoing = outgoing.body(body_bytes);
    }

    // 5. Send the request to the addon sidecar
    let response = outgoing.send().await.map_err(|e| {
        AppError::ServiceUnavailable(format!(
            "Failed to reach addon '{}': {}",
            addon_id, e
        ))
    })?;

    // 6. Convert the reqwest response back to an axum response
    let status = StatusCode::from_u16(response.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut response_headers = HeaderMap::new();

    // Preserve Content-Type from the addon's response
    if let Some(ct) = response.headers().get("content-type") {
        if let Ok(ct_val) = HeaderValue::from_bytes(ct.as_bytes()) {
            response_headers.insert("content-type", ct_val);
        }
    }

    // Preserve Content-Disposition (for file downloads)
    if let Some(cd) = response.headers().get("content-disposition") {
        if let Ok(cd_val) = HeaderValue::from_bytes(cd.as_bytes()) {
            response_headers.insert("content-disposition", cd_val);
        }
    }

    let response_body = response.bytes().await.map_err(|e| {
        AppError::Internal(format!(
            "Failed to read response from addon '{}': {}",
            addon_id, e
        ))
    })?;

    Ok((status, response_headers, Body::from(response_body)).into_response())
}
