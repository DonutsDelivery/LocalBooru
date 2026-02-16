use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Json};

/// Application error type that converts to proper HTTP responses.
#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    BadRequest(String),
    Unauthorized(String),
    Forbidden(String),
    TooManyRequests(String),
    Internal(String),
    ServiceUnavailable(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::NotFound(m) => write!(f, "Not found: {}", m),
            AppError::BadRequest(m) => write!(f, "Bad request: {}", m),
            AppError::Unauthorized(m) => write!(f, "Unauthorized: {}", m),
            AppError::Forbidden(m) => write!(f, "Forbidden: {}", m),
            AppError::TooManyRequests(m) => write!(f, "Too many requests: {}", m),
            AppError::Internal(m) => write!(f, "Internal error: {}", m),
            AppError::ServiceUnavailable(m) => write!(f, "Service unavailable: {}", m),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::NotFound(m) => (StatusCode::NOT_FOUND, m),
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            AppError::Unauthorized(m) => (StatusCode::UNAUTHORIZED, m),
            AppError::Forbidden(m) => (StatusCode::FORBIDDEN, m),
            AppError::TooManyRequests(m) => (StatusCode::TOO_MANY_REQUESTS, m),
            AppError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
            AppError::ServiceUnavailable(m) => (StatusCode::SERVICE_UNAVAILABLE, m),
        };
        (status, Json(serde_json::json!({"detail": message}))).into_response()
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(e: rusqlite::Error) -> Self {
        AppError::Internal(format!("Database error: {}", e))
    }
}

impl From<r2d2::Error> for AppError {
    fn from(e: r2d2::Error) -> Self {
        AppError::Internal(format!("Connection pool error: {}", e))
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Internal(format!("IO error: {}", e))
    }
}

impl From<tokio::task::JoinError> for AppError {
    fn from(e: tokio::task::JoinError) -> Self {
        AppError::Internal(format!("Task join error: {}", e))
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for AppError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        AppError::Internal(e.to_string())
    }
}

impl From<Box<dyn std::error::Error>> for AppError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        AppError::Internal(e.to_string())
    }
}
