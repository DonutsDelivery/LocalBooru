use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response, Json},
};
use serde::{Deserialize, Serialize};

use crate::server::state::AppState;

// ─── JWT shared types ─────────────────────────────────────────────────────────

/// JWT claims payload. Shared between auth middleware and user routes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub user_id: i64,
    pub username: String,
    pub access_level: String,
    pub can_write: bool,
    pub exp: i64,
}

/// Create a signed JWT token for the given user.
pub fn create_jwt(
    user_id: i64,
    username: &str,
    access_level: &str,
    can_write: bool,
    secret: &str,
) -> Result<String, crate::server::error::AppError> {
    let exp = chrono::Utc::now().timestamp() + 86400 * 30; // 30 days
    let claims = Claims {
        user_id,
        username: username.into(),
        access_level: access_level.into(),
        can_write,
        exp,
    };

    jsonwebtoken::encode(
        &jsonwebtoken::Header::default(),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|e| crate::server::error::AppError::Internal(format!("JWT error: {}", e)))
}

/// Decode and validate a JWT token, returning the claims if valid.
pub fn decode_jwt(token: &str, secret: &str) -> Result<Claims, crate::server::error::AppError> {
    let mut validation = jsonwebtoken::Validation::default();
    validation.validate_exp = true;

    jsonwebtoken::decode::<Claims>(
        token,
        &jsonwebtoken::DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .map(|data| data.claims)
    .map_err(|_| {
        crate::server::error::AppError::Unauthorized("Invalid or expired token".into())
    })
}

// ─── Auth extractor ───────────────────────────────────────────────────────────

/// Authenticated user information extracted from a valid JWT Bearer token.
///
/// Use this as a handler parameter to require authentication:
///
/// ```ignore
/// async fn protected_route(user: AuthUser) -> impl IntoResponse { ... }
/// ```
///
/// Or wrap in `Option` for optional authentication:
///
/// ```ignore
/// async fn optional_auth(user: Option<AuthUser>) -> impl IntoResponse { ... }
/// ```
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: i64,
    pub username: String,
    pub access_level: String,
    pub can_write: bool,
}

impl AuthUser {
    fn from_claims(claims: Claims) -> Self {
        Self {
            user_id: claims.user_id,
            username: claims.username,
            access_level: claims.access_level,
            can_write: claims.can_write,
        }
    }
}

/// Rejection type for AuthUser extraction failures.
pub struct AuthRejection {
    message: String,
}

impl IntoResponse for AuthRejection {
    fn into_response(self) -> Response {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "detail": self.message
            })),
        )
            .into_response()
    }
}

impl FromRequestParts<AppState> for AuthUser {
    type Rejection = AuthRejection;

    fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> impl std::future::Future<Output = Result<Self, Self::Rejection>> + Send {
        let secret = state.jwt_secret().to_owned();
        async move {
            // Extract the Authorization header
            let auth_header = parts
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| AuthRejection {
                    message: "Missing Authorization header".into(),
                })?;

            // Must be Bearer <token>
            let token = auth_header
                .strip_prefix("Bearer ")
                .or_else(|| auth_header.strip_prefix("bearer "))
                .ok_or_else(|| AuthRejection {
                    message: "Invalid Authorization header format. Expected: Bearer <token>".into(),
                })?;

            // Decode and validate the JWT
            let claims = decode_jwt(token, &secret).map_err(|e| AuthRejection {
                message: format!("{}", e),
            })?;

            Ok(AuthUser::from_claims(claims))
        }
    }
}

// Optional auth: use `Option<AuthUser>` as a handler parameter.
// Axum automatically provides `Option<T>` extraction for any `T: FromRequestParts`,
// returning `None` when extraction fails (no token / invalid token).
