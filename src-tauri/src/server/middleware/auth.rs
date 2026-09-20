use axum::{
    extract::FromRequestParts,
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};

use crate::db::pool::DbPool;
use crate::server::state::AppState;

// ─── JWT shared types ─────────────────────────────────────────────────────────

/// JWT claims payload. Shared between auth middleware and user routes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub user_id: i64,
    pub username: String,
    pub access_level: String,
    pub can_write: bool,
    /// Ordinary sessions remain time-limited. Paired-device credentials omit
    /// this claim and are instead valid until their database record is revoked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exp: Option<i64>,
    /// Token scope. `None` (absent) = full session token. `Some("media")` =
    /// short-lived, read-only token valid only for GET requests on media routes
    /// (used in `<img>/<video>` src URLs so the 30-day session JWT never appears
    /// in a URL). `#[serde(default)]` keeps pre-existing tokens (no field) valid.
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub device_id: Option<String>,
}

/// Scope value identifying a read-only, media-routes-only token.
pub const MEDIA_SCOPE: &str = "media";

impl Claims {
    /// True when this token is restricted to read-only media access.
    pub fn is_media_scoped(&self) -> bool {
        self.scope.as_deref() == Some(MEDIA_SCOPE)
    }
}

/// Revalidate the durable identity behind a token and refresh mutable user
/// permissions from the database.
pub(crate) fn refresh_claim_identity(mut claims: Claims, db: &DbPool) -> Option<Claims> {
    let conn = db.get().ok()?;
    if let Some(device_id) = claims.device_id.as_deref() {
        let active = conn
            .query_row(
                "SELECT revoked_at IS NULL FROM paired_devices WHERE device_id = ?1",
                rusqlite::params![device_id],
                |row| row.get::<_, bool>(0),
            )
            .ok()?;
        return active.then_some(claims);
    }

    // Compatibility for already-issued legacy QR credentials. New QR scans are
    // persisted as revocable devices below, but existing 30-day phone tokens
    // must remain usable long enough to authorize the replacement pairing.
    if claims.user_id == 0 && claims.username == "qr_paired_device" {
        return Some(claims);
    }

    let (username, is_active, access_level, can_write): (String, bool, String, bool) = conn
        .query_row(
            "SELECT username, is_active, access_level, can_write FROM users WHERE id = ?1",
            rusqlite::params![claims.user_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .ok()?;
    if !is_active {
        return None;
    }
    claims.username = username;
    claims.access_level = access_level;
    claims.can_write = can_write;
    Some(claims)
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
        exp: Some(exp),
        scope: None,
        device_id: None,
    };

    jsonwebtoken::encode(
        &jsonwebtoken::Header::default(),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|e| crate::server::error::AppError::Internal(format!("JWT error: {}", e)))
}

/// Create a full session token bound to one revocable paired-device record.
pub fn create_device_jwt(
    device_id: &str,
    display_name: &str,
    secret: &str,
) -> Result<String, crate::server::error::AppError> {
    let claims = Claims {
        user_id: 0,
        username: display_name.into(),
        access_level: "local_network".into(),
        can_write: true,
        exp: None,
        scope: None,
        device_id: Some(device_id.into()),
    };
    jsonwebtoken::encode(
        &jsonwebtoken::Header::default(),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|e| crate::server::error::AppError::Internal(format!("JWT error: {}", e)))
}

/// Lifetime of a media-scoped token, in seconds (24h).
pub const MEDIA_TOKEN_TTL_SECS: i64 = 86400;

/// Create a short-lived, read-only media token for use in `<img>/<video>` src
/// URLs (the `?token=` query param). Forces `can_write = false` and
/// `scope = Some("media")`, so it is rejected by `AuthUser` and by the Bearer
/// path of the access-control middleware — it is only honored for GET requests
/// on media routes. Derived from a caller that already holds a valid session JWT.
pub fn create_media_jwt(
    user_id: i64,
    username: &str,
    access_level: &str,
    device_id: Option<&str>,
    secret: &str,
) -> Result<String, crate::server::error::AppError> {
    let exp = chrono::Utc::now().timestamp() + MEDIA_TOKEN_TTL_SECS;
    let claims = Claims {
        user_id,
        username: username.into(),
        access_level: access_level.into(),
        can_write: false,
        exp: Some(exp),
        scope: Some(MEDIA_SCOPE.into()),
        device_id: device_id.map(str::to_owned),
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
    // Paired-device JWTs deliberately have no expiry. They are checked against
    // `paired_devices` on every request, so explicit revocation is immediate.
    // When an `exp` claim is present (user/media tokens), jsonwebtoken still
    // validates it normally.
    validation.required_spec_claims.remove("exp");

    jsonwebtoken::decode::<Claims>(
        token,
        &jsonwebtoken::DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .map(|data| data.claims)
    .map_err(|_| crate::server::error::AppError::Unauthorized("Invalid or expired token".into()))
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
    pub device_id: Option<String>,
}

impl AuthUser {
    fn from_claims(claims: Claims) -> Self {
        Self {
            user_id: claims.user_id,
            username: claims.username,
            access_level: claims.access_level,
            can_write: claims.can_write,
            device_id: claims.device_id,
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

            // Only unscoped session tokens are accepted as Bearer credentials.
            // Media and any future/unknown scopes fail closed here.
            if claims.scope.is_some() {
                return Err(AuthRejection {
                    message: "Scoped token cannot be used for authenticated requests".into(),
                });
            }
            let claims =
                refresh_claim_identity(claims, state.main_db()).ok_or_else(|| AuthRejection {
                    message: "Credential identity is disabled or revoked".into(),
                })?;
            Ok(AuthUser::from_claims(claims))
        }
    }
}

// Optional auth: use `Option<AuthUser>` as a handler parameter.
// Axum automatically provides `Option<T>` extraction for any `T: FromRequestParts`,
// returning `None` when extraction fails (no token / invalid token).

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request, routing::get, Router};
    use tower::ServiceExt;

    async fn protected(_: AuthUser) -> StatusCode {
        StatusCode::OK
    }

    async fn write_protected(user: AuthUser) -> StatusCode {
        if user.can_write {
            StatusCode::OK
        } else {
            StatusCode::FORBIDDEN
        }
    }

    #[tokio::test]
    async fn paired_device_revocation_invalidates_its_existing_jwt() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-device-auth-test-{}",
            uuid::Uuid::new_v4()
        ));
        let state = AppState::new(&data_dir, 0).unwrap();
        let device_id = uuid::Uuid::new_v4().to_string();
        state
            .main_db()
            .get()
            .unwrap()
            .execute(
                "INSERT INTO paired_devices (device_id, display_name, public_key_spki, public_key_fingerprint, last_seen_at) VALUES (?1, 'Bedroom Desktop', 'spki', 'fingerprint', datetime('now'))",
                rusqlite::params![device_id],
            )
            .unwrap();
        let token = create_device_jwt(&device_id, "Bedroom Desktop", state.jwt_secret()).unwrap();
        assert_eq!(decode_jwt(&token, state.jwt_secret()).unwrap().exp, None);
        let app = Router::new()
            .route("/protected", get(protected))
            .with_state(state.clone());
        let request = || {
            Request::builder()
                .uri("/protected")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap()
        };

        assert_eq!(
            app.clone().oneshot(request()).await.unwrap().status(),
            StatusCode::OK
        );
        state
            .main_db()
            .get()
            .unwrap()
            .execute(
                "UPDATE paired_devices SET revoked_at = datetime('now') WHERE device_id = ?1",
                rusqlite::params![device_id],
            )
            .unwrap();
        assert_eq!(
            app.oneshot(request()).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );

        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test]
    async fn ordinary_user_tokens_refresh_disabled_and_write_permissions() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-user-auth-refresh-test-{}",
            uuid::Uuid::new_v4()
        ));
        let state = AppState::new(&data_dir, 0).unwrap();
        state
            .main_db()
            .get()
            .unwrap()
            .execute(
                "INSERT INTO users (id, username, password_hash, is_active, access_level, can_write) VALUES (11, 'curator', 'unused', 1, 'local_network', 1)",
                [],
            )
            .unwrap();
        let token = create_jwt(11, "curator", "local_network", true, state.jwt_secret()).unwrap();
        let app = Router::new()
            .route("/protected", get(write_protected))
            .with_state(state.clone());
        let request = || {
            Request::builder()
                .uri("/protected")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap()
        };

        assert_eq!(
            app.clone().oneshot(request()).await.unwrap().status(),
            StatusCode::OK
        );
        state
            .main_db()
            .get()
            .unwrap()
            .execute("UPDATE users SET can_write = 0 WHERE id = 11", [])
            .unwrap();
        assert_eq!(
            app.clone().oneshot(request()).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        state
            .main_db()
            .get()
            .unwrap()
            .execute("UPDATE users SET is_active = 0 WHERE id = 11", [])
            .unwrap();
        assert_eq!(
            app.oneshot(request()).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
        let _ = std::fs::remove_dir_all(data_dir);
    }
}
