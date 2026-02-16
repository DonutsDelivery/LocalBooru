use axum::extract::{ConnectInfo, Path as AxumPath, State};
use axum::http::HeaderMap;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;

use crate::server::error::AppError;
use crate::server::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(list_users).post(create_user))
        .route(
            "/{user_id}",
            get(get_user).patch(update_user).delete(delete_user),
        )
        .route("/login", post(login))
        .route("/verify", post(verify_token))
}

// ─── Password hashing ────────────────────────────────────────────────────────

fn hash_password(password: &str) -> Result<String, AppError> {
    use argon2::password_hash::{rand_core::OsRng, PasswordHasher, SaltString};
    use argon2::Argon2;

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| AppError::Internal(format!("Password hashing error: {}", e)))?;

    Ok(hash.to_string())
}

fn verify_password(password: &str, stored_hash: &str) -> Result<bool, AppError> {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    use argon2::Argon2;

    let parsed_hash = PasswordHash::new(stored_hash)
        .map_err(|e| AppError::Internal(format!("Invalid stored hash: {}", e)))?;

    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok())
}

fn validate_password(password: &str) -> Result<(), AppError> {
    if password.len() < 8 {
        return Err(AppError::BadRequest(
            "Password must be at least 8 characters".into(),
        ));
    }
    if !password.chars().any(|c| c.is_uppercase()) {
        return Err(AppError::BadRequest(
            "Password must contain at least one uppercase letter".into(),
        ));
    }
    if !password.chars().any(|c| c.is_lowercase()) {
        return Err(AppError::BadRequest(
            "Password must contain at least one lowercase letter".into(),
        ));
    }
    if !password.chars().any(|c| c.is_ascii_digit()) {
        return Err(AppError::BadRequest(
            "Password must contain at least one number".into(),
        ));
    }
    Ok(())
}

// ─── Request models ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct UserCreate {
    username: String,
    password: String,
    access_level: Option<String>,
    can_write: Option<bool>,
}

#[derive(Deserialize)]
struct UserUpdate {
    password: Option<String>,
    is_active: Option<bool>,
    access_level: Option<String>,
    can_write: Option<bool>,
}

#[derive(Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

#[derive(Deserialize)]
struct VerifyTokenRequest {
    token: String,
}

// ─── IP extraction ────────────────────────────────────────────────────────────

/// Extract the client IP from the request, checking X-Forwarded-For first
/// (for proxied connections), then falling back to ConnectInfo.
fn extract_client_ip(headers: &HeaderMap, connect_info: &ConnectInfo<SocketAddr>) -> String {
    // Check X-Forwarded-For header (first IP is the original client)
    if let Some(forwarded_for) = headers.get("x-forwarded-for") {
        if let Ok(value) = forwarded_for.to_str() {
            if let Some(first_ip) = value.split(',').next() {
                let trimmed = first_ip.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    // Fall back to direct connection IP
    connect_info.0.ip().to_string()
}

// ─── Rate limit constants ─────────────────────────────────────────────────────

/// Maximum failed login attempts per IP before rate limiting kicks in.
const LOGIN_MAX_ATTEMPTS: u32 = 5;
/// Window in seconds for the login rate limiter (15 minutes).
const LOGIN_WINDOW_SECS: u64 = 900;

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /api/users
async fn list_users(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let mut stmt = conn.prepare(
            "SELECT id, username, is_active, access_level, can_write, created_at, last_login
             FROM users ORDER BY username",
        )?;

        let users: Vec<Value> = stmt
            .query_map([], |row| {
                Ok(json!({
                    "id": row.get::<_, i64>(0)?,
                    "username": row.get::<_, String>(1)?,
                    "is_active": row.get::<_, bool>(2)?,
                    "access_level": row.get::<_, String>(3)?,
                    "can_write": row.get::<_, bool>(4)?,
                    "created_at": row.get::<_, Option<String>>(5)?,
                    "last_login": row.get::<_, Option<String>>(6)?
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok::<_, AppError>(Json(json!({ "users": users })))
    })
    .await?
}

/// POST /api/users
async fn create_user(
    State(state): State<AppState>,
    Json(body): Json<UserCreate>,
) -> Result<Json<Value>, AppError> {
    validate_password(&body.password)?;

    let password_hash = hash_password(&body.password)?;
    let access_level = body.access_level.unwrap_or_else(|| "local_network".into());
    let can_write = body.can_write.unwrap_or(false);

    let state_clone = state.clone();
    let username = body.username.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        // Check duplicate
        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM users WHERE username = ?1",
                params![&username],
                |row| row.get::<_, i64>(0).map(|c| c > 0),
            )
            .unwrap_or(false);
        if exists {
            return Err(AppError::BadRequest("Username already exists".into()));
        }

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO users (username, password_hash, is_active, access_level, can_write, created_at)
             VALUES (?1, ?2, 1, ?3, ?4, ?5)",
            params![&username, &password_hash, &access_level, can_write, &now],
        )?;

        let id = conn.last_insert_rowid();
        Ok::<_, AppError>(Json(json!({
            "success": true,
            "user": {
                "id": id,
                "username": username,
                "is_active": true,
                "access_level": access_level,
                "can_write": can_write
            }
        })))
    })
    .await?
}

/// GET /api/users/:user_id
async fn get_user(
    State(state): State<AppState>,
    AxumPath(user_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        conn.query_row(
            "SELECT id, username, is_active, access_level, can_write, created_at, last_login
             FROM users WHERE id = ?1",
            params![user_id],
            |row| {
                Ok(Json(json!({
                    "id": row.get::<_, i64>(0)?,
                    "username": row.get::<_, String>(1)?,
                    "is_active": row.get::<_, bool>(2)?,
                    "access_level": row.get::<_, String>(3)?,
                    "can_write": row.get::<_, bool>(4)?,
                    "created_at": row.get::<_, Option<String>>(5)?,
                    "last_login": row.get::<_, Option<String>>(6)?
                })))
            },
        )
        .map_err(|_| AppError::NotFound("User not found".into()))
    })
    .await?
}

/// PATCH /api/users/:user_id
async fn update_user(
    State(state): State<AppState>,
    AxumPath(user_id): AxumPath<i64>,
    Json(body): Json<UserUpdate>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let mut sets = Vec::new();
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(ref password) = body.password {
            validate_password(password)?;
            let hash = hash_password(password)?;
            sql_params.push(Box::new(hash));
            sets.push(format!("password_hash = ?{}", sql_params.len()));
        }
        if let Some(is_active) = body.is_active {
            sql_params.push(Box::new(is_active));
            sets.push(format!("is_active = ?{}", sql_params.len()));
        }
        if let Some(ref al) = body.access_level {
            sql_params.push(Box::new(al.clone()));
            sets.push(format!("access_level = ?{}", sql_params.len()));
        }
        if let Some(cw) = body.can_write {
            sql_params.push(Box::new(cw));
            sets.push(format!("can_write = ?{}", sql_params.len()));
        }

        if sets.is_empty() {
            return Err(AppError::BadRequest("No fields to update".into()));
        }

        sql_params.push(Box::new(user_id));
        let sql = format!(
            "UPDATE users SET {} WHERE id = ?{}",
            sets.join(", "),
            sql_params.len()
        );
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        let updated = conn.execute(&sql, param_refs.as_slice())?;

        if updated == 0 {
            return Err(AppError::NotFound("User not found".into()));
        }
        Ok::<_, AppError>(Json(json!({ "success": true })))
    })
    .await?
}

/// DELETE /api/users/:user_id
async fn delete_user(
    State(state): State<AppState>,
    AxumPath(user_id): AxumPath<i64>,
) -> Result<Json<Value>, AppError> {
    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;
        let deleted = conn.execute("DELETE FROM users WHERE id = ?1", params![user_id])?;
        if deleted == 0 {
            return Err(AppError::NotFound("User not found".into()));
        }
        Ok::<_, AppError>(Json(json!({ "success": true, "deleted_user_id": user_id })))
    })
    .await?
}

/// POST /api/users/login
async fn login(
    State(state): State<AppState>,
    connect_info: ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<LoginRequest>,
) -> Result<Json<Value>, AppError> {
    // Extract client IP for rate limiting
    let client_ip = extract_client_ip(&headers, &connect_info);
    let rate_limit_key = format!("login:{}", client_ip);

    // Check rate limit before processing (counts all attempts including successful ones)
    state
        .rate_limiter()
        .check_rate_limit(&rate_limit_key, LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECS)?;

    let state_clone = state.clone();
    tokio::task::spawn_blocking(move || {
        let conn = state_clone.main_db().get()?;

        let user_row = conn.query_row(
            "SELECT id, username, password_hash, is_active, access_level, can_write FROM users WHERE username = ?1",
            params![&body.username],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, bool>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, bool>(5)?,
                ))
            },
        );

        let (id, username, password_hash, is_active, access_level, can_write) = match user_row {
            Ok(row) => row,
            Err(_) => return Err(AppError::Unauthorized("Invalid username or password".into())),
        };

        if !is_active {
            return Err(AppError::Unauthorized("Account is disabled".into()));
        }

        if !verify_password(&body.password, &password_hash)? {
            return Err(AppError::Unauthorized("Invalid username or password".into()));
        }

        // Update last login
        let now = chrono::Utc::now().to_rfc3339();
        let _ = conn.execute(
            "UPDATE users SET last_login = ?1 WHERE id = ?2",
            params![&now, id],
        );

        // Generate JWT token
        let token = create_jwt(id, &username, &access_level, can_write)?;

        Ok::<_, AppError>(Json(json!({
            "success": true,
            "token": token,
            "user": {
                "id": id,
                "username": username,
                "access_level": access_level,
                "can_write": can_write
            }
        })))
    })
    .await?
}

/// POST /api/users/verify
async fn verify_token(Json(body): Json<VerifyTokenRequest>) -> Result<Json<Value>, AppError> {
    let claims = decode_jwt(&body.token)?;
    Ok(Json(json!({
        "valid": true,
        "user": {
            "id": claims.user_id,
            "username": claims.username,
            "access_level": claims.access_level,
            "can_write": claims.can_write
        }
    })))
}

// ─── JWT helpers (delegated to shared auth module) ───────────────────────────

use crate::server::middleware::auth::{create_jwt, decode_jwt};
