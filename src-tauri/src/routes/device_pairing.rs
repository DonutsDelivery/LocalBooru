use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::Json;
use axum::routing::{delete, get, post};
use axum::Router;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use base64::Engine;
use dashmap::DashMap;
use p256::ecdsa::signature::{Signer, Verifier};
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use p256::pkcs8::{DecodePublicKey, EncodePublicKey};
use rand::RngCore;
use rsa::traits::PublicKeyParts;
use rsa::RsaPublicKey;
use rusqlite::{params, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::server::error::AppError;
use crate::server::middleware::auth::{create_device_jwt, AuthUser};
use crate::server::state::AppState;
use crate::server::utils::get_local_ip;

const DESKTOP_SESSION_TTL: Duration = Duration::from_secs(120);
const GRANT_TTL_SECS: i64 = 120;
const MAX_DELIVERIES: usize = 32;
const MAX_ENCRYPTED_PAYLOAD: usize = 16 * 1024;

#[derive(Clone)]
pub struct DesktopPairingSessions(Arc<DashMap<String, DesktopPairingSession>>);

#[derive(Clone)]
struct DesktopPairingSession {
    secret_hash: [u8; 32],
    display_name: String,
    public_key_fingerprint: String,
    callback_signing_key: SigningKey,
    expires_at_unix: i64,
    created_at: Instant,
    deliveries: Vec<DesktopPairingDelivery>,
    delivered_server_ids: HashSet<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopPairingDelivery {
    server_id: String,
    encrypted_payload: String,
}

pub fn create_desktop_pairing_sessions() -> DesktopPairingSessions {
    DesktopPairingSessions(Arc::new(DashMap::new()))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/desktop-sessions", post(create_desktop_session))
        .route(
            "/desktop-sessions/{session_id}",
            get(get_desktop_session).delete(cancel_desktop_session),
        )
        .route(
            "/desktop-sessions/{session_id}/deliver",
            post(deliver_to_desktop),
        )
        .route(
            "/desktop-sessions/{session_id}/verify",
            post(verify_desktop_callback),
        )
        .route("/grants", post(issue_grant))
        .route("/exchange", post(exchange_grant))
        .route("/devices", get(list_devices))
        .route("/devices/{device_id}", delete(revoke_device))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateDesktopSessionRequest {
    display_name: String,
    signing_public_key_spki: String,
    encryption_public_key_spki: String,
    public_key_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeliverRequest {
    session_secret: String,
    server_id: String,
    encrypted_payload: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerifyDesktopCallbackRequest {
    session_secret: String,
    public_key_fingerprint: String,
    callback_challenge: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct IssueGrantRequest {
    desktop_session_id: String,
    display_name: String,
    signing_public_key_spki: String,
    encryption_public_key_spki: String,
    public_key_fingerprint: String,
    requested_scope: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExchangeGrantRequest {
    grant_id: String,
    grant_secret: String,
    signature: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GrantResponse {
    server_id: String,
    server_name: String,
    grant_id: String,
    grant_secret: String,
    challenge: String,
    expires_at: i64,
}

fn random_secret() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn hash_bytes(value: &[u8]) -> [u8; 32] {
    Sha256::digest(value).into()
}

fn authorize_desktop_session(
    headers: &HeaderMap,
    session: &DesktopPairingSession,
) -> Result<(), AppError> {
    let supplied = headers
        .get("x-localbooru-pairing-secret")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| AppError::NotFound("Pairing session expired or does not exist".into()))?;
    if !bool::from(hash_bytes(supplied.as_bytes()).ct_eq(&session.secret_hash)) {
        return Err(AppError::NotFound(
            "Pairing session expired or does not exist".into(),
        ));
    }
    Ok(())
}

fn normalize_display_name(value: &str) -> Result<String, AppError> {
    let value = value.trim();
    if value.is_empty() || value.chars().count() > 80 {
        return Err(AppError::BadRequest(
            "Device name must be between 1 and 80 characters".into(),
        ));
    }
    Ok(value.to_string())
}

fn expected_fingerprint(signing_spki: &str, encryption_spki: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(signing_spki.as_bytes());
    hasher.update([0]);
    hasher.update(encryption_spki.as_bytes());
    URL_SAFE_NO_PAD.encode(hasher.finalize())
}

fn validate_public_keys(
    signing_spki: &str,
    encryption_spki: &str,
    fingerprint: &str,
) -> Result<(), AppError> {
    if signing_spki.len() > 2_048 || encryption_spki.len() > 4_096 {
        return Err(AppError::BadRequest(
            "Pairing public key is too large".into(),
        ));
    }
    let signing_der = STANDARD
        .decode(signing_spki)
        .map_err(|_| AppError::BadRequest("Invalid signing public key encoding".into()))?;
    VerifyingKey::from_public_key_der(&signing_der)
        .map_err(|_| AppError::BadRequest("Signing key must be a P-256 public key".into()))?;
    let encryption_der = STANDARD
        .decode(encryption_spki)
        .map_err(|_| AppError::BadRequest("Invalid encryption public key encoding".into()))?;
    let encryption_key = RsaPublicKey::from_public_key_der(&encryption_der)
        .map_err(|_| AppError::BadRequest("Encryption key must be an RSA public key".into()))?;
    if encryption_key.n().bits() < 2_048 {
        return Err(AppError::BadRequest(
            "Encryption key must be at least 2048-bit RSA".into(),
        ));
    }
    let expected = expected_fingerprint(signing_spki, encryption_spki);
    if !bool::from(expected.as_bytes().ct_eq(fingerprint.as_bytes())) {
        return Err(AppError::BadRequest(
            "Desktop key fingerprint does not match its public keys".into(),
        ));
    }
    Ok(())
}

fn sweep_desktop_sessions(sessions: &DesktopPairingSessions) {
    let expired: Vec<String> = sessions
        .0
        .iter()
        .filter(|entry| entry.value().created_at.elapsed() > DESKTOP_SESSION_TTL)
        .map(|entry| entry.key().clone())
        .collect();
    for id in expired {
        sessions.0.remove(&id);
    }
}

async fn create_desktop_session(
    State(state): State<AppState>,
    Json(body): Json<CreateDesktopSessionRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    state
        .rate_limiter()
        .check_rate_limit("device-pairing:create", 60, 60)?;
    let display_name = normalize_display_name(&body.display_name)?;
    validate_public_keys(
        &body.signing_public_key_spki,
        &body.encryption_public_key_spki,
        &body.public_key_fingerprint,
    )?;
    let sessions = state.desktop_pairing_sessions();
    sweep_desktop_sessions(sessions);
    let session_id = uuid::Uuid::new_v4().to_string();
    let session_secret = random_secret();
    let callback_signing_key = SigningKey::random(&mut rand::thread_rng());
    let callback_public_key_spki = STANDARD.encode(
        callback_signing_key
            .verifying_key()
            .to_public_key_der()
            .map_err(|_| AppError::Internal("Failed to encode callback proof key".into()))?
            .as_bytes(),
    );
    let expires_at = chrono::Utc::now().timestamp() + DESKTOP_SESSION_TTL.as_secs() as i64;
    sessions.0.insert(
        session_id.clone(),
        DesktopPairingSession {
            secret_hash: hash_bytes(session_secret.as_bytes()),
            display_name: display_name.clone(),
            public_key_fingerprint: body.public_key_fingerprint.clone(),
            callback_signing_key,
            expires_at_unix: expires_at,
            created_at: Instant::now(),
            deliveries: Vec::new(),
            delivered_server_ids: HashSet::new(),
        },
    );

    let mut callback_urls = vec![format!("http://127.0.0.1:{}", state.port())];
    if state.is_lan_enabled() {
        if let Some(ip) = get_local_ip() {
            callback_urls.insert(0, format!("http://{}:{}", ip, state.port()));
        }
    }
    Ok(Json(json!({
        "type": "localbooru-desktop-pairing",
        "version": 1,
        "sessionId": session_id,
        "sessionSecret": session_secret,
        "displayName": display_name,
        "signingPublicKeySpki": body.signing_public_key_spki,
        "encryptionPublicKeySpki": body.encryption_public_key_spki,
        "publicKeyFingerprint": body.public_key_fingerprint,
        "callbackPublicKeySpki": callback_public_key_spki,
        "callbackUrls": callback_urls,
        "expiresAt": expires_at,
        "requestedScope": "local_network_write"
    })))
}

async fn get_desktop_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    if session_id.len() > 128 {
        return Err(AppError::NotFound(
            "Pairing session expired or does not exist".into(),
        ));
    }
    state
        .rate_limiter()
        .check_rate_limit("device-pairing:unauthenticated", 1000, 120)?;
    state.rate_limiter().check_rate_limit(
        &format!("device-pairing:poll:{session_id}"),
        180,
        120,
    )?;
    sweep_desktop_sessions(state.desktop_pairing_sessions());
    let session = state
        .desktop_pairing_sessions()
        .0
        .get(&session_id)
        .ok_or_else(|| AppError::NotFound("Pairing session expired or does not exist".into()))?;
    authorize_desktop_session(&headers, &session)?;
    Ok(Json(json!({
        "status": if session.deliveries.is_empty() { "waiting" } else { "authorized" },
        "displayName": session.display_name,
        "publicKeyFingerprint": session.public_key_fingerprint,
        "expiresAt": session.expires_at_unix,
        "deliveries": session.deliveries
    })))
}

async fn cancel_desktop_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, AppError> {
    let sessions = state.desktop_pairing_sessions();
    let session = sessions
        .0
        .get(&session_id)
        .ok_or_else(|| AppError::NotFound("Pairing session expired or does not exist".into()))?;
    authorize_desktop_session(&headers, &session)?;
    drop(session);
    sessions.0.remove(&session_id);
    Ok(Json(json!({"success": true})))
}

async fn deliver_to_desktop(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(body): Json<DeliverRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    state.rate_limiter().check_rate_limit(
        &format!("device-pairing:deliver:{session_id}"),
        64,
        120,
    )?;
    if session_id.len() > 128
        || body.session_secret.len() > 128
        || body.encrypted_payload.len() > MAX_ENCRYPTED_PAYLOAD
        || body.server_id.len() > 128
    {
        return Err(AppError::BadRequest("Pairing delivery is too large".into()));
    }
    state
        .rate_limiter()
        .check_rate_limit("device-pairing:unauthenticated", 1000, 120)?;
    sweep_desktop_sessions(state.desktop_pairing_sessions());
    let mut session = state
        .desktop_pairing_sessions()
        .0
        .get_mut(&session_id)
        .ok_or_else(|| AppError::NotFound("Pairing session expired or does not exist".into()))?;
    let supplied = hash_bytes(body.session_secret.as_bytes());
    if !bool::from(supplied.ct_eq(&session.secret_hash)) {
        return Err(AppError::NotFound(
            "Pairing session expired or does not exist".into(),
        ));
    }
    if session.deliveries.len() >= MAX_DELIVERIES {
        return Err(AppError::BadRequest(
            "Pairing session has reached its server limit".into(),
        ));
    }
    if session.delivered_server_ids.insert(body.server_id.clone()) {
        session.deliveries.push(DesktopPairingDelivery {
            server_id: body.server_id,
            encrypted_payload: body.encrypted_payload,
        });
    }
    Ok(Json(json!({"success": true})))
}

async fn verify_desktop_callback(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(body): Json<VerifyDesktopCallbackRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    if session_id.len() > 128
        || body.session_secret.len() > 128
        || body.public_key_fingerprint.len() > 128
        || body.callback_challenge.len() > 128
        || body.callback_challenge.len() < 16
    {
        return Err(AppError::NotFound(
            "Pairing session expired or does not exist".into(),
        ));
    }
    state
        .rate_limiter()
        .check_rate_limit("device-pairing:unauthenticated", 1000, 120)?;
    state.rate_limiter().check_rate_limit(
        &format!("device-pairing:verify:{session_id}"),
        32,
        120,
    )?;
    sweep_desktop_sessions(state.desktop_pairing_sessions());
    let session = state
        .desktop_pairing_sessions()
        .0
        .get(&session_id)
        .ok_or_else(|| AppError::NotFound("Pairing session expired or does not exist".into()))?;
    let supplied_secret = hash_bytes(body.session_secret.as_bytes());
    if !bool::from(supplied_secret.ct_eq(&session.secret_hash))
        || !bool::from(
            session
                .public_key_fingerprint
                .as_bytes()
                .ct_eq(body.public_key_fingerprint.as_bytes()),
        )
    {
        return Err(AppError::NotFound(
            "Pairing session expired or does not exist".into(),
        ));
    }
    let callback_message = format!(
        "localbooru-callback-v1\0{}\0{}\0{}\0{}",
        session_id,
        session.public_key_fingerprint,
        body.callback_challenge,
        session.expires_at_unix
    );
    let callback_signature: Signature = session
        .callback_signing_key
        .sign(callback_message.as_bytes());
    Ok(Json(json!({
        "sessionId": session_id,
        "publicKeyFingerprint": session.public_key_fingerprint,
        "expiresAt": session.expires_at_unix,
        "callbackChallenge": body.callback_challenge,
        "callbackSignature": URL_SAFE_NO_PAD.encode(callback_signature.to_bytes())
    })))
}

async fn issue_grant(
    State(state): State<AppState>,
    user: AuthUser,
    Json(body): Json<IssueGrantRequest>,
) -> Result<Json<GrantResponse>, AppError> {
    state.rate_limiter().check_rate_limit(
        &format!("device-pairing:issue:{}", user.username),
        30,
        120,
    )?;
    if !user.can_write {
        return Err(AppError::Forbidden(
            "Pairing another device requires write access".into(),
        ));
    }
    if body.requested_scope != "local_network_write" {
        return Err(AppError::BadRequest(
            "Unsupported desktop pairing scope".into(),
        ));
    }
    let display_name = normalize_display_name(&body.display_name)?;
    if body.desktop_session_id.len() > 128 {
        return Err(AppError::BadRequest(
            "Desktop pairing session ID is too large".into(),
        ));
    }
    validate_public_keys(
        &body.signing_public_key_spki,
        &body.encryption_public_key_spki,
        &body.public_key_fingerprint,
    )?;
    let grant_id = uuid::Uuid::new_v4().to_string();
    let grant_secret = random_secret();
    let challenge = random_secret();
    let expires_at = chrono::Utc::now().timestamp() + GRANT_TTL_SECS;
    let secret_hash = URL_SAFE_NO_PAD.encode(hash_bytes(grant_secret.as_bytes()));
    let conn = state.main_db().get()?;
    conn.execute(
        "DELETE FROM device_pairing_grants WHERE expires_at < ?1 OR consumed_at IS NOT NULL",
        params![chrono::Utc::now().timestamp()],
    )?;
    conn.execute(
        "INSERT INTO device_pairing_grants (grant_id, grant_secret_hash, desktop_session_id, display_name, public_key_spki, public_key_fingerprint, challenge, authorized_by, authorized_by_user_id, authorized_by_device_id, expires_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![grant_id, secret_hash, body.desktop_session_id, display_name, body.signing_public_key_spki, body.public_key_fingerprint, challenge, user.username, (user.device_id.is_none()).then_some(user.user_id), user.device_id, expires_at],
    )?;
    conn.execute(
        "INSERT INTO device_pairing_audit (event_type, public_key_fingerprint, detail) VALUES ('grant_issued', ?1, ?2)",
        params![body.public_key_fingerprint, format!("authorized by {} for session {}", user.username, body.desktop_session_id)],
    )?;
    Ok(Json(GrantResponse {
        server_id: state.library_manager().primary().uuid.clone(),
        server_name: state.library_manager().primary().name.clone(),
        grant_id,
        grant_secret,
        challenge,
        expires_at,
    }))
}

fn verify_challenge_signature(
    public_key_spki: &str,
    challenge: &str,
    signature: &str,
) -> Result<(), AppError> {
    let public_der = STANDARD
        .decode(public_key_spki)
        .map_err(|_| AppError::BadRequest("Invalid stored pairing public key".into()))?;
    let key = VerifyingKey::from_public_key_der(&public_der)
        .map_err(|_| AppError::BadRequest("Invalid stored pairing public key".into()))?;
    let signature_bytes = URL_SAFE_NO_PAD
        .decode(signature)
        .map_err(|_| AppError::BadRequest("Invalid pairing proof encoding".into()))?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| AppError::BadRequest("Invalid pairing proof".into()))?;
    key.verify(challenge.as_bytes(), &signature)
        .map_err(|_| AppError::Unauthorized("Desktop proof-of-possession failed".into()))
}

async fn exchange_grant(
    State(state): State<AppState>,
    Json(body): Json<ExchangeGrantRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    if body.grant_id.len() > 64 || body.grant_secret.len() > 128 || body.signature.len() > 256 {
        return Err(AppError::BadRequest(
            "Pairing exchange field is too large".into(),
        ));
    }
    state
        .rate_limiter()
        .check_rate_limit("device-pairing:unauthenticated", 1000, 120)?;
    state.rate_limiter().check_rate_limit(
        &format!("device-pairing:exchange:{}", body.grant_id),
        12,
        120,
    )?;
    let now = chrono::Utc::now().timestamp();
    let mut conn = state.main_db().get()?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let grant = tx
        .query_row(
            "SELECT grant_secret_hash, display_name, public_key_spki, public_key_fingerprint, challenge, expires_at, consumed_at, authorized_by_user_id, authorized_by_device_id FROM device_pairing_grants WHERE grant_id = ?1",
            params![body.grant_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?, row.get::<_, String>(4)?, row.get::<_, i64>(5)?, row.get::<_, Option<String>>(6)?, row.get::<_, Option<i64>>(7)?, row.get::<_, Option<String>>(8)?)),
        )
        .optional()?
        .ok_or_else(|| AppError::NotFound("Pairing grant does not exist".into()))?;
    if grant.6.is_some() {
        return Err(AppError::BadRequest(
            "Pairing grant has already been used".into(),
        ));
    }
    if grant.5 <= now {
        return Err(AppError::BadRequest("Pairing grant has expired".into()));
    }
    let authorizer_active = if let Some(device_id) = grant.8.as_deref() {
        tx.query_row(
            "SELECT revoked_at IS NULL FROM paired_devices WHERE device_id = ?1",
            params![device_id],
            |row| row.get::<_, bool>(0),
        )
        .optional()?
        .unwrap_or(false)
    } else if let Some(user_id) = grant.7 {
        tx.query_row(
            "SELECT is_active AND can_write FROM users WHERE id = ?1",
            params![user_id],
            |row| row.get::<_, bool>(0),
        )
        .optional()?
        .unwrap_or(false)
    } else {
        false
    };
    if !authorizer_active {
        return Err(AppError::Unauthorized(
            "Pairing authorization was revoked".into(),
        ));
    }
    let supplied_hash = URL_SAFE_NO_PAD.encode(hash_bytes(body.grant_secret.as_bytes()));
    if !bool::from(supplied_hash.as_bytes().ct_eq(grant.0.as_bytes())) {
        return Err(AppError::NotFound("Pairing grant does not exist".into()));
    }
    verify_challenge_signature(&grant.2, &grant.4, &body.signature)?;
    let device_id = uuid::Uuid::new_v4().to_string();
    tx.execute(
        "INSERT INTO paired_devices (device_id, display_name, public_key_spki, public_key_fingerprint, last_seen_at) VALUES (?1, ?2, ?3, ?4, datetime('now')) ON CONFLICT(public_key_fingerprint) DO UPDATE SET device_id = excluded.device_id, display_name = excluded.display_name, public_key_spki = excluded.public_key_spki, last_seen_at = datetime('now'), revoked_at = NULL",
        params![device_id, grant.1, grant.2, grant.3],
    )?;
    let stored_device_id: String = tx.query_row(
        "SELECT device_id FROM paired_devices WHERE public_key_fingerprint = ?1",
        params![grant.3],
        |row| row.get(0),
    )?;
    tx.execute(
        "UPDATE device_pairing_grants SET consumed_at = datetime('now') WHERE grant_id = ?1 AND consumed_at IS NULL",
        params![body.grant_id],
    )?;
    tx.execute(
        "INSERT INTO device_pairing_audit (event_type, device_id, public_key_fingerprint, detail) VALUES ('grant_exchanged', ?1, ?2, 'desktop credential issued')",
        params![stored_device_id, grant.3],
    )?;
    tx.commit()?;
    let token = create_device_jwt(&stored_device_id, &grant.1, state.jwt_secret())?;
    Ok(Json(json!({
        "success": true,
        "deviceId": stored_device_id,
        "token": token,
        "serverId": state.library_manager().primary().uuid,
        "serverName": state.library_manager().primary().name
    })))
}

async fn list_devices(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    let conn = state.main_db().get()?;
    let mut stmt = conn.prepare("SELECT device_id, display_name, public_key_fingerprint, created_at, last_seen_at, revoked_at FROM paired_devices ORDER BY created_at DESC")?;
    let devices = stmt
        .query_map([], |row| {
            Ok(json!({
                "deviceId": row.get::<_, String>(0)?,
                "displayName": row.get::<_, String>(1)?,
                "publicKeyFingerprint": row.get::<_, String>(2)?,
                "createdAt": row.get::<_, String>(3)?,
                "lastSeenAt": row.get::<_, Option<String>>(4)?,
                "revokedAt": row.get::<_, Option<String>>(5)?
            }))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Json(json!({"devices": devices})))
}

async fn revoke_device(
    State(state): State<AppState>,
    Path(device_id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let conn = state.main_db().get()?;
    let changed = conn.execute(
        "UPDATE paired_devices SET revoked_at = datetime('now') WHERE device_id = ?1 AND revoked_at IS NULL",
        params![device_id],
    )?;
    if changed == 0 {
        return Err(AppError::NotFound(
            "Paired device does not exist or is already revoked".into(),
        ));
    }
    conn.execute(
        "INSERT INTO device_pairing_audit (event_type, device_id, detail) VALUES ('device_revoked', ?1, 'revoked from host settings')",
        params![device_id],
    )?;
    conn.execute(
        "DELETE FROM device_pairing_grants WHERE authorized_by_device_id = ?1 AND consumed_at IS NULL",
        params![device_id],
    )?;
    Ok(Json(json!({"success": true})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use p256::ecdsa::signature::Signer;
    use p256::ecdsa::SigningKey;
    use p256::pkcs8::EncodePublicKey;
    use rsa::RsaPrivateKey;
    use tower::ServiceExt;

    #[test]
    fn fingerprint_binds_both_desktop_keys() {
        assert_ne!(
            expected_fingerprint("sign-a", "enc"),
            expected_fingerprint("sign-b", "enc")
        );
        assert_ne!(
            expected_fingerprint("sign", "enc-a"),
            expected_fingerprint("sign", "enc-b")
        );
    }

    #[test]
    fn secret_hash_comparison_rejects_wrong_secret() {
        let expected = hash_bytes(b"correct");
        assert!(bool::from(expected.ct_eq(&hash_bytes(b"correct"))));
        assert!(!bool::from(expected.ct_eq(&hash_bytes(b"wrong"))));
    }

    #[tokio::test]
    async fn desktop_poll_and_cancel_require_the_session_secret() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-desktop-session-test-{}",
            uuid::Uuid::new_v4()
        ));
        let state = AppState::new(&data_dir, 0).unwrap();
        let signing_key = SigningKey::random(&mut rand::thread_rng());
        let signing_spki = STANDARD.encode(
            signing_key
                .verifying_key()
                .to_public_key_der()
                .unwrap()
                .as_bytes(),
        );
        let rsa_private = RsaPrivateKey::new(&mut rand::thread_rng(), 2048).unwrap();
        let encryption_spki = STANDARD.encode(
            rsa::RsaPublicKey::from(&rsa_private)
                .to_public_key_der()
                .unwrap()
                .as_bytes(),
        );
        let fingerprint = expected_fingerprint(&signing_spki, &encryption_spki);
        let app = router().with_state(state);
        let create = Request::builder()
            .method("POST")
            .uri("/desktop-sessions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "displayName": "Bedroom Desktop",
                    "signingPublicKeySpki": signing_spki,
                    "encryptionPublicKeySpki": encryption_spki,
                    "publicKeyFingerprint": fingerprint
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(create).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let created: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let session_id = created["sessionId"].as_str().unwrap();
        let session_secret = created["sessionSecret"].as_str().unwrap();
        let callback_public_key_der = STANDARD
            .decode(created["callbackPublicKeySpki"].as_str().unwrap())
            .unwrap();
        let callback_public_key =
            VerifyingKey::from_public_key_der(&callback_public_key_der).unwrap();

        let callback_challenge = random_secret();
        let callback_verify = Request::builder()
            .method("POST")
            .uri(format!("/desktop-sessions/{session_id}/verify"))
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "sessionSecret": session_secret,
                    "publicKeyFingerprint": created["publicKeyFingerprint"],
                    "callbackChallenge": callback_challenge
                })
                .to_string(),
            ))
            .unwrap();
        let callback_response = app.clone().oneshot(callback_verify).await.unwrap();
        assert_eq!(callback_response.status(), StatusCode::OK);
        let callback_proof: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(callback_response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let callback_message = format!(
            "localbooru-callback-v1\0{}\0{}\0{}\0{}",
            session_id,
            created["publicKeyFingerprint"].as_str().unwrap(),
            callback_challenge,
            callback_proof["expiresAt"].as_i64().unwrap()
        );
        let callback_signature = Signature::from_slice(
            &URL_SAFE_NO_PAD
                .decode(callback_proof["callbackSignature"].as_str().unwrap())
                .unwrap(),
        )
        .unwrap();
        callback_public_key
            .verify(callback_message.as_bytes(), &callback_signature)
            .unwrap();

        let poll = |secret: Option<&str>| {
            let mut request = Request::builder()
                .method("GET")
                .uri(format!("/desktop-sessions/{session_id}"));
            if let Some(secret) = secret {
                request = request.header("x-localbooru-pairing-secret", secret);
            }
            request.body(Body::empty()).unwrap()
        };
        assert_eq!(
            app.clone().oneshot(poll(None)).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            app.clone()
                .oneshot(poll(Some("wrong")))
                .await
                .unwrap()
                .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            app.clone()
                .oneshot(poll(Some(session_secret)))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        let wrong_cancel = Request::builder()
            .method("DELETE")
            .uri(format!("/desktop-sessions/{session_id}"))
            .header("x-localbooru-pairing-secret", "wrong")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(wrong_cancel).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            app.oneshot(poll(Some(session_secret)))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );

        let _ = std::fs::remove_dir_all(data_dir);
    }

    #[tokio::test]
    async fn grant_requires_desktop_proof_and_is_single_use() {
        let data_dir = std::env::temp_dir().join(format!(
            "localbooru-device-pairing-test-{}",
            uuid::Uuid::new_v4()
        ));
        let state = AppState::new(&data_dir, 0).unwrap();
        state
            .main_db()
            .get()
            .unwrap()
            .execute(
                "INSERT INTO users (id, username, password_hash, is_active, access_level, can_write) VALUES (7, 'phone-owner', 'unused', 1, 'local_network', 1)",
                [],
            )
            .unwrap();
        let signing_key = SigningKey::random(&mut rand::thread_rng());
        let signing_spki = STANDARD.encode(
            signing_key
                .verifying_key()
                .to_public_key_der()
                .unwrap()
                .as_bytes(),
        );
        let rsa_private = RsaPrivateKey::new(&mut rand::thread_rng(), 2048).unwrap();
        let encryption_spki = STANDARD.encode(
            rsa::RsaPublicKey::from(&rsa_private)
                .to_public_key_der()
                .unwrap()
                .as_bytes(),
        );
        let fingerprint = expected_fingerprint(&signing_spki, &encryption_spki);
        let user_token = crate::server::middleware::auth::create_jwt(
            7,
            "phone-owner",
            "local_network",
            true,
            state.jwt_secret(),
        )
        .unwrap();
        let app = router().with_state(state.clone());
        let request = Request::builder()
            .method("POST")
            .uri("/grants")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {user_token}"))
            .body(Body::from(
                json!({
                    "desktopSessionId": "desktop-session",
                    "displayName": "Bedroom Desktop",
                    "signingPublicKeySpki": signing_spki,
                    "encryptionPublicKeySpki": encryption_spki,
                    "publicKeyFingerprint": fingerprint,
                    "requestedScope": "local_network_write"
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let grant: GrantResponse = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let signature: Signature = signing_key.sign(grant.challenge.as_bytes());
        let exchange_body = json!({
            "grantId": grant.grant_id,
            "grantSecret": grant.grant_secret,
            "signature": URL_SAFE_NO_PAD.encode(signature.to_bytes())
        })
        .to_string();
        let exchange_request = || {
            Request::builder()
                .method("POST")
                .uri("/exchange")
                .header("content-type", "application/json")
                .body(Body::from(exchange_body.clone()))
                .unwrap()
        };

        let first = app.clone().oneshot(exchange_request()).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let second = app.oneshot(exchange_request()).await.unwrap();
        assert_eq!(second.status(), StatusCode::BAD_REQUEST);

        let _ = std::fs::remove_dir_all(data_dir);
    }
}
