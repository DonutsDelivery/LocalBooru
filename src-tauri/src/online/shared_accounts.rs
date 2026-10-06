// SPDX-License-Identifier: GPL-3.0-only
//! Shared secrets are attached only after reciprocal pinned identities and a
//! fresh authenticated catalog check. A sibling alias never changes its source.
use super::{hub, models::RemoteSource, transport};
use crate::{routes::online as routes, server::error::AppError};
use axum::http::{HeaderValue, StatusCode};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashSet, path::Path};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Identity {
    origin: String,
    instance_id: String,
    realm: String,
    peer_id: String,
    relay_url: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Authorization {
    version: u32,
    realms: Vec<Identity>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct Binding {
    identity: Identity,
    mode: transport::TransportMode,
    credential_ref: String,
}
#[derive(Clone)]
struct Snapshot {
    source: RemoteSource,
    config: routes::HubConfig,
    generation: i64,
    blocked: bool,
}
#[derive(Clone)]
struct Donor {
    snapshot: Snapshot,
    connection_id: String,
    credential_ref: String,
    account: Option<String>,
    credential: routes::StoredCredential,
}

pub(crate) fn schema(conn: &Connection) -> Result<(), AppError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS remote_account_state(
           source_id TEXT PRIMARY KEY REFERENCES remote_sources(source_id) ON DELETE CASCADE,
           generation INTEGER NOT NULL DEFAULT 0, automatic_blocked INTEGER NOT NULL DEFAULT 0);
         CREATE TABLE IF NOT EXISTS remote_credential_aliases(
           connection_id TEXT PRIMARY KEY REFERENCES remote_connections(connection_id) ON DELETE CASCADE,
           binding_json TEXT NOT NULL);
         INSERT OR IGNORE INTO remote_account_state(source_id,automatic_blocked) SELECT source_id,1 FROM remote_sources;
         CREATE TRIGGER IF NOT EXISTS account_source_insert AFTER INSERT ON remote_sources BEGIN
           INSERT OR IGNORE INTO remote_account_state(source_id) VALUES(NEW.source_id); END;
         CREATE TRIGGER IF NOT EXISTS account_source_update AFTER UPDATE ON remote_sources BEGIN
           UPDATE remote_account_state SET generation=generation+1 WHERE source_id=NEW.source_id; END;
         CREATE TRIGGER IF NOT EXISTS account_transport_update AFTER UPDATE ON remote_hub_transport BEGIN
           UPDATE remote_account_state SET generation=generation+1 WHERE source_id=NEW.source_id; END;
         CREATE TRIGGER IF NOT EXISTS account_connection_insert AFTER INSERT ON remote_connections BEGIN
           UPDATE remote_account_state SET generation=generation+1 WHERE source_id=NEW.source_id; END;
         CREATE TRIGGER IF NOT EXISTS account_connection_update AFTER UPDATE ON remote_connections BEGIN
           UPDATE remote_account_state SET generation=generation+1 WHERE source_id=NEW.source_id; END;
         CREATE TRIGGER IF NOT EXISTS account_connection_delete AFTER DELETE ON remote_connections BEGIN
           UPDATE remote_account_state SET generation=generation+1,automatic_blocked=1 WHERE source_id=OLD.source_id; END;",
    )?;
    Ok(())
}
pub(crate) fn generation(conn: &Connection, sid: &str) -> Result<i64, AppError> {
    Ok(conn.query_row(
        "SELECT generation FROM remote_account_state WHERE source_id=?1",
        params![sid],
        |r| r.get(0),
    )?)
}
fn snapshot(data_dir: &Path, sid: &str) -> Result<Snapshot, AppError> {
    let source = routes::load_source(data_dir, sid)?;
    if !source.enabled {
        return Err(AppError::Forbidden(
            "Community connection is disabled".into(),
        ));
    }
    let config = routes::load_hub(data_dir, sid)?
        .ok_or_else(|| AppError::BadRequest("Source is not an enrolled community".into()))?;
    if config.transport_mode == transport::TransportMode::Anonymous {
        return Err(AppError::Forbidden(
            "Shared sign-in is unavailable in Anonymous mode".into(),
        ));
    }
    let (generation, blocked) = routes::open_db(data_dir)?.query_row(
        "SELECT generation,automatic_blocked FROM remote_account_state WHERE source_id=?1",
        params![sid],
        |r| Ok((r.get(0)?, r.get::<_, bool>(1)?)),
    )?;
    Ok(Snapshot {
        source,
        config,
        generation,
        blocked,
    })
}
fn identity(source: &RemoteSource, config: &routes::HubConfig) -> Result<Identity, AppError> {
    let realm = config
        .descriptor
        .pointer("/instance/realm")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty() && v.len() <= 64)
        .ok_or_else(|| {
            AppError::Forbidden("Community has no explicit shared sign-in realm".into())
        })?;
    Ok(Identity {
        origin: source.normalized_base_url.clone(),
        instance_id: config.instance_id.clone(),
        realm: realm.into(),
        peer_id: config.bootstrap.peer_id.clone(),
        relay_url: config.bootstrap.relay_url.clone(),
    })
}
fn authorization(descriptor: &Value, own: &Identity) -> Result<Authorization, AppError> {
    let value = descriptor
        .pointer("/authentication/shared_credentials")
        .ok_or_else(|| AppError::Forbidden("Community does not authorize shared sign-in".into()))?;
    let auth: Authorization = serde_json::from_value(value.clone()).map_err(|_| {
        AppError::Forbidden("Community shared sign-in authorization is invalid".into())
    })?;
    // The v1 contract explicitly names two independent realms, not a transitive group.
    if auth.version != 1
        || auth.realms.len() != 2
        || !auth.realms.contains(own)
        || auth.realms[0].origin == auth.realms[1].origin
        || auth.realms[0].peer_id == auth.realms[1].peer_id
        || auth.realms[0].realm == auth.realms[1].realm
        || auth.realms.iter().any(|id| {
            id.instance_id != own.instance_id
                || id.realm.is_empty()
                || id.realm.len() > 64
                || id.peer_id.len() != 64
                || !id.peer_id.bytes().all(|c| c.is_ascii_hexdigit())
                || hub::origin(&id.origin, false).ok().as_deref() != Some(id.origin.as_str())
                || reqwest::Url::parse(&id.relay_url).ok().is_none_or(|u| {
                    u.scheme() != "https"
                        || u.host_str().is_none()
                        || u.password().is_some()
                        || !u.username().is_empty()
                        || u.query().is_some()
                        || u.fragment().is_some()
                })
        })
    {
        return Err(AppError::Forbidden(
            "Shared sign-in must name two exact independently pinned realms".into(),
        ));
    }
    Ok(auth)
}
fn reciprocal(
    a: &Identity,
    a_descriptor: &Value,
    b: &Identity,
    b_descriptor: &Value,
) -> Result<(), AppError> {
    let aa = authorization(a_descriptor, a)?;
    let bb = authorization(b_descriptor, b)?;
    if a == b
        || !aa.realms.contains(b)
        || !bb.realms.contains(a)
        || !aa.realms.iter().all(|entry| bb.realms.contains(entry))
    {
        return Err(AppError::Forbidden(
            "Communities do not reciprocally authorize these exact sign-in identities".into(),
        ));
    }
    Ok(())
}

/// Checked before any alias bearer is loaded or sent. Donor removal does not
/// revoke a surviving alias: its protected reference has independent ownership.
pub(crate) fn validate_alias(
    data_dir: &Path,
    sid: &str,
    cid: &str,
    reference: &str,
) -> Result<(), AppError> {
    let binding: Option<String> = routes::open_db(data_dir)?
        .query_row(
            "SELECT binding_json FROM remote_credential_aliases WHERE connection_id=?1",
            params![cid],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(raw) = binding {
        let binding: Binding = serde_json::from_str(&raw).map_err(|_| {
            AppError::Unauthorized("Shared account binding is corrupt; reconnect it".into())
        })?;
        let config = routes::load_hub(data_dir, sid)?.ok_or_else(|| {
            AppError::Unauthorized("Shared account community was disconnected".into())
        })?;
        let source = routes::load_source(data_dir, sid)?;
        if binding.credential_ref != reference
            || binding.mode != config.transport_mode
            || binding.identity != identity(&source, &config)?
        {
            return Err(AppError::Unauthorized(
                "Shared account identity or connection mode changed; reconnect it".into(),
            ));
        }
    }
    Ok(())
}
pub(crate) fn remove_unused_secret(data_dir: &Path, reference: &str) -> Result<(), AppError> {
    let mut conn = routes::open_db(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let in_use: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM remote_connections WHERE credential_ref=?1)",
        params![reference],
        |r| r.get(0),
    )?;
    if !in_use {
        let _ = std::fs::remove_file(routes::credential_path(data_dir, reference)?);
    }
    tx.commit()?;
    Ok(())
}
fn has_accounts(conn: &Connection, sid: &str) -> Result<bool, AppError> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM remote_connections WHERE source_id=?1)",
        params![sid],
        |r| r.get(0),
    )?)
}
fn donors(data_dir: &Path, target: &Snapshot) -> Result<Vec<Donor>, AppError> {
    let conn = routes::open_db(data_dir)?;
    let rows = {
        let mut statement = conn.prepare("SELECT c.source_id,c.connection_id,c.credential_ref,c.account_display_name FROM remote_connections c JOIN remote_hub_transport h USING(source_id) WHERE c.source_id!=?1 AND c.credential_ref IS NOT NULL AND c.trust_state='verified' ORDER BY c.last_probe_at DESC,c.connection_id LIMIT 65")?;
        let rows = statement.query_map(params![target.source.source_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<String>>(3)?,
            ))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    if rows.len() > 64 {
        return Err(AppError::BadRequest(
            "Too many connected accounts; select an account explicitly after reducing the list"
                .into(),
        ));
    }
    let mut refs = HashSet::new();
    let mut candidates = vec![];
    for (sid, connection_id, credential_ref, account) in rows {
        let Ok(snap) = snapshot(data_dir, &sid) else {
            continue;
        };
        let (Ok(a), Ok(b)) = (
            identity(&snap.source, &snap.config),
            identity(&target.source, &target.config),
        ) else {
            continue;
        };
        // Stored metadata is only a candidate filter, never authority to send a secret.
        if a.instance_id != b.instance_id
            || a.origin == b.origin
            || validate_alias(data_dir, &sid, &connection_id, &credential_ref).is_err()
        {
            continue;
        }
        let Ok(credential) = routes::read_credential(data_dir, &credential_ref) else {
            continue;
        };
        if credential.auth_kind != "bearer"
            || credential.secret.is_empty()
            || credential
                .expires_at
                .is_some_and(|e| e <= Utc::now().timestamp())
        {
            continue;
        }
        if !refs.insert(credential_ref.clone()) {
            continue;
        }
        candidates.push(Donor {
            snapshot: snap,
            connection_id,
            credential_ref,
            account,
            credential,
        });
    }
    Ok(candidates)
}
pub(crate) async fn status(data_dir: &Path, sid: &str) -> Result<Value, AppError> {
    let target = snapshot(data_dir, sid)?;
    if has_accounts(&routes::open_db(data_dir)?, sid)? {
        return Ok(json!({"state":"account_present","candidates":[]}));
    }
    let candidates = eligible_donors(data_dir, &target).await?;
    Ok(
        json!({"state":if candidates.is_empty(){"unavailable"}else if candidates.len()>1{"selection_required"}else{"available"},
        "candidates":candidates.iter().map(|d| json!({"connection_id":d.connection_id,"source_id":d.snapshot.source.source_id,"community_name":d.snapshot.source.display_name,"account_display_name":d.account})).collect::<Vec<_>>() }),
    )
}
async fn fresh_descriptor(data_dir: &Path, snap: &Snapshot) -> Result<Value, AppError> {
    let response = routes::hub_request(
        data_dir,
        &snap.source,
        &snap.config,
        "GET",
        "/api/client/v1/capabilities",
        vec![],
        vec![],
    )
    .await?;
    let descriptor = routes::hub_response_json(response).await?;
    let own = identity(&snap.source, &snap.config)?;
    if hub::validate_descriptor(&own.origin, &descriptor)? != own.instance_id
        || descriptor
            .pointer("/instance/realm")
            .and_then(Value::as_str)
            != Some(&own.realm)
        || descriptor
            .pointer("/transports/iroh/peer_id")
            .and_then(Value::as_str)
            != Some(&own.peer_id)
        || descriptor
            .pointer("/transports/iroh/relay_url")
            .and_then(Value::as_str)
            != Some(&own.relay_url)
    {
        return Err(AppError::Forbidden(
            "Shared sign-in community identity differs from its independent pin".into(),
        ));
    }
    Ok(descriptor)
}
async fn eligible_donors(data_dir: &Path, target: &Snapshot) -> Result<Vec<Donor>, AppError> {
    let candidates = donors(data_dir, target)?;
    if candidates.is_empty() {
        return Ok(vec![]);
    }
    let b = fresh_descriptor(data_dir, target).await?;
    let own = identity(&target.source, &target.config)?;
    // Absence is optional capability support, never authorization by inference.
    if authorization(&b, &own).is_err() {
        return Ok(vec![]);
    }
    let mut eligible = vec![];
    for donor in candidates {
        let a = fresh_descriptor(data_dir, &donor.snapshot).await?;
        if reciprocal(
            &identity(&donor.snapshot.source, &donor.snapshot.config)?,
            &a,
            &own,
            &b,
        )
        .is_ok()
        {
            let conn = routes::open_db(data_dir)?;
            if snapshot_current(&conn, target)?
                && snapshot_current(&conn, &donor.snapshot)?
                && !has_accounts(&conn, &target.source.source_id)?
            {
                eligible.push(donor);
            }
        }
    }
    Ok(eligible)
}
fn snapshot_current(conn: &Connection, snap: &Snapshot) -> Result<bool, AppError> {
    let sid = &snap.source.source_id;
    let row: Option<(i64, bool, String, bool, String, String, String, String, String, String)> = conn.query_row(
        "SELECT a.generation,s.enabled,s.normalized_base_url,s.allow_local_network,s.policy_profile,h.instance_id,h.peer_id,h.relay_url,h.transport_mode,h.descriptor_json FROM remote_account_state a JOIN remote_sources s USING(source_id) JOIN remote_hub_transport h USING(source_id) WHERE source_id=?1",
        params![sid], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?,r.get(8)?,r.get(9)?)),
    ).optional()?;
    let mode = serde_json::to_value(snap.config.transport_mode).unwrap();
    Ok(row.is_some_and(
        |(g, e, origin, local, policy, instance, peer, relay, m, descriptor)| {
            g == snap.generation
                && e
                && origin == snap.source.normalized_base_url
                && local == snap.source.allow_local_network
                && policy == snap.source.policy_profile
                && instance == snap.config.instance_id
                && peer == snap.config.bootstrap.peer_id
                && relay == snap.config.bootstrap.relay_url
                && mode.as_str() == Some(m.as_str())
                && descriptor == snap.config.descriptor.to_string()
        },
    ))
}
fn attach_checked(data_dir: &Path, target: &Snapshot, donor: &Donor) -> Result<Value, AppError> {
    let mut conn = routes::open_db(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let donor_present: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM remote_connections WHERE source_id=?1 AND connection_id=?2 AND credential_ref=?3 AND trust_state='verified')",
        params![donor.snapshot.source.source_id,donor.connection_id,donor.credential_ref], |r|r.get(0))?;
    let current = routes::read_credential(data_dir, &donor.credential_ref)?;
    if !snapshot_current(&tx, target)?
        || !snapshot_current(&tx, &donor.snapshot)?
        || !donor_present
        || has_accounts(&tx, &target.source.source_id)?
        || current != donor.credential
        || current
            .expires_at
            .is_some_and(|e| e <= Utc::now().timestamp())
    {
        return Err(AppError::Upstream(
            StatusCode::CONFLICT,
            "Account or community changed while checking shared sign-in; try again".into(),
        ));
    }
    let cid = Uuid::new_v4().to_string();
    let binding = Binding {
        identity: identity(&target.source, &target.config)?,
        mode: target.config.transport_mode,
        credential_ref: donor.credential_ref.clone(),
    };
    tx.execute(
        "INSERT INTO remote_connections VALUES(?1,?2,?3,?4,'verified',?5)",
        params![
            cid,
            target.source.source_id,
            donor.account,
            donor.credential_ref,
            Utc::now().to_rfc3339()
        ],
    )?;
    tx.execute(
        "INSERT INTO remote_credential_aliases VALUES(?1,?2)",
        params![cid, serde_json::to_string(&binding).unwrap()],
    )?;
    tx.execute(
        "DELETE FROM remote_cache WHERE source_id=?1",
        params![target.source.source_id],
    )?;
    tx.execute(
        "DELETE FROM remote_item_cache WHERE source_id=?1",
        params![target.source.source_id],
    )?;
    tx.commit()?;
    Ok(
        json!({"state":"attached","source_id":target.source.source_id,"connection_id":cid,"account_display_name":donor.account}),
    )
}
pub(crate) async fn reuse(
    data_dir: &Path,
    sid: &str,
    selected: Option<&str>,
    automatic: bool,
) -> Result<Value, AppError> {
    let target = snapshot(data_dir, sid)?;
    if has_accounts(&routes::open_db(data_dir)?, sid)? {
        return Ok(json!({"state":"account_present"}));
    }
    if automatic && target.blocked {
        return Ok(json!({"state":"disconnected"}));
    }
    let candidates = eligible_donors(data_dir, &target).await?;
    let donor = if let Some(cid) = selected {
        candidates
            .into_iter()
            .find(|d| d.connection_id == cid)
            .ok_or_else(|| {
                AppError::BadRequest("Selected shared account is no longer eligible".into())
            })?
    } else if candidates.len() == 1 {
        candidates.into_iter().next().unwrap()
    } else {
        return Ok(
            json!({"state":if candidates.is_empty(){"unavailable"}else{"selection_required"}}),
        );
    };
    let a = fresh_descriptor(data_dir, &donor.snapshot).await?;
    let b = fresh_descriptor(data_dir, &target).await?;
    reciprocal(
        &identity(&donor.snapshot.source, &donor.snapshot.config)?,
        &a,
        &identity(&target.source, &target.config)?,
        &b,
    )?;
    routes::validate_relay(&target.config.bootstrap, target.source.allow_local_network).await?;
    // Recheck all mutable authorization state before transmitting the secret.
    {
        let conn = routes::open_db(data_dir)?;
        if !snapshot_current(&conn, &target)?
            || !snapshot_current(&conn, &donor.snapshot)?
            || has_accounts(&conn, sid)?
            || routes::read_credential(data_dir, &donor.credential_ref)? != donor.credential
            || donor
                .credential
                .expires_at
                .is_some_and(|expiry| expiry <= Utc::now().timestamp())
        {
            return Err(AppError::Upstream(
                StatusCode::CONFLICT,
                "Account or connection changed during shared sign-in".into(),
            ));
        }
    }
    HeaderValue::from_str(&donor.credential.secret)
        .map_err(|_| AppError::Unauthorized("Existing account has an invalid credential".into()))?;
    let response = transport::request(
        &target.config.bootstrap,
        target.config.transport_mode,
        transport::HubRequest {
            method: "GET".into(),
            path: "/api/client/v1/catalog?per_page=1".into(),
            headers: vec![(
                "Authorization".into(),
                format!("Bearer {}", donor.credential.secret),
            )],
            body: vec![],
        },
    )
    .await
    .map_err(AppError::ServiceUnavailable)?;
    accept_catalog(response).await?;
    attach_checked(data_dir, &target, &donor)
}
async fn accept_catalog(response: transport::HubResponse) -> Result<(), AppError> {
    if response.status != 200 {
        return Err(AppError::Upstream(StatusCode::from_u16(response.status).unwrap_or(StatusCode::BAD_GATEWAY),"Existing account was not accepted by this community; its current account was preserved".into()));
    }
    let bytes = response
        .into_bytes(256 * 1024)
        .await
        .map_err(AppError::ServiceUnavailable)?;
    let catalog: Value = serde_json::from_slice(&bytes).map_err(|_| {
        AppError::ServiceUnavailable("Shared sign-in catalog check returned invalid JSON".into())
    })?;
    if !catalog
        .get("items")
        .and_then(Value::as_array)
        .is_some_and(|items| items.len() <= 1)
    {
        return Err(AppError::ServiceUnavailable(
            "Shared sign-in catalog check was not a bounded catalog page".into(),
        ));
    }
    Ok(())
}
pub(crate) async fn reuse_after_signin(data_dir: &Path, signed_in_sid: &str) {
    let ids = (|| -> Result<Vec<String>, AppError> {
        let conn = routes::open_db(data_dir)?;
        let mut stmt=conn.prepare("SELECT h.source_id FROM remote_hub_transport h JOIN remote_sources s USING(source_id) WHERE s.enabled=1 AND h.source_id!=?1 AND NOT EXISTS(SELECT 1 FROM remote_connections c WHERE c.source_id=h.source_id) ORDER BY h.source_id LIMIT 64")?;
        let rows = stmt.query_map(params![signed_in_sid], |r| r.get(0))?;
        Ok(rows.collect::<Result<_, _>>()?)
    })();
    if let Ok(ids) = ids {
        for sid in ids {
            let _ = reuse(data_dir, &sid, None, true).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::online::provider;

    fn pair() -> (Identity, Identity) {
        let a = Identity {
            origin: "https://main.example".into(),
            instance_id: "00000000-0000-0000-0000-000000000001".into(),
            realm: "main".into(),
            peer_id: "a".repeat(64),
            relay_url: "https://relay.example".into(),
        };
        let mut b = a.clone();
        b.origin = "https://adult.example".into();
        b.realm = "adult".into();
        b.peer_id = "b".repeat(64);
        (a, b)
    }
    fn descriptor(own: &Identity, a: &Identity, b: &Identity) -> Value {
        json!({"protocol":"booru-node-client","version":"1.1","instance":{"id":own.instance_id,"name":own.realm,"origin":own.origin,"realm":own.realm},"access":{"catalog":"members","registration":"closed","submission":"members","moderation":"required","retention":"hosted"},"authentication":{"methods":["bearer"],"pkce_methods":["S256"],"authorization_endpoint":"/api/client/v1/authorize","token_endpoint":"/api/client/v1/token","shared_credentials":{"version":1,"realms":[a,b]}},"capabilities":{"catalog":true,"publish":true,"idempotency":true,"revisions":true,"tombstones":true,"byte_ranges":true,"native_authorization":true},"transports":{"iroh":{"peer_id":own.peer_id,"relay_url":own.relay_url}}})
    }
    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let conn = routes::open_db(dir.path()).unwrap();
        let (a, b) = pair();
        for (sid, own, policy) in [("a", &a, "safe"), ("b", &b, "unrestricted")] {
            conn.execute("INSERT INTO remote_sources VALUES(?1,'donutbooru',?1,?2,?3,1,0,?4,NULL,'synthetic')",params![sid,own.origin,policy,serde_json::to_string(&provider::capabilities("donutbooru")).unwrap()]).unwrap();
            conn.execute(
                "INSERT INTO remote_hub_transport VALUES(?1,?2,?3,?4,'relayed',?5)",
                params![
                    sid,
                    own.instance_id,
                    own.peer_id,
                    own.relay_url,
                    descriptor(own, &a, &b).to_string()
                ],
            )
            .unwrap();
        }
        store(
            dir.path(),
            "account.json",
            "synthetic-bearer",
            Some(Utc::now().timestamp() + 3600),
        );
        conn.execute("INSERT INTO remote_connections VALUES('donor','a','Test account','account.json','verified','synthetic')",[]).unwrap();
        dir
    }
    fn store(dir: &Path, reference: &str, secret: &str, expires_at: Option<i64>) {
        let credential = routes::StoredCredential {
            auth_kind: "bearer".into(),
            secret: secret.into(),
            expires_at,
        };
        crate::server::credentials::store_application_secret(
            &routes::credential_path(dir, reference).unwrap(),
            &serde_json::to_vec(&credential).unwrap(),
        )
        .unwrap();
    }
    fn ready(dir: &Path) -> (Snapshot, Donor) {
        let target = snapshot(dir, "b").unwrap();
        let donor = donors(dir, &target).unwrap().remove(0);
        (target, donor)
    }
    // AC: @verified-shared-hub-account ac-2
    #[test]
    fn migration_requires_explicit_reuse_for_legacy_empty_communities() {
        let dir = fixture();
        let conn = routes::open_db(dir.path()).unwrap();
        conn.execute("DELETE FROM remote_account_state WHERE source_id='b'", [])
            .unwrap();
        schema(&conn).unwrap();
        assert!(snapshot(dir.path(), "b").unwrap().blocked);
        // A newly confirmed enrollment opts into automatic reuse.
        assert!(!snapshot(dir.path(), "a").unwrap().blocked);
    }
    // AC: @verified-shared-hub-account ac-3
    #[test]
    fn reciprocal_authorization_requires_every_exact_field_and_both_descriptors() {
        let (a, b) = pair();
        let ad = descriptor(&a, &a, &b);
        let bd = descriptor(&b, &a, &b);
        assert!(reciprocal(&a, &ad, &b, &bd).is_ok());
        for field in ["origin", "instance_id", "realm", "peer_id", "relay_url"] {
            let mut changed = bd.clone();
            changed["authentication"]["shared_credentials"]["realms"][0][field] =
                json!("different");
            assert!(reciprocal(&a, &ad, &b, &changed).is_err(), "{field}");
        }
        for invalid in [
            json!(null),
            json!({"version":2,"realms":[a,b]}),
            json!({"version":1,"realms":[a]}),
        ] {
            let mut changed = bd.clone();
            changed["authentication"]["shared_credentials"] = invalid;
            assert!(reciprocal(&a, &ad, &b, &changed).is_err());
        }
        let mut third = b.clone();
        third.origin = "https://third.example".into();
        assert!(reciprocal(&a, &ad, &third, &descriptor(&third, &a, &third)).is_err());
    }
    // AC: @verified-shared-hub-account ac-1
    // AC: @verified-shared-hub-account ac-2
    #[test]
    fn attachment_shares_one_protected_reference_without_changing_sources() {
        let dir = fixture();
        let (target, donor) = ready(dir.path());
        let a_before = routes::load_source(dir.path(), "a").unwrap();
        let b_before = routes::load_source(dir.path(), "b").unwrap();
        let result = attach_checked(dir.path(), &target, &donor).unwrap();
        let conn = routes::open_db(dir.path()).unwrap();
        let refs: i64 = conn
            .query_row(
                "SELECT COUNT(DISTINCT credential_ref) FROM remote_connections",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(refs, 1);
        assert_eq!(
            std::fs::read_dir(
                routes::credential_path(dir.path(), "account.json")
                    .unwrap()
                    .parent()
                    .unwrap()
            )
            .unwrap()
            .count(),
            1
        );
        assert_eq!(a_before, routes::load_source(dir.path(), "a").unwrap());
        assert_eq!(b_before, routes::load_source(dir.path(), "b").unwrap());
        validate_alias(
            dir.path(),
            "b",
            result["connection_id"].as_str().unwrap(),
            "account.json",
        )
        .unwrap();
        assert!(result.get("secret").is_none());
        assert!(result.get("credential_ref").is_none());
    }
    // AC: @verified-shared-hub-account ac-6
    #[test]
    fn explicit_expired_target_account_is_never_replaced() {
        let dir = fixture();
        let (target, donor) = ready(dir.path());
        store(
            dir.path(),
            "explicit.json",
            "different-expired-account",
            Some(1),
        );
        routes::open_db(dir.path()).unwrap().execute("INSERT INTO remote_connections VALUES('explicit','b','Expired explicit account','explicit.json','verified','synthetic')",[]).unwrap();
        assert!(attach_checked(dir.path(), &target, &donor).is_err());
        let conn = routes::open_db(dir.path()).unwrap();
        assert_eq!(
            conn.query_row(
                "SELECT credential_ref FROM remote_connections WHERE source_id='b'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
            "explicit.json"
        );
    }
    // AC: @verified-shared-hub-account ac-5
    #[test]
    fn disconnect_identity_mode_and_secret_changes_invalidate_inflight_attach() {
        for mutation in [
            "DELETE FROM remote_connections WHERE connection_id='donor'",
            "UPDATE remote_connections SET trust_state='unverified' WHERE connection_id='donor'",
            "UPDATE remote_sources SET enabled=0 WHERE source_id='b'",
            "UPDATE remote_sources SET policy_profile='safe' WHERE source_id='b'",
            "UPDATE remote_hub_transport SET transport_mode='standard' WHERE source_id='b'",
            "UPDATE remote_hub_transport SET peer_id='changed' WHERE source_id='a'",
            "UPDATE remote_hub_transport SET descriptor_json='{}' WHERE source_id='b'",
        ] {
            let dir = fixture();
            let (target, donor) = ready(dir.path());
            routes::open_db(dir.path())
                .unwrap()
                .execute(mutation, [])
                .unwrap();
            assert!(
                attach_checked(dir.path(), &target, &donor).is_err(),
                "{mutation}"
            );
            assert!(!has_accounts(&routes::open_db(dir.path()).unwrap(), "b").unwrap());
        }
        let dir = fixture();
        let (target, donor) = ready(dir.path());
        store(
            dir.path(),
            "account.json",
            "changed-bearer",
            Some(Utc::now().timestamp() + 3600),
        );
        assert!(attach_checked(dir.path(), &target, &donor).is_err());
    }
    // AC: @verified-shared-hub-account ac-5
    #[test]
    fn surviving_alias_owns_secret_and_disconnect_creates_tombstone() {
        let dir = fixture();
        let (target, donor) = ready(dir.path());
        let attached = attach_checked(dir.path(), &target, &donor).unwrap();
        let conn = routes::open_db(dir.path()).unwrap();
        conn.execute(
            "DELETE FROM remote_connections WHERE connection_id='donor'",
            [],
        )
        .unwrap();
        remove_unused_secret(dir.path(), "account.json").unwrap();
        assert!(routes::credential_path(dir.path(), "account.json")
            .unwrap()
            .exists());
        assert!(snapshot(dir.path(), "a").unwrap().blocked);
        validate_alias(
            dir.path(),
            "b",
            attached["connection_id"].as_str().unwrap(),
            "account.json",
        )
        .unwrap();
        assert_eq!(
            routes::read_credential(dir.path(), "account.json")
                .unwrap()
                .secret,
            "synthetic-bearer"
        );
        conn.execute("DELETE FROM remote_sources WHERE source_id='b'", [])
            .unwrap();
        remove_unused_secret(dir.path(), "account.json").unwrap();
        assert!(!routes::credential_path(dir.path(), "account.json")
            .unwrap()
            .exists());
    }
    // AC: @verified-shared-hub-account ac-3
    #[test]
    fn alias_binding_is_checked_before_a_changed_pin_can_use_its_credential() {
        for mutation in [
            "UPDATE remote_hub_transport SET transport_mode='standard' WHERE source_id='b'",
            "UPDATE remote_hub_transport SET instance_id='other' WHERE source_id='b'",
            "UPDATE remote_hub_transport SET relay_url='https://other.example' WHERE source_id='b'",
        ] {
            let dir = fixture();
            let (target, donor) = ready(dir.path());
            let attached = attach_checked(dir.path(), &target, &donor).unwrap();
            routes::open_db(dir.path())
                .unwrap()
                .execute(mutation, [])
                .unwrap();
            assert!(validate_alias(
                dir.path(),
                "b",
                attached["connection_id"].as_str().unwrap(),
                "account.json"
            )
            .is_err());
        }
    }
    // AC: @verified-shared-hub-account ac-6
    #[test]
    fn distinct_accounts_remain_ambiguous_and_identical_references_are_deduplicated() {
        let dir = fixture();
        let conn = routes::open_db(dir.path()).unwrap();
        conn.execute("INSERT INTO remote_connections VALUES('same','a','Same shared secret','account.json','verified','synthetic')",[]).unwrap();
        let target = snapshot(dir.path(), "b").unwrap();
        assert_eq!(donors(dir.path(), &target).unwrap().len(), 1);
        store(
            dir.path(),
            "another.json",
            "another-synthetic-account",
            None,
        );
        conn.execute("INSERT INTO remote_connections VALUES('another','a','Another account','another.json','verified','synthetic')",[]).unwrap();
        let target = snapshot(dir.path(), "b").unwrap();
        assert_eq!(donors(dir.path(), &target).unwrap().len(), 2);
        assert!(!has_accounts(&conn, "b").unwrap());
    }
    // AC: @verified-shared-hub-account ac-5
    #[tokio::test]
    async fn automatic_reuse_does_not_reconnect_a_disconnected_or_legacy_source() {
        let dir = fixture();
        let conn = routes::open_db(dir.path()).unwrap();
        conn.execute(
            "UPDATE remote_account_state SET automatic_blocked=1 WHERE source_id='b'",
            [],
        )
        .unwrap();
        let result = reuse(dir.path(), "b", None, true).await.unwrap();
        assert_eq!(result["state"], "disconnected");
        assert!(!has_accounts(&conn, "b").unwrap());
    }
    fn response(status: u16, bytes: Vec<u8>) -> transport::HubResponse {
        transport::HubResponse {
            status,
            headers: vec![],
            body: Box::pin(futures_util::stream::once(async move { Ok(bytes) })),
        }
    }
    // AC: @verified-shared-hub-account ac-4
    #[tokio::test]
    async fn acceptance_requires_successful_bounded_catalog_not_auth_fallback_or_arbitrary_json() {
        accept_catalog(response(200, br#"{"items":[]}"#.to_vec()))
            .await
            .unwrap();
        for status in [401, 403, 302, 500] {
            assert!(
                accept_catalog(response(status, br#"{"items":[]}"#.to_vec()))
                    .await
                    .is_err()
            );
        }
        for bytes in [
            br#"{"items":[{},{}]}"#.to_vec(),
            br#"{"ok":true}"#.to_vec(),
            b"invalid JSON".to_vec(),
            vec![b' '; 256 * 1024 + 1],
        ] {
            assert!(accept_catalog(response(200, bytes)).await.is_err());
        }
    }
}
