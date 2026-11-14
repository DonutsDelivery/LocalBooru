// SPDX-License-Identifier: GPL-3.0-only
//! Explicit, source-qualified personal copies. Stored bytes never stand in for
//! a hub authorization grant; community serving has a separate fresh grant.
use crate::online::{
    hub::CatalogQuery,
    models::{RemoteItem, RemotePage, RemoteSource},
};
use crate::server::{error::AppError, state::AppState};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

pub const MAX_PENDING_ITEMS: usize = 500;
pub const MAX_VARIANT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_STORAGE_BYTES: u64 = 1024 * 1024 * 1024 * 1024;

pub fn open(data_dir: &Path) -> Result<Connection, AppError> {
    let conn = Connection::open(data_dir.join("online.db"))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS offline_sources(
        source_id TEXT PRIMARY KEY REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        enabled INTEGER NOT NULL DEFAULT 0,sync_all_creators INTEGER NOT NULL DEFAULT 0,
        storage_limit_bytes INTEGER NOT NULL DEFAULT 0,paused INTEGER NOT NULL DEFAULT 0,
        peer_hosting_enabled INTEGER NOT NULL DEFAULT 0,state TEXT NOT NULL DEFAULT 'disabled',
        error TEXT,last_sync_at TEXT,peer_error TEXT,peer_reported_at INTEGER,peer_reported_items INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS offline_creators(
        source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        creator_id TEXT NOT NULL,name TEXT NOT NULL,directory_json TEXT NOT NULL DEFAULT '{}',
        following INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(source_id,creator_id));
      CREATE TABLE IF NOT EXISTS offline_selections(
        source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        creator_id TEXT NOT NULL,media_type TEXT NOT NULL DEFAULT 'all',enabled INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(source_id,creator_id,media_type));
      CREATE TABLE IF NOT EXISTS offline_scans(
        source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        creator_id TEXT NOT NULL,media_type TEXT NOT NULL,page INTEGER NOT NULL DEFAULT 1,total INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(source_id,creator_id,media_type));
      CREATE TABLE IF NOT EXISTS offline_queue(
        source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        post_id TEXT NOT NULL,creator_id TEXT,media_type TEXT NOT NULL,revision TEXT NOT NULL,
        item_json TEXT NOT NULL,metadata_json TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'queued',error TEXT,
        PRIMARY KEY(source_id,post_id));
      CREATE TABLE IF NOT EXISTS offline_items(
        source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        post_id TEXT NOT NULL,creator_id TEXT,media_type TEXT NOT NULL,revision TEXT NOT NULL,
        item_json TEXT NOT NULL,metadata_json TEXT NOT NULL,withdrawn INTEGER NOT NULL DEFAULT 0,
        last_verified_at INTEGER NOT NULL,PRIMARY KEY(source_id,post_id));
      CREATE TABLE IF NOT EXISTS offline_variants(
        source_id TEXT NOT NULL REFERENCES remote_sources(source_id) ON DELETE CASCADE,
        post_id TEXT NOT NULL,revision TEXT NOT NULL,kind TEXT NOT NULL,
        sha256 TEXT NOT NULL,authoritative_digest INTEGER NOT NULL DEFAULT 0,
        bytes INTEGER NOT NULL,reserved_bytes INTEGER NOT NULL DEFAULT 0,
        relative_path TEXT,state TEXT NOT NULL,mime_type TEXT,
        PRIMARY KEY(source_id,post_id,revision,kind));
      CREATE INDEX IF NOT EXISTS offline_queue_state ON offline_queue(source_id,state);
      CREATE INDEX IF NOT EXISTS offline_items_creator ON offline_items(source_id,creator_id,media_type);
      CREATE INDEX IF NOT EXISTS offline_variants_path ON offline_variants(relative_path);")?;
    for (column, definition) in [
        ("peer_reported_at", "INTEGER"),
        ("peer_reported_items", "INTEGER NOT NULL DEFAULT 0"),
    ] {
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('offline_sources') WHERE name=?1)",
            params![column],
            |r| r.get(0),
        )?;
        if !exists {
            conn.execute(
                &format!("ALTER TABLE offline_sources ADD COLUMN {column} {definition}"),
                [],
            )?;
        }
    }
    Ok(conn)
}

#[derive(Debug, Default, Deserialize)]
pub struct SettingsPatch {
    pub sync_all_creators: Option<bool>,
    #[serde(alias = "storage_budget_bytes")]
    pub storage_limit_bytes: Option<u64>,
    pub paused: Option<bool>,
    pub peer_hosting_enabled: Option<bool>,
    #[serde(default)]
    pub confirmed_download: bool,
    #[serde(default)]
    pub confirmed_hosting: bool,
}
#[derive(Debug, Default, Deserialize)]
pub struct CreatorPatch {
    pub following: Option<bool>,
    pub offline_enabled: Option<bool>,
    pub media_type: Option<String>,
    #[serde(alias = "storage_budget_bytes")]
    pub storage_limit_bytes: Option<u64>,
    pub peer_hosting_enabled: Option<bool>,
    #[serde(default)]
    pub confirmed_download: bool,
    #[serde(default)]
    pub confirmed_hosting: bool,
}
fn media_type(raw: Option<&str>) -> Result<&str, AppError> {
    match raw {
        None | Some("all") => Ok("all"),
        Some("image") => Ok("image"),
        Some("video") => Ok("video"),
        _ => Err(AppError::BadRequest("Invalid creator media_type".into())),
    }
}
pub fn validate_creator_id(id: &str) -> Result<(), AppError> {
    if id.is_empty()
        || id.len() > 128
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err(AppError::BadRequest(
            "Invalid source creator identity".into(),
        ));
    }
    Ok(())
}
fn ensure_settings(conn: &Connection, source_id: &str) -> Result<(), AppError> {
    conn.execute(
        "INSERT OR IGNORE INTO offline_sources(source_id) VALUES(?1)",
        params![source_id],
    )?;
    Ok(())
}
fn set_cap(conn: &Connection, source_id: &str, cap: Option<u64>) -> Result<(), AppError> {
    if let Some(cap) = cap {
        if cap == 0 || cap > MAX_STORAGE_BYTES {
            return Err(AppError::BadRequest(
                "Storage limit must be positive and at most 1 TiB".into(),
            ));
        }
        let used:i64=conn.query_row("SELECT COALESCE(SUM(CASE WHEN state='ready' THEN bytes ELSE reserved_bytes END),0) FROM offline_variants WHERE source_id=?1",params![source_id],|r|r.get(0))?;
        if cap < used as u64 {
            return Err(AppError::BadRequest("Storage limit is below retained and reserved bytes; remove copies before lowering it".into()));
        }
        conn.execute(
            "UPDATE offline_sources SET storage_limit_bytes=?2 WHERE source_id=?1",
            params![source_id, cap as i64],
        )?;
    }
    Ok(())
}
fn hosting_patch(
    conn: &Connection,
    source_id: &str,
    enabled: Option<bool>,
    confirmed: bool,
) -> Result<(), AppError> {
    if enabled == Some(true) && !confirmed {
        return Err(AppError::BadRequest(
            "Confirm community hosting of selected copies while DMC runs".into(),
        ));
    }
    if let Some(enabled) = enabled {
        conn.execute(
            "UPDATE offline_sources SET peer_hosting_enabled=?2 WHERE source_id=?1",
            params![source_id, enabled],
        )?;
    }
    Ok(())
}
fn update_enabled(conn: &Connection, source_id: &str) -> Result<(), AppError> {
    conn.execute("UPDATE offline_sources SET enabled=(sync_all_creators=1 OR EXISTS(SELECT 1 FROM offline_selections s WHERE s.source_id=offline_sources.source_id AND s.enabled=1)),state=CASE WHEN paused=1 THEN 'paused' WHEN sync_all_creators=1 OR EXISTS(SELECT 1 FROM offline_selections s WHERE s.source_id=offline_sources.source_id AND s.enabled=1) THEN 'queued' ELSE 'disabled' END,error=NULL WHERE source_id=?1",params![source_id])?;
    Ok(())
}
pub fn patch_settings(
    data_dir: &Path,
    source_id: &str,
    patch: &SettingsPatch,
) -> Result<Value, AppError> {
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    ensure_settings(&tx, source_id)?;
    if patch.sync_all_creators == Some(true) && !patch.confirmed_download {
        return Err(AppError::BadRequest(
            "Confirm downloading all creators with an explicit storage limit".into(),
        ));
    }
    set_cap(&tx, source_id, patch.storage_limit_bytes)?;
    if patch.sync_all_creators == Some(true) {
        let cap: i64 = tx.query_row(
            "SELECT storage_limit_bytes FROM offline_sources WHERE source_id=?1",
            params![source_id],
            |r| r.get(0),
        )?;
        if cap <= 0 {
            return Err(AppError::BadRequest(
                "Choose a storage limit before enabling offline sync".into(),
            ));
        }
    }
    if let Some(all) = patch.sync_all_creators {
        tx.execute(
            "UPDATE offline_sources SET sync_all_creators=?2 WHERE source_id=?1",
            params![source_id, all],
        )?;
    }
    if let Some(paused) = patch.paused {
        tx.execute(
            "UPDATE offline_sources SET paused=?2 WHERE source_id=?1",
            params![source_id, paused],
        )?;
    }
    hosting_patch(
        &tx,
        source_id,
        patch.peer_hosting_enabled,
        patch.confirmed_hosting,
    )?;
    update_enabled(&tx, source_id)?;
    tx.commit()?;
    settings(data_dir, source_id)
}
pub fn remember_creator(data_dir: &Path, source_id: &str, row: &Value) -> Result<(), AppError> {
    let id = row
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::ServiceUnavailable("Hub creator lacks identity".into()))?;
    validate_creator_id(id)?;
    let name = row
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty() && n.len() <= 256)
        .ok_or_else(|| {
            AppError::ServiceUnavailable("Hub creator lacks a valid display name".into())
        })?;
    let mut row = row.clone();
    if row.get("availability").is_some() {
        row["directory_verified_at"] = json!(Utc::now().timestamp());
    }
    open(data_dir)?.execute("INSERT INTO offline_creators(source_id,creator_id,name,directory_json) VALUES(?1,?2,?3,?4) ON CONFLICT(source_id,creator_id) DO UPDATE SET name=excluded.name,directory_json=CASE WHEN json_type(excluded.directory_json,'$.availability') IS NULL THEN offline_creators.directory_json ELSE excluded.directory_json END",params![source_id,id,name,serde_json::to_string(&row).unwrap()])?;
    Ok(())
}
pub fn patch_creator(
    data_dir: &Path,
    source_id: &str,
    id: &str,
    patch: &CreatorPatch,
) -> Result<(), AppError> {
    validate_creator_id(id)?;
    let media = media_type(patch.media_type.as_deref())?;
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    ensure_settings(&tx, source_id)?;
    if patch.offline_enabled == Some(true) && !patch.confirmed_download {
        return Err(AppError::BadRequest(
            "Confirm downloading this creator with an explicit storage limit".into(),
        ));
    }
    if !tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM offline_creators WHERE source_id=?1 AND creator_id=?2)",
        params![source_id, id],
        |r| r.get::<_, bool>(0),
    )? {
        return Err(AppError::NotFound(
            "Creator is not in this source directory".into(),
        ));
    }
    set_cap(&tx, source_id, patch.storage_limit_bytes)?;
    if patch.offline_enabled == Some(true) {
        let cap: i64 = tx.query_row(
            "SELECT storage_limit_bytes FROM offline_sources WHERE source_id=?1",
            params![source_id],
            |r| r.get(0),
        )?;
        if cap <= 0 {
            return Err(AppError::BadRequest(
                "Choose a storage limit before enabling offline sync".into(),
            ));
        }
    }
    if let Some(following) = patch.following {
        tx.execute(
            "UPDATE offline_creators SET following=?3 WHERE source_id=?1 AND creator_id=?2",
            params![source_id, id, following],
        )?;
    }
    if let Some(enabled) = patch.offline_enabled {
        tx.execute("INSERT INTO offline_selections(source_id,creator_id,media_type,enabled) VALUES(?1,?2,?3,?4) ON CONFLICT(source_id,creator_id,media_type) DO UPDATE SET enabled=excluded.enabled",params![source_id,id,media,enabled])?;
    }
    hosting_patch(
        &tx,
        source_id,
        patch.peer_hosting_enabled,
        patch.confirmed_hosting,
    )?;
    update_enabled(&tx, source_id)?;
    tx.commit()?;
    Ok(())
}
pub fn settings(data_dir: &Path, source_id: &str) -> Result<Value, AppError> {
    let conn = open(data_dir)?;
    ensure_settings(&conn, source_id)?;
    let mut result=conn.query_row("SELECT enabled,sync_all_creators,storage_limit_bytes,paused,peer_hosting_enabled,state,error,last_sync_at,peer_error,peer_reported_at,peer_reported_items FROM offline_sources WHERE source_id=?1",params![source_id],|r|Ok(json!({"enabled":r.get::<_,bool>(0)?,"sync_all_creators":r.get::<_,bool>(1)?,"storage_limit_bytes":r.get::<_,i64>(2)?,"paused":r.get::<_,bool>(3)?,"peer_hosting_enabled":r.get::<_,bool>(4)?,"state":r.get::<_,String>(5)?,"error":r.get::<_,Option<String>>(6)?,"last_sync_at":r.get::<_,Option<String>>(7)?,"peer_hosting_error":r.get::<_,Option<String>>(8)?,"peer_reported_at":r.get::<_,Option<i64>>(9)?,"peer_reported_items":r.get::<_,i64>(10)?})))?;
    let (bytes,reserved,variants):(i64,i64,i64)=conn.query_row("SELECT COALESCE(SUM(CASE WHEN state='ready' THEN bytes ELSE 0 END),0),COALESCE(SUM(CASE WHEN state!='ready' THEN reserved_bytes ELSE 0 END),0),COALESCE(SUM(state='ready'),0) FROM offline_variants WHERE source_id=?1",params![source_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
    let items: i64 = conn.query_row(
        "SELECT COUNT(*) FROM offline_items i WHERE source_id=?1 AND json_array_length(item_json,'$.media')>0 AND NOT EXISTS(SELECT 1 FROM json_each(i.item_json,'$.media') m WHERE NOT EXISTS(SELECT 1 FROM offline_variants v WHERE v.source_id=i.source_id AND v.post_id=i.post_id AND v.revision=i.revision AND v.kind=json_extract(m.value,'$.kind') AND v.state='ready'))",
        params![source_id],
        |r| r.get(0),
    )?;
    let pending: i64 = conn.query_row(
        "SELECT COUNT(*) FROM offline_queue WHERE source_id=?1",
        params![source_id],
        |r| r.get(0),
    )?;
    let expected: i64 = conn.query_row(
        "SELECT COALESCE(SUM(total),0) FROM offline_scans WHERE source_id=?1",
        params![source_id],
        |r| r.get(0),
    )?;
    let total_variants:i64=conn.query_row("SELECT COALESCE(SUM(json_array_length(item_json,'$.media')),0) FROM offline_items WHERE source_id=?1",params![source_id],|r|r.get(0))?;
    result["downloaded_bytes"] = json!(bytes);
    result["reserved_bytes"] = json!(reserved);
    result["downloaded_items"] = json!(items);
    result["pending_items"] = json!(pending);
    result["total_items"] = json!(expected.max(items + pending));
    result["downloaded_variants"] = json!(variants);
    result["total_variants"] = json!(total_variants.max(variants));
    result["partial"] = json!(
        result["state"] != "ready" || variants < total_variants || items < expected || pending > 0
    );
    let source_enabled: bool = conn.query_row(
        "SELECT enabled FROM remote_sources WHERE source_id=?1",
        params![source_id],
        |r| r.get(0),
    )?;
    result["peer_hosting_state"] = json!(if !result["peer_hosting_enabled"]
        .as_bool()
        .unwrap_or(false)
    {
        "disabled"
    } else if result["peer_hosting_error"].is_string() {
        "error"
    } else if !source_enabled
        || result["enabled"] == false
        || result["peer_reported_items"].as_u64().unwrap_or(0) == 0
        || result["peer_reported_at"]
            .as_i64()
            .is_none_or(|timestamp| timestamp <= Utc::now().timestamp() - 120)
    {
        "idle"
    } else {
        "hosting"
    });
    Ok(result)
}
pub fn state(
    data_dir: &Path,
    source_id: &str,
    status: &str,
    error: Option<&str>,
) -> Result<(), AppError> {
    open(data_dir)?.execute(
        "UPDATE offline_sources SET state=?2,error=?3 WHERE source_id=?1",
        params![source_id, status, error],
    )?;
    Ok(())
}
pub fn stop_source(data_dir: &Path, source_id: &str) -> Result<(), AppError> {
    open(data_dir)?.execute(
        "UPDATE offline_sources SET paused=1,state='paused' WHERE source_id=?1",
        params![source_id],
    )?;
    Ok(())
}
pub fn active(data_dir: &Path, source_id: &str) -> Result<bool, AppError> {
    Ok(open(data_dir)?
        .query_row(
            "SELECT enabled=1 AND paused=0 FROM offline_sources WHERE source_id=?1",
            params![source_id],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(false))
}
pub fn selected(
    data_dir: &Path,
    sid: &str,
    creator: Option<&str>,
    media: &str,
) -> Result<bool, AppError> {
    let conn = open(data_dir)?;
    let explicit: Option<bool> = if let Some(creator) = creator {
        conn.query_row("SELECT enabled FROM offline_selections WHERE source_id=?1 AND creator_id=?2 AND media_type IN('all',?3) ORDER BY media_type=?3 DESC LIMIT 1",params![sid,creator,media],|r|r.get(0)).optional()?
    } else {
        None
    };
    if let Some(enabled) = explicit {
        return Ok(enabled);
    }
    Ok(conn
        .query_row(
            "SELECT sync_all_creators FROM offline_sources WHERE source_id=?1",
            params![sid],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(false))
}
pub fn scopes(data_dir: &Path, source_id: &str) -> Result<Vec<(Option<String>, String)>, AppError> {
    let settings = settings(data_dir, source_id)?;
    if settings["sync_all_creators"] == true {
        return Ok(vec![(None, "all".into())]);
    }
    let conn = open(data_dir)?;
    let mut stmt=conn.prepare("SELECT creator_id,media_type FROM offline_selections WHERE source_id=?1 AND enabled=1 ORDER BY creator_id,media_type")?;
    let rows = stmt.query_map(params![source_id], |r| Ok((Some(r.get(0)?), r.get(1)?)))?;
    Ok(rows.collect::<Result<_, _>>()?)
}
pub fn scan_cursor(
    data_dir: &Path,
    sid: &str,
    creator: Option<&str>,
    media: &str,
) -> Result<u32, AppError> {
    let conn = open(data_dir)?;
    conn.execute(
        "INSERT OR IGNORE INTO offline_scans(source_id,creator_id,media_type) VALUES(?1,?2,?3)",
        params![sid, creator.unwrap_or(""), media],
    )?;
    Ok(conn.query_row(
        "SELECT page FROM offline_scans WHERE source_id=?1 AND creator_id=?2 AND media_type=?3",
        params![sid, creator.unwrap_or(""), media],
        |r| r.get(0),
    )?)
}
pub fn scan_advance(
    data_dir: &Path,
    sid: &str,
    creator: Option<&str>,
    media: &str,
    page: u32,
    total: u64,
) -> Result<(), AppError> {
    open(data_dir)?.execute("UPDATE offline_scans SET page=?4,total=?5 WHERE source_id=?1 AND creator_id=?2 AND media_type=?3",params![sid,creator.unwrap_or(""),media,page,total as i64])?;
    Ok(())
}
pub fn pending_count(data_dir: &Path, sid: &str) -> Result<usize, AppError> {
    Ok(open(data_dir)?.query_row(
        "SELECT COUNT(*) FROM offline_queue WHERE source_id=?1",
        params![sid],
        |r| r.get::<_, i64>(0),
    )? as usize)
}
#[derive(Clone)]
pub struct QueuedItem {
    pub item: RemoteItem,
    pub metadata: Value,
    pub revision: String,
}
fn revision(item: &RemoteItem) -> String {
    item.remote_revision.clone().unwrap_or_else(|| {
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(item).unwrap_or_default())
        )
    })
}
pub fn item_media(item: &RemoteItem) -> &str {
    if item.media.iter().any(|v| {
        v.kind == "original"
            && v.mime_type
                .as_deref()
                .is_some_and(|m| m.starts_with("video/"))
    }) {
        "video"
    } else {
        "image"
    }
}
pub fn enqueue(
    data_dir: &Path,
    sid: &str,
    item: &RemoteItem,
    metadata: &Value,
) -> Result<bool, AppError> {
    let rev = revision(item);
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let current: Option<(String, String)> = tx
        .query_row(
            "SELECT revision,item_json FROM offline_items WHERE source_id=?1 AND post_id=?2",
            params![sid, item.remote_post_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    // If a provider corrects the authoritative identity without changing its
    // publication revision, keep the old personal snapshot until a verified
    // replacement is complete, using a private storage generation key.
    if current.as_ref().is_some_and(|(current, _)| current == &rev)
        && item.media.iter().any(|variant| {
            variant_ready(&tx, sid, &item.remote_post_id, &rev, &variant.kind).unwrap_or(false)
                && !variant_matches(&tx, sid, &item.remote_post_id, &rev, variant).unwrap_or(false)
        })
    {
        let retained = format!("{}-retained-{}", rev, uuid::Uuid::new_v4());
        tx.execute("UPDATE offline_variants SET revision=?4 WHERE source_id=?1 AND post_id=?2 AND revision=?3",params![sid,item.remote_post_id,rev,retained])?;
        tx.execute(
            "UPDATE offline_items SET revision=?3 WHERE source_id=?1 AND post_id=?2",
            params![sid, item.remote_post_id, retained],
        )?;
    }
    if current.as_ref().is_some_and(|(current, _)| current == &rev)
        && item
            .media
            .iter()
            .all(|v| variant_matches(&tx, sid, &item.remote_post_id, &rev, v).unwrap_or(false))
    {
        tx.execute("UPDATE offline_items SET last_verified_at=?3,withdrawn=0,metadata_json=?4 WHERE source_id=?1 AND post_id=?2",params![sid,item.remote_post_id,Utc::now().timestamp(),serde_json::to_string(metadata).unwrap()])?;
        tx.commit()?;
        refresh_metadata(data_dir, sid, item, metadata)?;
        return Ok(false);
    }
    let pending: i64 = tx.query_row(
        "SELECT COUNT(*) FROM offline_queue WHERE source_id=?1",
        params![sid],
        |r| r.get(0),
    )?;
    if pending >= MAX_PENDING_ITEMS as i64 {
        return Ok(false);
    }
    tx.execute("INSERT INTO offline_queue(source_id,post_id,creator_id,media_type,revision,item_json,metadata_json) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(source_id,post_id) DO UPDATE SET revision=excluded.revision,item_json=excluded.item_json,metadata_json=excluded.metadata_json,state='queued',error=NULL",params![sid,item.remote_post_id,item.creator.as_ref().map(|c|&c.id),item_media(item),rev,serde_json::to_string(item).unwrap(),serde_json::to_string(metadata).unwrap()])?;
    tx.commit()?;
    Ok(true)
}
pub fn queue(data_dir: &Path, sid: &str, limit: usize) -> Result<Vec<QueuedItem>, AppError> {
    let conn = open(data_dir)?;
    let mut stmt=conn.prepare("SELECT item_json,metadata_json,revision FROM offline_queue WHERE source_id=?1 ORDER BY state='error',post_id LIMIT ?2")?;
    let rows = stmt.query_map(params![sid, limit as i64], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    rows.map(|r| {
        let (item, metadata, revision) = r?;
        Ok(QueuedItem {
            item: serde_json::from_str(&item)
                .map_err(|_| AppError::Internal("Corrupt offline queue metadata".into()))?,
            metadata: serde_json::from_str(&metadata)
                .map_err(|_| AppError::Internal("Corrupt offline queue metadata".into()))?,
            revision,
        })
    })
    .collect()
}
fn variant_ready(
    conn: &Connection,
    sid: &str,
    pid: &str,
    rev: &str,
    kind: &str,
) -> Result<bool, AppError> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM offline_variants WHERE source_id=?1 AND post_id=?2 AND revision=?3 AND kind=?4 AND state='ready')",params![sid,pid,rev,kind],|r|r.get(0))?)
}
fn variant_matches(
    conn: &Connection,
    sid: &str,
    pid: &str,
    rev: &str,
    variant: &crate::online::models::RemoteMediaVariant,
) -> Result<bool, AppError> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM offline_variants WHERE source_id=?1 AND post_id=?2 AND revision=?3 AND kind=?4 AND state='ready' AND (?5 IS NULL OR sha256=?5) AND (?6 IS NULL OR bytes=?6))",params![sid,pid,rev,variant.kind,variant.sha256,variant.byte_size.map(|n|n as i64)],|r|r.get(0))?)
}
pub fn can_reuse_variant(
    data_dir: &Path,
    sid: &str,
    pid: &str,
    rev: &str,
    variant: &crate::online::models::RemoteMediaVariant,
) -> Result<bool, AppError> {
    variant_matches(&open(data_dir)?, sid, pid, rev, variant)
}
pub fn is_ready(
    data_dir: &Path,
    sid: &str,
    pid: &str,
    rev: &str,
    kind: &str,
) -> Result<bool, AppError> {
    variant_ready(&open(data_dir)?, sid, pid, rev, kind)
}
pub fn reserve(
    data_dir: &Path,
    sid: &str,
    pid: &str,
    rev: &str,
    kind: &str,
    bytes: u64,
) -> Result<(), AppError> {
    if bytes > MAX_VARIANT_BYTES {
        return Err(AppError::BadRequest("Offline variant exceeds 2 GiB".into()));
    }
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (enabled, cap): (bool, i64) = tx.query_row(
        "SELECT enabled=1 AND paused=0,storage_limit_bytes FROM offline_sources WHERE source_id=?1",
        params![sid],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if !enabled {
        return Err(AppError::Forbidden("Offline sync is stopped".into()));
    }
    let used:i64=tx.query_row("SELECT COALESCE(SUM(CASE WHEN state='ready' THEN bytes ELSE reserved_bytes END),0) FROM offline_variants WHERE source_id=?1 AND NOT(post_id=?2 AND revision=?3 AND kind=?4 AND state!='ready')",params![sid,pid,rev,kind],|r|r.get(0))?;
    if used as u64 + bytes > cap as u64 {
        return Err(AppError::BadRequest("Offline storage limit reached".into()));
    }
    tx.execute("INSERT INTO offline_variants(source_id,post_id,revision,kind,sha256,bytes,reserved_bytes,state) VALUES(?1,?2,?3,?4,'',0,?5,'downloading') ON CONFLICT(source_id,post_id,revision,kind) DO UPDATE SET reserved_bytes=excluded.reserved_bytes,state='downloading'",params![sid,pid,rev,kind,bytes as i64])?;
    tx.commit()?;
    Ok(())
}
pub fn release(
    data_dir: &Path,
    sid: &str,
    pid: &str,
    rev: &str,
    kind: &str,
) -> Result<(), AppError> {
    open(data_dir)?.execute("DELETE FROM offline_variants WHERE source_id=?1 AND post_id=?2 AND revision=?3 AND kind=?4 AND state!='ready'",params![sid,pid,rev,kind])?;
    Ok(())
}
pub fn cache_root(data_dir: &Path, sid: &str) -> PathBuf {
    data_dir
        .join("online-offline")
        .join(format!("{:x}", Sha256::digest(sid.as_bytes())))
}
pub fn finish(
    data_dir: &Path,
    sid: &str,
    queued: &QueuedItem,
    kind: &str,
    digest: &str,
    bytes: u64,
    path: &Path,
    authoritative: bool,
    mime: Option<&str>,
) -> Result<(), AppError> {
    let relative = path
        .strip_prefix(data_dir.join("online-offline"))
        .map_err(|_| AppError::Internal("Offline file is outside cache".into()))?
        .to_string_lossy()
        .to_string();
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let selection: bool = tx.query_row("SELECT COALESCE((SELECT enabled FROM offline_selections WHERE source_id=?1 AND creator_id=?2 AND media_type IN('all',?3) ORDER BY media_type=?3 DESC LIMIT 1),(SELECT sync_all_creators FROM offline_sources WHERE source_id=?1),0) AND EXISTS(SELECT 1 FROM remote_sources WHERE source_id=?1 AND enabled=1)",params![sid,queued.item.creator.as_ref().map(|c|c.id.as_str()),item_media(&queued.item)],|r|r.get(0))?;
    if !selection {
        return Err(AppError::Forbidden(
            "Offline selection stopped before commit".into(),
        ));
    }
    let changed = tx.execute("UPDATE offline_variants SET sha256=?5,bytes=?6,reserved_bytes=0,relative_path=?7,state='ready',authoritative_digest=?8,mime_type=?9 WHERE source_id=?1 AND post_id=?2 AND revision=?3 AND kind=?4 AND state='downloading' AND EXISTS(SELECT 1 FROM offline_sources WHERE source_id=?1 AND enabled=1 AND paused=0)",params![sid,queued.item.remote_post_id,queued.revision,kind,digest,bytes as i64,relative,authoritative,mime])?;
    if changed != 1 {
        return Err(AppError::Forbidden(
            "Offline reservation was stopped or removed".into(),
        ));
    }
    // Publish metadata only after the complete new snapshot is retained. A
    // failed update keeps the prior personal snapshot and its policy together.
    let complete = queued.item.media.iter().all(|variant| {
        variant_ready(
            &tx,
            sid,
            &queued.item.remote_post_id,
            &queued.revision,
            &variant.kind,
        )
        .unwrap_or(false)
    });
    if complete {
        tx.execute("INSERT INTO offline_items(source_id,post_id,creator_id,media_type,revision,item_json,metadata_json,last_verified_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(source_id,post_id) DO UPDATE SET revision=excluded.revision,item_json=excluded.item_json,metadata_json=excluded.metadata_json,last_verified_at=excluded.last_verified_at,withdrawn=0",params![sid,queued.item.remote_post_id,queued.item.creator.as_ref().map(|c|&c.id),item_media(&queued.item),queued.revision,serde_json::to_string(&queued.item).unwrap(),serde_json::to_string(&queued.metadata).unwrap(),Utc::now().timestamp()])?;
    }
    tx.commit()?;
    Ok(())
}
pub fn complete_queue(data_dir: &Path, sid: &str, pid: &str) -> Result<(), AppError> {
    open(data_dir)?.execute(
        "DELETE FROM offline_queue WHERE source_id=?1 AND post_id=?2",
        params![sid, pid],
    )?;
    Ok(())
}
pub fn fail_queue(data_dir: &Path, sid: &str, pid: &str, error: &str) -> Result<(), AppError> {
    open(data_dir)?.execute(
        "UPDATE offline_queue SET state='error',error=?3 WHERE source_id=?1 AND post_id=?2",
        params![sid, pid, error],
    )?;
    Ok(())
}
pub fn withdraw(data_dir: &Path, sid: &str, pid: &str) -> Result<(), AppError> {
    let conn = open(data_dir)?;
    conn.execute(
        "UPDATE offline_items SET withdrawn=1 WHERE source_id=?1 AND post_id=?2",
        params![sid, pid],
    )?;
    conn.execute(
        "DELETE FROM offline_queue WHERE source_id=?1 AND post_id=?2",
        params![sid, pid],
    )?;
    Ok(())
}

pub fn safe_rating(rating: Option<&str>, profile: &str, family: bool) -> bool {
    if family || matches!(profile, "safe" | "adult-hidden") {
        matches!(rating.unwrap_or("pg"), "pg" | "pg13")
    } else {
        true
    }
}
#[derive(Debug, Clone)]
pub struct RetainedVariant {
    pub source_id: String,
    pub remote_post_id: String,
    pub kind: String,
    pub revision: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub mime_type: Option<String>,
    pub community_hosting: bool,
    pub authoritative_digest: bool,
    pub canonical_origin: String,
    pub instance_id: String,
}
pub fn verify_file(record: &RetainedVariant) -> Result<(), String> {
    use std::io::Read;
    let mut file = std::fs::File::open(&record.path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let read = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        if bytes > record.bytes {
            return Err("Retained media size changed".into());
        }
        hash.update(&buffer[..read]);
    }
    if bytes != record.bytes || format!("{:x}", hash.finalize()) != record.sha256 {
        return Err("Retained media digest changed".into());
    }
    Ok(())
}
pub fn personal_variant(
    data_dir: &Path,
    sid: &str,
    pid: &str,
    kind: &str,
) -> Result<Option<RetainedVariant>, AppError> {
    let conn = open(data_dir)?;
    let row:Option<(String,String,i64,String,Option<String>,bool)>=conn.query_row("SELECT v.revision,v.sha256,v.bytes,v.relative_path,v.mime_type,v.authoritative_digest FROM offline_variants v JOIN offline_items i ON i.source_id=v.source_id AND i.post_id=v.post_id WHERE v.source_id=?1 AND v.post_id=?2 AND v.kind=?3 AND v.state='ready' AND v.revision=i.revision ORDER BY v.rowid DESC LIMIT 1",params![sid,pid,kind],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).optional()?;
    let Some((revision, sha256, bytes, relative, mime_type, authoritative_digest)) = row else {
        return Ok(None);
    };
    let root = data_dir.join("online-offline");
    let path = root.join(relative);
    let Ok(path) = std::fs::canonicalize(&path) else {
        return Ok(None);
    };
    let root = std::fs::canonicalize(root)?;
    if !path.starts_with(&root)
        || !path.is_file()
        || std::fs::metadata(&path)?.len() != bytes as u64
    {
        return Ok(None);
    }
    let (origin,instance):(String,String)=conn.query_row("SELECT s.normalized_base_url,h.instance_id FROM remote_sources s JOIN remote_hub_transport h USING(source_id) WHERE source_id=?1",params![sid],|r|Ok((r.get(0)?,r.get(1)?)))?;
    Ok(Some(RetainedVariant {
        source_id: sid.into(),
        remote_post_id: pid.into(),
        kind: kind.into(),
        revision,
        path,
        sha256,
        bytes: bytes as u64,
        mime_type,
        community_hosting: false,
        authoritative_digest,
        canonical_origin: origin,
        instance_id: instance,
    }))
}
pub fn personal_item(
    data_dir: &Path,
    sid: &str,
    pid: &str,
) -> Result<Option<RemoteItem>, AppError> {
    let text: Option<String> = open(data_dir)?
        .query_row(
            "SELECT item_json FROM offline_items WHERE source_id=?1 AND post_id=?2",
            params![sid, pid],
            |r| r.get(0),
        )
        .optional()?;
    text.map(|text| {
        serde_json::from_str(&text).map_err(|_| AppError::Internal("Corrupt offline item".into()))
    })
    .transpose()
}
pub fn retained_variant(
    data_dir: &Path,
    sid: &str,
    pid: &str,
    kind: &str,
    family_locked: bool,
) -> Result<Option<RetainedVariant>, String> {
    let result = (|| -> Result<Option<RetainedVariant>, AppError> {
        let conn = open(data_dir)?;
        let eligible:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM offline_items i JOIN offline_sources o USING(source_id) JOIN remote_sources s USING(source_id) WHERE i.source_id=?1 AND i.post_id=?2 AND i.withdrawn=0 AND o.enabled=1 AND o.peer_hosting_enabled=1 AND o.state NOT IN('auth_required','error') AND s.enabled=1 )",params![sid,pid],|r|r.get(0))?;
        if !eligible {
            return Ok(None);
        };
        let profile: String = conn.query_row(
            "SELECT policy_profile FROM remote_sources WHERE source_id=?1",
            params![sid],
            |r| r.get(0),
        )?;
        let Some(item) = personal_item(data_dir, sid, pid)? else {
            return Ok(None);
        };
        if !selected(
            data_dir,
            sid,
            item.creator.as_ref().map(|c| c.id.as_str()),
            item_media(&item),
        )? || !safe_rating(item.rating.as_deref(), &profile, family_locked)
        {
            return Ok(None);
        }
        let Some(mut record) = personal_variant(data_dir, sid, pid, kind)? else {
            return Ok(None);
        };
        if !record.authoritative_digest || record.revision != revision(&item) {
            return Ok(None);
        };
        record.community_hosting = true;
        Ok(Some(record))
    })();
    result.map_err(|e| e.to_string())
}
pub fn hosted_manifest(
    data_dir: &Path,
    sid: &str,
    family_locked: bool,
) -> Result<Vec<RetainedVariant>, String> {
    hosted_manifest_page(data_dir, sid, family_locked, "", "").map(|(records, _)| records)
}
/// Iterate the retained catalog in bounded pages. Advance over raw identities
/// even when a record is currently ineligible, so later providers stay reachable.
pub fn hosted_manifest_page(
    data_dir: &Path,
    sid: &str,
    family_locked: bool,
    after_post: &str,
    after_kind: &str,
) -> Result<(Vec<RetainedVariant>, Option<(String, String)>), String> {
    let rows=(||->Result<Vec<(String,String)>,AppError>{
        let conn=open(data_dir)?;
        let mut stmt=conn.prepare("SELECT DISTINCT post_id,kind FROM offline_variants WHERE source_id=?1 AND state='ready' AND authoritative_digest=1 AND (post_id>?2 OR (post_id=?2 AND kind>?3)) ORDER BY post_id,kind LIMIT 100")?;
        let rows=stmt.query_map(params![sid,after_post,after_kind],|r|Ok((r.get(0)?,r.get(1)?)))?;
        Ok(rows.collect::<Result<_,_>>()?)
    })().map_err(|e|e.to_string())?;
    let next = if rows.len() == 100 {
        rows.last().cloned()
    } else {
        None
    };
    let mut records = Vec::with_capacity(rows.len());
    for (pid, kind) in rows {
        if let Some(record) = retained_variant(data_dir, sid, &pid, &kind, family_locked)? {
            records.push(record);
        }
    }
    Ok((records, next))
}

/// SQL applies filters/counts/order before loading a bounded page of metadata.
/// User values always bind; only fixed expressions enter the SQL text.
fn local_filter(
    sid: &str,
    q: &CatalogQuery,
    source: &RemoteSource,
    family: bool,
) -> (String, Vec<rusqlite::types::Value>) {
    use rusqlite::types::Value as SqlValue;
    let mut conditions=vec!["i.source_id=?".to_owned(),"EXISTS(SELECT 1 FROM offline_variants v WHERE v.source_id=i.source_id AND v.post_id=i.post_id AND v.revision=i.revision AND v.state='ready')".to_owned()];
    let mut values = vec![SqlValue::Text(sid.into())];
    let mut bind = |condition: &str, value: SqlValue| {
        conditions.push(condition.into());
        values.push(value);
    };
    if family || matches!(source.policy_profile.as_str(), "safe" | "adult-hidden") {
        bind(
            "COALESCE(json_extract(i.item_json,'$.rating'),'pg') IN('pg','pg13') AND ?=1",
            SqlValue::Integer(1),
        );
    }
    if let Some(id) = &q.creator_id {
        bind("i.creator_id=?", SqlValue::Text(id.clone()));
    }
    if let Some(media) = &q.media_type {
        bind("i.media_type=?", SqlValue::Text(media.clone()));
    }
    if let Some(ratings) = &q.rating {
        bind(
            "instr(',' || ? || ',', ',' || json_extract(i.item_json,'$.rating') || ',')>0",
            SqlValue::Text(ratings.clone()),
        );
    }
    if q.favorites_only {
        bind(
            "COALESCE(json_extract(i.metadata_json,'$.is_favorite'),0)=?",
            SqlValue::Integer(1),
        );
    }
    for term in q.q.as_deref().unwrap_or_default().split_whitespace() {
        let exclude = term.starts_with('-');
        bind(
            if exclude {
                "NOT EXISTS(SELECT 1 FROM json_each(i.item_json,'$.tags') t WHERE lower(t.value)=?)"
            } else {
                "EXISTS(SELECT 1 FROM json_each(i.item_json,'$.tags') t WHERE lower(t.value)=?)"
            },
            SqlValue::Text(term.trim_start_matches(['+', '-']).to_lowercase()),
        );
    }
    if let Some(min) = q.min_width {
        bind(
            "COALESCE(json_extract(i.item_json,'$.width'),0)>=?",
            SqlValue::Integer(min as i64),
        );
    }
    if let Some(min) = q.min_height {
        bind("min(COALESCE(json_extract(i.item_json,'$.width'),0),COALESCE(json_extract(i.item_json,'$.height'),0))>=?",SqlValue::Integer(min as i64));
    }
    if let Some(orientation) = q.orientation.as_deref() {
        bind(match orientation{"landscape"=>"json_extract(i.item_json,'$.width')>json_extract(i.item_json,'$.height') AND ?=1","portrait"=>"json_extract(i.item_json,'$.height')>json_extract(i.item_json,'$.width') AND ?=1",_=>"json_extract(i.item_json,'$.width')=json_extract(i.item_json,'$.height') AND ?=1"},SqlValue::Integer(1));
    }
    if let Some(min) = q.min_duration {
        bind(
            "json_extract(i.item_json,'$.duration')>=?",
            SqlValue::Real(min),
        );
    }
    if let Some(max) = q.max_duration {
        bind(
            "json_extract(i.item_json,'$.duration')<=?",
            SqlValue::Real(max),
        );
    }
    if let Some(time) = q.timeframe.as_deref() {
        let seconds = match time {
            "day" => 86400,
            "week" => 7 * 86400,
            "month" => 30 * 86400,
            _ => 365 * 86400,
        };
        bind(
            "unixepoch(json_extract(i.metadata_json,'$.created_at'))>=?",
            SqlValue::Integer(Utc::now().timestamp() - seconds),
        );
    }
    (conditions.join(" AND "), values)
}
fn local_order(sort: Option<&str>) -> Result<&'static str, AppError> {
    Ok(match sort.unwrap_or("newest"){
        "newest"=>"unixepoch(json_extract(i.metadata_json,'$.created_at')) DESC",
        "oldest"=>"unixepoch(json_extract(i.metadata_json,'$.created_at')) ASC",
        "filename_asc"=>"json_extract(i.metadata_json,'$.filename') ASC",
        "filename_desc"=>"json_extract(i.metadata_json,'$.filename') DESC",
        "filesize_largest"=>"json_extract(i.metadata_json,'$.file_size') DESC",
        "filesize_smallest"=>"json_extract(i.metadata_json,'$.file_size') ASC",
        "resolution_high"=>"json_extract(i.item_json,'$.width') * json_extract(i.item_json,'$.height') DESC",
        "resolution_low"=>"json_extract(i.item_json,'$.width') * json_extract(i.item_json,'$.height') ASC",
        "duration_longest"=>"json_extract(i.item_json,'$.duration') DESC",
        "duration_shortest"=>"json_extract(i.item_json,'$.duration') IS NULL, json_extract(i.item_json,'$.duration') ASC",
        "random"=>"(CAST(i.post_id AS INTEGER)*1103515245+12345)%2147483647",
        _=>return Err(AppError::BadRequest("Offline copies do not retain community vote/view ranking".into()))
    })
}
pub fn local_page(
    data_dir: &Path,
    sid: &str,
    q: &CatalogQuery,
    source: &RemoteSource,
    family: bool,
) -> Result<RemotePage, AppError> {
    let conn = open(data_dir)?;
    let (filter, mut values) = local_filter(sid, q, source, family);
    let total: u64 = conn.query_row(
        &format!("SELECT COUNT(*) FROM offline_items i WHERE {filter}"),
        rusqlite::params_from_iter(values.iter()),
        |r| r.get(0),
    )?;
    let mut ratings_stmt=conn.prepare(&format!("SELECT DISTINCT json_extract(i.item_json,'$.rating') FROM offline_items i WHERE {filter} ORDER BY 1"))?;
    let available_ratings = ratings_stmt
        .query_map(rusqlite::params_from_iter(values.iter()), |r| {
            r.get::<_, Option<String>>(0)
        })?
        .filter_map(|r| r.transpose())
        .collect::<Result<Vec<_>, _>>()?;
    let page = q.page.unwrap_or(1);
    let per = q.per_page.unwrap_or(40);
    let start = (page as u64 - 1) * per as u64;
    let order = local_order(q.sort.as_deref())?;
    values.push(rusqlite::types::Value::Integer(per as i64));
    values.push(rusqlite::types::Value::Integer(start as i64));
    let mut stmt=conn.prepare(&format!("SELECT i.item_json FROM offline_items i WHERE {filter} ORDER BY {order},i.post_id LIMIT ? OFFSET ?"))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(values.iter()), |r| {
        r.get::<_, String>(0)
    })?;
    let mut items = Vec::new();
    let mut missing = false;
    for row in rows {
        let mut item: RemoteItem = serde_json::from_str(&row?)
            .map_err(|_| AppError::Internal("Corrupt offline metadata".into()))?;
        let count = item.media.len();
        item.media.retain(|variant| {
            personal_variant(data_dir, sid, &item.remote_post_id, &variant.kind)
                .ok()
                .flatten()
                .is_some()
        });
        missing |= item.media.len() != count;
        if !item.media.is_empty() {
            items.push(item);
        }
    }
    let prefs = settings(data_dir, sid)?;
    Ok(RemotePage {
        items,
        page,
        per_page: per,
        total: Some(total),
        available_ratings,
        stale: true,
        offline: true,
        partial: missing
            || prefs["downloaded_variants"] != prefs["total_variants"]
            || prefs["state"] != "ready"
            || prefs["pending_items"].as_u64().unwrap_or(0) > 0
            || prefs["total_items"].as_u64().unwrap_or(0)
                > prefs["downloaded_items"].as_u64().unwrap_or(0),
        next_cursor: if start + (per as u64) < total {
            Some((page + 1).to_string())
        } else {
            None
        },
    })
}
pub fn local_tags(
    data_dir: &Path,
    sid: &str,
    q: &CatalogQuery,
    source: &RemoteSource,
    family: bool,
    prefix: &str,
    limit: usize,
) -> Result<Value, AppError> {
    let conn = open(data_dir)?;
    let (filter, mut values) = local_filter(sid, q, source, family);
    if prefix.len() > 100 {
        return Err(AppError::BadRequest("Tag prefix exceeds 100 bytes".into()));
    }
    let mut ratings=conn.prepare(&format!("SELECT DISTINCT json_extract(i.item_json,'$.rating') FROM offline_items i WHERE {filter} ORDER BY 1"))?;
    let ratings = ratings
        .query_map(rusqlite::params_from_iter(values.iter()), |r| {
            r.get::<_, Option<String>>(0)
        })?
        .filter_map(|r| r.transpose())
        .collect::<Result<Vec<_>, _>>()?;
    values.push(rusqlite::types::Value::Text(prefix.to_lowercase()));
    values.push(rusqlite::types::Value::Integer(limit.min(50) as i64));
    let mut stmt=conn.prepare(&format!("SELECT t.value,COUNT(DISTINCT i.post_id) FROM offline_items i,json_each(i.item_json,'$.tags') t WHERE {filter} AND substr(lower(t.value),1,length(?))=? GROUP BY t.value ORDER BY 2 DESC,1 LIMIT ?"))?;
    // The prefix occurs twice in this SQL expression.
    values.insert(
        values.len() - 1,
        rusqlite::types::Value::Text(prefix.to_lowercase()),
    );
    let tags=stmt.query_map(rusqlite::params_from_iter(values.iter()),|r|Ok(json!({"name":r.get::<_,String>(0)?,"category":"general","count":r.get::<_,u64>(1)?})))?.collect::<Result<Vec<_>,_>>()?;
    Ok(json!({"tags":tags,"available_ratings":ratings,"offline":true,"partial":true}))
}

pub fn creator_rows(
    data_dir: &Path,
    sid: &str,
    media: Option<&str>,
    safe_only: bool,
) -> Result<Vec<Value>, AppError> {
    let media = media_type(media)?;
    let conn = open(data_dir)?;
    let prefs = settings(data_dir, sid)?;
    let mut stmt = conn.prepare(
        "SELECT creator_id,name,directory_json,following FROM offline_creators WHERE source_id=?1",
    )?;
    let rows = stmt.query_map(params![sid], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, bool>(3)?,
        ))
    })?;
    let mut creators = Vec::new();
    for row in rows {
        let (id, name, directory, following) = row?;
        let snapshots: Value = serde_json::from_str(&directory).unwrap_or(json!({}));
        let key = format!("{}:{}", media, if safe_only { "safe" } else { "all" });
        let directory = snapshots
            .get("snapshots")
            .and_then(|v| v.get(&key))
            .cloned()
            .unwrap_or(json!({}));
        let enabled = selected(data_dir, sid, Some(&id), media)?;
        let (items,bytes):(i64,i64)=conn.query_row("SELECT COUNT(DISTINCT i.post_id),COALESCE(SUM(v.bytes),0) FROM offline_items i JOIN offline_variants v ON v.source_id=i.source_id AND v.post_id=i.post_id AND v.state='ready' WHERE i.source_id=?1 AND i.creator_id=?2 AND (?3='all' OR i.media_type=?3) AND (?4=0 OR COALESCE(json_extract(i.item_json,'$.rating'),'pg') IN('pg','pg13'))",params![sid,id,media,safe_only],|r|Ok((r.get(0)?,r.get(1)?)))?;
        let total = directory
            .get("post_count")
            .or_else(|| directory.get("item_count"))
            .and_then(Value::as_u64)
            .unwrap_or(items as u64);
        let fresh = directory["directory_verified_at"]
            .as_i64()
            .is_some_and(|t| t > Utc::now().timestamp() - 120);
        let hosted = if fresh {
            directory
                .pointer("/availability/hosted_posts")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        } else {
            0
        };
        let availability = if !fresh {
            "unknown"
        } else if hosted >= total && total > 0 {
            "available"
        } else if hosted > 0 {
            "partial"
        } else if directory
            .pointer("/availability/state")
            .and_then(Value::as_str)
            == Some("offline")
        {
            "offline"
        } else {
            "unknown"
        };
        creators.push(json!({"id":id,"name":name,"item_count":total,"availability":availability,"provider_availability":if fresh{directory.get("availability")}else{None},"central_readiness":if hosted>=total && total>0 {"ready"} else if hosted>0 {"partial"} else {"unavailable"},"hub_hosted":hosted>0,"peer_availability":if fresh {directory.pointer("/availability/peer_status").and_then(Value::as_str).unwrap_or("unknown")} else {"unknown"},"following":following,"offline":{"enabled":enabled,"state":if enabled{prefs["state"].as_str().unwrap_or("queued")}else{"disabled"},"downloaded_items":items,"total_items":total,"partial":prefs["state"]!="ready" || prefs["downloaded_variants"]!=prefs["total_variants"],"bytes":bytes,"error":prefs["error"]}}));
    }
    Ok(creators)
}
pub fn cleanup_source(data_dir: &Path, sid: &str) -> Result<(), AppError> {
    let root = cache_root(data_dir, sid);
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    Ok(())
}

struct RuntimeInner {
    state: AppState,
    cancel: Arc<AtomicBool>,
}
pub struct OfflineRuntime {
    inner: Arc<RuntimeInner>,
}
impl Drop for OfflineRuntime {
    fn drop(&mut self) {
        self.inner.cancel.store(true, Ordering::Release);
    }
}
static RUNTIMES: OnceLock<Mutex<HashMap<PathBuf, Weak<RuntimeInner>>>> = OnceLock::new();
static JOBS: OnceLock<Mutex<HashSet<(PathBuf, String)>>> = OnceLock::new();
pub fn start(state: AppState) -> Result<OfflineRuntime, String> {
    crate::edition::require_community()?;
    crate::routes::online::initialize_offline_database(state.data_dir())?;
    open(state.data_dir()).map_err(|e| e.to_string())?;
    let handle = tauri::async_runtime::handle();
    let inner = Arc::new(RuntimeInner {
        state,
        cancel: Arc::new(AtomicBool::new(false)),
    });
    let weak = Arc::downgrade(&inner);
    RUNTIMES
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "Offline runtime lock poisoned")?
        .insert(inner.state.data_dir().to_path_buf(), weak.clone());
    reconcile_startup(inner.state.data_dir()).map_err(|e| e.to_string())?;
    handle.spawn(async move{let mut timer=tokio::time::interval(std::time::Duration::from_secs(60));loop{timer.tick().await;let Some(runtime)=weak.upgrade()else{break};if runtime.cancel.load(Ordering::Acquire){break}let state=runtime.state.clone();drop(runtime);let sources=(||->Result<Vec<String>,AppError>{let conn=open(state.data_dir())?;let mut stmt=conn.prepare("SELECT o.source_id FROM offline_sources o JOIN remote_sources s USING(source_id) WHERE o.enabled=1 AND o.paused=0 AND s.enabled=1")?;let rows=stmt.query_map([],|r|r.get(0))?;Ok(rows.collect::<Result<_,_>>()?)})();match sources{Ok(sources)=>for sid in sources{if let Err(error)=kick(&state,&sid){let _=self::state(state.data_dir(),&sid,"error",Some(&error.to_string()));}},Err(error)=>log::error!("Offline supervisor failed: {error}")}}});
    Ok(OfflineRuntime { inner })
}
struct JobGuard((PathBuf, String));
impl Drop for JobGuard {
    fn drop(&mut self) {
        if let Some(jobs) = JOBS.get() {
            if let Ok(mut jobs) = jobs.lock() {
                jobs.remove(&self.0);
            }
        }
    }
}
pub fn kick(state: &AppState, sid: &str) -> Result<(), AppError> {
    crate::edition::require_community().map_err(AppError::Forbidden)?;
    let runtime = RUNTIMES
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| AppError::Internal("Offline runtime lock poisoned".into()))?
        .get(state.data_dir())
        .and_then(Weak::upgrade)
        .ok_or_else(|| {
            AppError::ServiceUnavailable(
                "Offline runtime did not start; restart DMC and review its startup error".into(),
            )
        })?;
    if runtime.cancel.load(Ordering::Acquire) {
        return Err(AppError::ServiceUnavailable(
            "Offline runtime is stopping".into(),
        ));
    }
    let key = (state.data_dir().to_path_buf(), sid.to_string());
    let mut jobs = JOBS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| AppError::Internal("Offline job lock poisoned".into()))?;
    if !jobs.insert(key.clone()) {
        return Ok(());
    }
    drop(jobs);
    let state = state.clone();
    let sid = sid.to_string();
    let cancel = runtime.cancel.clone();
    drop(runtime);
    tauri::async_runtime::spawn(async move {
        let _guard = JobGuard(key);
        if let Err(error) = crate::routes::online::sync_offline_source(&state, &sid, &cancel).await
        {
            if !active(state.data_dir(), &sid).unwrap_or(false) {
                return;
            }
            let status = if matches!(error, AppError::Unauthorized(_) | AppError::Forbidden(_))
                || matches!(&error,AppError::Upstream(status,_)if status.as_u16()==401||status.as_u16()==403)
            {
                "auth_required"
            } else if error.to_string().contains("storage limit") {
                "budget_exceeded"
            } else {
                "error"
            };
            if let Err(save) = self::state(state.data_dir(), &sid, status, Some(&error.to_string()))
            {
                log::error!("Offline progress persistence failed: {save}");
            }
        }
    });
    Ok(())
}

/// Explicitly destroys managed personal copies; it never touches imports or library files.
pub fn remove_copies(data_dir: &Path, sid: &str) -> Result<(), AppError> {
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    ensure_settings(&tx, sid)?;
    tx.execute("UPDATE offline_sources SET enabled=0,sync_all_creators=0,paused=1,peer_hosting_enabled=0,state='disabled',error=NULL WHERE source_id=?1",params![sid])?;
    tx.execute(
        "UPDATE offline_selections SET enabled=0 WHERE source_id=?1",
        params![sid],
    )?;
    for table in [
        "offline_queue",
        "offline_variants",
        "offline_items",
        "offline_scans",
    ] {
        tx.execute(
            &format!("DELETE FROM {table} WHERE source_id=?1"),
            params![sid],
        )?;
    }
    tx.commit()?;
    cleanup_source(data_dir, sid)
}

/// Promote only identical retained bytes after the authoritative catalog gains a digest.
pub fn refresh_metadata(
    data_dir: &Path,
    sid: &str,
    item: &RemoteItem,
    metadata: &Value,
) -> Result<(), AppError> {
    let mut conn = open(data_dir)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let rev = revision(item);
    if item.media.is_empty()
        || !item.media.iter().all(|variant| {
            variant_matches(&tx, sid, &item.remote_post_id, &rev, variant).unwrap_or(false)
        })
    {
        return Err(AppError::ServiceUnavailable(
            "Canonical metadata changed after transfer; retained snapshot preserved for retry"
                .into(),
        ));
    }
    tx.execute("UPDATE offline_items SET item_json=?3,metadata_json=?4,last_verified_at=?5,withdrawn=0 WHERE source_id=?1 AND post_id=?2 AND revision=?6",params![sid,item.remote_post_id,serde_json::to_string(item).unwrap(),serde_json::to_string(metadata).unwrap(),Utc::now().timestamp(),rev])?;
    for variant in &item.media {
        if let (Some(digest), Some(bytes)) = (&variant.sha256, variant.byte_size) {
            tx.execute("UPDATE offline_variants SET authoritative_digest=1 WHERE source_id=?1 AND post_id=?2 AND revision=?3 AND kind=?4 AND sha256=?5 AND bytes=?6 AND state='ready'",params![sid,item.remote_post_id,rev,variant.kind,digest,bytes as i64])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn remember_creator_directory(
    data_dir: &Path,
    sid: &str,
    row: &Value,
    media: Option<&str>,
    safe_only: bool,
) -> Result<(), AppError> {
    remember_creator(data_dir, sid, &json!({"id":row["id"],"name":row["name"]}))?;
    let media = media_type(media)?;
    let conn = open(data_dir)?;
    let mut directory: Value = conn
        .query_row(
            "SELECT directory_json FROM offline_creators WHERE source_id=?1 AND creator_id=?2",
            params![sid, row["id"].as_str()],
            |r| r.get::<_, String>(0),
        )
        .map(|v| serde_json::from_str(&v).unwrap_or(json!({})))?;
    if !directory["snapshots"].is_object() {
        directory["snapshots"] = json!({});
    }
    let key = format!("{}:{}", media, if safe_only { "safe" } else { "all" });
    let mut snapshot = row.clone();
    snapshot["directory_verified_at"] = json!(Utc::now().timestamp());
    directory["snapshots"][key] = snapshot;
    conn.execute(
        "UPDATE offline_creators SET directory_json=?3 WHERE source_id=?1 AND creator_id=?2",
        params![
            sid,
            row["id"].as_str(),
            serde_json::to_string(&directory).unwrap()
        ],
    )?;
    Ok(())
}

/// Crashes can leave a file between rename and SQLite commit. Reconcile managed
/// paths individually rather than loading the catalog or trusting stale reservations.
pub(crate) fn reconcile_startup(data_dir: &Path) -> Result<(), AppError> {
    // Unfinished reservations belong to vanished staging files after restart.
    open(data_dir)?.execute("DELETE FROM offline_variants WHERE state!='ready'", [])?;
    recover_staging(data_dir)
}

fn recover_staging(data_dir: &Path) -> Result<(), AppError> {
    let root = data_dir.join("online-offline");
    if !root.exists() {
        return Ok(());
    }
    let conn = open(data_dir)?;
    let mut ready = conn.prepare(
        "SELECT EXISTS(SELECT 1 FROM offline_variants WHERE relative_path=?1 AND state='ready')",
    )?;
    for directory in std::fs::read_dir(&root)? {
        let directory = directory?;
        if !directory.file_type()?.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(directory.path())? {
            let file = file?;
            if !file.file_type()?.is_file() {
                continue;
            }
            let path = file.path();
            let relative = path
                .strip_prefix(&root)
                .map_err(|e| AppError::Internal(e.to_string()))?
                .to_string_lossy();
            if !ready.query_row(params![relative.as_ref()], |r| r.get::<_, bool>(0))? {
                std::fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

/// Record observed successful advertisement coverage, rather than inferring it
/// from configuration or possession of personal bytes. Zero withdraws coverage.
pub fn hosting_reported(data_dir: &Path, sid: &str, count: usize) -> Result<(), AppError> {
    open(data_dir)?.execute(
        "UPDATE offline_sources SET peer_reported_at=?2,peer_reported_items=?3 WHERE source_id=?1",
        params![
            sid,
            if count == 0 {
                None
            } else {
                Some(Utc::now().timestamp())
            },
            count as i64
        ],
    )?;
    Ok(())
}

/// Available discovery includes recent provider reports without promising that
/// an unprobed peer is reachable. Expired snapshots expose neither nested claims
/// nor a recent peer status, so they naturally leave the default view.
pub fn creator_potentially_available(row: &Value) -> bool {
    matches!(row["availability"].as_str(), Some("available" | "partial"))
        || (row["peer_availability"].as_str() == Some("recently_reported")
            && row
                .pointer("/provider_availability/peer_reported_posts")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                > 0)
}
