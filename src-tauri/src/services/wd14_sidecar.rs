use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, params_from_iter, TransactionBehavior};
use serde::{Deserialize, Serialize};
#[cfg(not(unix))]
use tempfile::NamedTempFile;

use crate::db::library::LibraryContext;
use crate::server::error::AppError;
use crate::server::state::AppState;

const MAX_DIRECTORIES: usize = 100;
const MAX_MEDIA_CANDIDATES: usize = 50_000;
const MAX_SIDECAR_BYTES: u64 = 1024 * 1024;
const MAX_TAGS_PER_SIDECAR: usize = 10_000;
const MAX_TAG_BYTES: usize = 1024;
const MAX_LIBRARY_DIRECTORIES: usize = 1_000;
const MAX_RECONCILIATION_QUERIES: usize = MAX_MEDIA_CANDIDATES * MAX_LIBRARY_DIRECTORIES;
const TAG_LOOKUP_CHUNK: usize = 500;

static WD14_OPERATION_LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct DirectorySelection {
    pub library_id: String,
    pub directory_id: i64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Wd14Operation {
    Import,
    Absorb,
    Export,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Wd14Request {
    pub directories: Vec<DirectorySelection>,
    #[serde(default)]
    pub overwrite: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SidecarStatus {
    Imported,
    Absorbed,
    Exported,
    SkippedMissing,
    SkippedExists,
    ImportedNotRemoved,
    FailedValidation,
    FailedRead,
    FailedDatabase,
    FailedWrite,
    ConflictingMediaStem,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct MediaTarget {
    pub library_id: String,
    pub directory_id: i64,
    pub image_id: i64,
    pub media_path: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SidecarResult {
    pub sidecar_path: String,
    pub targets: Vec<MediaTarget>,
    pub status: SidecarStatus,
    pub tags_parsed: usize,
    pub tags_added: usize,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Wd14Summary {
    pub directories: usize,
    pub media_candidates: usize,
    pub sidecars_found: usize,
    pub sidecars_succeeded: usize,
    pub sidecars_skipped: usize,
    pub sidecars_failed: usize,
    pub tags_parsed: usize,
    pub tags_added: usize,
    pub sidecars_written: usize,
    pub sidecars_removed: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Wd14Response {
    pub operation: Wd14Operation,
    pub summary: Wd14Summary,
    pub results: Vec<SidecarResult>,
}

#[derive(Clone, Debug)]
struct PreparedTarget {
    locator: MediaTarget,
    root: PathBuf,
}

#[derive(Clone, Debug)]
struct SidecarGroup {
    path: PathBuf,
    targets: Vec<PreparedTarget>,
    preparation_errors: Vec<String>,
}

#[derive(Debug)]
struct OpenedSidecar {
    bytes: Vec<u8>,
    identity: FileIdentity,
}

#[derive(Debug)]
struct ApplyFailure {
    error: AppError,
    added: usize,
}

struct DirectoryWriteLocks {
    connections: Vec<(i64, PooledConnection<SqliteConnectionManager>)>,
}

impl DirectoryWriteLocks {
    fn acquire(library: &LibraryContext, directory_ids: &[i64]) -> Result<Self, AppError> {
        let mut locks = Self {
            connections: Vec::with_capacity(directory_ids.len()),
        };
        for directory_id in directory_ids {
            let pool = library.directory_db.get_pool(*directory_id)?;
            let connection = pool.get()?;
            connection.execute_batch("BEGIN IMMEDIATE")?;
            locks.connections.push((*directory_id, connection));
        }
        Ok(locks)
    }

    fn get(&self, directory_id: i64) -> Option<&rusqlite::Connection> {
        self.connections
            .iter()
            .find(|(candidate, _)| *candidate == directory_id)
            .map(|(_, connection)| &**connection)
    }

    fn commit(mut self) -> Result<(), AppError> {
        for (_, connection) in &self.connections {
            connection.execute_batch("COMMIT")?;
        }
        self.connections.clear();
        Ok(())
    }
}

impl Drop for DirectoryWriteLocks {
    fn drop(&mut self) {
        for (_, connection) in &self.connections {
            let _ = connection.execute_batch("ROLLBACK");
        }
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[cfg(not(unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileIdentity {
    len: u64,
    modified_nanos: u128,
}

pub fn run_operation(
    state: &AppState,
    operation: Wd14Operation,
    request: Wd14Request,
) -> Result<Wd14Response, AppError> {
    if request.directories.is_empty() {
        return Err(AppError::BadRequest(
            "Select at least one registered directory".into(),
        ));
    }
    if request.directories.len() > MAX_DIRECTORIES {
        return Err(AppError::BadRequest(format!(
            "At most {MAX_DIRECTORIES} directories may be processed at once"
        )));
    }
    if operation != Wd14Operation::Export && request.overwrite {
        return Err(AppError::BadRequest(
            "overwrite is only valid for export".into(),
        ));
    }
    let _operation_guard = WD14_OPERATION_LOCK
        .lock()
        .map_err(|_| AppError::Internal("WD14 operation lock is poisoned".into()))?;

    let selections: Vec<_> = request
        .directories
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let (groups, preparation_failures, media_candidates) = prepare_groups(state, &selections)?;

    let mut response = Wd14Response {
        operation,
        summary: Wd14Summary {
            directories: selections.len(),
            media_candidates,
            ..Wd14Summary::default()
        },
        results: preparation_failures,
    };
    for result in &response.results {
        record_result(&mut response.summary, result);
    }

    let mut reconciliation_budget = MAX_RECONCILIATION_QUERIES;
    for group in groups.into_values() {
        let result = if !group.preparation_errors.is_empty() {
            result(
                &group,
                SidecarStatus::FailedValidation,
                0,
                0,
                Some(group.preparation_errors.join("; ")),
            )
        } else {
            match operation {
                Wd14Operation::Import => {
                    import_group(state, &group, false, &mut reconciliation_budget)
                }
                Wd14Operation::Absorb => {
                    import_group(state, &group, true, &mut reconciliation_budget)
                }
                Wd14Operation::Export => export_group(state, &group, request.overwrite),
            }
        };
        record_result(&mut response.summary, &result);
        response.results.push(result);
    }
    response.results.sort_by(|left, right| {
        result_rank(left.status)
            .cmp(&result_rank(right.status))
            .then_with(|| left.sidecar_path.cmp(&right.sidecar_path))
    });
    Ok(response)
}

fn prepare_groups(
    state: &AppState,
    selections: &[DirectorySelection],
) -> Result<(BTreeMap<Vec<u8>, SidecarGroup>, Vec<SidecarResult>, usize), AppError> {
    let mut candidates = Vec::new();
    for selection in selections {
        let library = state.resolve_library(Some(&selection.library_id))?;
        if library.directory_db.get_all_directory_ids().len() > MAX_LIBRARY_DIRECTORIES {
            return Err(AppError::BadRequest(format!(
                "Library {} exceeds the {MAX_LIBRARY_DIRECTORIES}-directory WD14 limit",
                selection.library_id
            )));
        }
        if !library.directory_db.db_exists(selection.directory_id) {
            return Err(AppError::NotFound(format!(
                "Directory database not found for {}:{}",
                selection.library_id, selection.directory_id
            )));
        }
        let main = library.main_pool.get()?;
        let root: String = main
            .query_row(
                "SELECT path FROM watch_directories WHERE id = ?1",
                params![selection.directory_id],
                |row| row.get(0),
            )
            .map_err(|_| {
                AppError::NotFound(format!(
                    "Registered directory not found for {}:{}",
                    selection.library_id, selection.directory_id
                ))
            })?;
        let root = fs::canonicalize(&root).map_err(|error| {
            AppError::BadRequest(format!("Registered directory is inaccessible: {error}"))
        })?;
        if !root.is_dir() {
            return Err(AppError::BadRequest(format!(
                "Registered path is not a directory: {}",
                root.display()
            )));
        }

        let remaining = MAX_MEDIA_CANDIDATES.saturating_sub(candidates.len());
        let pool = library.directory_db.get_pool(selection.directory_id)?;
        let connection = pool.get()?;
        let mut statement = connection.prepare(
            "SELECT image_id, original_path
             FROM image_files
             WHERE file_exists = 1 AND curation_discarded_at IS NULL
             ORDER BY original_path, image_id
             LIMIT ?1",
        )?;
        let rows = statement
            .query_map(params![(remaining + 1) as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        if rows.len() > remaining {
            return Err(AppError::BadRequest(format!(
                "Operation exceeds the {MAX_MEDIA_CANDIDATES}-media candidate limit"
            )));
        }
        for (image_id, media_path) in rows {
            candidates.push((selection.clone(), root.clone(), image_id, media_path));
        }
    }

    let media_candidates = candidates.len();
    let mut groups = BTreeMap::new();
    let mut sensitivity_by_parent: BTreeMap<PathBuf, Result<bool, String>> = BTreeMap::new();
    for (selection, root, image_id, recorded_path) in candidates {
        let recorded_path = PathBuf::from(recorded_path);
        let sidecar_path = sidecar_path_for(&recorded_path);
        let grouping_parent = sidecar_path
            .parent()
            .and_then(|parent| fs::canonicalize(parent).ok());
        let mut grouping_error = None;
        let sidecar_metadata = fs::symlink_metadata(&sidecar_path).ok();
        let is_regular_sidecar = sidecar_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink());
        let case_insensitive = match grouping_parent.as_ref() {
            Some(parent) => {
                match sensitivity_by_parent
                    .entry(parent.clone())
                    .or_insert_with(|| {
                        case_insensitive_grouping(parent).map_err(|error| {
                            format!(
                                "Cannot determine filesystem case behavior for {}: {error}",
                                parent.display()
                            )
                        })
                    })
                    .clone()
                {
                    Ok(case_insensitive) => case_insensitive,
                    Err(error) => {
                        grouping_error = Some(error);
                        false
                    }
                }
            }
            None => {
                grouping_error = Some(format!(
                    "Cannot resolve sidecar parent for {}",
                    sidecar_path.display()
                ));
                false
            }
        };
        let grouping_path = if is_regular_sidecar {
            match fs::canonicalize(&sidecar_path) {
                Ok(canonical) => canonical,
                Err(error) => {
                    grouping_error = Some(format!(
                        "Cannot resolve existing sidecar {}: {error}",
                        sidecar_path.display()
                    ));
                    sidecar_path.clone()
                }
            }
        } else {
            grouping_parent
                .as_ref()
                .and_then(|parent| sidecar_path.file_name().map(|name| parent.join(name)))
                .unwrap_or_else(|| sidecar_path.clone())
        };
        let key =
            directory_entry_key(&grouping_path, case_insensitive).map_err(AppError::BadRequest)?;
        let (media_path, preparation_error) = match validate_media_path(&recorded_path, &root) {
            Ok(media_path) => (media_path, None),
            Err(error) => (recorded_path.clone(), Some(error)),
        };
        let target = PreparedTarget {
            locator: MediaTarget {
                library_id: selection.library_id.clone(),
                directory_id: selection.directory_id,
                image_id,
                media_path: display_path(&media_path),
            },
            root,
        };
        let group = groups.entry(key).or_insert_with(|| SidecarGroup {
            path: sidecar_path.clone(),
            targets: Vec::new(),
            preparation_errors: Vec::new(),
        });
        if path_key(&sidecar_path, false) < path_key(&group.path, false) {
            group.path = sidecar_path;
        }
        group.targets.push(target);
        if let Some(error) = grouping_error {
            group.preparation_errors.push(error);
        }
        if let Some(error) = preparation_error {
            group.preparation_errors.push(format!(
                "Media target {}:{}:{image_id} failed validation: {error}",
                selection.library_id, selection.directory_id
            ));
        }
    }
    Ok((groups, Vec::new(), media_candidates))
}

fn import_group(
    state: &AppState,
    group: &SidecarGroup,
    absorb: bool,
    reconciliation_budget: &mut usize,
) -> SidecarResult {
    let targets = serialized_targets(group);
    let opened = match read_existing_sidecar(group) {
        Ok(Some(opened)) => opened,
        Ok(None) => return result(group, SidecarStatus::SkippedMissing, 0, 0, None),
        Err(error) => return result(group, SidecarStatus::FailedRead, 0, 0, Some(error)),
    };
    let tags = match parse_tags(&opened.bytes) {
        Ok(tags) => tags,
        Err(error) => return result(group, SidecarStatus::FailedRead, 0, 0, Some(error)),
    };

    let mut added = 0usize;
    for target in &group.targets {
        let library = match state.resolve_library(Some(&target.locator.library_id)) {
            Ok(library) => library,
            Err(error) => {
                return SidecarResult {
                    sidecar_path: display_path(&group.path),
                    targets,
                    status: SidecarStatus::FailedDatabase,
                    tags_parsed: tags.len(),
                    tags_added: added,
                    error: Some(error.to_string()),
                }
            }
        };
        match apply_tags(
            &library,
            target.locator.directory_id,
            target.locator.image_id,
            &tags,
            reconciliation_budget,
        ) {
            Ok(count) => added += count,
            Err(failure) => {
                added += failure.added;
                return SidecarResult {
                    sidecar_path: display_path(&group.path),
                    targets,
                    status: SidecarStatus::FailedDatabase,
                    tags_parsed: tags.len(),
                    tags_added: added,
                    error: Some(failure.error.to_string()),
                };
            }
        }
    }

    if absorb {
        if let Err(error) = remove_absorbed_sidecar(group, &opened) {
            return result(
                group,
                SidecarStatus::ImportedNotRemoved,
                tags.len(),
                added,
                Some(error),
            );
        }
        result(group, SidecarStatus::Absorbed, tags.len(), added, None)
    } else {
        result(group, SidecarStatus::Imported, tags.len(), added, None)
    }
}

fn export_group(state: &AppState, group: &SidecarGroup, overwrite: bool) -> SidecarResult {
    let parent_identity = match validate_export_target(group) {
        Ok(identity) => identity,
        Err(error) => return result(group, SidecarStatus::FailedValidation, 0, 0, Some(error)),
    };

    let mut tag_sets = Vec::new();
    for target in &group.targets {
        let library = match state.resolve_library(Some(&target.locator.library_id)) {
            Ok(library) => library,
            Err(error) => {
                return result(
                    group,
                    SidecarStatus::FailedDatabase,
                    0,
                    0,
                    Some(error.to_string()),
                )
            }
        };
        match load_tag_set(
            &library,
            target.locator.directory_id,
            target.locator.image_id,
        ) {
            Ok(tags) => tag_sets.push(tags),
            Err(error) => {
                return result(
                    group,
                    SidecarStatus::FailedDatabase,
                    0,
                    0,
                    Some(error.to_string()),
                )
            }
        }
    }
    let Some(tags) = common_tag_set(&tag_sets) else {
        return result(
            group,
            SidecarStatus::ConflictingMediaStem,
            0,
            0,
            Some("Media sharing this sidecar stem have different tag sets".into()),
        );
    };
    if let Err(error) = validate_serializable_tags(tags) {
        return result(
            group,
            SidecarStatus::FailedValidation,
            tags.len(),
            0,
            Some(error),
        );
    }
    let bytes = serialize_tags(tags);
    match atomic_write_sidecar(&group.path, &bytes, overwrite, parent_identity) {
        Ok(WriteOutcome::Written) => result(group, SidecarStatus::Exported, tags.len(), 0, None),
        Ok(WriteOutcome::SkippedExists) => {
            result(group, SidecarStatus::SkippedExists, tags.len(), 0, None)
        }
        Err(error) => result(
            group,
            SidecarStatus::FailedWrite,
            tags.len(),
            0,
            Some(error.to_string()),
        ),
    }
}

fn apply_tags(
    library: &LibraryContext,
    directory_id: i64,
    image_id: i64,
    tags: &BTreeSet<String>,
    reconciliation_budget: &mut usize,
) -> Result<usize, ApplyFailure> {
    let directory_ids = library.directory_db.get_all_directory_ids();
    if directory_ids.len() > *reconciliation_budget {
        return Err(ApplyFailure {
            error: AppError::BadRequest(format!(
                "WD14 count reconciliation exceeds the {MAX_RECONCILIATION_QUERIES}-query operation limit"
            )),
            added: 0,
        });
    }
    *reconciliation_budget -= directory_ids.len();

    let write_associations = || -> Result<(usize, Vec<i64>), AppError> {
        let tag_ids = {
            let mut main = library.main_pool.get()?;
            let transaction = main.transaction()?;
            let mut tag_ids = Vec::with_capacity(tags.len());
            {
                let mut insert = transaction.prepare(
                    "INSERT OR IGNORE INTO tags (name, category) VALUES (?1, 'general')",
                )?;
                let mut select = transaction.prepare("SELECT id FROM tags WHERE name = ?1")?;
                for tag in tags {
                    insert.execute(params![tag])?;
                    tag_ids.push(select.query_row(params![tag], |row| row.get::<_, i64>(0))?);
                }
            }
            transaction.commit()?;
            tag_ids
        };

        let pool = library.directory_db.get_pool(directory_id)?;
        let mut connection = pool.get()?;
        let transaction = connection.transaction()?;
        let mut added = 0usize;
        {
            let mut insert = transaction.prepare(
                "INSERT OR IGNORE INTO image_tags (image_id, tag_id, confidence, is_manual)
                 VALUES (?1, ?2, NULL, 1)",
            )?;
            for tag_id in &tag_ids {
                added += insert.execute(params![image_id, tag_id])?;
            }
        }
        transaction.commit()?;
        Ok((added, tag_ids))
    };

    let (added, tag_ids) =
        write_associations().map_err(|error| ApplyFailure { error, added: 0 })?;
    reconcile_counts(library, directory_id, &tag_ids, &directory_ids)
        .map_err(|error| ApplyFailure { error, added })?;
    Ok(added)
}

fn reconcile_counts(
    library: &LibraryContext,
    directory_id: i64,
    tag_ids: &[i64],
    directory_ids: &[i64],
) -> Result<(), AppError> {
    let mut main = library.main_pool.get()?;
    let transaction = main.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let directory_locks = DirectoryWriteLocks::acquire(library, directory_ids)?;
    let unique_tag_ids: Vec<_> = tag_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut post_counts: BTreeMap<i64, i64> = unique_tag_ids
        .iter()
        .copied()
        .map(|tag_id| (tag_id, 0))
        .collect();
    if !unique_tag_ids.is_empty() {
        let placeholders = std::iter::repeat("?")
            .take(unique_tag_ids.len())
            .collect::<Vec<_>>()
            .join(",");
        let query = format!(
            "SELECT tag_id, COUNT(*) FROM image_tags
             WHERE tag_id IN ({placeholders}) GROUP BY tag_id"
        );
        for candidate_directory_id in directory_ids {
            let connection = directory_locks
                .get(*candidate_directory_id)
                .ok_or_else(|| AppError::Internal("Missing locked directory database".into()))?;
            let mut statement = connection.prepare(&query)?;
            let rows = statement.query_map(params_from_iter(unique_tag_ids.iter()), |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
            })?;
            for row in rows {
                let (tag_id, count) = row?;
                *post_counts.entry(tag_id).or_default() += count;
            }
        }
    }

    let directory_connection = directory_locks
        .get(directory_id)
        .ok_or_else(|| AppError::Internal("Missing selected directory lock".into()))?;
    let tagged_count: i64 = directory_connection.query_row(
        "SELECT COUNT(DISTINCT image_id) FROM image_tags",
        [],
        |row| row.get(0),
    )?;

    {
        let mut update = transaction.prepare("UPDATE tags SET post_count = ?1 WHERE id = ?2")?;
        for (tag_id, count) in post_counts {
            update.execute(params![count, tag_id])?;
        }
    }
    transaction.execute(
        "UPDATE watch_directories SET tagged_count = ?1 WHERE id = ?2",
        params![tagged_count, directory_id],
    )?;
    transaction.commit()?;
    directory_locks.commit()?;
    Ok(())
}

fn load_tag_set(
    library: &LibraryContext,
    directory_id: i64,
    image_id: i64,
) -> Result<BTreeSet<String>, AppError> {
    let pool = library.directory_db.get_pool(directory_id)?;
    let connection = pool.get()?;
    let mut statement = connection
        .prepare("SELECT tag_id FROM image_tags WHERE image_id = ?1 ORDER BY tag_id LIMIT ?2")?;
    let tag_ids: BTreeSet<i64> = statement
        .query_map(
            params![image_id, (MAX_TAGS_PER_SIDECAR + 1) as i64],
            |row| row.get::<_, i64>(0),
        )?
        .collect::<Result<_, _>>()?;
    drop(statement);
    drop(connection);
    if tag_ids.len() > MAX_TAGS_PER_SIDECAR {
        return Err(AppError::BadRequest(format!(
            "Tag set exceeds the {MAX_TAGS_PER_SIDECAR}-tag export limit"
        )));
    }
    if tag_ids.is_empty() {
        return Ok(BTreeSet::new());
    }

    let main = library.main_pool.get()?;
    let mut tags = BTreeSet::new();
    for chunk in tag_ids
        .iter()
        .copied()
        .collect::<Vec<_>>()
        .chunks(TAG_LOOKUP_CHUNK)
    {
        let placeholders = std::iter::repeat("?")
            .take(chunk.len())
            .collect::<Vec<_>>()
            .join(",");
        let mut statement = main.prepare(&format!(
            "SELECT name FROM tags WHERE id IN ({placeholders}) ORDER BY name"
        ))?;
        let names = statement
            .query_map(params_from_iter(chunk.iter()), |row| {
                row.get::<_, String>(0)
            })?
            .collect::<Result<Vec<_>, _>>()?;
        tags.extend(names);
    }
    if tags.len() != tag_ids.len() {
        return Err(AppError::Internal(
            "Image references a missing global tag definition".into(),
        ));
    }
    Ok(tags)
}

fn parse_tags(bytes: &[u8]) -> Result<BTreeSet<String>, String> {
    if bytes.len() as u64 > MAX_SIDECAR_BYTES {
        return Err(format!(
            "Sidecar exceeds the {MAX_SIDECAR_BYTES}-byte limit"
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "Sidecar is not valid UTF-8".to_string())?;
    let mut tags = BTreeSet::new();
    for value in text.split(',') {
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        if value.chars().any(char::is_control) {
            return Err("Tags may not contain control characters".into());
        }
        let value = value.to_lowercase();
        if value.len() > MAX_TAG_BYTES {
            return Err(format!("Tag exceeds the {MAX_TAG_BYTES}-byte limit"));
        }
        tags.insert(value);
        if tags.len() > MAX_TAGS_PER_SIDECAR {
            return Err(format!(
                "Sidecar exceeds the {MAX_TAGS_PER_SIDECAR}-tag limit"
            ));
        }
    }
    Ok(tags)
}

fn validate_serializable_tags(tags: &BTreeSet<String>) -> Result<(), String> {
    if tags.len() > MAX_TAGS_PER_SIDECAR {
        return Err(format!(
            "Tag set exceeds the {MAX_TAGS_PER_SIDECAR}-tag export limit"
        ));
    }
    for tag in tags {
        if tag.is_empty()
            || tag.trim() != tag
            || tag.contains(',')
            || tag.chars().any(char::is_control)
        {
            return Err(format!(
                "Tag cannot be represented in WD14 text format: {tag:?}"
            ));
        }
        if tag.len() > MAX_TAG_BYTES {
            return Err(format!("Tag exceeds the {MAX_TAG_BYTES}-byte export limit"));
        }
    }
    if serialize_tags(tags).len() as u64 > MAX_SIDECAR_BYTES {
        return Err(format!(
            "Serialized tags exceed the {MAX_SIDECAR_BYTES}-byte export limit"
        ));
    }
    Ok(())
}

fn serialize_tags(tags: &BTreeSet<String>) -> Vec<u8> {
    if tags.is_empty() {
        Vec::new()
    } else {
        format!("{}\n", tags.iter().cloned().collect::<Vec<_>>().join(", ")).into_bytes()
    }
}

fn common_tag_set(tag_sets: &[BTreeSet<String>]) -> Option<&BTreeSet<String>> {
    let first = tag_sets.first()?;
    tag_sets.iter().all(|tags| tags == first).then_some(first)
}

fn validate_media_path(path: &Path, root: &Path) -> Result<PathBuf, String> {
    let canonical =
        fs::canonicalize(path).map_err(|error| format!("Media path is inaccessible: {error}"))?;
    if !canonical.starts_with(root) {
        return Err("Media path escapes the registered directory".into());
    }
    if !canonical.is_file() {
        return Err("Media path is not a regular file".into());
    }
    Ok(canonical)
}

fn same_sidecar(expected: &OpenedSidecar, current: &OpenedSidecar) -> bool {
    expected.identity == current.identity && expected.bytes == current.bytes
}

fn remove_absorbed_sidecar(group: &SidecarGroup, expected: &OpenedSidecar) -> Result<(), String> {
    let parent = group
        .path
        .parent()
        .ok_or_else(|| "Sidecar path has no parent directory".to_string())?;
    let file_name = group
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("sidecar.txt");
    let quarantine = parent.join(format!(".{file_name}.{}.absorbing", uuid::Uuid::new_v4()));
    fs::rename(&group.path, &quarantine)
        .map_err(|error| format!("Could not claim sidecar for removal: {error}"))?;
    let quarantined_group = SidecarGroup {
        path: quarantine.clone(),
        targets: group.targets.clone(),
        preparation_errors: Vec::new(),
    };
    let current = read_existing_sidecar(&quarantined_group);
    let matches = matches!(&current, Ok(Some(current)) if same_sidecar(expected, current));
    if matches {
        return match fs::remove_file(&quarantine) {
            Ok(()) => Ok(()),
            Err(error) => {
                let restore_error =
                    restore_quarantined_sidecar(&quarantine, &group.path, Some(&expected.bytes))
                        .err();
                Err(format!(
                    "Could not remove imported sidecar: {error}{}",
                    restore_error
                        .map(|error| format!("; restoring its same-stem name failed: {error}"))
                        .unwrap_or_default()
                ))
            }
        };
    }

    let restore_bytes = current
        .as_ref()
        .ok()
        .and_then(|current| current.as_ref())
        .map(|current| current.bytes.as_slice());
    if let Err(error) = restore_quarantined_sidecar(&quarantine, &group.path, restore_bytes) {
        return Err(format!(
            "Sidecar changed after import and was preserved at {}; restoring its original name failed: {error}",
            quarantine.display()
        ));
    }
    match current {
        Ok(Some(_)) => Err("Sidecar changed after it was imported".into()),
        Ok(None) => Err("Sidecar disappeared while it was being removed".into()),
        Err(error) => Err(error),
    }
}

fn restore_quarantined_sidecar(
    quarantine: &Path,
    original: &Path,
    bytes: Option<&[u8]>,
) -> io::Result<()> {
    match fs::hard_link(quarantine, original) {
        Ok(()) => return fs::remove_file(quarantine),
        Err(error) if original.exists() => return Err(error),
        Err(_) => {}
    }

    let bytes = bytes.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Other,
            "sidecar bytes are unavailable for safe restoration",
        )
    })?;
    let parent = original
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    let parent_identity = file_identity(&fs::metadata(parent)?);
    match atomic_write_sidecar(original, bytes, false, parent_identity)? {
        WriteOutcome::Written => fs::remove_file(quarantine),
        WriteOutcome::SkippedExists => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "same-stem sidecar already exists",
        )),
    }
}

fn read_existing_sidecar(group: &SidecarGroup) -> Result<Option<OpenedSidecar>, String> {
    let mut file = match open_sidecar_nofollow(&group.path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Could not open sidecar safely: {error}")),
    };
    let metadata = file
        .metadata()
        .map_err(|error| format!("Could not inspect opened sidecar: {error}"))?;
    if !metadata.is_file() {
        return Err("Sidecar must be a regular non-symlink file".into());
    }
    let path_metadata = fs::symlink_metadata(&group.path)
        .map_err(|error| format!("Could not inspect sidecar path: {error}"))?;
    if path_metadata.file_type().is_symlink()
        || file_identity(&metadata) != file_identity(&path_metadata)
    {
        return Err("Sidecar changed or became a symlink while it was opened".into());
    }
    let canonical = fs::canonicalize(&group.path)
        .map_err(|error| format!("Could not resolve sidecar: {error}"))?;
    if group
        .targets
        .iter()
        .any(|target| !canonical.starts_with(&target.root))
    {
        return Err("Sidecar path escapes a registered directory".into());
    }

    let mut bytes = Vec::with_capacity(metadata.len().min(MAX_SIDECAR_BYTES + 1) as usize);
    Read::by_ref(&mut file)
        .take(MAX_SIDECAR_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Could not read sidecar: {error}"))?;
    if bytes.len() as u64 > MAX_SIDECAR_BYTES {
        return Err(format!(
            "Sidecar exceeds the {MAX_SIDECAR_BYTES}-byte limit"
        ));
    }
    Ok(Some(OpenedSidecar {
        bytes,
        identity: file_identity(&metadata),
    }))
}

#[cfg(unix)]
fn open_sidecar_nofollow(path: &Path) -> io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;

    fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(not(unix))]
fn open_sidecar_nofollow(path: &Path) -> io::Result<fs::File> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sidecar is a symlink",
        ));
    }
    fs::File::open(path)
}

#[cfg(unix)]
fn file_identity(metadata: &fs::Metadata) -> FileIdentity {
    use std::os::unix::fs::MetadataExt;

    FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

#[cfg(not(unix))]
fn file_identity(metadata: &fs::Metadata) -> FileIdentity {
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    FileIdentity {
        len: metadata.len(),
        modified_nanos,
    }
}

fn validate_existing_sidecar(group: &SidecarGroup) -> Result<Option<fs::Metadata>, String> {
    let metadata = match fs::symlink_metadata(&group.path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Could not inspect sidecar: {error}")),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("Sidecar must be a regular non-symlink file".into());
    }
    let canonical = fs::canonicalize(&group.path)
        .map_err(|error| format!("Could not resolve sidecar: {error}"))?;
    if group
        .targets
        .iter()
        .any(|target| !canonical.starts_with(&target.root))
    {
        return Err("Sidecar path escapes a registered directory".into());
    }
    Ok(Some(metadata))
}

fn validate_export_target(group: &SidecarGroup) -> Result<FileIdentity, String> {
    validate_existing_sidecar(group)?;
    let parent = group
        .path
        .parent()
        .ok_or_else(|| "Sidecar path has no parent directory".to_string())?;
    let canonical = fs::canonicalize(parent)
        .map_err(|error| format!("Could not resolve sidecar directory: {error}"))?;
    if group
        .targets
        .iter()
        .any(|target| !canonical.starts_with(&target.root))
    {
        return Err("Sidecar directory escapes a registered directory".into());
    }
    let metadata = fs::metadata(&canonical)
        .map_err(|error| format!("Could not inspect sidecar directory: {error}"))?;
    if !metadata.is_dir() {
        return Err("Sidecar parent is not a directory".into());
    }
    Ok(file_identity(&metadata))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteOutcome {
    Written,
    SkippedExists,
}

#[cfg(unix)]
fn atomic_write_sidecar(
    path: &Path,
    bytes: &[u8],
    overwrite: bool,
    expected_parent: FileIdentity,
) -> io::Result<WriteOutcome> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    let destination = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let destination = CString::new(destination.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "file name contains NUL"))?;
    let directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(parent)?;
    if file_identity(&directory.metadata()?) != expected_parent {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "sidecar directory changed after validation",
        ));
    }

    let temporary_name = CString::new(format!(".wd14-{}.tmp", uuid::Uuid::new_v4())).unwrap();
    let temporary_fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            temporary_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    if temporary_fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut temporary = unsafe { fs::File::from_raw_fd(temporary_fd) };
    let write_result = temporary
        .write_all(bytes)
        .and_then(|()| temporary.sync_all());
    drop(temporary);
    if let Err(error) = write_result {
        unsafe {
            libc::unlinkat(directory.as_raw_fd(), temporary_name.as_ptr(), 0);
        }
        return Err(error);
    }

    let result = unsafe {
        if overwrite {
            libc::renameat(
                directory.as_raw_fd(),
                temporary_name.as_ptr(),
                directory.as_raw_fd(),
                destination.as_ptr(),
            )
        } else {
            libc::linkat(
                directory.as_raw_fd(),
                temporary_name.as_ptr(),
                directory.as_raw_fd(),
                destination.as_ptr(),
                0,
            )
        }
    };
    if result != 0 {
        let error = io::Error::last_os_error();
        unsafe {
            libc::unlinkat(directory.as_raw_fd(), temporary_name.as_ptr(), 0);
        }
        if !overwrite && error.kind() == io::ErrorKind::AlreadyExists {
            return Ok(WriteOutcome::SkippedExists);
        }
        return Err(error);
    }
    if !overwrite {
        let unlink_result =
            unsafe { libc::unlinkat(directory.as_raw_fd(), temporary_name.as_ptr(), 0) };
        if unlink_result != 0 {
            log::warn!(
                "WD14 export wrote {} but could not remove its temporary link: {}",
                path.display(),
                io::Error::last_os_error()
            );
        }
    }
    directory.sync_all()?;
    Ok(WriteOutcome::Written)
}

#[cfg(not(unix))]
fn atomic_write_sidecar(
    path: &Path,
    bytes: &[u8],
    overwrite: bool,
    expected_parent: FileIdentity,
) -> io::Result<WriteOutcome> {
    if path.exists() && !overwrite {
        return Ok(WriteOutcome::SkippedExists);
    }
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    if file_identity(&fs::metadata(parent)?) != expected_parent {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "sidecar directory changed after validation",
        ));
    }
    let mut temporary = NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;

    if overwrite {
        let temporary = temporary.into_temp_path();
        replace_file(temporary.as_ref(), path)?;
        sync_parent_directory(parent);
        Ok(WriteOutcome::Written)
    } else {
        match temporary.persist_noclobber(path) {
            Ok(_) => {
                sync_parent_directory(parent);
                Ok(WriteOutcome::Written)
            }
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
                Ok(WriteOutcome::SkippedExists)
            }
            Err(error) => Err(error.error),
        }
    }
}

#[cfg(target_os = "windows")]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) {}

fn sidecar_path_for(media_path: &Path) -> PathBuf {
    media_path.with_extension("txt")
}

#[cfg(target_os = "windows")]
fn case_insensitive_grouping(root: &Path) -> io::Result<bool> {
    use std::mem::{size_of, zeroed};
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileCaseSensitiveInfo, GetFileInformationByHandleEx, FILE_CASE_SENSITIVE_INFO,
        FILE_FLAG_BACKUP_SEMANTICS,
    };

    let directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(root)?;
    let mut info: FILE_CASE_SENSITIVE_INFO = unsafe { zeroed() };
    let result = unsafe {
        GetFileInformationByHandleEx(
            directory.as_raw_handle() as _,
            FileCaseSensitiveInfo,
            (&mut info as *mut FILE_CASE_SENSITIVE_INFO).cast(),
            size_of::<FILE_CASE_SENSITIVE_INFO>() as u32,
        )
    };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    const FILE_CS_FLAG_CASE_SENSITIVE_DIR: u32 = 1;
    Ok(info.Flags & FILE_CS_FLAG_CASE_SENSITIVE_DIR == 0)
}

#[cfg(target_os = "macos")]
fn case_insensitive_grouping(root: &Path) -> io::Result<bool> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(root.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
    let result = unsafe { libc::pathconf(path.as_ptr(), libc::_PC_CASE_SENSITIVE) };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result == 0)
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn case_insensitive_grouping(root: &Path) -> io::Result<bool> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;

    let directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open(root)?;
    let mut filesystem: libc::statfs = unsafe { std::mem::zeroed() };
    let filesystem_type = (unsafe { libc::fstatfs(directory.as_raw_fd(), &mut filesystem) } == 0)
        .then_some(filesystem.f_type as libc::c_long);
    const EXFAT_SUPER_MAGIC: libc::c_long = 0x2011_bab0;
    const NTFS_SB_MAGIC: libc::c_long = 0x5346_544e;
    if filesystem_type.is_some_and(|filesystem_type| {
        matches!(
            filesystem_type,
            libc::MSDOS_SUPER_MAGIC | EXFAT_SUPER_MAGIC | NTFS_SB_MAGIC
        )
    }) {
        return Ok(true);
    }

    let mut flags: libc::c_long = 0;
    let ioctl_result = unsafe {
        libc::ioctl(
            directory.as_raw_fd(),
            libc::FS_IOC_GETFLAGS as _,
            &mut flags,
        )
    };
    const FS_CASEFOLD_FL: libc::c_long = 0x4000_0000;
    if ioctl_result == 0
        && filesystem_type.is_some_and(|filesystem_type| {
            matches!(
                filesystem_type,
                libc::EXT4_SUPER_MAGIC | libc::F2FS_SUPER_MAGIC
            )
        })
    {
        return Ok(flags & FS_CASEFOLD_FL != 0);
    }

    let probe = tempfile::Builder::new()
        .prefix(".LocalBooru-WD14-Case-Probe-")
        .tempfile_in(root)?;
    let probe_path = probe.path();
    let alternate_name = probe_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_ascii_lowercase)
        .filter(|name| Some(name.as_str()) != probe_path.file_name().and_then(|name| name.to_str()))
        .ok_or_else(|| io::Error::other("case probe name could not be transformed"))?;
    let alternate_path = root.join(alternate_name);
    match fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(alternate_path)
    {
        Ok(alternate) => Ok(
            file_identity(&probe.as_file().metadata()?) == file_identity(&alternate.metadata()?)
        ),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn case_insensitive_grouping(_root: &Path) -> io::Result<bool> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "filesystem case behavior is not supported on this platform",
    ))
}

#[cfg(unix)]
fn directory_entry_key(path: &Path, case_insensitive: bool) -> Result<Vec<u8>, String> {
    use std::os::unix::ffi::OsStrExt;

    let parent = path
        .parent()
        .ok_or_else(|| format!("Sidecar has no parent directory: {}", path.display()))?;
    let name = path
        .file_name()
        .ok_or_else(|| format!("Sidecar has no file name: {}", path.display()))?
        .as_bytes();
    if case_insensitive && !name.is_ascii() {
        return Err(format!(
            "Cannot safely group a non-ASCII sidecar name on this case-insensitive filesystem: {}",
            path.display()
        ));
    }
    let parent_metadata = fs::metadata(parent).map_err(|error| {
        format!(
            "Cannot identify sidecar parent directory {}: {error}",
            parent.display()
        )
    })?;
    let identity = file_identity(&parent_metadata);
    let mut key = b"\0WD14-ENTRY\0".to_vec();
    key.extend_from_slice(&identity.device.to_le_bytes());
    key.extend_from_slice(&identity.inode.to_le_bytes());
    if case_insensitive {
        key.extend(name.iter().map(u8::to_ascii_lowercase));
    } else {
        key.extend_from_slice(name);
    }
    Ok(key)
}

#[cfg(not(unix))]
fn directory_entry_key(path: &Path, case_insensitive: bool) -> Result<Vec<u8>, String> {
    if case_insensitive && !path.to_string_lossy().is_ascii() {
        return Err(format!(
            "Cannot safely group a non-ASCII sidecar path on this case-insensitive filesystem: {}",
            path.display()
        ));
    }
    Ok(path_key(path, case_insensitive))
}

#[cfg(unix)]
fn path_key(path: &Path, case_insensitive: bool) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;

    let bytes = path.as_os_str().as_bytes();
    if !case_insensitive {
        return bytes.to_vec();
    }
    match std::str::from_utf8(bytes) {
        Ok(path) => path.to_lowercase().into_bytes(),
        Err(_) => bytes.iter().map(u8::to_ascii_lowercase).collect(),
    }
}

#[cfg(target_os = "windows")]
fn path_key(path: &Path, case_insensitive: bool) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;

    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let normalized: Vec<u16> = if case_insensitive {
        match String::from_utf16(&wide) {
            Ok(path) => path.to_lowercase().encode_utf16().collect(),
            Err(_) => wide
                .into_iter()
                .map(|unit| {
                    if (b'A' as u16..=b'Z' as u16).contains(&unit) {
                        unit + (b'a' - b'A') as u16
                    } else {
                        unit
                    }
                })
                .collect(),
        }
    } else {
        wide
    };
    normalized.into_iter().flat_map(u16::to_le_bytes).collect()
}

#[cfg(not(any(unix, target_os = "windows")))]
fn path_key(path: &Path, case_insensitive: bool) -> Vec<u8> {
    let path = path.as_os_str().to_string_lossy();
    if case_insensitive {
        path.to_lowercase().into_bytes()
    } else {
        path.as_bytes().to_vec()
    }
}

fn serialized_targets(group: &SidecarGroup) -> Vec<MediaTarget> {
    group
        .targets
        .iter()
        .map(|target| target.locator.clone())
        .collect()
}

fn result(
    group: &SidecarGroup,
    status: SidecarStatus,
    tags_parsed: usize,
    tags_added: usize,
    error: Option<String>,
) -> SidecarResult {
    SidecarResult {
        sidecar_path: display_path(&group.path),
        targets: serialized_targets(group),
        status,
        tags_parsed,
        tags_added,
        error,
    }
}

#[cfg(unix)]
fn display_path(path: &Path) -> String {
    use std::os::unix::ffi::OsStrExt;

    match path.to_str() {
        Some(path) => path.to_owned(),
        None => {
            let mut escaped = String::from("unix-bytes:");
            for byte in path.as_os_str().as_bytes() {
                use std::fmt::Write as _;
                let _ = write!(escaped, "{byte:02X}");
            }
            escaped
        }
    }
}

#[cfg(not(unix))]
fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn result_rank(status: SidecarStatus) -> u8 {
    match status {
        SidecarStatus::FailedValidation
        | SidecarStatus::FailedRead
        | SidecarStatus::FailedDatabase
        | SidecarStatus::FailedWrite
        | SidecarStatus::ImportedNotRemoved
        | SidecarStatus::ConflictingMediaStem => 0,
        SidecarStatus::SkippedMissing | SidecarStatus::SkippedExists => 1,
        SidecarStatus::Imported | SidecarStatus::Absorbed | SidecarStatus::Exported => 2,
    }
}

fn record_result(summary: &mut Wd14Summary, result: &SidecarResult) {
    summary.tags_parsed += result.tags_parsed;
    summary.tags_added += result.tags_added;
    match result.status {
        SidecarStatus::Imported => {
            summary.sidecars_found += 1;
            summary.sidecars_succeeded += 1;
        }
        SidecarStatus::Absorbed => {
            summary.sidecars_found += 1;
            summary.sidecars_succeeded += 1;
            summary.sidecars_removed += 1;
        }
        SidecarStatus::Exported => {
            summary.sidecars_succeeded += 1;
            summary.sidecars_written += 1;
        }
        SidecarStatus::SkippedMissing => summary.sidecars_skipped += 1,
        SidecarStatus::SkippedExists => {
            summary.sidecars_found += 1;
            summary.sidecars_skipped += 1;
        }
        SidecarStatus::ImportedNotRemoved => {
            summary.sidecars_found += 1;
            summary.sidecars_failed += 1;
        }
        SidecarStatus::FailedRead | SidecarStatus::FailedDatabase => {
            summary.sidecars_found += 1;
            summary.sidecars_failed += 1;
        }
        SidecarStatus::FailedValidation
        | SidecarStatus::FailedWrite
        | SidecarStatus::ConflictingMediaStem => summary.sidecars_failed += 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        _temp: tempfile::TempDir,
        state: AppState,
        library_id: String,
        media_root: PathBuf,
    }

    impl Fixture {
        fn new(files: &[(i64, &str, &str)]) -> Self {
            let temp = tempfile::tempdir().unwrap();
            let data_dir = temp.path().join("data");
            let media_root = temp.path().join("media");
            fs::create_dir_all(&media_root).unwrap();
            let state = AppState::new(&data_dir, 0).unwrap();
            let library = state.resolve_library(None).unwrap();
            let library_id = library.uuid.clone();
            let main = library.main_pool.get().unwrap();
            main.execute(
                "INSERT INTO watch_directories (id, path) VALUES (1, ?1)",
                params![media_root.to_string_lossy()],
            )
            .unwrap();
            drop(main);

            let pool = library.directory_db.get_pool(1).unwrap();
            let connection = pool.get().unwrap();
            for (image_id, filename, hash) in files {
                let media_path = media_root.join(filename);
                fs::write(&media_path, b"media").unwrap();
                connection
                    .execute(
                        "INSERT INTO images (id, filename, file_hash) VALUES (?1, ?2, ?3)",
                        params![image_id, filename, hash],
                    )
                    .unwrap();
                connection
                    .execute(
                        "INSERT INTO image_files (image_id, original_path) VALUES (?1, ?2)",
                        params![image_id, media_path.to_string_lossy()],
                    )
                    .unwrap();
            }
            drop(connection);
            drop(library);
            Self {
                _temp: temp,
                state,
                library_id,
                media_root,
            }
        }

        fn request(&self, overwrite: bool) -> Wd14Request {
            Wd14Request {
                directories: vec![DirectorySelection {
                    library_id: self.library_id.clone(),
                    directory_id: 1,
                }],
                overwrite,
            }
        }

        fn tags(&self, image_id: i64) -> BTreeSet<String> {
            let library = self.state.resolve_library(Some(&self.library_id)).unwrap();
            load_tag_set(&library, 1, image_id).unwrap()
        }

        fn add_tags(&self, image_id: i64, tags: &[&str]) {
            let library = self.state.resolve_library(Some(&self.library_id)).unwrap();
            let mut reconciliation_budget = MAX_RECONCILIATION_QUERIES;
            apply_tags(
                &library,
                1,
                image_id,
                &tags.iter().map(|tag| (*tag).to_string()).collect(),
                &mut reconciliation_budget,
            )
            .unwrap();
        }
    }

    // AC: @wd14-sidecar-exchange-contract ac-text-contract
    #[test]
    fn import_parses_normalizes_and_bounds_real_sidecars() {
        let fixture = Fixture::new(&[(1, "sample.jpg", "one")]);
        let sidecar = fixture.media_root.join("sample.txt");
        fs::write(&sidecar, "  Silver_Hair, blue eyes, SILVER_HAIR, , tail\n").unwrap();
        let response = run_operation(
            &fixture.state,
            Wd14Operation::Import,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::Imported);
        assert_eq!(
            fixture.tags(1),
            BTreeSet::from([
                "blue eyes".to_string(),
                "silver_hair".to_string(),
                "tail".to_string(),
            ])
        );

        fs::write(&sidecar, vec![b'x'; MAX_SIDECAR_BYTES as usize + 1]).unwrap();
        let response = run_operation(
            &fixture.state,
            Wd14Operation::Import,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::FailedRead);
    }

    // AC: @wd14-sidecar-exchange-contract ac-additive-import
    // AC: @wd14-managed-filesystem-safety ac-idempotent-retry
    #[test]
    fn import_is_additive_manual_and_idempotent_without_rewriting_existing_tags() {
        let fixture = Fixture::new(&[(1, "sample.jpg", "one")]);
        let library = fixture
            .state
            .resolve_library(Some(&fixture.library_id))
            .unwrap();
        let main = library.main_pool.get().unwrap();
        main.execute(
            "INSERT INTO tags (id, name, category, post_count) VALUES (1, 'existing', 'artist', 1)",
            [],
        )
        .unwrap();
        main.execute(
            "INSERT INTO tags (id, name, category, post_count) VALUES (2, 'unrelated', 'meta', 1)",
            [],
        )
        .unwrap();
        main.execute(
            "UPDATE watch_directories SET tagged_count = 1 WHERE id = 1",
            [],
        )
        .unwrap();
        drop(main);
        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO image_tags (image_id, tag_id, confidence, is_manual) VALUES (1, 1, 0.9, 0)",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO image_tags (image_id, tag_id, confidence, is_manual) VALUES (1, 2, NULL, 1)",
                [],
            )
            .unwrap();
        drop(connection);
        drop(library);
        fs::write(fixture.media_root.join("sample.txt"), "existing, new_tag").unwrap();

        for _ in 0..2 {
            run_operation(
                &fixture.state,
                Wd14Operation::Import,
                fixture.request(false),
            )
            .unwrap();
        }

        let library = fixture
            .state
            .resolve_library(Some(&fixture.library_id))
            .unwrap();
        let main = library.main_pool.get().unwrap();
        let existing: (String, i64) = main
            .query_row(
                "SELECT category, post_count FROM tags WHERE name = 'existing'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(existing, ("artist".into(), 1));
        let new_id: i64 = main
            .query_row("SELECT id FROM tags WHERE name = 'new_tag'", [], |row| {
                row.get(0)
            })
            .unwrap();
        let new_count: i64 = main
            .query_row(
                "SELECT post_count FROM tags WHERE id = ?1",
                params![new_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(new_count, 1);
        drop(main);
        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        let new_metadata: (Option<f64>, bool) = connection
            .query_row(
                "SELECT confidence, is_manual FROM image_tags WHERE image_id = 1 AND tag_id = ?1",
                params![new_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(new_metadata, (None, true));
        let unrelated_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM image_tags WHERE image_id = 1 AND tag_id = 2",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(unrelated_count, 1);
    }

    // AC: @wd14-sidecar-exchange-contract ac-safe-absorb
    #[test]
    fn absorb_retains_sidecar_until_every_shared_target_commits() {
        let fixture = Fixture::new(&[(1, "shared.jpg", "one"), (2, "shared.png", "two")]);
        let sidecar = fixture.media_root.join("shared.txt");
        fs::write(&sidecar, "safe_tag").unwrap();
        let library = fixture
            .state
            .resolve_library(Some(&fixture.library_id))
            .unwrap();
        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute_batch(
                "PRAGMA foreign_keys = OFF;
             DELETE FROM images WHERE id = 2;
             PRAGMA foreign_keys = ON;",
            )
            .unwrap();
        drop(connection);
        drop(library);

        let response = run_operation(
            &fixture.state,
            Wd14Operation::Absorb,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::FailedDatabase);
        assert!(sidecar.exists());
        assert!(fixture.tags(1).contains("safe_tag"));

        let library = fixture
            .state
            .resolve_library(Some(&fixture.library_id))
            .unwrap();
        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (2, 'shared.png', 'two-restored')",
                [],
            )
            .unwrap();
        drop(connection);
        drop(library);
        let response = run_operation(
            &fixture.state,
            Wd14Operation::Absorb,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::Absorbed);
        assert!(!sidecar.exists());
        assert!(fixture.tags(2).contains("safe_tag"));
    }

    // AC: @wd14-sidecar-exchange-contract ac-deterministic-export
    #[test]
    fn export_writes_complete_same_stem_tags_and_respects_overwrite_policy() {
        let fixture = Fixture::new(&[(1, "archive.photo.webp", "one")]);
        fixture.add_tags(1, &["z_tag", "a tag"]);
        let sidecar = fixture.media_root.join("archive.photo.txt");

        let response = run_operation(
            &fixture.state,
            Wd14Operation::Export,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::Exported);
        assert_eq!(fs::read_to_string(&sidecar).unwrap(), "a tag, z_tag\n");

        fs::write(&sidecar, "external").unwrap();
        let response = run_operation(
            &fixture.state,
            Wd14Operation::Export,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::SkippedExists);
        assert_eq!(fs::read_to_string(&sidecar).unwrap(), "external");
        let response =
            run_operation(&fixture.state, Wd14Operation::Export, fixture.request(true)).unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::Exported);
        assert_eq!(fs::read_to_string(&sidecar).unwrap(), "a tag, z_tag\n");
    }

    // AC: @wd14-sidecar-exchange-contract ac-shared-stem
    #[test]
    fn shared_stem_import_fans_out_and_conflicting_export_changes_nothing() {
        let fixture = Fixture::new(&[(1, "shared.jpg", "one"), (2, "shared.png", "two")]);
        let sidecar = fixture.media_root.join("shared.txt");
        fs::write(&sidecar, "shared_tag").unwrap();
        let response = run_operation(
            &fixture.state,
            Wd14Operation::Import,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].targets.len(), 2);
        assert!(fixture.tags(1).contains("shared_tag"));
        assert!(fixture.tags(2).contains("shared_tag"));

        fixture.add_tags(2, &["different"]);
        fs::write(&sidecar, "leave untouched").unwrap();
        let response =
            run_operation(&fixture.state, Wd14Operation::Export, fixture.request(true)).unwrap();
        assert_eq!(
            response.results[0].status,
            SidecarStatus::ConflictingMediaStem
        );
        assert_eq!(fs::read_to_string(&sidecar).unwrap(), "leave untouched");
    }

    // AC: @wd14-sidecar-exchange-contract ac-safe-absorb
    // AC: @wd14-sidecar-exchange-contract ac-shared-stem
    // AC: @wd14-managed-filesystem-safety ac-idempotent-retry
    #[test]
    fn invalid_shared_target_preserves_one_sidecar_and_blocks_every_target() {
        let fixture = Fixture::new(&[(1, "shared.jpg", "one"), (2, "shared.png", "two")]);
        let sidecar = fixture.media_root.join("shared.txt");
        fs::write(&sidecar, "shared_tag").unwrap();
        fs::remove_file(fixture.media_root.join("shared.png")).unwrap();

        let response = run_operation(
            &fixture.state,
            Wd14Operation::Absorb,
            fixture.request(false),
        )
        .unwrap();

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].status, SidecarStatus::FailedValidation);
        assert_eq!(response.results[0].targets.len(), 2);
        assert_eq!(response.summary.sidecars_failed, 1);
        assert!(sidecar.exists());
        assert!(fixture.tags(1).is_empty());
        assert!(fixture.tags(2).is_empty());
    }

    // AC: @wd14-sidecar-exchange-contract ac-shared-stem
    #[test]
    fn grouping_keys_fold_case_only_for_case_insensitive_filesystems() {
        let lower = Path::new("dataset/foo.txt");
        let upper = Path::new("dataset/FOO.txt");
        assert_ne!(path_key(lower, false), path_key(upper, false));
        assert_eq!(path_key(lower, true), path_key(upper, true));
    }

    // AC: @wd14-sidecar-exchange-contract ac-shared-stem
    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn filesystem_case_probe_is_non_destructive_on_case_sensitive_directories() {
        let directory = tempfile::tempdir().unwrap();
        assert!(!case_insensitive_grouping(directory.path()).unwrap());
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    // AC: @wd14-sidecar-exchange-contract ac-round-trip
    #[test]
    fn exported_file_imports_into_an_untagged_database_record() {
        let fixture = Fixture::new(&[(1, "source.jpg", "one"), (2, "destination.jpg", "two")]);
        fixture.add_tags(1, &["blue eyes", "silver_hair"]);
        run_operation(
            &fixture.state,
            Wd14Operation::Export,
            fixture.request(false),
        )
        .unwrap();
        fs::copy(
            fixture.media_root.join("source.txt"),
            fixture.media_root.join("destination.txt"),
        )
        .unwrap();
        run_operation(
            &fixture.state,
            Wd14Operation::Import,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(fixture.tags(2), fixture.tags(1));
    }

    #[test]
    fn export_rejects_unrepresentable_tags_without_touching_the_sidecar() {
        let fixture = Fixture::new(&[(1, "sample.jpg", "one")]);
        let library = fixture
            .state
            .resolve_library(Some(&fixture.library_id))
            .unwrap();
        let main = library.main_pool.get().unwrap();
        main.execute(
            "INSERT INTO tags (id, name, category) VALUES (1, 'bad,tag', 'general')",
            [],
        )
        .unwrap();
        drop(main);
        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO image_tags (image_id, tag_id) VALUES (1, 1)",
                [],
            )
            .unwrap();
        drop(connection);
        drop(library);
        let sidecar = fixture.media_root.join("sample.txt");
        fs::write(&sidecar, "keep").unwrap();
        let response =
            run_operation(&fixture.state, Wd14Operation::Export, fixture.request(true)).unwrap();
        assert_eq!(response.results[0].status, SidecarStatus::FailedValidation);
        assert_eq!(fs::read_to_string(sidecar).unwrap(), "keep");
    }

    // AC: @wd14-sidecar-exchange-contract ac-safe-absorb
    #[test]
    fn absorb_identity_guard_detects_a_replaced_sidecar() {
        let fixture = Fixture::new(&[(1, "sample.jpg", "one")]);
        let sidecar = fixture.media_root.join("sample.txt");
        fs::write(&sidecar, "original").unwrap();
        let group = SidecarGroup {
            path: sidecar.clone(),
            targets: vec![PreparedTarget {
                locator: MediaTarget {
                    library_id: fixture.library_id.clone(),
                    directory_id: 1,
                    image_id: 1,
                    media_path: fixture
                        .media_root
                        .join("sample.jpg")
                        .to_string_lossy()
                        .into_owned(),
                },
                root: fs::canonicalize(&fixture.media_root).unwrap(),
            }],
            preparation_errors: Vec::new(),
        };
        let original = read_existing_sidecar(&group).unwrap().unwrap();
        let replacement = fixture.media_root.join("replacement.tmp");
        fs::write(&replacement, "replacement").unwrap();
        fs::rename(replacement, &sidecar).unwrap();
        assert!(remove_absorbed_sidecar(&group, &original).is_err());
        assert_eq!(fs::read_to_string(sidecar).unwrap(), "replacement");
        assert_eq!(
            fs::read_dir(&fixture.media_root)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().contains(".absorbing"))
                .count(),
            0
        );
    }

    // AC: @wd14-managed-filesystem-safety ac-compound-scope
    #[test]
    fn compound_scope_keeps_duplicate_image_ids_in_their_selected_library() {
        let fixture = Fixture::new(&[(1, "primary.jpg", "primary")]);
        let auxiliary_data = fixture._temp.path().join("auxiliary-data");
        let auxiliary_media = fixture._temp.path().join("auxiliary-media");
        fs::create_dir_all(&auxiliary_data).unwrap();
        fs::create_dir_all(&auxiliary_media).unwrap();
        let auxiliary = LibraryContext::open(&auxiliary_data, "Auxiliary").unwrap();
        let auxiliary_id = auxiliary.uuid.clone();
        auxiliary
            .main_pool
            .get()
            .unwrap()
            .execute(
                "INSERT INTO watch_directories (id, path) VALUES (1, ?1)",
                params![auxiliary_media.to_string_lossy()],
            )
            .unwrap();
        let auxiliary_path = auxiliary_media.join("auxiliary.jpg");
        fs::write(&auxiliary_path, b"media").unwrap();
        let pool = auxiliary.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (1, 'auxiliary.jpg', 'auxiliary')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO image_files (image_id, original_path) VALUES (1, ?1)",
                params![auxiliary_path.to_string_lossy()],
            )
            .unwrap();
        drop(connection);
        drop(pool);
        fixture.state.library_manager().mount(auxiliary);
        fs::write(auxiliary_media.join("auxiliary.txt"), "scoped").unwrap();

        let response = run_operation(
            &fixture.state,
            Wd14Operation::Import,
            Wd14Request {
                directories: vec![DirectorySelection {
                    library_id: auxiliary_id.clone(),
                    directory_id: 1,
                }],
                overwrite: false,
            },
        )
        .unwrap();
        assert_eq!(response.summary.sidecars_succeeded, 1);
        assert_eq!(response.results[0].targets[0].library_id, auxiliary_id);
        assert!(fixture.tags(1).is_empty());
        let auxiliary = fixture
            .state
            .resolve_library(Some(&response.results[0].targets[0].library_id))
            .unwrap();
        assert_eq!(
            load_tag_set(&auxiliary, 1, 1).unwrap(),
            BTreeSet::from(["scoped".to_string()])
        );
    }

    // AC: @wd14-managed-filesystem-safety ac-batch-results
    #[test]
    fn mixed_batch_reports_each_sidecar_and_accurate_aggregate_counts() {
        let fixture = Fixture::new(&[
            (1, "good.jpg", "good"),
            (2, "invalid.jpg", "invalid"),
            (3, "missing.jpg", "missing"),
            (4, "inaccessible.jpg", "inaccessible"),
        ]);
        fs::write(fixture.media_root.join("good.txt"), "one, two").unwrap();
        fs::write(fixture.media_root.join("invalid.txt"), [0xff]).unwrap();
        fs::remove_file(fixture.media_root.join("inaccessible.jpg")).unwrap();

        let response = run_operation(
            &fixture.state,
            Wd14Operation::Import,
            fixture.request(false),
        )
        .unwrap();
        assert_eq!(response.summary.media_candidates, 4);
        assert_eq!(response.summary.sidecars_succeeded, 1);
        assert_eq!(response.summary.sidecars_failed, 2);
        assert_eq!(response.summary.sidecars_skipped, 1);
        assert_eq!(response.summary.tags_added, 2);
        assert_eq!(response.results.len(), 4);
        assert!(response
            .results
            .iter()
            .any(|result| result.status == SidecarStatus::Imported && result.tags_added == 2));
        assert!(response
            .results
            .iter()
            .any(|result| result.status == SidecarStatus::FailedRead && result.error.is_some()));
        assert!(response
            .results
            .iter()
            .any(|result| result.status == SidecarStatus::SkippedMissing));
        assert!(response
            .results
            .iter()
            .any(|result| result.status == SidecarStatus::FailedValidation));
        assert!(response
            .results
            .iter()
            .all(|result| result.targets.len() == 1
                && result.targets[0].library_id == fixture.library_id));

        let collision = Fixture::new(&[(1, "same.jpg", "one"), (2, "same.png", "two")]);
        collision.add_tags(1, &["first"]);
        collision.add_tags(2, &["second"]);
        let collision_response = run_operation(
            &collision.state,
            Wd14Operation::Export,
            collision.request(false),
        )
        .unwrap();
        assert_eq!(collision_response.summary.sidecars_failed, 1);
        assert_eq!(
            collision_response.results[0].status,
            SidecarStatus::ConflictingMediaStem
        );
        assert!(!collision.media_root.join("same.txt").exists());
    }

    // AC: @wd14-managed-filesystem-safety ac-managed-paths
    #[cfg(unix)]
    #[test]
    fn validation_rejects_escaping_media_sidecar_symlinks_and_lossy_key_collisions() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_media = outside.path().join("outside.jpg");
        fs::write(&outside_media, b"media").unwrap();
        let media_link = root.path().join("image.jpg");
        symlink(&outside_media, &media_link).unwrap();
        let canonical_root = fs::canonicalize(root.path()).unwrap();
        assert!(validate_media_path(&media_link, &canonical_root).is_err());

        let inside_media = root.path().join("inside.jpg");
        fs::write(&inside_media, b"media").unwrap();
        let outside_sidecar = outside.path().join("outside.txt");
        fs::write(&outside_sidecar, b"tag").unwrap();
        let sidecar_link = root.path().join("inside.txt");
        symlink(&outside_sidecar, &sidecar_link).unwrap();
        let target = PreparedTarget {
            locator: MediaTarget {
                library_id: "test".into(),
                directory_id: 1,
                image_id: 1,
                media_path: inside_media.to_string_lossy().into_owned(),
            },
            root: canonical_root,
        };
        let group = SidecarGroup {
            path: sidecar_link,
            targets: vec![target],
            preparation_errors: Vec::new(),
        };
        assert!(read_existing_sidecar(&group).is_err());

        let first = PathBuf::from(OsString::from_vec(vec![b'x', 0xff]));
        let second = PathBuf::from(OsString::from_vec(vec![b'x', 0xfe]));
        assert_ne!(path_key(&first, false), path_key(&second, false));
        assert_ne!(display_path(&first), display_path(&second));
    }

    // AC: @wd14-managed-filesystem-safety ac-managed-paths
    #[cfg(unix)]
    #[test]
    fn export_rejects_a_parent_directory_replaced_after_validation() {
        let fixture = Fixture::new(&[(1, "sample.jpg", "one")]);
        let sidecar = fixture.media_root.join("sample.txt");
        let parent_identity = file_identity(&fs::metadata(&fixture.media_root).unwrap());
        let moved = fixture._temp.path().join("moved-media");
        fs::rename(&fixture.media_root, &moved).unwrap();
        fs::create_dir(&fixture.media_root).unwrap();

        assert!(atomic_write_sidecar(&sidecar, b"tag\n", false, parent_identity).is_err());
        assert!(!sidecar.exists());
        assert!(!moved.join("sample.txt").exists());
    }
}
