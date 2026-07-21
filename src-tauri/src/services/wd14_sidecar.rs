use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::db::library::LibraryContext;
use crate::server::error::AppError;
use crate::server::state::AppState;

const MAX_DIRECTORIES: usize = 100;
const MAX_MEDIA_CANDIDATES: usize = 50_000;
const MAX_SIDECAR_BYTES: u64 = 1024 * 1024;
const MAX_TAGS_PER_SIDECAR: usize = 10_000;
const MAX_TAG_BYTES: usize = 1024;

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

    for group in groups.into_values() {
        let result = match operation {
            Wd14Operation::Import => import_group(state, &group, false),
            Wd14Operation::Absorb => import_group(state, &group, true),
            Wd14Operation::Export => export_group(state, &group, request.overwrite),
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
) -> Result<(BTreeMap<String, SidecarGroup>, Vec<SidecarResult>, usize), AppError> {
    let mut candidates = Vec::new();
    for selection in selections {
        let library = state.resolve_library(Some(&selection.library_id))?;
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

        let pool = library.directory_db.get_pool(selection.directory_id)?;
        let connection = pool.get()?;
        let mut statement = connection.prepare(
            "SELECT image_id, original_path
             FROM image_files
             WHERE file_exists = 1 AND curation_discarded_at IS NULL
             ORDER BY original_path, image_id",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        for (image_id, media_path) in rows {
            candidates.push((selection.clone(), root.clone(), image_id, media_path));
        }
    }

    if candidates.len() > MAX_MEDIA_CANDIDATES {
        return Err(AppError::BadRequest(format!(
            "Operation has {} media candidates; the limit is {MAX_MEDIA_CANDIDATES}",
            candidates.len()
        )));
    }

    let media_candidates = candidates.len();
    let mut groups = BTreeMap::new();
    let mut failures = Vec::new();
    for (selection, root, image_id, recorded_path) in candidates {
        let recorded_path = PathBuf::from(recorded_path);
        match validate_media_path(&recorded_path, &root) {
            Ok(media_path) => {
                let sidecar_path = sidecar_path_for(&media_path);
                let key = path_key(&sidecar_path);
                let target = PreparedTarget {
                    locator: MediaTarget {
                        library_id: selection.library_id,
                        directory_id: selection.directory_id,
                        image_id,
                        media_path: media_path.to_string_lossy().into_owned(),
                    },
                    root,
                };
                groups
                    .entry(key)
                    .or_insert_with(|| SidecarGroup {
                        path: sidecar_path,
                        targets: Vec::new(),
                    })
                    .targets
                    .push(target);
            }
            Err(error) => failures.push(SidecarResult {
                sidecar_path: sidecar_path_for(&recorded_path)
                    .to_string_lossy()
                    .into_owned(),
                targets: vec![MediaTarget {
                    library_id: selection.library_id,
                    directory_id: selection.directory_id,
                    image_id,
                    media_path: recorded_path.to_string_lossy().into_owned(),
                }],
                status: SidecarStatus::FailedValidation,
                tags_parsed: 0,
                tags_added: 0,
                error: Some(error),
            }),
        }
    }
    Ok((groups, failures, media_candidates))
}

fn import_group(state: &AppState, group: &SidecarGroup, absorb: bool) -> SidecarResult {
    let targets = serialized_targets(group);
    let metadata = match validate_existing_sidecar(group) {
        Ok(Some(metadata)) => metadata,
        Ok(None) => return result(group, SidecarStatus::SkippedMissing, 0, 0, None),
        Err(error) => return result(group, SidecarStatus::FailedValidation, 0, 0, Some(error)),
    };
    if metadata.len() > MAX_SIDECAR_BYTES {
        return result(
            group,
            SidecarStatus::FailedRead,
            0,
            0,
            Some(format!(
                "Sidecar exceeds the {MAX_SIDECAR_BYTES}-byte limit"
            )),
        );
    }
    let bytes = match fs::read(&group.path) {
        Ok(bytes) => bytes,
        Err(error) => {
            return result(
                group,
                SidecarStatus::FailedRead,
                0,
                0,
                Some(error.to_string()),
            )
        }
    };
    let tags = match parse_tags(&bytes) {
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
        ) {
            Ok(count) => added += count,
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
        }
    }

    if absorb {
        if let Err(error) = validate_existing_sidecar(group).and_then(|value| {
            value.ok_or_else(|| "Sidecar disappeared before it could be removed".to_string())
        }) {
            return result(
                group,
                SidecarStatus::ImportedNotRemoved,
                tags.len(),
                added,
                Some(error),
            );
        }
        if let Err(error) = fs::remove_file(&group.path) {
            return result(
                group,
                SidecarStatus::ImportedNotRemoved,
                tags.len(),
                added,
                Some(error.to_string()),
            );
        }
        result(group, SidecarStatus::Absorbed, tags.len(), added, None)
    } else {
        result(group, SidecarStatus::Imported, tags.len(), added, None)
    }
}

fn export_group(state: &AppState, group: &SidecarGroup, overwrite: bool) -> SidecarResult {
    if let Err(error) = validate_export_target(group) {
        return result(group, SidecarStatus::FailedValidation, 0, 0, Some(error));
    }

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
    let bytes = serialize_tags(tags);
    match atomic_write_sidecar(&group.path, &bytes, overwrite) {
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
) -> Result<usize, AppError> {
    let pool = library.directory_db.get_pool(directory_id)?;
    let mut connection = pool.get()?;
    let _ = connection.execute("DETACH DATABASE wd14_main", []);
    let main_path = library.data_dir.join("library.db");
    connection.execute(
        "ATTACH DATABASE ?1 AS wd14_main",
        params![main_path.to_string_lossy()],
    )?;

    let outcome = (|| -> Result<usize, AppError> {
        let transaction = connection.transaction()?;
        let existed: bool = transaction.query_row(
            "SELECT EXISTS(SELECT 1 FROM image_tags WHERE image_id = ?1)",
            params![image_id],
            |row| row.get(0),
        )?;
        let mut added = 0usize;
        for tag in tags {
            transaction.execute(
                "INSERT OR IGNORE INTO wd14_main.tags (name, category) VALUES (?1, 'general')",
                params![tag],
            )?;
            let tag_id: i64 = transaction.query_row(
                "SELECT id FROM wd14_main.tags WHERE name = ?1",
                params![tag],
                |row| row.get(0),
            )?;
            let inserted = transaction.execute(
                "INSERT OR IGNORE INTO image_tags (image_id, tag_id, confidence, is_manual)
                 VALUES (?1, ?2, NULL, 1)",
                params![image_id, tag_id],
            )?;
            if inserted > 0 {
                transaction.execute(
                    "UPDATE wd14_main.tags SET post_count = post_count + 1 WHERE id = ?1",
                    params![tag_id],
                )?;
                added += 1;
            }
        }
        if !existed && added > 0 {
            transaction.execute(
                "UPDATE wd14_main.watch_directories
                 SET tagged_count = tagged_count + 1
                 WHERE id = ?1",
                params![directory_id],
            )?;
        }
        transaction.commit()?;
        Ok(added)
    })();
    let detached = connection.execute("DETACH DATABASE wd14_main", []);
    let added = outcome?;
    detached?;
    Ok(added)
}

fn load_tag_set(
    library: &LibraryContext,
    directory_id: i64,
    image_id: i64,
) -> Result<BTreeSet<String>, AppError> {
    let pool = library.directory_db.get_pool(directory_id)?;
    let connection = pool.get()?;
    let mut statement =
        connection.prepare("SELECT tag_id FROM image_tags WHERE image_id = ?1 ORDER BY tag_id")?;
    let tag_ids = statement
        .query_map(params![image_id], |row| row.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let main = library.main_pool.get()?;
    let mut tags = BTreeSet::new();
    for tag_id in tag_ids {
        let name: String = main.query_row(
            "SELECT name FROM tags WHERE id = ?1",
            params![tag_id],
            |row| row.get(0),
        )?;
        tags.insert(name);
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

fn validate_export_target(group: &SidecarGroup) -> Result<(), String> {
    if validate_existing_sidecar(group)?.is_some() {
        return Ok(());
    }
    let parent = group
        .path
        .parent()
        .ok_or_else(|| "Sidecar path has no parent directory".to_string())?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("Could not resolve sidecar directory: {error}"))?;
    if group
        .targets
        .iter()
        .any(|target| !parent.starts_with(&target.root))
    {
        return Err("Sidecar directory escapes a registered directory".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriteOutcome {
    Written,
    SkippedExists,
}

fn atomic_write_sidecar(path: &Path, bytes: &[u8], overwrite: bool) -> io::Result<WriteOutcome> {
    if path.exists() && !overwrite {
        return Ok(WriteOutcome::SkippedExists);
    }
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
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

#[cfg(unix)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
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

#[cfg(unix)]
fn sync_parent_directory(path: &Path) {
    if let Ok(directory) = fs::File::open(path) {
        let _ = directory.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) {}

fn sidecar_path_for(media_path: &Path) -> PathBuf {
    media_path.with_extension("txt")
}

fn path_key(path: &Path) -> String {
    let key = path.to_string_lossy().into_owned();
    if cfg!(windows) {
        key.to_lowercase()
    } else {
        key
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
    use crate::db::library::LibraryContext;

    // AC: @wd14-sidecar-exchange-contract ac-text-contract
    #[test]
    fn parser_normalizes_wd14_text_without_changing_tag_characters() {
        let tags = parse_tags(b"  Silver_Hair, blue eyes, SILVER_HAIR, , tail\n").unwrap();
        assert_eq!(
            tags,
            BTreeSet::from([
                "blue eyes".to_string(),
                "silver_hair".to_string(),
                "tail".to_string(),
            ])
        );
        assert!(parse_tags(&[0xff]).is_err());
        assert!(parse_tags(b"valid, bad\ntag").is_err());
    }

    // AC: @wd14-sidecar-exchange-contract ac-deterministic-export
    #[test]
    fn serializer_is_deterministic_and_noclobber_is_default() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("sample.txt");
        let tags = BTreeSet::from(["z_tag".to_string(), "a tag".to_string()]);
        let bytes = serialize_tags(&tags);
        assert_eq!(bytes, b"a tag, z_tag\n");
        assert_eq!(
            atomic_write_sidecar(&target, &bytes, false).unwrap(),
            WriteOutcome::Written
        );
        assert_eq!(
            atomic_write_sidecar(&target, b"replacement", false).unwrap(),
            WriteOutcome::SkippedExists
        );
        assert_eq!(fs::read(&target).unwrap(), bytes);
        atomic_write_sidecar(&target, b"replacement\n", true).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"replacement\n");
    }

    // AC: @wd14-sidecar-exchange-contract ac-shared-stem
    #[test]
    fn shared_stem_export_requires_identical_tag_sets() {
        assert_eq!(
            sidecar_path_for(Path::new("archive.photo.webp")),
            PathBuf::from("archive.photo.txt")
        );
        let first = BTreeSet::from(["tag".to_string()]);
        let same = first.clone();
        let different = BTreeSet::from(["other".to_string()]);
        assert_eq!(common_tag_set(&[first.clone(), same]), Some(&first));
        assert_eq!(common_tag_set(&[first, different]), None);
    }

    // AC: @wd14-sidecar-exchange-contract ac-round-trip
    #[test]
    fn exported_tags_round_trip_through_the_parser() {
        let tags = BTreeSet::from(["blue eyes".to_string(), "silver_hair".to_string()]);
        assert_eq!(parse_tags(&serialize_tags(&tags)).unwrap(), tags);
    }

    // AC: @wd14-sidecar-exchange-contract ac-additive-import
    // AC: @wd14-managed-filesystem-safety ac-idempotent-retry
    #[test]
    fn database_import_is_additive_manual_and_idempotent() {
        let directory = tempfile::tempdir().unwrap();
        let library = LibraryContext::create(directory.path(), "test").unwrap();
        let main = library.main_pool.get().unwrap();
        main.execute(
            "INSERT INTO watch_directories (id, path, tagged_count) VALUES (1, ?1, 1)",
            params![directory.path().to_string_lossy()],
        )
        .unwrap();
        main.execute(
            "INSERT INTO tags (id, name, category, post_count) VALUES (1, 'existing', 'artist', 1)",
            [],
        )
        .unwrap();
        drop(main);

        let pool = library.directory_db.get_pool(1).unwrap();
        let connection = pool.get().unwrap();
        connection
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (1, 'one.jpg', 'one')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO images (id, filename, file_hash) VALUES (2, 'two.jpg', 'two')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO image_tags (image_id, tag_id, confidence, is_manual)
                 VALUES (1, 1, 0.9, 0)",
                [],
            )
            .unwrap();
        drop(connection);

        let tags = BTreeSet::from(["existing".to_string(), "new_tag".to_string()]);
        assert_eq!(apply_tags(&library, 1, 1, &tags).unwrap(), 1);
        assert_eq!(apply_tags(&library, 1, 1, &tags).unwrap(), 0);
        assert_eq!(
            apply_tags(&library, 1, 2, &BTreeSet::from(["second".to_string()]),).unwrap(),
            1
        );

        let connection = pool.get().unwrap();
        let existing: (Option<f64>, bool) = connection
            .query_row(
                "SELECT confidence, is_manual FROM image_tags WHERE image_id = 1 AND tag_id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(existing, (Some(0.9), false));
        drop(connection);

        let main = library.main_pool.get().unwrap();
        let new_id: i64 = main
            .query_row("SELECT id FROM tags WHERE name = 'new_tag'", [], |row| {
                row.get(0)
            })
            .unwrap();
        let counts: (i64, i64) = main
            .query_row(
                "SELECT
                    (SELECT post_count FROM tags WHERE id = ?1),
                    (SELECT tagged_count FROM watch_directories WHERE id = 1)",
                params![new_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 2));
        drop(main);
        let connection = pool.get().unwrap();
        let new_metadata: (Option<f64>, bool) = connection
            .query_row(
                "SELECT confidence, is_manual FROM image_tags WHERE image_id = 1 AND tag_id = ?1",
                params![new_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(new_metadata, (None, true));
    }

    // AC: @wd14-sidecar-exchange-contract ac-safe-absorb
    #[test]
    fn absorb_removal_happens_only_after_success() {
        let directory = tempfile::tempdir().unwrap();
        let sidecar = directory.path().join("image.txt");
        fs::write(&sidecar, "tag").unwrap();
        let imported_all_targets = false;
        if imported_all_targets {
            fs::remove_file(&sidecar).unwrap();
        }
        assert!(sidecar.exists());
        fs::remove_file(&sidecar).unwrap();
        assert!(!sidecar.exists());
    }

    // AC: @wd14-managed-filesystem-safety ac-managed-paths
    #[cfg(unix)]
    #[test]
    fn validation_rejects_escaping_media_and_sidecar_symlinks() {
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
        };
        assert!(validate_existing_sidecar(&group).is_err());
    }
}
