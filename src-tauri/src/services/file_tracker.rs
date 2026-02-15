use std::path::Path;

use rusqlite::params;

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::importer::{self, ImportStatus};

/// File availability status.
#[derive(Debug, Clone, PartialEq)]
pub enum FileStatus {
    Available,
    Missing,
    DriveOffline,
}

/// Check if a file is available, distinguishing between:
/// - File exists and accessible
/// - File deleted but parent directory exists (confirmed missing)
/// - Parent directory/drive is unavailable (drive offline)
pub fn check_file_availability(file_path: &str) -> FileStatus {
    let path = Path::new(file_path);

    if path.exists() {
        return FileStatus::Available;
    }

    // Walk up parent chain to distinguish missing vs drive offline
    let mut parent = path.parent();
    while let Some(p) = parent {
        if p == Path::new("") || p == Path::new("/") {
            break;
        }
        if p.exists() {
            // Parent exists but file doesn't = file was deleted
            return FileStatus::Missing;
        }
        parent = p.parent();
    }

    // No parent directories exist = drive/mount point is offline
    FileStatus::DriveOffline
}

/// Check if a watch directory's drive/mount point is available.
pub fn is_drive_available(watch_directory_path: &str) -> bool {
    let mut path = std::path::PathBuf::from(watch_directory_path);
    loop {
        if path.exists() {
            return true;
        }
        if !path.pop() {
            break;
        }
    }
    false
}

/// Stats returned by scan/verify/clean operations.
#[derive(Debug, Default)]
pub struct ScanStats {
    pub found: i64,
    pub imported: i64,
    pub duplicates: i64,
    pub errors: i64,
    pub cleaned: i64,
    pub removed: i64,
}

/// Scan a directory for media files and import them.
///
/// Uses streaming file discovery to start processing while scanning.
pub fn scan_directory(
    state: &AppState,
    directory_id: i64,
    directory_path: &str,
    recursive: bool,
    clean_deleted: bool,
) -> Result<ScanStats, AppError> {
    let mut stats = ScanStats::default();

    let path = Path::new(directory_path);
    if !path.exists() || !path.is_dir() {
        return Err(AppError::BadRequest(format!(
            "Directory does not exist: {}",
            directory_path
        )));
    }

    // Clean deleted files first if requested
    if clean_deleted {
        stats.removed = clean_deleted_files(state, directory_id)? as i64;
    }

    // Walk the directory for media files
    let walker: Box<dyn Iterator<Item = walkdir::DirEntry>> = if recursive {
        Box::new(
            walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok()),
        )
    } else {
        Box::new(
            walkdir::WalkDir::new(path)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok()),
        )
    };

    for entry in walker {
        if !entry.file_type().is_file() {
            continue;
        }
        if !importer::is_media_file(entry.path()) {
            continue;
        }

        stats.found += 1;

        let file_path = entry.path().to_string_lossy().to_string();
        match importer::import_image(state, &file_path, directory_id) {
            Ok(result) => match result.status {
                ImportStatus::Imported => stats.imported += 1,
                ImportStatus::Duplicate => stats.duplicates += 1,
                ImportStatus::Error => stats.errors += 1,
            },
            Err(e) => {
                log::warn!("Import error for {}: {}", entry.path().display(), e);
                stats.errors += 1;
            }
        }
    }

    // Update last_scanned_at in main DB
    let main_conn = state.main_db().get()?;
    let now = chrono::Utc::now().to_rfc3339();
    main_conn.execute(
        "UPDATE watch_directories SET last_scanned_at = ?1 WHERE id = ?2",
        params![&now, directory_id],
    )?;

    Ok(stats)
}

/// Clean up deleted files from a directory database.
///
/// Returns the number of removed file references.
pub fn clean_deleted_files(state: &AppState, directory_id: i64) -> Result<i64, AppError> {
    let dir_pool = state.directory_db().get_pool(directory_id)?;
    let conn = dir_pool.get()?;
    let mut removed: i64 = 0;

    // Get all file references
    let mut stmt = conn.prepare(
        "SELECT id, image_id, original_path FROM image_files",
    )?;

    let files: Vec<(i64, i64, String)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (file_id, image_id, original_path) in files {
        if Path::new(&original_path).exists() {
            continue;
        }

        // Check if this is the only file for this image
        let file_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM image_files WHERE image_id = ?1",
                params![image_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if file_count <= 1 {
            // Last file — delete the image and thumbnail
            if let Ok(hash) = conn.query_row(
                "SELECT file_hash FROM images WHERE id = ?1",
                params![image_id],
                |row| row.get::<_, String>(0),
            ) {
                let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
                let thumb_path = state.thumbnails_dir().join(&thumb_name);
                let _ = std::fs::remove_file(&thumb_path);
            }
            conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
        }

        conn.execute("DELETE FROM image_files WHERE id = ?1", params![file_id])?;
        removed += 1;
    }

    Ok(removed)
}

/// Verify files in a specific directory still exist at their recorded locations.
#[derive(Debug, Default)]
pub struct VerifyStats {
    pub verified: i64,
    pub deleted: i64,
    pub drive_offline: i64,
}

pub fn verify_directory_files(
    state: &AppState,
    directory_id: i64,
) -> Result<VerifyStats, AppError> {
    let mut stats = VerifyStats::default();

    let dir_pool = state.directory_db().get_pool(directory_id)?;
    let conn = dir_pool.get()?;

    let mut stmt =
        conn.prepare("SELECT id, image_id, original_path FROM image_files WHERE file_status != 'drive_offline'")?;

    let files: Vec<(i64, i64, String)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (file_id, image_id, original_path) in files {
        match check_file_availability(&original_path) {
            FileStatus::Available => {
                conn.execute(
                    "UPDATE image_files SET file_exists = 1, file_status = 'available' WHERE id = ?1",
                    params![file_id],
                )?;
                stats.verified += 1;
            }
            FileStatus::DriveOffline => {
                let now = chrono::Utc::now().to_rfc3339();
                conn.execute(
                    "UPDATE image_files SET file_status = 'drive_offline', last_verified_at = ?1 WHERE id = ?2",
                    params![&now, file_id],
                )?;
                stats.drive_offline += 1;
            }
            FileStatus::Missing => {
                // Check for other file references
                let other_count: i64 = conn
                    .query_row(
                        "SELECT COUNT(*) FROM image_files WHERE image_id = ?1 AND id != ?2",
                        params![image_id, file_id],
                        |row| row.get(0),
                    )
                    .unwrap_or(0);

                conn.execute("DELETE FROM image_files WHERE id = ?1", params![file_id])?;

                if other_count == 0 {
                    // Delete the image and thumbnail
                    if let Ok(hash) = conn.query_row(
                        "SELECT file_hash FROM images WHERE id = ?1",
                        params![image_id],
                        |row| row.get::<_, String>(0),
                    ) {
                        let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
                        let thumb_path = state.thumbnails_dir().join(&thumb_name);
                        let _ = std::fs::remove_file(&thumb_path);
                    }
                    conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
                }

                stats.deleted += 1;
            }
        }
    }

    Ok(stats)
}

/// Mark a file as missing (called when file watcher detects deletion).
///
/// Deletes the ImageFile entry. If no other references exist, deletes the Image too.
pub fn mark_file_missing(
    state: &AppState,
    file_path: &str,
    directory_id: i64,
) -> Result<(), AppError> {
    let dir_pool = state.directory_db().get_pool(directory_id)?;
    let conn = dir_pool.get()?;

    // Find the file record
    let file_info: Option<(i64, i64)> = conn
        .query_row(
            "SELECT id, image_id FROM image_files WHERE original_path = ?1",
            params![file_path],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok();

    let (file_id, image_id) = match file_info {
        Some(info) => info,
        None => return Ok(()), // File not tracked
    };

    // Check for other file references
    let other_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM image_files WHERE image_id = ?1 AND id != ?2",
            params![image_id, file_id],
            |row| row.get(0),
        )
        .unwrap_or(0);

    conn.execute("DELETE FROM image_files WHERE id = ?1", params![file_id])?;

    if other_count == 0 {
        // Delete image and thumbnail
        if let Ok(hash) = conn.query_row(
            "SELECT file_hash FROM images WHERE id = ?1",
            params![image_id],
            |row| row.get::<_, String>(0),
        ) {
            let thumb_name = format!("{}.webp", &hash[..16.min(hash.len())]);
            let thumb_path = state.thumbnails_dir().join(&thumb_name);
            let _ = std::fs::remove_file(&thumb_path);
        }
        conn.execute("DELETE FROM images WHERE id = ?1", params![image_id])?;
    }

    Ok(())
}
