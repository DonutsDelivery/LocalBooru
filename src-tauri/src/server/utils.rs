use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rusqlite::Connection;

use crate::server::error::AppError;
use crate::server::middleware::AccessTier;
use crate::server::state::AppState;

/// Lexically normalize a path: resolve `.` and `..` components *without* touching
/// the filesystem or following symlinks. This neutralizes path traversal while
/// preserving the path as written.
///
/// Android only: on desktop the strict `canonicalize` comparison is used instead.
#[cfg(target_os = "android")]
fn normalize_lexical(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Canonicalize a client-supplied file path and confirm it resides inside one of
/// the configured watch directories.
///
/// Returns the canonicalized path on success. This guards every endpoint that
/// hands a client-supplied path to ffmpeg/ffprobe or serves it directly
/// (cast/play, video-info, dimensions, audio-gain, transcode, interpolated
/// stream, whisper) against reading or probing arbitrary files outside the
/// media library.
///
/// `canonicalize` requires the target to exist, so callers that already perform
/// an availability/existence check should run this *after* it, so an offline
/// drive still surfaces its own error rather than a generic "invalid path".
///
/// On Android, scoped/FUSE storage resolves the canonical path under a different
/// root than the app-visible watch-dir path (`/storage/…`), so a lexical
/// containment fallback is added there to avoid spurious "not within a watched
/// directory" rejections. Desktop behavior is unchanged (canonicalize only).
pub fn validate_path_in_watch_dir(
    state: &AppState,
    client_path: &str,
) -> Result<PathBuf, AppError> {
    let resolved = Path::new(client_path)
        .canonicalize()
        .map_err(|_| AppError::BadRequest("Invalid file path".into()))?;

    for library in state.library_manager().all_mounted() {
        let main_conn = library.main_pool.get()?;
        let mut stmt = main_conn
            .prepare("SELECT path FROM watch_directories")
            .map_err(|e| AppError::Internal(format!("Failed to query watch directories: {}", e)))?;
        let paths: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| AppError::Internal(format!("Failed to read watch directories: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        for wd_path in &paths {
            if let Ok(wd_resolved) = Path::new(wd_path).canonicalize() {
                if resolved.starts_with(&wd_resolved) {
                    return Ok(resolved);
                }
            }

            // Android scoped/FUSE storage: match the app-visible path lexically.
            #[cfg(target_os = "android")]
            {
                let client = Path::new(client_path);
                if normalize_lexical(client).starts_with(normalize_lexical(Path::new(wd_path))) {
                    return Ok(normalize_lexical(client));
                }
            }
        }
    }

    Err(AppError::Forbidden(
        "Path is not within a watched directory".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::library::LibraryContext;

    fn temp_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "localbooru-utils-{}-{}",
            name,
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn accepts_files_from_mounted_auxiliary_library() {
        let primary_dir = temp_test_dir("primary");
        let auxiliary_dir = temp_test_dir("auxiliary");
        let watch_dir = auxiliary_dir.join("media");
        let media_path = watch_dir.join("image.png");
        std::fs::create_dir_all(&primary_dir).unwrap();
        std::fs::create_dir_all(&watch_dir).unwrap();
        std::fs::write(&media_path, b"test").unwrap();

        let state = AppState::new(&primary_dir, 0).unwrap();
        let auxiliary = LibraryContext::create(&auxiliary_dir, "Auxiliary").unwrap();
        {
            let conn = auxiliary.main_pool.get().unwrap();
            conn.execute(
                "INSERT INTO watch_directories (path, name) VALUES (?1, 'Media')",
                rusqlite::params![watch_dir.to_string_lossy()],
            )
            .unwrap();
        }
        state.library_manager().mount(auxiliary);

        let resolved = validate_path_in_watch_dir(&state, media_path.to_str().unwrap()).unwrap();
        assert_eq!(resolved, media_path.canonicalize().unwrap());

        drop(state);
        let _ = std::fs::remove_dir_all(primary_dir);
        let _ = std::fs::remove_dir_all(auxiliary_dir);
    }
}

/// Detect the primary local (non-loopback) IPv4 address.
///
/// Connects a UDP socket to a public IP (no data sent) to determine which
/// local interface the OS would route through.
pub fn get_local_ip() -> Option<String> {
    use std::net::UdpSocket;
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}

/// Return the set of directory IDs visible to the given access tier and
/// family-mode lock state.
///
/// Returns `None` when no filtering is needed (localhost + family mode
/// unlocked), meaning ALL directories are visible. Otherwise returns
/// `Some(HashSet<i64>)` containing the visible directory IDs.
pub fn get_visible_directory_ids(
    main_conn: &Connection,
    tier: AccessTier,
    family_locked: bool,
) -> Result<Option<HashSet<i64>>, AppError> {
    // Localhost with family mode unlocked → no filtering needed
    if tier == AccessTier::Localhost && !family_locked {
        return Ok(None);
    }

    let mut conditions: Vec<&str> = Vec::new();

    // Family mode: only show family-safe directories
    if family_locked {
        conditions.push("family_safe = 1");
    }

    // Network visibility based on access tier
    match tier {
        AccessTier::Localhost => {
            // Localhost sees all (only family_safe filter applies if locked)
        }
        AccessTier::LocalNetwork => {
            conditions.push("lan_visible = 1");
        }
        AccessTier::Public => {
            conditions.push("public_access = 1");
        }
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    let sql = format!("SELECT id FROM watch_directories{}", where_clause);
    let mut stmt = main_conn
        .prepare(&sql)
        .map_err(|e| AppError::Internal(format!("Failed to query visible directories: {}", e)))?;

    let ids: HashSet<i64> = stmt
        .query_map([], |row| row.get(0))
        .map_err(|e| AppError::Internal(format!("Failed to read directory IDs: {}", e)))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Some(ids))
}
