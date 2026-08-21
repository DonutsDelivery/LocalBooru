use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::routes::settings::get_config_section;

pub const PREVIEW_SNAPSHOT_PREFIX: &str = "localbooru-preview-source-";
const OPERATION_STALE_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Resolve and prepare the private scratch directory used for large preview snapshots.
///
/// Resolution order:
/// 1. `LOCALBOORU_SCRATCH_DIR`
/// 2. `storage.scratch_dir` in settings.json
/// 3. `<app data>/tmp`
pub fn prepare_preview_scratch_dir(data_dir: &Path) -> io::Result<PathBuf> {
    let configured = std::env::var_os("LOCALBOORU_SCRATCH_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            get_config_section(data_dir, "storage")
                .get("scratch_dir")
                .and_then(|value| value.as_str())
                .filter(|value| !value.trim().is_empty())
                .map(PathBuf::from)
        });
    let root = match configured {
        Some(path) if path.is_absolute() => path,
        Some(path) => data_dir.join(path),
        None => data_dir.join("tmp"),
    };

    fs::create_dir_all(&root)?;
    let metadata = fs::symlink_metadata(&root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "preview scratch root is not a real directory: {}",
                root.display()
            ),
        ));
    }
    set_private_directory_permissions(&root)?;

    // No preview generation is active while AppState is being constructed, so every
    // matching snapshot left by an earlier process is stale.
    cleanup_preview_snapshots(&root, None)?;
    Ok(root)
}

/// Remove snapshots old enough to predate a normal operation. Active snapshots from
/// concurrent preview requests are left alone.
pub fn cleanup_stale_preview_snapshots(root: &Path) -> io::Result<usize> {
    cleanup_preview_snapshots(root, Some(OPERATION_STALE_AGE))
}

/// Remove snapshots left in the process temp directory by releases that wrote
/// previews there. Called only by the real desktop startup, never by AppState
/// unit-test construction.
pub fn cleanup_legacy_process_temp_snapshots() -> io::Result<usize> {
    cleanup_preview_snapshots(&std::env::temp_dir(), None)
}

fn cleanup_preview_snapshots(root: &Path, minimum_age: Option<Duration>) -> io::Result<usize> {
    let mut removed = 0;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_file()
            || file_type.is_symlink()
            || !is_preview_snapshot_name(&entry.file_name())
        {
            continue;
        }
        if let Some(age) = minimum_age {
            let modified = entry
                .metadata()?
                .modified()
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if SystemTime::now()
                .duration_since(modified)
                .unwrap_or_default()
                < age
            {
                continue;
            }
        }
        match fs::remove_file(entry.path()) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(removed)
}

fn is_preview_snapshot_name(name: &OsStr) -> bool {
    name.to_str()
        .is_some_and(|name| name.starts_with(PREVIEW_SNAPSHOT_PREFIX))
}

/// Copy a candidate into the configured storage-backed scratch root and keep the
/// snapshot alive only for the duration of `operation`. NamedTempFile removes it
/// on every normal return and unwinding error path.
pub fn with_preview_source_snapshot<T>(
    source: &Path,
    scratch_root: &Path,
    extension: &str,
    operation: impl FnOnce(&Path) -> T,
) -> io::Result<T> {
    let _ = cleanup_stale_preview_snapshots(scratch_root);
    let suffix = format!(".{}", extension.trim_start_matches('.'));
    let mut snapshot = tempfile::Builder::new()
        .prefix(PREVIEW_SNAPSHOT_PREFIX)
        .suffix(&suffix)
        .tempfile_in(scratch_root)?;
    let mut input = fs::File::open(source)?;
    io::copy(&mut input, snapshot.as_file_mut())?;
    Ok(operation(snapshot.path()))
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_files(root: &Path) -> Vec<PathBuf> {
        fs::read_dir(root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| is_preview_snapshot_name(name))
            })
            .collect()
    }

    #[test]
    fn configured_scratch_root_avoids_process_temp_directory() {
        let fixture = tempfile::tempdir().unwrap();
        let data_dir = fixture.path().join("data");
        let configured = fixture.path().join("storage-backed-scratch");
        fs::create_dir_all(&data_dir).unwrap();
        fs::write(
            data_dir.join("settings.json"),
            serde_json::to_vec(&serde_json::json!({
                "storage": { "scratch_dir": configured }
            }))
            .unwrap(),
        )
        .unwrap();

        let resolved = prepare_preview_scratch_dir(&data_dir).unwrap();

        assert_eq!(resolved, fixture.path().join("storage-backed-scratch"));
        assert_ne!(resolved, std::env::temp_dir());
    }

    #[test]
    fn snapshot_is_removed_after_success() {
        let fixture = tempfile::tempdir().unwrap();
        let scratch = fixture.path().join("scratch");
        fs::create_dir_all(&scratch).unwrap();
        let source = fixture.path().join("source.mp4");
        fs::write(&source, b"video fixture").unwrap();

        let copied = with_preview_source_snapshot(&source, &scratch, "mp4", |snapshot| {
            assert_eq!(snapshot.parent(), Some(scratch.as_path()));
            assert_eq!(fs::read(snapshot).unwrap(), b"video fixture");
            true
        })
        .unwrap();

        assert!(copied);
        assert!(snapshot_files(&scratch).is_empty());
    }

    #[test]
    fn snapshot_is_removed_after_operation_error() {
        let fixture = tempfile::tempdir().unwrap();
        let scratch = fixture.path().join("scratch");
        fs::create_dir_all(&scratch).unwrap();
        let source = fixture.path().join("source.mp4");
        fs::write(&source, b"video fixture").unwrap();

        let result = with_preview_source_snapshot(&source, &scratch, "mp4", |_snapshot| {
            Err::<(), _>(io::Error::other("preview failed"))
        })
        .unwrap();

        assert!(result.is_err());
        assert!(snapshot_files(&scratch).is_empty());
    }

    #[test]
    fn startup_cleanup_removes_only_owned_snapshot_names() {
        let fixture = tempfile::tempdir().unwrap();
        let data_dir = fixture.path().join("data");
        let scratch = data_dir.join("tmp");
        fs::create_dir_all(&scratch).unwrap();
        let stale = scratch.join(format!("{PREVIEW_SNAPSHOT_PREFIX}stale.mp4"));
        let unrelated = scratch.join("keep-me.mp4");
        fs::write(&stale, b"stale").unwrap();
        fs::write(&unrelated, b"keep").unwrap();

        assert_eq!(prepare_preview_scratch_dir(&data_dir).unwrap(), scratch);
        assert!(!stale.exists());
        assert!(unrelated.exists());
    }
}
