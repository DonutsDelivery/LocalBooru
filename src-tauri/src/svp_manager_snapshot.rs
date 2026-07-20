use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ManagerGraphSnapshot {
    pub kind: &'static str,
    pub revision: u64,
    pub snapshot_path: String,
    pub snapshot_sha256: String,
}

#[derive(Clone)]
pub struct ManagerGraphSnapshotStore {
    inner: Arc<SnapshotStoreInner>,
}

struct SnapshotStoreInner {
    root: PathBuf,
    state: Mutex<SnapshotState>,
}

#[derive(Default)]
struct SnapshotState {
    revision: u64,
    current: Option<ManagerGraphSnapshot>,
}

impl ManagerGraphSnapshotStore {
    pub fn new(root: PathBuf) -> Self {
        Self {
            inner: Arc::new(SnapshotStoreInner {
                root,
                state: Mutex::new(SnapshotState::default()),
            }),
        }
    }

    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    pub fn current(&self) -> Option<ManagerGraphSnapshot> {
        self.inner.state.lock().ok()?.current.clone()
    }

    pub fn prepare_file(&self, source: &Path) -> io::Result<(ManagerGraphSnapshot, bool)> {
        let bytes = fs::read(source)?;
        self.prepare_bytes(&bytes)
    }

    fn prepare_bytes(&self, bytes: &[u8]) -> io::Result<(ManagerGraphSnapshot, bool)> {
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| io::Error::other("snapshot state poisoned"))?;
        if let Some(current) = state.current.as_ref() {
            if current.snapshot_sha256 == sha256 {
                return Ok((current.clone(), false));
            }
        }

        prepare_private_directory(&self.inner.root)?;
        state.revision = state
            .revision
            .checked_add(1)
            .ok_or_else(|| io::Error::other("snapshot revision exhausted"))?;
        let filename = format!("graph-{}-{}.vpy", state.revision, sha256);
        let path = self.inner.root.join(filename);
        write_private_atomic(&path, bytes)?;
        let snapshot = ManagerGraphSnapshot {
            kind: "manager_snapshot",
            revision: state.revision,
            snapshot_path: path.to_string_lossy().into_owned(),
            snapshot_sha256: sha256,
        };
        Ok((snapshot, true))
    }

    pub fn commit(&self, snapshot: ManagerGraphSnapshot) -> io::Result<()> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| io::Error::other("snapshot state poisoned"))?;
        if snapshot.revision != state.revision {
            return Err(io::Error::other("snapshot revision was not prepared"));
        }
        state.current = Some(snapshot);
        Ok(())
    }

    #[cfg(test)]
    fn publish_bytes(&self, bytes: &[u8]) -> io::Result<(ManagerGraphSnapshot, bool)> {
        let (snapshot, changed) = self.prepare_bytes(bytes)?;
        if changed {
            self.commit(snapshot.clone())?;
        }
        Ok((snapshot, changed))
    }

    pub fn clear_current(&self) -> bool {
        let Ok(mut state) = self.inner.state.lock() else {
            return false;
        };
        state.current.take().is_some()
    }
}

fn prepare_private_directory(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "snapshot root is not a private directory",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => fs::create_dir(path)?,
        Err(error) => return Err(error),
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let metadata = fs::metadata(path)?;
        if metadata.uid() != unsafe { libc::geteuid() } {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "snapshot root is owned by another user",
            ));
        }
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn write_private_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid snapshot path"))?;
    let temporary = path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()));
    fs::write(&temporary, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store(name: &str) -> ManagerGraphSnapshotStore {
        ManagerGraphSnapshotStore::new(std::env::temp_dir().join(format!(
            "localbooru-svp-snapshot-{name}-{}",
            uuid::Uuid::new_v4()
        )))
    }

    // AC: @svp-manager-transitions ac-idempotent-changes
    #[test]
    fn identical_bytes_reuse_the_published_revision() {
        let store = test_store("idempotent");
        let (first, first_changed) = store.publish_bytes(b"video_in.set_output()\n").unwrap();
        let (second, second_changed) = store.publish_bytes(b"video_in.set_output()\n").unwrap();

        assert!(first_changed);
        assert!(!second_changed);
        assert_eq!(first, second);
        assert_eq!(
            first.snapshot_sha256,
            "b5ef1a284ac0281f2272ef56592d03aa5eb4aee87004a6d48c7aac66501c7128"
        );
        let _ = fs::remove_dir_all(store.root());
    }

    // AC: @svp-manager-transitions ac-changed-graph
    #[test]
    fn changed_bytes_publish_one_new_immutable_snapshot() {
        let store = test_store("changed");
        let (first, _) = store.publish_bytes(b"first").unwrap();
        let (second, changed) = store.publish_bytes(b"second").unwrap();

        assert!(changed);
        assert_eq!(second.revision, first.revision + 1);
        assert_ne!(second.snapshot_path, first.snapshot_path);
        assert_eq!(fs::read(&first.snapshot_path).unwrap(), b"first");
        assert_eq!(fs::read(&second.snapshot_path).unwrap(), b"second");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&second.snapshot_path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(store.root()).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        let _ = fs::remove_dir_all(store.root());
    }

    #[test]
    fn prepared_snapshot_is_not_visible_until_committed() {
        let store = test_store("transaction");
        let (snapshot, changed) = store.prepare_bytes(b"prepared").unwrap();

        assert!(changed);
        assert!(store.current().is_none());
        store.commit(snapshot.clone()).unwrap();
        assert_eq!(store.current(), Some(snapshot));
        let _ = fs::remove_dir_all(store.root());
    }

    // AC: @svp-manager-transitions ac-no-active-graph
    #[test]
    fn clearing_current_keeps_published_snapshots_available_to_pinned_sessions() {
        let store = test_store("clear");
        let (snapshot, _) = store.publish_bytes(b"pinned").unwrap();

        assert!(store.clear_current());
        assert!(store.current().is_none());
        assert_eq!(fs::read(&snapshot.snapshot_path).unwrap(), b"pinned");
        assert!(!store.clear_current());
        let _ = fs::remove_dir_all(store.root());
    }
}
