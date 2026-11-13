//! Durable security-scoped bookmark lifecycle. The platform adapter owns active scopes.
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub trait Backend {
    type Scope;
    /// Resolve and start access. A stale bookmark must be replaced while access is active.
    fn activate(&mut self, bookmark: &[u8]) -> Result<(Self::Scope, bool), String>;
    fn refresh(&mut self, scope: &Self::Scope) -> Result<Vec<u8>, String>;
}

pub struct Grants<B: Backend> {
    directory: PathBuf,
    backend: B,
    active: HashMap<String, B::Scope>,
    failed: HashSet<String>,
}

impl<B: Backend> Grants<B> {
    pub fn new(directory: PathBuf, backend: B) -> Self {
        Self {
            directory,
            backend,
            active: HashMap::new(),
            failed: HashSet::new(),
        }
    }

    /// The caller must obtain this bookmark from the native picker, never a webview.
    pub fn persist_and_activate(&mut self, id: &str, bookmark: &[u8]) -> Result<(), String> {
        let path = self.path(id)?;
        let (scope, stale) = self.backend.activate(bookmark)?;
        let replacement = if stale {
            self.backend.refresh(&scope)?
        } else {
            bookmark.to_vec()
        };
        atomic_write(&path, &replacement).map_err(|e| e.to_string())?;
        self.active.insert(id.into(), scope);
        self.failed.remove(id);
        Ok(())
    }

    /// Restore every reference independently; failures leave the original bytes intact.
    pub fn restore(&mut self) -> io::Result<Vec<String>> {
        fs::create_dir_all(&self.directory)?;
        let mut failures = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("bookmark") {
                continue;
            }
            let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if self.path(id).is_err() {
                continue;
            }
            let result = (|| -> Result<(), String> {
                if !entry.file_type().map_err(|e| e.to_string())?.is_file()
                    || entry.metadata().map_err(|e| e.to_string())?.len() > 1024 * 1024
                {
                    return Err("Invalid folder bookmark".into());
                }
                let data = fs::read(&path).map_err(|e| e.to_string())?;
                let (scope, stale) = self.backend.activate(&data)?;
                if stale {
                    let replacement = self.backend.refresh(&scope)?;
                    atomic_write(&path, &replacement).map_err(|e| e.to_string())?;
                }
                self.active.insert(id.into(), scope);
                Ok(())
            })();
            if result.is_err() {
                self.failed.insert(id.into());
                failures
                    .push("Folder access could not be restored; select the folder again.".into());
            } else {
                self.failed.remove(id);
            }
        }
        Ok(failures)
    }

    pub fn failed_count(&self) -> usize {
        self.failed.len()
    }

    fn path(&self, id: &str) -> Result<PathBuf, String> {
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err("Invalid grant identifier".into());
        }
        Ok(self.directory.join(format!("{id}.bookmark")))
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if bytes.is_empty() || bytes.len() > 1024 * 1024 {
        return Err(io::Error::other("Invalid folder bookmark"));
    }
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("missing parent"))?;
    fs::create_dir_all(dir)?;
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    use std::io::Write;
    temp.write_all(bytes)?;
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|e| e.error)?;
    #[cfg(unix)]
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

pub fn required_for(os: &str, lite: bool) -> bool {
    lite && os == "macos"
}

#[cfg(all(target_os = "macos", feature = "store-lite"))]
#[path = "macos_grants_apple.rs"]
pub(crate) mod apple;

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake {
        denied: bool,
        stale: bool,
        refresh_fails: bool,
    }
    impl Backend for Fake {
        type Scope = ();
        fn activate(&mut self, _: &[u8]) -> Result<((), bool), String> {
            if self.denied {
                Err("denied".into())
            } else {
                Ok(((), self.stale))
            }
        }
        fn refresh(&mut self, _: &()) -> Result<Vec<u8>, String> {
            if self.refresh_fails {
                Err("refresh failed".into())
            } else {
                Ok(b"refreshed".to_vec())
            }
        }
    }
    fn fake(denied: bool, stale: bool, refresh_fails: bool) -> Fake {
        Fake {
            denied,
            stale,
            refresh_fails,
        }
    }
    #[test]
    fn restart_refresh_and_recovery_preserve_references() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("grants");
        Grants::new(dir.clone(), fake(false, false, false))
            .persist_and_activate("one", b"original")
            .unwrap();
        let mut denied = Grants::new(dir.clone(), fake(true, false, false));
        assert_eq!(denied.restore().unwrap().len(), 1);
        assert_eq!(fs::read(dir.join("one.bookmark")).unwrap(), b"original");
        let mut failed_refresh = Grants::new(dir.clone(), fake(false, true, true));
        assert_eq!(failed_refresh.restore().unwrap().len(), 1);
        assert_eq!(fs::read(dir.join("one.bookmark")).unwrap(), b"original");
        let mut recovered = Grants::new(dir.clone(), fake(false, true, false));
        assert!(recovered.restore().unwrap().is_empty());
        assert_eq!(fs::read(dir.join("one.bookmark")).unwrap(), b"refreshed");
        assert_eq!(recovered.active.len(), 1);
    }

    #[test]
    fn failed_replacement_preserves_existing_bookmark_and_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let mut grants = Grants::new(tmp.path().join("grants"), fake(false, false, false));
        grants.persist_and_activate("one", b"original").unwrap();
        assert!(grants.persist_and_activate("one", b"").is_err());
        assert_eq!(fs::read(grants.path("one").unwrap()).unwrap(), b"original");
        assert_eq!(grants.active.len(), 1);
        grants.persist_and_activate("one", b"replacement").unwrap();
        assert_eq!(grants.active.len(), 1);
    }

    #[test]
    fn reselection_recovers_the_same_reference_without_removing_other_failures() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("grants");
        let mut grants = Grants::new(dir.clone(), fake(false, false, false));
        for id in ["one", "two"] {
            grants.persist_and_activate(id, b"original").unwrap();
        }
        let mut restored = Grants::new(dir, fake(true, false, false));
        assert_eq!(restored.restore().unwrap().len(), 2);
        restored.backend = fake(false, false, false);
        restored
            .persist_and_activate("one", b"new permission")
            .unwrap();
        assert_eq!(restored.failed_count(), 1);
        assert!(restored.restore().unwrap().is_empty());
        assert_eq!(restored.failed_count(), 0);
    }

    #[test]
    fn full_and_other_platforms_do_not_require_grants() {
        assert!(required_for("macos", true));
        for (os, lite) in [("macos", false), ("linux", true), ("windows", true)] {
            assert!(!required_for(os, lite));
        }
    }

    #[test]
    fn invalid_identifiers_and_nonregular_bookmarks_are_not_read() {
        let tmp = tempfile::tempdir().unwrap();
        let mut grants = Grants::new(tmp.path().join("grants"), fake(false, false, false));
        assert!(grants.persist_and_activate("../escape", b"data").is_err());
        fs::create_dir_all(grants.directory.join("directory.bookmark")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            "/does-not-exist-synthetic",
            grants.directory.join("link.bookmark"),
        )
        .unwrap();
        assert!(!grants.restore().unwrap().is_empty());
        assert!(grants.active.is_empty());
    }

    #[test]
    fn replacement_failure_and_shutdown_release_each_started_scope() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct TrackedScope(Arc<AtomicUsize>);
        impl Drop for TrackedScope {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        struct Tracked(Arc<AtomicUsize>);
        impl Backend for Tracked {
            type Scope = TrackedScope;
            fn activate(&mut self, _: &[u8]) -> Result<(TrackedScope, bool), String> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok((TrackedScope(self.0.clone()), false))
            }
            fn refresh(&mut self, _: &TrackedScope) -> Result<Vec<u8>, String> {
                unreachable!()
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let mut grants = Grants::new(tmp.path().join("grants"), Tracked(count.clone()));
        grants.persist_and_activate("one", b"first").unwrap();
        grants.persist_and_activate("one", b"second").unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(grants.persist_and_activate("one", b"").is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
        drop(grants);
        assert_eq!(count.load(Ordering::SeqCst), 0);
        let obstruction = tmp.path().join("not-a-directory");
        fs::write(&obstruction, b"fixture").unwrap();
        let mut denied = Grants::new(obstruction, Tracked(count.clone()));
        assert!(denied.persist_and_activate("one", b"bookmark").is_err());
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }
}
