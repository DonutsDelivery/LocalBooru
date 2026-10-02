//! Private registration of the WebProcess hosting the selected DMC video.
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::Mutex,
};

pub(crate) const HOST_PREFIX: &str = "localbooru-svp-host-";

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VideoResolutionBounds {
    pub max_width: u32,
    pub max_height: u32,
}

pub(crate) fn valid_host_id(id: &str) -> bool {
    id.strip_prefix(HOST_PREFIX).is_some_and(|token| {
        !token.is_empty()
            && token.len() <= 80
            && token
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    })
}

pub(crate) fn manager_socket_path(pid: u32) -> PathBuf {
    PathBuf::from(format!("/tmp/mpvSockets/{pid}"))
}

pub(crate) struct VideoHostLease {
    root: PathBuf,
    state: Mutex<LeaseState>,
}

#[derive(Default)]
struct LeaseState {
    epoch: u64,
    revision: u64,
    current: Option<String>,
    resize_revision: u64,
    resize_current: Option<String>,
    resize_bounds: Option<VideoResolutionBounds>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VideoHostTicket {
    pub(crate) host_id: String,
    pub(crate) epoch: u64,
    pub(crate) revision: u64,
    pub(crate) pid: u32,
}

impl VideoHostLease {
    pub(crate) fn new(root: PathBuf) -> Self {
        Self {
            root,
            state: Mutex::new(LeaseState::default()),
        }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn prepare(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let metadata = fs::symlink_metadata(&self.root)?;
        if !metadata.is_dir()
            || metadata.file_type().is_symlink()
            || metadata.uid() != unsafe { libc::geteuid() }
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "unowned video-host directory",
            ));
        }
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
        let _ = fs::remove_file(self.root.join("active"));
        let _ = fs::remove_file(self.root.join("resize-active"));
        Ok(())
    }

    pub(crate) fn acquire_epoch(&self) -> io::Result<u64> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("video host state poisoned"))?;
        state.epoch = state
            .epoch
            .checked_add(1)
            .ok_or_else(|| io::Error::other("video host epoch exhausted"))?;
        state.revision = 0;
        state.current = None;
        state.resize_revision = 0;
        state.resize_bounds = None;
        if let Some(id) = state.resize_current.take() {
            let _ = fs::remove_file(self.root.join(format!("{id}.resize")));
            let _ = fs::remove_file(self.root.join(format!("{id}.geometry")));
        }
        let _ = fs::remove_file(self.root.join("resize-active"));
        let _ = fs::remove_file(self.root.join("active"));
        Ok(state.epoch)
    }

    /// Prepare before attaching the source. Raw resize never advertises an MPV
    /// endpoint, and stale Manager cleanup cannot remove the selected resolution.
    pub(crate) fn configure_resolution(
        &self,
        id: &str,
        epoch: u64,
        revision: u64,
        bounds: Option<VideoResolutionBounds>,
    ) -> io::Result<bool> {
        if !valid_host_id(id)
            || bounds.is_some_and(|bounds| {
                !(2..=16384).contains(&bounds.max_width)
                    || !(2..=16384).contains(&bounds.max_height)
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid video resolution contract",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("video host state poisoned"))?;
        if epoch == 0
            || epoch != state.epoch
            || revision < state.resize_revision
            || (revision == state.resize_revision && state.resize_current.as_deref() != Some(id))
        {
            return Ok(false);
        }
        let requested_bounds = bounds;
        let bounds = bounds.unwrap_or(VideoResolutionBounds {
            max_width: 0,
            max_height: 0,
        });
        let target = self.root.join(format!("{id}.resize"));
        let temporary = self.root.join("resize.tmp");
        fs::write(
            &temporary,
            format!(
                "{epoch}\n{revision}\n{}\n{}\n",
                bounds.max_width, bounds.max_height
            ),
        )?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        fs::rename(&temporary, target)?;
        fs::write(&temporary, format!("{id}\n{epoch}\n{revision}\n"))?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        fs::rename(&temporary, self.root.join("resize-active"))?;
        if let Some(previous) = state
            .resize_current
            .replace(id.to_owned())
            .filter(|previous| previous != id)
        {
            let _ = fs::remove_file(self.root.join(format!("{previous}.resize")));
            let _ = fs::remove_file(self.root.join(format!("{previous}.geometry")));
        }
        state.resize_revision = revision;
        state.resize_bounds = requested_bounds;
        Ok(true)
    }

    pub(crate) fn playback_geometry(
        &self,
        id: Option<&str>,
        epoch: Option<u64>,
        resize_revision: Option<u64>,
    ) -> io::Result<Option<(u32, u32)>> {
        let state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("video host state poisoned"))?;
        if state.resize_current.is_none() {
            return Ok(None);
        }
        if state.resize_current.as_deref() != id
            || epoch != Some(state.epoch)
            || resize_revision != Some(state.resize_revision)
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "Native video geometry is not ready for this resolution owner",
            ));
        }
        if state.resize_bounds.is_none() {
            return Ok(None);
        }
        drop(state);
        self.input_geometry(id, epoch).map(Some).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                "Native video geometry is not ready; matching scaler runtime is required",
            )
        })
    }

    pub(crate) fn verify_resolution(&self, id: &str, epoch: u64, revision: u64) -> io::Result<()> {
        let state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("video host state poisoned"))?;
        if state.resize_current.as_deref() != Some(id)
            || state.epoch != epoch
            || state.resize_revision != revision
        {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "Native video geometry is not ready for this resolution owner",
            ));
        }
        drop(state);
        self.playback_geometry(Some(id), Some(epoch), Some(revision))
            .map(|_| ())
    }

    /// The scaler reports its actual negotiated input to the Manager graph;
    /// never substitute that graph's possibly different output dimensions.
    pub(crate) fn input_geometry(
        &self,
        id: Option<&str>,
        epoch: Option<u64>,
    ) -> Option<(u32, u32)> {
        let state = self.state.lock().ok()?;
        let id = id.filter(|id| valid_host_id(id))?;
        if epoch != Some(state.epoch) || state.resize_current.as_deref() != Some(id) {
            return None;
        }
        let path = self.root.join(format!("{id}.geometry"));
        let metadata = fs::symlink_metadata(&path).ok()?;
        if !metadata.is_file()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.len() > 128
        {
            return None;
        }
        let raw = fs::read_to_string(path).ok()?;
        let fields: Vec<u64> = raw.lines().map(str::parse).collect::<Result<_, _>>().ok()?;
        if fields.len() != 6
            || fields[0] != state.epoch
            || fields[1] != state.resize_revision
            || fields[2..]
                .iter()
                .any(|value| *value == 0 || *value > 16384)
        {
            return None;
        }
        if let Some(bounds) = state.resize_bounds {
            if fields[4] > u64::from(bounds.max_width)
                || fields[5] > u64::from(bounds.max_height)
                || fields[4] > fields[2]
                || fields[5] > fields[3]
            {
                return None;
            }
        }
        Some((fields[4] as u32, fields[5] as u32))
    }

    /// Both late enables and stale cleanup are rejected, including old documents.
    pub(crate) fn update(
        &self,
        enabled: bool,
        id: Option<&str>,
        epoch: Option<u64>,
        revision: Option<u64>,
    ) -> io::Result<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("video host state poisoned"))?;
        let Some((epoch, revision)) = epoch.zip(revision) else {
            return Ok(false);
        };
        if epoch != state.epoch
            || epoch == 0
            || revision < state.revision
            || (enabled && revision == state.revision && state.current.is_none())
        {
            return Ok(false);
        }
        if !enabled {
            let _ = fs::remove_file(self.root.join("active"));
            state.revision = revision;
            state.current = None;
            return Ok(true);
        }
        if revision == state.revision && state.current.as_deref() != id {
            return Ok(false);
        }
        let id = id
            .filter(|id| valid_host_id(id))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid video host ID"))?;
        let pid = self.registered_pid(id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "Native SVP video host is not registered; matching WebKit runtime is required",
            )
        })?;
        let temporary = self.root.join(format!("active.tmp-{}", std::process::id()));
        fs::write(&temporary, format!("{id}\n{pid}\n{epoch}\n{revision}\n"))?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        fs::rename(temporary, self.root.join("active"))?;
        state.revision = revision;
        state.current = Some(id.to_owned());
        Ok(true)
    }

    fn registered_pid(&self, id: &str) -> Option<u32> {
        let path = self.root.join(id);
        let metadata = fs::symlink_metadata(&path).ok()?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.len() > 16
        {
            return None;
        }
        fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|pid| *pid > 0)
    }

    pub(crate) fn ticket(&self) -> Option<VideoHostTicket> {
        let state = self.state.lock().ok()?;
        let id = state.current.as_deref()?;
        Some(VideoHostTicket {
            host_id: id.to_owned(),
            epoch: state.epoch,
            revision: state.revision,
            pid: self.registered_pid(id)?,
        })
    }

    pub(crate) fn active_pid(&self) -> Option<u32> {
        self.ticket().map(|ticket| ticket.pid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> VideoHostLease {
        let lease = VideoHostLease::new(
            std::env::temp_dir().join(format!("dmc-synthetic-video-host-{}", uuid::Uuid::new_v4())),
        );
        lease.prepare().unwrap();
        lease.acquire_epoch().unwrap();
        lease
    }

    // AC: @local-decoded-video-resolution ac-local-raw
    // AC: @local-decoded-video-resolution ac-ownership
    #[test]
    fn raw_resolution_prepares_before_registration_without_advertising_manager() {
        let lease = fixture();
        let first = "localbooru-svp-host-first";
        let second = "localbooru-svp-host-second";
        let bounds = Some(VideoResolutionBounds {
            max_width: 1280,
            max_height: 720,
        });
        assert!(lease.verify_resolution(first, 1, 2).is_err());
        assert!(lease.configure_resolution(first, 1, 2, bounds).unwrap());
        assert_eq!(
            fs::read_to_string(lease.root.join(format!("{first}.resize"))).unwrap(),
            "1\n2\n1280\n720\n"
        );
        assert!(!lease.root.join("active").exists());
        assert!(lease.update(false, Some(first), Some(1), Some(2)).unwrap());
        assert!(lease.root.join(format!("{first}.resize")).exists());
        assert!(lease.configure_resolution(second, 1, 3, None).unwrap());
        assert!(lease.verify_resolution(second, 1, 3).is_ok());
        assert!(lease.verify_resolution(first, 1, 2).is_err());
        assert!(!lease.configure_resolution(first, 1, 2, bounds).unwrap());
        assert!(!lease.configure_resolution(first, 1, 3, bounds).unwrap());
        assert!(!lease.root.join(format!("{first}.resize")).exists());
        assert_eq!(
            fs::read_to_string(lease.root.join(format!("{second}.resize"))).unwrap(),
            "1\n3\n0\n0\n"
        );
        assert_eq!(lease.acquire_epoch().unwrap(), 2);
        assert!(!lease.configure_resolution(first, 1, 100, bounds).unwrap());
        assert!(!lease.root.join("resize-active").exists());
        assert!(lease.configure_resolution(first, 2, 1, bounds).unwrap());
        assert!(lease
            .playback_geometry(Some(first), Some(2), Some(1))
            .is_err());
        fs::write(
            lease.root.join(format!("{first}.geometry")),
            "2\n1\n2160\n3840\n404\n720\n",
        )
        .unwrap();
        assert_eq!(lease.input_geometry(Some(first), Some(2)), Some((404, 720)));
        assert!(lease.verify_resolution(first, 2, 1).is_ok());
        assert_eq!(
            lease
                .playback_geometry(Some(first), Some(2), Some(1))
                .unwrap(),
            Some((404, 720))
        );
        assert!(lease
            .playback_geometry(Some(first), Some(2), Some(2))
            .is_err());
        fs::write(
            lease.root.join(format!("{first}.geometry")),
            "2\n1\n3840\n2160\n3840\n2160\n",
        )
        .unwrap();
        assert!(lease
            .playback_geometry(Some(first), Some(2), Some(1))
            .is_err());
        assert_eq!(lease.input_geometry(Some(second), Some(2)), None);
        assert_eq!(lease.input_geometry(Some(first), Some(1)), None);
        fs::write(
            lease.root.join(format!("{first}.geometry")),
            "2\n0\n2160\n3840\n404\n720\n",
        )
        .unwrap();
        assert_eq!(lease.input_geometry(Some(first), Some(2)), None);
        assert!(lease.configure_resolution("../bad", 2, 2, bounds).is_err());
        assert!(lease
            .configure_resolution(
                first,
                2,
                2,
                Some(VideoResolutionBounds {
                    max_width: 0,
                    max_height: 720
                })
            )
            .is_err());
        fs::remove_dir_all(lease.root()).unwrap();
    }

    // AC: @svp-platform-routing ac-linux-route
    #[test]
    fn active_id_selects_only_its_registered_graph_host() {
        let lease = fixture();
        fs::write(lease.root.join("localbooru-svp-host-main"), "101").unwrap();
        fs::write(lease.root.join("localbooru-svp-host-iframe"), "202").unwrap();
        assert_eq!(lease.active_pid(), None);
        lease
            .update(true, Some("localbooru-svp-host-main"), Some(1), Some(1))
            .unwrap();
        assert_eq!(lease.active_pid(), Some(101));
        lease
            .update(true, Some("localbooru-svp-host-iframe"), Some(1), Some(2))
            .unwrap();
        assert_eq!(lease.active_pid(), Some(202));
        assert!(!lease
            .update(false, Some("localbooru-svp-host-main"), Some(1), Some(1))
            .unwrap());
        assert_eq!(lease.active_pid(), Some(202));
        assert!(lease
            .update(false, Some("localbooru-svp-host-iframe"), Some(1), Some(2))
            .unwrap());
        assert_eq!(lease.active_pid(), None);
        assert!(!lease.root.join("active").exists());
        fs::remove_dir_all(lease.root()).unwrap();
    }

    #[test]
    fn invalid_ids_pid_markers_and_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;
        let lease = fixture();
        assert!(lease
            .update(true, Some("../outside"), Some(1), Some(1))
            .is_err());
        let path = lease.root.join("localbooru-svp-host-main");
        fs::write(&path, "101").unwrap();
        lease
            .update(true, Some("localbooru-svp-host-main"), Some(1), Some(1))
            .unwrap();
        for data in ["0", "invalid", "4294967296", "12345678901234567"] {
            fs::write(&path, data).unwrap();
            assert_eq!(lease.active_pid(), None);
        }
        fs::remove_file(&path).unwrap();
        let target = lease.root.join("synthetic-marker");
        fs::write(&target, "303").unwrap();
        symlink(target, path).unwrap();
        assert_eq!(lease.active_pid(), None);
        fs::remove_dir_all(lease.root()).unwrap();
    }

    #[test]
    fn late_enable_tombstone_and_document_reload_cannot_take_the_lease() {
        let lease = fixture();
        let first = "localbooru-svp-host-first";
        let next = "localbooru-svp-host-next";
        fs::write(lease.root.join(first), "101").unwrap();
        fs::write(lease.root.join(next), "101").unwrap();
        assert!(lease.update(true, Some(first), Some(1), Some(5)).unwrap());
        let old_ticket = lease.ticket().unwrap();
        assert!(lease.update(true, Some(next), Some(1), Some(6)).unwrap());
        assert_ne!(lease.ticket().unwrap(), old_ticket); // Same PID, different video.
        assert!(!lease.update(true, Some(first), Some(1), Some(5)).unwrap());
        assert!(!lease.update(true, Some(first), Some(1), Some(6)).unwrap());
        assert!(lease.update(false, Some(next), Some(1), Some(6)).unwrap());
        assert!(!lease.update(true, Some(next), Some(1), Some(6)).unwrap());
        assert!(lease.update(true, Some(next), Some(1), Some(7)).unwrap());
        assert_eq!(lease.acquire_epoch().unwrap(), 2);
        assert_eq!(lease.ticket(), None);
        assert!(lease.update(true, Some(first), Some(2), Some(1)).unwrap());
        assert!(!lease.update(true, Some(next), Some(1), Some(100)).unwrap());
        assert!(!lease.update(false, Some(next), Some(1), Some(100)).unwrap());
        assert_eq!(lease.ticket().unwrap().host_id, first);
        fs::remove_dir_all(lease.root()).unwrap();
    }

    #[test]
    fn missing_registration_and_missing_epoch_never_publish() {
        let lease = fixture();
        assert_eq!(
            lease
                .update(true, Some("localbooru-svp-host-missing"), Some(1), Some(1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        assert!(!lease.update(false, None, None, Some(1)).unwrap());
        assert!(!lease.root.join("active").exists());
        assert_eq!(
            fs::metadata(lease.root()).unwrap().permissions().mode() & 0o777,
            0o700
        );
        fs::remove_dir_all(lease.root()).unwrap();
    }

    #[test]
    fn manager_endpoint_is_a_pid_socket_without_default_alias() {
        assert_eq!(
            manager_socket_path(101),
            PathBuf::from("/tmp/mpvSockets/101")
        );
        assert!(valid_host_id("localbooru-svp-host-abc-123"));
        assert!(!valid_host_id("localbooru-svp-host-../mpvsocket"));
    }
}
