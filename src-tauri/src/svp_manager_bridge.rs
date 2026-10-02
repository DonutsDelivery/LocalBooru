use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::path::PathBuf;
use std::{
    fs, io,
    path::Path,
    sync::{Arc, Mutex},
};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
#[cfg(target_os = "windows")]
use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};

use crate::svp_manager_snapshot::ManagerGraphSnapshotStore;

#[cfg(target_os = "macos")]
const MPV_SOCKET_PATH: &str = "/tmp/mpvsocket";
#[cfg(target_os = "windows")]
const MPV_SOCKET_PATH: &str = r"\\.\pipe\mpvpipe";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SvpPlaybackUpdate {
    pub enabled: bool,
    pub path: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f64>,
    pub duration: Option<f64>,
    pub paused: Option<bool>,
    pub media_key: Option<String>,
    pub host_id: Option<String>,
    pub host_epoch: Option<u64>,
    pub host_revision: Option<u64>,
    pub resize_revision: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalVideoResolutionUpdate {
    pub host_id: String,
    pub host_epoch: u64,
    pub host_revision: u64,
    #[cfg(target_os = "linux")]
    pub bounds: Option<crate::svp_video_host::VideoResolutionBounds>,
    #[cfg(not(target_os = "linux"))]
    pub bounds: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FilterChanged {
    #[serde(flatten)]
    owner: PlaybackOwner,
    enabled: bool,
    script_path: Option<String>,
    graph_revision: Option<u64>,
    media_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PlaybackPaused {
    #[serde(flatten)]
    owner: PlaybackOwner,
    paused: bool,
    media_key: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct PlaybackOwner {
    host_id: Option<String>,
    host_epoch: Option<u64>,
    host_revision: Option<u64>,
}

#[derive(Debug)]
struct PlaybackState {
    enabled: bool,
    path: Option<String>,
    media_key: Option<String>,
    width: u32,
    height: u32,
    fps: f64,
    duration: f64,
    paused: bool,
    filter_active: bool,
    script_path: Option<String>,
    output_fps: f64,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            enabled: false,
            path: None,
            media_key: None,
            width: 0,
            height: 0,
            fps: 0.0,
            duration: 0.0,
            paused: true,
            filter_active: false,
            script_path: None,
            output_fps: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct SvpManagerBridge {
    playback: Arc<Mutex<PlaybackState>>,
    transition: Arc<Mutex<()>>,
    controller: Arc<Mutex<Option<(u32, usize)>>>,
    snapshots: ManagerGraphSnapshotStore,
    #[cfg(target_os = "linux")]
    filter_file: PathBuf,
    #[cfg(target_os = "linux")]
    script_file: PathBuf,
    #[cfg(target_os = "linux")]
    video_hosts: Arc<crate::svp_video_host::VideoHostLease>,
    #[cfg(unix)]
    control_socket: PathBuf,
}

impl SvpManagerBridge {
    pub fn new(snapshots: ManagerGraphSnapshotStore) -> Self {
        #[cfg(target_os = "linux")]
        let uid = unsafe { libc::geteuid() };
        #[cfg(target_os = "linux")]
        let pid = std::process::id();
        Self {
            playback: Arc::new(Mutex::new(PlaybackState::default())),
            transition: Arc::new(Mutex::new(())),
            controller: Arc::new(Mutex::new(None)),
            #[cfg(target_os = "linux")]
            video_hosts: Arc::new(crate::svp_video_host::VideoHostLease::new(
                snapshots.root().join("video-hosts"),
            )),
            snapshots,
            #[cfg(target_os = "linux")]
            filter_file: PathBuf::from(format!("/tmp/localbooru-svp-filter-{uid}-{pid}")),
            #[cfg(target_os = "linux")]
            script_file: PathBuf::from(format!("/tmp/localbooru-svp-script-{uid}-{pid}")),
            #[cfg(target_os = "linux")]
            control_socket: PathBuf::from(format!("/tmp/localbooru-mpv-control-{uid}-{pid}")),
            #[cfg(target_os = "macos")]
            control_socket: PathBuf::from(MPV_SOCKET_PATH),
        }
    }

    pub fn configure_environment(&self) {
        std::env::set_var("LOCALBOORU_SVP_SNAPSHOT_ROOT", self.snapshots.root());
        #[cfg(target_os = "linux")]
        {
            std::env::remove_var("LOCALBOORU_MPV_CONTROL_UPSTREAM");
            std::env::set_var("WEBKIT_GST_VIDEO_FILTER_FILE", &self.filter_file);
            std::env::set_var("LOCALBOORU_VS_SCRIPT_FILE", &self.script_file);
            let native_svp_enabled =
                std::env::var("LOCALBOORU_ENABLE_NATIVE_SVP").as_deref() == Ok("1");
            if self.video_hosts.prepare().is_ok() {
                std::env::set_var("LOCALBOORU_MPV_CONTROL_HOST_ROOT", self.video_hosts.root());
                if native_svp_enabled {
                    std::env::set_var("LOCALBOORU_MPV_CONTROL_UPSTREAM_V2", &self.control_socket);
                } else {
                    std::env::remove_var("LOCALBOORU_MPV_CONTROL_UPSTREAM_V2");
                }
            } else {
                std::env::remove_var("LOCALBOORU_MPV_CONTROL_UPSTREAM_V2");
                std::env::remove_var("LOCALBOORU_MPV_CONTROL_HOST_ROOT");
            }

            if let Some(home) = dirs::home_dir() {
                let plugin_dir = home.join(".local/lib/localbooru");
                let mut paths = vec![plugin_dir];
                if let Some(existing) = std::env::var_os("GST_PLUGIN_PATH") {
                    paths.extend(std::env::split_paths(&existing));
                }
                if let Ok(joined) = std::env::join_paths(paths) {
                    std::env::set_var("GST_PLUGIN_PATH", joined);
                }
            }
            let _ = fs::remove_file(&self.filter_file);
            let _ = fs::remove_file(&self.script_file);
        }
    }

    pub fn start(&self, app: AppHandle) {
        let bridge = self.clone();
        tauri::async_runtime::spawn(async move {
            if let Err(error) = bridge.run_server(app).await {
                log::warn!("[SVPManager] bridge unavailable: {error}");
            }
        });
    }

    #[cfg(unix)]
    async fn run_server(self, app: AppHandle) -> io::Result<()> {
        if UnixStream::connect(&self.control_socket).await.is_ok() {
            log::warn!(
                "[SVPManager] {} is already owned by another process",
                self.control_socket.display()
            );
            return Ok(());
        }
        remove_stale_control_socket(&self.control_socket)?;

        let listener = UnixListener::bind(&self.control_socket)?;
        fs::set_permissions(&self.control_socket, fs::Permissions::from_mode(0o600))?;
        log::info!(
            "[SVPManager] MPV-compatible control backend listening on {}",
            self.control_socket.display()
        );

        loop {
            let (stream, _) = listener.accept().await?;
            let Some(peer_pid) = unix_peer_pid(&stream) else {
                log::warn!("[SVPManager] rejecting control connection without peer PID");
                continue;
            };
            #[cfg(target_os = "linux")]
            let ticket = {
                let _transition = self
                    .transition
                    .lock()
                    .map_err(|_| io::Error::other("transition state poisoned"))?;
                let Some(ticket) = self
                    .video_hosts
                    .ticket()
                    .filter(|ticket| ticket.pid == peer_pid)
                else {
                    continue;
                };
                if !self.claim_controller(peer_pid) {
                    continue;
                }
                ticket
            };
            let bridge = self.clone();
            #[cfg(not(target_os = "linux"))]
            if !bridge.claim_controller(peer_pid) {
                log::warn!("[SVPManager] rejecting competing controller process {peer_pid}");
                continue;
            }
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                let result = bridge
                    .handle_connection(
                        stream,
                        app,
                        #[cfg(target_os = "linux")]
                        ticket,
                    )
                    .await;
                bridge.release_controller(peer_pid);
                if let Err(error) = result {
                    log::debug!("[SVPManager] control connection closed: {error}");
                }
            });
        }
    }

    #[cfg(target_os = "windows")]
    async fn run_server(self, app: AppHandle) -> io::Result<()> {
        let mut first_instance = true;
        loop {
            let server = ServerOptions::new()
                .first_pipe_instance(first_instance)
                .create(MPV_SOCKET_PATH)?;
            first_instance = false;
            server.connect().await?;
            let Some(peer_pid) = windows_named_pipe_client_pid(&server) else {
                log::warn!("[SVPManager] rejecting named-pipe connection without client PID");
                continue;
            };
            let bridge = self.clone();
            if !bridge.claim_controller(peer_pid) {
                log::warn!("[SVPManager] rejecting competing controller process {peer_pid}");
                continue;
            }
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                let result = bridge.handle_connection(server, app).await;
                bridge.release_controller(peer_pid);
                if let Err(error) = result {
                    log::debug!("[SVPManager] control connection closed: {error}");
                }
            });
        }
    }

    fn claim_controller(&self, pid: u32) -> bool {
        let Ok(mut controller) = self.controller.lock() else {
            return false;
        };
        match controller.as_mut() {
            Some((owner, connections)) if *owner == pid => {
                *connections += 1;
                true
            }
            #[cfg(target_os = "linux")]
            Some(_) if self.video_hosts.active_pid() == Some(pid) => {
                // The previous graph host may retain an idle connection. Its
                // captured lease ticket is invalid, so only the active host can
                // replace this stale process claim.
                *controller = Some((pid, 1));
                true
            }
            Some(_) => false,
            None => {
                *controller = Some((pid, 1));
                true
            }
        }
    }

    fn release_controller(&self, pid: u32) {
        let Ok(mut controller) = self.controller.lock() else {
            return;
        };
        if let Some((owner, connections)) = controller.as_mut() {
            if *owner != pid {
                return;
            }
            if *connections > 1 {
                *connections -= 1;
            } else {
                *controller = None;
            }
        }
    }

    async fn handle_connection<S>(
        &self,
        stream: S,
        app: AppHandle,
        #[cfg(target_os = "linux")] ticket: crate::svp_video_host::VideoHostTicket,
    ) -> io::Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let (reader, mut writer) = tokio::io::split(stream);
        let mut lines = BufReader::new(reader).lines();
        while let Some(line) = lines.next_line().await? {
            let request: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(_) => continue,
            };
            let request_id = request.get("request_id").cloned().unwrap_or(Value::Null);
            let command = request
                .get("command")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let result = {
                let _transition = self
                    .transition
                    .lock()
                    .map_err(|_| io::Error::other("transition state poisoned"))?;
                #[cfg(target_os = "linux")]
                if self.video_hosts.ticket().as_ref() != Some(&ticket) {
                    break;
                }
                self.handle_command(&command, &app)
            };
            let response = match result {
                Ok(data) => json!({"request_id": request_id, "error": "success", "data": data}),
                Err(error) => json!({"request_id": request_id, "error": error}),
            };
            writer.write_all(response.to_string().as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        Ok(())
    }

    fn handle_command(&self, command: &[Value], app: &AppHandle) -> Result<Value, &'static str> {
        let name = command.first().and_then(Value::as_str).unwrap_or_default();
        log::debug!("[SVPManager] command {}", Value::Array(command.to_vec()));
        match name {
            "client_name" => Ok(json!({"name": "mpv", "version": "0.41.0"})),
            "observe_property" | "unobserve_property" => Ok(Value::Null),
            "get_property" => {
                let property = command.get(1).and_then(Value::as_str).unwrap_or_default();
                self.get_property(property, app)
            }
            "set_property" => {
                let property = command.get(1).and_then(Value::as_str).unwrap_or_default();
                let value = command.get(2).cloned().unwrap_or(Value::Null);
                self.set_property(property, value, app)
            }
            "vf" => self.handle_vf(command, app),
            _ => Ok(Value::Null),
        }
    }

    fn get_property(&self, property: &str, app: &AppHandle) -> Result<Value, &'static str> {
        let state = self.playback.lock().map_err(|_| "unavailable")?;
        let active = state.enabled && state.path.is_some();
        match property {
            "path" if active => Ok(json!(state.path)),
            "path" => Err("property unavailable"),
            "mpv-version" => Ok(json!("mpv v0.41.0")),
            #[cfg(target_os = "linux")]
            "input-ipc-server" => self
                .video_hosts
                .active_pid()
                .map(|pid| json!(crate::svp_video_host::manager_socket_path(pid)))
                .ok_or("property unavailable"),
            #[cfg(not(target_os = "linux"))]
            "input-ipc-server" => Ok(json!(MPV_SOCKET_PATH)),
            "working-directory" => Ok(json!(std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned())),
            "display-names" => {
                let display_name = app
                    .get_webview_window("main")
                    .and_then(|window| window.current_monitor().ok().flatten())
                    .and_then(|monitor| monitor.name().cloned())
                    .unwrap_or_default();
                Ok(json!([display_name]))
            }
            "video-format" if active => Ok(json!("h264")),
            "video-codec" if active => Ok(json!("H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10")),
            "video-params" if active => Ok(json!({
                "pixelformat": "yuv420p",
                "w": state.width,
                "h": state.height,
                "dw": state.width,
                "dh": state.height,
                "crop-x": 0,
                "crop-y": 0,
                "crop-w": state.width,
                "crop-h": state.height,
                "average-bpp": 12,
                "aspect": if state.height > 0 { state.width as f64 / state.height as f64 } else { 1.0 },
                "par": 1.0,
                "sar": if state.height > 0 { state.width as f64 / state.height as f64 } else { 1.0 },
                "colormatrix": "bt.709",
                "colorlevels": "limited",
                "primaries": "bt.709",
                "gamma": "bt.1886",
                "sig-peak": 0.0,
                "light": "display",
                "chroma-location": "mpeg2/4/h264",
                "stereo-in": "mono",
                "rotate": 0,
                "alpha": "none",
            })),
            "video-frame-info" if active => Ok(json!({
                "picture-type": "B",
                "interlaced": false,
                "tff": false,
                "repeat": false,
            })),
            "container-fps" if active => Ok(json!(state.fps)),
            "estimated-vf-fps" if active => {
                Ok(json!(if state.filter_active && state.output_fps > 0.0 {
                    state.output_fps
                } else {
                    state.fps
                }))
            }
            "duration" if active => Ok(json!(state.duration)),
            "user-data" => Ok(json!({
                "osc": {
                    "visibility": "auto",
                    "margins": {"t": 0, "l": 0, "r": 0, "b": 0}
                }
            })),
            "pause" => Ok(json!(state.paused)),
            "vf" => {
                if state.filter_active {
                    Ok(json!([{
                        "name": "vapoursynth",
                        "label": "svp",
                        "params": {
                            "file": state.script_path,
                            "buffered-frames": "4",
                            "concurrent-frames": "25",
                        }
                    }]))
                } else {
                    Ok(json!([]))
                }
            }
            _ => Err("property unavailable"),
        }
    }

    fn event_owner(&self) -> PlaybackOwner {
        #[cfg(target_os = "linux")]
        if let Some(ticket) = self.video_hosts.ticket() {
            return PlaybackOwner {
                host_id: Some(ticket.host_id),
                host_epoch: Some(ticket.epoch),
                host_revision: Some(ticket.revision),
            };
        }
        PlaybackOwner::default()
    }

    fn set_property(
        &self,
        property: &str,
        value: Value,
        app: &AppHandle,
    ) -> Result<Value, &'static str> {
        if property == "pause" {
            let paused = value.as_bool().unwrap_or(false);
            let media_key = if let Ok(mut state) = self.playback.lock() {
                state.paused = paused;
                state.media_key.clone()
            } else {
                None
            };
            let _ = app.emit(
                "svp-manager-set-paused",
                PlaybackPaused {
                    owner: self.event_owner(),
                    paused,
                    media_key,
                },
            );
        }
        Ok(Value::Null)
    }

    fn handle_vf(&self, command: &[Value], app: &AppHandle) -> Result<Value, &'static str> {
        let action = command.get(1).and_then(Value::as_str).unwrap_or_default();
        let spec = command.get(2).and_then(Value::as_str).unwrap_or_default();
        if action == "add" && spec.starts_with("@svp:vapoursynth=") {
            let value = spec.trim_start_matches("@svp:vapoursynth=");
            let script_path = value.rsplitn(3, ':').last().unwrap_or(value);
            if !Path::new(script_path).is_file() {
                return Err("invalid parameter");
            }
            self.enable_filter_locked(script_path, app)
                .map_err(|_| "error")?;
        } else if matches!(action, "remove" | "del") && spec == "@svp" {
            self.disable_filter_locked(app).map_err(|_| "error")?;
        }
        Ok(Value::Null)
    }

    fn enable_filter_locked(&self, script_path: &str, app: &AppHandle) -> io::Result<()> {
        let (snapshot, changed) = self.snapshots.prepare_file(Path::new(script_path))?;
        if !changed
            && self
                .playback
                .lock()
                .map(|state| state.filter_active)
                .unwrap_or(false)
        {
            return Ok(());
        }
        let mut state = self
            .playback
            .lock()
            .map_err(|_| io::Error::other("state poisoned"))?;
        #[cfg(target_os = "linux")]
        if let Err(error) = (|| {
            write_runtime_file(&self.script_file, &snapshot.snapshot_path)?;
            write_runtime_file(&self.filter_file, "localbooruvs")
        })() {
            let _ = fs::remove_file(&self.script_file);
            let _ = fs::remove_file(&self.filter_file);
            return Err(error);
        }
        if changed {
            if let Err(error) = self.snapshots.commit(snapshot.clone()) {
                #[cfg(target_os = "linux")]
                {
                    let _ = fs::remove_file(&self.script_file);
                    let _ = fs::remove_file(&self.filter_file);
                }
                return Err(error);
            }
        }
        state.filter_active = true;
        state.script_path = Some(script_path.to_owned());
        state.output_fps =
            script_output_fps(&snapshot.snapshot_path, state.fps).unwrap_or(state.fps);
        let media_key = state.media_key.clone();
        drop(state);
        log::info!(
            "[SVPManager] enabling Manager graph revision {} from {script_path}",
            snapshot.revision
        );
        let _ = app.emit(
            "svp-manager-filter-changed",
            FilterChanged {
                owner: self.event_owner(),
                enabled: true,
                script_path: Some(script_path.to_owned()),
                graph_revision: Some(snapshot.revision),
                media_key,
            },
        );
        Ok(())
    }

    fn disable_filter_locked(&self, app: &AppHandle) -> io::Result<()> {
        if self
            .playback
            .lock()
            .map(|state| !state.filter_active)
            .unwrap_or(true)
        {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            let _ = fs::remove_file(&self.filter_file);
            let _ = fs::remove_file(&self.script_file);
        }
        self.snapshots.clear_current();
        let media_key = if let Ok(mut state) = self.playback.lock() {
            state.filter_active = false;
            state.script_path = None;
            state.output_fps = state.fps;
            state.media_key.clone()
        } else {
            None
        };
        log::info!("[SVPManager] disabling interpolation filter");
        let _ = app.emit(
            "svp-manager-filter-changed",
            FilterChanged {
                owner: self.event_owner(),
                enabled: false,
                script_path: None,
                graph_revision: None,
                media_key,
            },
        );
        Ok(())
    }
}

#[cfg(unix)]
fn remove_stale_control_socket(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if !metadata.file_type().is_socket() || metadata.uid() != unsafe { libc::geteuid() } {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "refusing to remove an unowned control-socket path",
        ));
    }
    fs::remove_file(path)
}

#[cfg(target_os = "linux")]
fn unix_peer_pid(stream: &UnixStream) -> Option<u32> {
    stream
        .peer_cred()
        .ok()
        .and_then(|credentials| credentials.pid())
        .and_then(|pid| u32::try_from(pid).ok())
}

#[cfg(target_os = "macos")]
fn unix_peer_pid(stream: &UnixStream) -> Option<u32> {
    use std::os::fd::AsRawFd;

    let mut pid: libc::pid_t = 0;
    let mut length = std::mem::size_of::<libc::pid_t>() as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_LOCAL,
            libc::LOCAL_PEERPID,
            (&mut pid as *mut libc::pid_t).cast(),
            &mut length,
        )
    };
    (result == 0).then(|| u32::try_from(pid).ok()).flatten()
}

#[cfg(target_os = "windows")]
fn windows_named_pipe_client_pid(server: &NamedPipeServer) -> Option<u32> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Pipes::GetNamedPipeClientProcessId;

    let mut pid = 0;
    let result = unsafe { GetNamedPipeClientProcessId(server.as_raw_handle() as HANDLE, &mut pid) };
    (result != 0).then_some(pid)
}

#[cfg(target_os = "linux")]
fn write_runtime_file(path: &Path, value: &str) -> io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid runtime path"))?;
    let temporary = path.with_file_name(format!("{file_name}.tmp-{}", std::process::id()));
    fs::write(&temporary, value)?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
    fs::rename(temporary, path)
}

fn script_output_fps(path: &str, source_fps: f64) -> Option<f64> {
    let script = fs::read_to_string(path).ok()?;
    let pattern = Regex::new(r"rate:\{num:(\d+),den:(\d+)\}").ok()?;
    let captures = pattern.captures(&script)?;
    let numerator: f64 = captures.get(1)?.as_str().parse().ok()?;
    let denominator: f64 = captures.get(2)?.as_str().parse().ok()?;
    (denominator > 0.0 && source_fps > 0.0).then_some(source_fps * numerator / denominator)
}

#[tauri::command]
pub fn acquire_svp_video_host_epoch(
    app: AppHandle,
    bridge: State<'_, SvpManagerBridge>,
) -> Result<u64, String> {
    #[cfg(target_os = "linux")]
    {
        let _transition = bridge
            .transition
            .lock()
            .map_err(|_| "SVP transition state unavailable")?;
        let epoch = bridge
            .video_hosts
            .acquire_epoch()
            .map_err(|error| error.to_string())?;
        bridge
            .disable_filter_locked(&app)
            .map_err(|error| error.to_string())?;
        let mut state = bridge
            .playback
            .lock()
            .map_err(|_| "SVP state unavailable")?;
        state.enabled = false;
        state.path = None;
        state.media_key = None;
        Ok(epoch)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (app, bridge);
        Ok(0)
    }
}

#[tauri::command]
pub fn configure_local_video_resolution(
    bridge: State<'_, SvpManagerBridge>,
    update: LocalVideoResolutionUpdate,
) -> Result<bool, String> {
    #[cfg(target_os = "linux")]
    {
        let _transition = bridge
            .transition
            .lock()
            .map_err(|_| "video transition state unavailable")?;
        bridge
            .video_hosts
            .configure_resolution(
                &update.host_id,
                update.host_epoch,
                update.host_revision,
                update.bounds,
            )
            .map_err(|error| error.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bridge, update);
        Err("Decoded frame resizing requires the Linux desktop player".into())
    }
}

#[tauri::command]
pub fn verify_local_video_resolution(
    bridge: State<'_, SvpManagerBridge>,
    update: LocalVideoResolutionUpdate,
) -> Result<bool, String> {
    #[cfg(target_os = "linux")]
    {
        let _transition = bridge
            .transition
            .lock()
            .map_err(|_| "video transition state unavailable")?;
        bridge
            .video_hosts
            .verify_resolution(&update.host_id, update.host_epoch, update.host_revision)
            .map(|_| true)
            .map_err(|error| error.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (bridge, update);
        Err("Decoded frame resizing requires the Linux desktop player".into())
    }
}

#[tauri::command]
pub fn update_svp_manager_playback(
    app: AppHandle,
    bridge: State<'_, SvpManagerBridge>,
    update: SvpPlaybackUpdate,
) -> Result<(), String> {
    let _transition = bridge
        .transition
        .lock()
        .map_err(|_| "SVP transition state unavailable")?;
    #[cfg(target_os = "linux")]
    let geometry = if update.enabled {
        bridge
            .video_hosts
            .playback_geometry(
                update.host_id.as_deref(),
                update.host_epoch,
                update.resize_revision,
            )
            .map_err(|error| error.to_string())?
    } else {
        None
    };
    #[cfg(not(target_os = "linux"))]
    let geometry: Option<(u32, u32)> = None;
    #[cfg(target_os = "linux")]
    if !bridge
        .video_hosts
        .update(
            update.enabled,
            update.host_id.as_deref(),
            update.host_epoch,
            update.host_revision,
        )
        .map_err(|error| error.to_string())?
    {
        return Ok(());
    }
    if !update.enabled {
        bridge
            .disable_filter_locked(&app)
            .map_err(|error| error.to_string())?;
    }
    let mut state = bridge
        .playback
        .lock()
        .map_err(|_| "SVP state unavailable")?;
    state.enabled = update.enabled;
    state.path = if update.enabled { update.path } else { None };
    state.media_key = if update.enabled {
        update.media_key
    } else {
        None
    };
    state.width = geometry
        .map(|(width, _)| width)
        .or(update.width)
        .unwrap_or(0);
    state.height = geometry
        .map(|(_, height)| height)
        .or(update.height)
        .unwrap_or(0);
    state.fps = update.fps.unwrap_or(0.0);
    state.duration = update.duration.unwrap_or(0.0);
    state.paused = update.paused.unwrap_or(true);
    if !update.enabled {
        state.output_fps = state.fps;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bridge() -> SvpManagerBridge {
        let snapshots = ManagerGraphSnapshotStore::new(std::env::temp_dir().join(format!(
            "localbooru-svp-bridge-test-{}",
            uuid::Uuid::new_v4()
        )));
        SvpManagerBridge::new(snapshots)
    }

    // AC: @ordinary-lightbox-media-rendering ac-3
    #[test]
    fn svp_events_include_the_owning_media_key() {
        let filter = serde_json::to_value(FilterChanged {
            owner: PlaybackOwner {
                host_id: Some("localbooru-svp-host-video".into()),
                host_epoch: Some(3),
                host_revision: Some(7),
            },
            enabled: true,
            script_path: Some("graph.vpy".into()),
            graph_revision: Some(7),
            media_key: Some("library:42".into()),
        })
        .unwrap();
        let paused = serde_json::to_value(PlaybackPaused {
            owner: PlaybackOwner {
                host_id: Some("localbooru-svp-host-video".into()),
                host_epoch: Some(3),
                host_revision: Some(7),
            },
            paused: true,
            media_key: Some("library:42".into()),
        })
        .unwrap();

        assert_eq!(filter["mediaKey"], "library:42");
        assert_eq!(paused["mediaKey"], "library:42");
        for event in [filter, paused] {
            assert_eq!(event["hostId"], "localbooru-svp-host-video");
            assert_eq!(event["hostEpoch"], 3);
            assert_eq!(event["hostRevision"], 7);
        }
    }

    // AC: @svp-manager-transitions ac-controller-ownership
    #[test]
    fn controller_ownership_allows_one_process_and_its_parallel_connections() {
        let bridge = test_bridge();
        assert!(bridge.claim_controller(101));
        assert!(bridge.claim_controller(101));
        assert!(!bridge.claim_controller(202));

        bridge.release_controller(101);
        assert!(!bridge.claim_controller(202));
        bridge.release_controller(101);
        assert!(bridge.claim_controller(202));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn new_active_graph_host_replaces_only_stale_controller_claim() {
        let bridge = test_bridge();
        bridge.video_hosts.prepare().unwrap();
        let epoch = bridge.video_hosts.acquire_epoch().unwrap();
        assert!(bridge.claim_controller(101));
        assert!(!bridge.claim_controller(202));
        fs::write(
            bridge.video_hosts.root().join("localbooru-svp-host-next"),
            "202",
        )
        .unwrap();
        assert!(bridge
            .video_hosts
            .update(true, Some("localbooru-svp-host-next"), Some(epoch), Some(1))
            .unwrap());
        assert!(bridge.claim_controller(202));
        bridge.release_controller(101);
        assert!(!bridge.claim_controller(303));
        bridge.release_controller(202);
        assert!(bridge.claim_controller(303));
        fs::remove_dir_all(bridge.snapshots.root()).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_transport_reports_the_controlling_process() {
        let path = PathBuf::from("/tmp").join(format!("lb-svp-{}", uuid::Uuid::new_v4().simple()));
        let listener = UnixListener::bind(&path).unwrap();
        let client = tokio::spawn({
            let path = path.clone();
            async move { UnixStream::connect(path).await.unwrap() }
        });
        let (server, _) = listener.accept().await.unwrap();

        assert_eq!(unix_peer_pid(&server), Some(std::process::id()));
        drop(client.await.unwrap());
        drop(listener);
        let _ = fs::remove_file(path);
    }

    #[cfg(target_os = "windows")]
    #[tokio::test]
    async fn named_pipe_transport_reports_the_controlling_process() {
        use tokio::net::windows::named_pipe::ClientOptions;

        let name = format!(r"\\.\pipe\localbooru-manager-peer-{}", uuid::Uuid::new_v4());
        let server = ServerOptions::new()
            .first_pipe_instance(true)
            .create(&name)
            .unwrap();
        let client = ClientOptions::new().open(&name).unwrap();
        server.connect().await.unwrap();

        assert_eq!(
            windows_named_pipe_client_pid(&server),
            Some(std::process::id())
        );
        drop(client);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn runtime_file_replace_is_atomic_and_private() {
        let path = std::env::temp_dir().join(format!(
            "localbooru-svp-runtime-test-{}",
            std::process::id()
        ));
        write_runtime_file(&path, "first").unwrap();
        write_runtime_file(&path, "second").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let _ = fs::remove_file(path);
    }
}
