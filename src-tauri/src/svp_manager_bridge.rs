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

#[cfg(unix)]
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
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FilterChanged {
    enabled: bool,
    script_path: Option<String>,
    graph_revision: Option<u64>,
}

#[derive(Debug)]
struct PlaybackState {
    enabled: bool,
    path: Option<String>,
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
    #[cfg(unix)]
    control_socket: PathBuf,
}

impl SvpManagerBridge {
    pub fn new(snapshots: ManagerGraphSnapshotStore) -> Self {
        #[cfg(target_os = "linux")]
        let uid = unsafe { libc::geteuid() };
        Self {
            playback: Arc::new(Mutex::new(PlaybackState::default())),
            transition: Arc::new(Mutex::new(())),
            controller: Arc::new(Mutex::new(None)),
            snapshots,
            #[cfg(target_os = "linux")]
            filter_file: PathBuf::from(format!("/tmp/localbooru-svp-filter-{uid}")),
            #[cfg(target_os = "linux")]
            script_file: PathBuf::from(format!("/tmp/localbooru-svp-script-{uid}")),
            #[cfg(target_os = "linux")]
            control_socket: PathBuf::from(format!("/tmp/localbooru-mpv-control-{uid}")),
            #[cfg(target_os = "macos")]
            control_socket: PathBuf::from(MPV_SOCKET_PATH),
        }
    }

    pub fn configure_environment(&self) {
        std::env::set_var("LOCALBOORU_SVP_SNAPSHOT_ROOT", self.snapshots.root());
        #[cfg(target_os = "linux")]
        {
            std::env::set_var("WEBKIT_GST_VIDEO_FILTER_FILE", &self.filter_file);
            std::env::set_var("LOCALBOORU_VS_SCRIPT_FILE", &self.script_file);
            let native_svp_enabled =
                std::env::var("LOCALBOORU_ENABLE_NATIVE_SVP").as_deref() == Ok("1");
            if native_svp_enabled {
                std::env::set_var("LOCALBOORU_MPV_CONTROL_UPSTREAM", &self.control_socket);
            } else {
                std::env::remove_var("LOCALBOORU_MPV_CONTROL_UPSTREAM");
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
            let bridge = self.clone();
            if !bridge.claim_controller(peer_pid) {
                log::warn!("[SVPManager] rejecting competing controller process {peer_pid}");
                continue;
            }
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                let result = bridge.handle_connection(stream, app).await;
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

    async fn handle_connection<S>(&self, stream: S, app: AppHandle) -> io::Result<()>
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
            let result = self.handle_command(&command, &app);
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

    fn set_property(
        &self,
        property: &str,
        value: Value,
        app: &AppHandle,
    ) -> Result<Value, &'static str> {
        if property == "pause" {
            let paused = value.as_bool().unwrap_or(false);
            if let Ok(mut state) = self.playback.lock() {
                state.paused = paused;
            }
            let _ = app.emit("svp-manager-set-paused", paused);
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
            self.enable_filter(script_path, app).map_err(|_| "error")?;
        } else if matches!(action, "remove" | "del") && spec == "@svp" {
            self.disable_filter(app).map_err(|_| "error")?;
        }
        Ok(Value::Null)
    }

    fn enable_filter(&self, script_path: &str, app: &AppHandle) -> io::Result<()> {
        let _transition = self
            .transition
            .lock()
            .map_err(|_| io::Error::other("transition state poisoned"))?;
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
        drop(state);
        log::info!(
            "[SVPManager] enabling Manager graph revision {} from {script_path}",
            snapshot.revision
        );
        let _ = app.emit(
            "svp-manager-filter-changed",
            FilterChanged {
                enabled: true,
                script_path: Some(script_path.to_owned()),
                graph_revision: Some(snapshot.revision),
            },
        );
        Ok(())
    }

    fn disable_filter(&self, app: &AppHandle) -> io::Result<()> {
        let _transition = self
            .transition
            .lock()
            .map_err(|_| io::Error::other("transition state poisoned"))?;
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
        if let Ok(mut state) = self.playback.lock() {
            state.filter_active = false;
            state.script_path = None;
            state.output_fps = state.fps;
        }
        log::info!("[SVPManager] disabling interpolation filter");
        let _ = app.emit(
            "svp-manager-filter-changed",
            FilterChanged {
                enabled: false,
                script_path: None,
                graph_revision: None,
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
pub fn update_svp_manager_playback(
    app: AppHandle,
    bridge: State<'_, SvpManagerBridge>,
    update: SvpPlaybackUpdate,
) -> Result<(), String> {
    if !update.enabled {
        bridge
            .disable_filter(&app)
            .map_err(|error| error.to_string())?;
    }
    let mut state = bridge
        .playback
        .lock()
        .map_err(|_| "SVP state unavailable")?;
    state.enabled = update.enabled;
    state.path = if update.enabled { update.path } else { None };
    state.width = update.width.unwrap_or(0);
    state.height = update.height.unwrap_or(0);
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
