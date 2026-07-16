use std::io::{BufRead, BufReader, Write};
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, OwnedFd};
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::protocol::{
    validate_protocol_version, NativeVideoCommand, NativeVideoEvent, PROTOCOL_VERSION,
};
#[cfg(target_os = "linux")]
use super::surface_channel::{
    receive_message as receive_surface_message, send_message as send_surface_message, socket_pair,
    ReceivedSurfaceMessage, SurfaceChannelMessage,
};
#[cfg(target_os = "linux")]
use super::surface_protocol::SurfaceFrameRelease;

const MAX_CONTROL_MESSAGE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, Default)]
pub struct HelperProcessOptions {
    pub force_copy: bool,
}

#[derive(Debug)]
pub struct HelperProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    protocol_version: u32,
    #[cfg(target_os = "linux")]
    surface_socket: OwnedFd,
}

impl HelperProcess {
    pub fn spawn(executable: &Path) -> Result<Self, String> {
        Self::spawn_with_options(executable, HelperProcessOptions::default())
    }

    pub fn spawn_with_options(
        executable: &Path,
        options: HelperProcessOptions,
    ) -> Result<Self, String> {
        #[cfg(target_os = "linux")]
        let (surface_socket, child_surface_socket) = socket_pair()
            .map_err(|error| format!("failed to create native surface channel: {error}"))?;

        let mut command = Command::new(executable);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        #[cfg(target_os = "linux")]
        command.env(
            "LOCALBOORU_NATIVE_DMABUF",
            if options.force_copy { "0" } else { "1" },
        );
        #[cfg(target_os = "linux")]
        {
            const CHILD_SURFACE_FD: libc::c_int = 3;
            let inherited_fd = child_surface_socket.as_raw_fd();
            unsafe {
                command.pre_exec(move || {
                    let parent_pid = libc::getppid();
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0
                        || libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != parent_pid {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "native video parent exited during helper launch",
                        ));
                    }
                    let nofile_limit = libc::rlimit {
                        rlim_cur: 256,
                        rlim_max: 256,
                    };
                    let core_limit = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::setrlimit(libc::RLIMIT_NOFILE, &nofile_limit) != 0
                        || libc::setrlimit(libc::RLIMIT_CORE, &core_limit) != 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::dup2(inherited_fd, CHILD_SURFACE_FD) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    let flags = libc::fcntl(CHILD_SURFACE_FD, libc::F_GETFD);
                    if flags < 0
                        || libc::fcntl(CHILD_SURFACE_FD, libc::F_SETFD, flags & !libc::FD_CLOEXEC)
                            < 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            command.env("LOCALBOORU_SURFACE_FD", CHILD_SURFACE_FD.to_string());
        }
        let mut child = command
            .spawn()
            .map_err(|error| format!("failed to spawn native video helper: {error}"))?;
        #[cfg(target_os = "linux")]
        log::info!(
            "[NativeVideo] spawned helper pid={} parent_surface_fd={} inherited_surface_fd={}",
            child.id(),
            surface_socket.as_raw_fd(),
            child_surface_socket.as_raw_fd()
        );
        #[cfg(target_os = "linux")]
        drop(child_surface_socket);

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "native video helper stdin unavailable".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "native video helper stdout unavailable".to_string())?;
        let mut stdout = BufReader::new(stdout);

        let hello = NativeVideoCommand::Hello {
            protocol_version: PROTOCOL_VERSION,
        };
        if let Err(error) = write_message(&mut stdin, &hello) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }

        let line = match read_bounded_line(&mut stdout, "helper handshake") {
            Ok(Some(line)) => line,
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("native video helper exited before handshake".to_string());
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };

        let event: NativeVideoEvent = match serde_json::from_slice(&line) {
            Ok(event) => event,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("invalid helper handshake: {error}"));
            }
        };
        let protocol_version = match event {
            NativeVideoEvent::Ready { protocol_version } => protocol_version,
            other => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("unexpected helper handshake event: {other:?}"));
            }
        };
        if let Err(error) = validate_protocol_version(protocol_version) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }

        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout,
            protocol_version,
            #[cfg(target_os = "linux")]
            surface_socket,
        })
    }

    pub fn protocol_version(&self) -> u32 {
        self.protocol_version
    }

    pub fn send(&mut self, command: &NativeVideoCommand) -> Result<(), String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "native video helper is shutting down".to_string())?;
        write_message(stdin, command)
    }

    pub fn read_event(&mut self) -> Result<Option<NativeVideoEvent>, String> {
        let Some(line) = read_bounded_line(&mut self.stdout, "helper event")? else {
            return Ok(None);
        };
        serde_json::from_slice(&line)
            .map(Some)
            .map_err(|error| format!("invalid helper event: {error}"))
    }

    #[cfg(target_os = "linux")]
    pub fn read_surface_message(&self) -> Result<ReceivedSurfaceMessage, String> {
        receive_surface_message(self.surface_socket.as_raw_fd())
            .map_err(|error| format!("failed to read helper surface message: {error}"))
    }

    #[cfg(target_os = "linux")]
    pub fn send_surface_release(&self, release: SurfaceFrameRelease) -> Result<(), String> {
        send_surface_message(
            self.surface_socket.as_raw_fd(),
            &SurfaceChannelMessage::FrameRelease { release },
            &[],
        )
        .map_err(|error| format!("failed to release helper surface: {error}"))
    }

    #[cfg(target_os = "linux")]
    pub fn control_event_fd(&self) -> std::os::fd::RawFd {
        self.stdout.get_ref().as_raw_fd()
    }

    #[cfg(target_os = "linux")]
    pub fn surface_event_fd(&self) -> std::os::fd::RawFd {
        self.surface_socket.as_raw_fd()
    }

    pub fn try_wait(&mut self) -> Result<Option<std::process::ExitStatus>, String> {
        self.child
            .try_wait()
            .map_err(|error| format!("failed to inspect native video helper: {error}"))
    }

    pub fn shutdown(&mut self) -> Result<(), String> {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        let status = loop {
            if let Some(status) = self
                .child
                .try_wait()
                .map_err(|error| format!("failed to wait for native video helper: {error}"))?
            {
                break status;
            }
            if Instant::now() >= deadline {
                self.child
                    .kill()
                    .map_err(|error| format!("failed to terminate native video helper: {error}"))?;
                self.child
                    .wait()
                    .map_err(|error| format!("failed to reap native video helper: {error}"))?;
                return Err("native video helper did not stop within 2 seconds".to_string());
            }
            thread::sleep(Duration::from_millis(10));
        };
        if status.success() {
            Ok(())
        } else {
            Err(format!("native video helper exited with {status}"))
        }
    }
}

impl Drop for HelperProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            self.stdin.take();
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn write_message(writer: &mut impl Write, message: &NativeVideoCommand) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, message)
        .map_err(|error| format!("failed to encode helper command: {error}"))?;
    writer
        .write_all(b"\n")
        .and_then(|_| writer.flush())
        .map_err(|error| format!("failed to write helper command: {error}"))
}

fn read_bounded_line(reader: &mut impl BufRead, context: &str) -> Result<Option<Vec<u8>>, String> {
    let mut message = Vec::new();
    loop {
        let available = reader
            .fill_buf()
            .map_err(|error| format!("failed to read {context}: {error}"))?;
        if available.is_empty() {
            return if message.is_empty() {
                Ok(None)
            } else {
                Ok(Some(message))
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |position| position + 1);
        let payload_bytes = newline.unwrap_or(available.len());
        if message.len() + payload_bytes > MAX_CONTROL_MESSAGE_BYTES {
            return Err(format!(
                "{context} exceeds the {MAX_CONTROL_MESSAGE_BYTES}-byte limit"
            ));
        }
        message.extend_from_slice(&available[..payload_bytes]);
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(Some(message));
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::native_video::surface_channel::SurfaceChannelMessage;
    use crate::native_video::surface_protocol::SurfaceFrameRelease;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_SCRIPT_ID: AtomicU64 = AtomicU64::new(1);

    fn helper_script(body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "localbooru-native-helper-{}-{}.sh",
            std::process::id(),
            NEXT_SCRIPT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, body).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_helper_launch_enforces_parent_and_resource_guards() {
        let script = helper_script(
            "#!/bin/sh\n[ \"$(awk '/NoNewPrivs:/ { print $2 }' /proc/self/status)\" = 1 ] || exit 21\n[ \"$(ulimit -n)\" = 256 ] || exit 22\n[ \"$(ulimit -c)\" = 0 ] || exit 23\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nwhile read line; do :; done\n",
        );
        let mut helper = HelperProcess::spawn(&script).unwrap();
        helper.shutdown().unwrap();
        let _ = std::fs::remove_file(script);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn force_copy_option_is_scoped_to_the_helper_process() {
        let script = helper_script(
            "#!/bin/sh\n[ \"$LOCALBOORU_NATIVE_DMABUF\" = 0 ] || exit 19\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nwhile read line; do :; done\n",
        );
        let mut helper =
            HelperProcess::spawn_with_options(&script, HelperProcessOptions { force_copy: true })
                .unwrap();
        helper.shutdown().unwrap();
        let _ = fs::remove_file(script);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn preferred_dmabuf_option_is_scoped_to_the_helper_process() {
        let script = helper_script(
            "#!/bin/sh\n[ \"$LOCALBOORU_NATIVE_DMABUF\" = 1 ] || exit 20\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nwhile read line; do :; done\n",
        );
        let mut helper = HelperProcess::spawn(&script).unwrap();
        helper.shutdown().unwrap();
        let _ = fs::remove_file(script);
    }

    #[test]
    fn missing_helper_returns_a_specific_error() {
        let error =
            HelperProcess::spawn(std::path::Path::new("/definitely/missing/helper")).unwrap_err();
        assert!(error.contains("failed to spawn native video helper"));
    }

    #[test]
    fn helper_negotiates_compatible_protocol_and_shuts_down() {
        let script = helper_script(
            "#!/bin/sh\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nwhile read line; do :; done\n",
        );
        let mut helper = HelperProcess::spawn(&script).unwrap();
        assert_eq!(helper.protocol_version(), PROTOCOL_VERSION);
        helper.shutdown().unwrap();
        let _ = fs::remove_file(script);
    }

    #[test]
    fn inherited_surface_socket_carries_seqpacket_messages() {
        let script = helper_script(
            "#!/bin/sh\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nprintf '%s' '{\"type\":\"frame_release\",\"release\":{\"generation\":4,\"buffer_id\":2,\"sequence\":9}}' >&3\nwhile read line; do :; done\n",
        );
        let mut helper = HelperProcess::spawn(&script).unwrap();
        let received = helper.read_surface_message().unwrap();
        assert!(received.fds.is_empty());
        assert_eq!(
            received.message,
            SurfaceChannelMessage::FrameRelease {
                release: SurfaceFrameRelease {
                    generation: 4,
                    buffer_id: 2,
                    sequence: 9,
                }
            }
        );
        helper.shutdown().unwrap();
        let _ = fs::remove_file(script);
    }

    #[test]
    fn oversized_handshake_is_rejected_before_json_parsing() {
        let script = helper_script("#!/bin/sh\nread hello\nprintf '%70000s\\n' x\n");
        let error = HelperProcess::spawn(&script).unwrap_err();
        assert!(error.contains("65536-byte limit"));
        let _ = fs::remove_file(script);
    }

    #[test]
    fn incompatible_protocol_is_rejected_and_child_is_stopped() {
        let script = helper_script(
            "#!/bin/sh\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":2000}'\n",
        );
        let error = HelperProcess::spawn(&script).unwrap_err();
        assert!(error.contains("incompatible native-video protocol"));
        let _ = fs::remove_file(script);
    }
}
