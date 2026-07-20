use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::helper_process::{HelperProcess, HelperProcessOptions};
use super::protocol::{NativeVideoCommand, NativeVideoEvent};
use super::surface_channel::ReceivedSurfaceMessage;
use super::surface_protocol::SurfaceFrameRelease;

#[derive(Debug)]
pub enum RuntimeNotification {
    Control(NativeVideoEvent),
    Surface(ReceivedSurfaceMessage),
    Exited { success: bool },
    Error(String),
}

enum RuntimeCommand {
    Helper(NativeVideoCommand),
    Release(SurfaceFrameRelease),
    Stop,
}

pub struct NativeVideoRuntime {
    commands: Sender<RuntimeCommand>,
    notifications: Option<Receiver<RuntimeNotification>>,
    worker: Option<JoinHandle<()>>,
}

#[derive(Clone)]
pub struct NativeVideoRuntimeHandle {
    commands: Sender<RuntimeCommand>,
}

impl NativeVideoRuntimeHandle {
    pub fn send(&self, command: NativeVideoCommand) -> Result<(), String> {
        self.commands
            .send(RuntimeCommand::Helper(command))
            .map_err(|_| "native video runtime is not running".to_string())
    }

    pub fn release(&self, release: SurfaceFrameRelease) -> Result<(), String> {
        self.commands
            .send(RuntimeCommand::Release(release))
            .map_err(|_| "native video runtime is not running".to_string())
    }
}

impl NativeVideoRuntime {
    pub fn start(executable: &Path) -> Result<Self, String> {
        Self::start_with_options(executable, HelperProcessOptions::default())
    }

    pub fn start_with_options(
        executable: &Path,
        options: HelperProcessOptions,
    ) -> Result<Self, String> {
        let executable = executable.to_path_buf();
        let (command_tx, command_rx) = mpsc::channel();
        let (notification_tx, notification_rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("native-video-runtime".to_string())
            .spawn(
                move || match HelperProcess::spawn_with_options(&executable, options) {
                    Ok(helper) => {
                        let _ = startup_tx.send(Ok(()));
                        runtime_loop(helper, command_rx, notification_tx);
                    }
                    Err(error) => {
                        let _ = startup_tx.send(Err(error));
                    }
                },
            )
            .map_err(|error| format!("failed to start native video runtime: {error}"))?;
        match startup_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let _ = worker.join();
                return Err(error);
            }
            Err(_) => {
                let _ = worker.join();
                return Err("native video runtime worker exited during startup".to_string());
            }
        }
        Ok(Self {
            commands: command_tx,
            notifications: Some(notification_rx),
            worker: Some(worker),
        })
    }

    pub fn send(&self, command: NativeVideoCommand) -> Result<(), String> {
        self.handle().send(command)
    }

    pub fn release(&self, release: SurfaceFrameRelease) -> Result<(), String> {
        self.handle().release(release)
    }

    pub fn handle(&self) -> NativeVideoRuntimeHandle {
        NativeVideoRuntimeHandle {
            commands: self.commands.clone(),
        }
    }

    pub fn take_notifications(&mut self) -> Result<Receiver<RuntimeNotification>, String> {
        self.notifications
            .take()
            .ok_or_else(|| "native video notifications were already taken".to_string())
    }

    pub fn stop(&mut self) {
        let _ = self.commands.send(RuntimeCommand::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for NativeVideoRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}

fn runtime_loop(
    mut helper: HelperProcess,
    commands: Receiver<RuntimeCommand>,
    notifications: Sender<RuntimeNotification>,
) {
    loop {
        loop {
            match commands.try_recv() {
                Ok(RuntimeCommand::Helper(command)) => {
                    if let Err(error) = helper.send(&command) {
                        let _ = notifications.send(RuntimeNotification::Error(error));
                        return;
                    }
                }
                Ok(RuntimeCommand::Release(release)) => {
                    if let Err(error) = helper.send_surface_release(release) {
                        let _ = notifications.send(RuntimeNotification::Error(error));
                        return;
                    }
                }
                Ok(RuntimeCommand::Stop) | Err(TryRecvError::Disconnected) => {
                    let result = helper.shutdown();
                    let _ = notifications.send(match result {
                        Ok(()) => RuntimeNotification::Exited { success: true },
                        Err(error) => RuntimeNotification::Error(error),
                    });
                    return;
                }
                Err(TryRecvError::Empty) => break,
            }
        }

        let mut poll_fds = [
            libc::pollfd {
                fd: helper.control_event_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: helper.surface_event_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // Surface releases share the runtime command channel. Keep the bounded
        // poll interval well below one 60 Hz frame so a three-surface pool does
        // not spend an extra frame waiting for an already-completed lease to be
        // returned to the helper.
        let poll_result =
            unsafe { libc::poll(poll_fds.as_mut_ptr(), poll_fds.len() as libc::nfds_t, 4) };
        if poll_result < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                let _ = notifications.send(RuntimeNotification::Error(format!(
                    "failed to poll native video helper: {error}"
                )));
                return;
            }
        }
        if poll_fds[0].revents & libc::POLLIN != 0 {
            match helper.read_event() {
                Ok(Some(event)) => {
                    if notifications
                        .send(RuntimeNotification::Control(event))
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = notifications.send(RuntimeNotification::Error(error));
                    return;
                }
            }
        }
        if poll_fds[1].revents & libc::POLLIN != 0 {
            match helper.read_surface_message() {
                Ok(message) => {
                    if notifications
                        .send(RuntimeNotification::Surface(message))
                        .is_err()
                    {
                        return;
                    }
                }
                Err(error) => {
                    let mut child_state = "helper still running".to_string();
                    for _ in 0..20 {
                        match helper.try_wait() {
                            Ok(Some(status)) => {
                                child_state = format!("helper exited with {status}");
                                break;
                            }
                            Ok(None) => thread::sleep(Duration::from_millis(5)),
                            Err(wait_error) => {
                                child_state = format!("helper status unavailable: {wait_error}");
                                break;
                            }
                        }
                    }
                    let _ = notifications.send(RuntimeNotification::Error(format!(
                        "{error}; {child_state}"
                    )));
                    return;
                }
            }
        }
        match helper.try_wait() {
            Ok(Some(status)) => {
                let _ = notifications.send(RuntimeNotification::Exited {
                    success: status.success(),
                });
                return;
            }
            Ok(None) => {}
            Err(error) => {
                let _ = notifications.send(RuntimeNotification::Error(error));
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use crate::native_video::surface_channel::SurfaceChannelMessage;

    static NEXT_SCRIPT_ID: AtomicU64 = AtomicU64::new(1);

    fn helper_script(body: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "localbooru-native-runtime-test-{}-{}",
            std::process::id(),
            NEXT_SCRIPT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, body).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    #[test]
    fn actor_multiplexes_control_surface_and_release_messages() {
        let script = helper_script(
            "#!/bin/sh\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nread command\necho '{\"type\":\"playback_state\",\"generation\":3,\"position\":1.0,\"duration\":2.0,\"paused\":true}'\nprintf '%s' '{\"type\":\"frame_release\",\"release\":{\"generation\":3,\"buffer_id\":1,\"sequence\":7}}' >&3\ndd bs=65536 count=1 <&3 >/dev/null 2>&1\n",
        );
        let mut runtime = NativeVideoRuntime::start(&script).unwrap();
        let notifications = runtime.take_notifications().unwrap();
        runtime
            .send(NativeVideoCommand::SetPaused { paused: true })
            .unwrap();

        let mut saw_control = false;
        let mut saw_surface = false;
        while !(saw_control && saw_surface) {
            match notifications.recv_timeout(Duration::from_secs(2)).unwrap() {
                RuntimeNotification::Control(NativeVideoEvent::PlaybackState {
                    generation,
                    ..
                }) => {
                    assert_eq!(generation, 3);
                    saw_control = true;
                }
                RuntimeNotification::Surface(message) => {
                    let SurfaceChannelMessage::FrameRelease { release } = message.message else {
                        panic!("unexpected surface message");
                    };
                    assert_eq!(release.sequence, 7);
                    saw_surface = true;
                }
                other => panic!("unexpected runtime notification: {other:?}"),
            }
        }
        runtime
            .release(SurfaceFrameRelease {
                generation: 3,
                buffer_id: 1,
                sequence: 7,
            })
            .unwrap();
        runtime.stop();
        let _ = fs::remove_file(script);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn helper_parent_thread_lives_for_the_runtime_lifetime() {
        let script = helper_script(
            "#!/bin/sh\nread hello\necho '{\"type\":\"ready\",\"protocol_version\":1000}'\nwhile read command; do echo '{\"type\":\"playback_state\",\"generation\":9,\"position\":4.0,\"duration\":8.0,\"paused\":true}'; done\n",
        );
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let launch_script = script.clone();
        std::thread::spawn(move || {
            sender
                .send(NativeVideoRuntime::start(&launch_script).unwrap())
                .unwrap();
        })
        .join()
        .unwrap();

        let mut runtime = receiver.recv().unwrap();
        let notifications = runtime.take_notifications().unwrap();
        runtime
            .send(NativeVideoCommand::SetPaused { paused: true })
            .unwrap();
        match notifications.recv_timeout(Duration::from_secs(2)).unwrap() {
            RuntimeNotification::Control(NativeVideoEvent::PlaybackState {
                generation,
                position,
                ..
            }) => {
                assert_eq!(generation, 9);
                assert_eq!(position, 4.0);
            }
            other => panic!("unexpected runtime notification: {other:?}"),
        }
        runtime.stop();
        let _ = fs::remove_file(script);
    }
}
