//! Linux recovery of a legacy listener belonging to this installed add-on.
//! Never adopt health alone or signal a PID discovered without installation proof.

use std::collections::HashSet;
use std::fs;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    pid: u32,
    parent: u32,
    group: u32,
    started: u64,
}

fn identity(pid: u32) -> Result<Identity, String> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))
        .map_err(|_| "Listener exited during verification")?;
    let fields: Vec<_> = stat
        .rsplit_once(") ")
        .ok_or("Invalid process identity")?
        .1
        .split_whitespace()
        .collect();
    Ok(Identity {
        pid,
        parent: fields
            .get(1)
            .ok_or("Missing parent identity")?
            .parse()
            .map_err(|_| "Invalid parent identity")?,
        group: fields
            .get(2)
            .ok_or("Missing process group")?
            .parse()
            .map_err(|_| "Invalid process group")?,
        started: fields
            .get(19)
            .ok_or("Missing process start identity")?
            .parse()
            .map_err(|_| "Invalid process start identity")?,
    })
}

fn listener_pid(port: u16) -> Result<u32, String> {
    let mut inodes = HashSet::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        for row in fs::read_to_string(table)
            .map_err(|_| "Cannot inspect listening sockets")?
            .lines()
            .skip(1)
        {
            let columns: Vec<_> = row.split_whitespace().collect();
            if columns.get(3) != Some(&"0A") {
                continue;
            }
            let bound_port = columns
                .get(1)
                .and_then(|address| address.rsplit_once(':'))
                .and_then(|(_, p)| u16::from_str_radix(p, 16).ok());
            if bound_port == Some(port) {
                if let Some(inode) = columns.get(9) {
                    inodes.insert(format!("socket:[{inode}]"));
                }
            }
        }
    }
    let mut owners = HashSet::new();
    for process in fs::read_dir("/proc")
        .map_err(|_| "Cannot inspect listener ownership")?
        .flatten()
    {
        let Some(pid) = process
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(descriptors) = fs::read_dir(process.path().join("fd")) else {
            continue;
        };
        for descriptor in descriptors.flatten() {
            if let Ok(target) = fs::read_link(descriptor.path()) {
                if inodes.contains(target.to_string_lossy().as_ref()) {
                    owners.insert(pid);
                }
            }
        }
    }
    if owners.len() != 1 {
        return Err("Listener ownership is missing or ambiguous; refusing recovery".into());
    }
    Ok(*owners.iter().next().unwrap())
}

// Init and the user's systemd subreaper can make their executable links
// unreadable. Identify PID1 structurally; identify user systemd additionally
// through the kernel credentials of its private socket, never comm alone.
fn is_system_reaper(pid: u32) -> bool {
    if pid == 1 {
        return fs::metadata("/proc/1").is_ok_and(|metadata| metadata.uid() == 0)
            && identity(1).is_ok_and(|process| process.parent == 0)
            && fs::read_to_string("/proc/1/comm")
                .is_ok_and(|name| matches!(name.trim(), "systemd" | "init"));
    }
    let uid = unsafe { libc::geteuid() };
    if !is_system_reaper(1)
        || !fs::metadata(format!("/proc/{pid}")).is_ok_and(|metadata| metadata.uid() == uid)
        || !identity(pid).is_ok_and(|process| process.parent == 1)
        || !fs::read_to_string(format!("/proc/{pid}/comm"))
            .is_ok_and(|name| name.trim() == "systemd")
    {
        return false;
    }
    // Nonblocking connect fails closed if systemd's accept queue is full. Send
    // no authentication or commands: SO_PEERCRED is supplied by the kernel.
    let raw = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if raw < 0 {
        return false;
    }
    let socket = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let path = format!("/run/user/{uid}/systemd/private");
    if path.len() >= address.sun_path.len() {
        return false;
    }
    for (target, byte) in address.sun_path.iter_mut().zip(path.bytes()) {
        *target = byte as libc::c_char;
    }
    if unsafe {
        libc::connect(
            socket.as_raw_fd(),
            &address as *const _ as *const libc::sockaddr,
            std::mem::size_of_val(&address) as libc::socklen_t,
        )
    } != 0
    {
        return false;
    }
    let mut peer: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of_val(&peer) as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            socket.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut peer as *mut _ as *mut libc::c_void,
            &mut size,
        )
    };
    result == 0
        && size as usize == std::mem::size_of_val(&peer)
        && peer.pid == pid as libc::pid_t
        && peer.uid == uid
}

fn verify(
    pid: u32,
    python: &Path,
    directory: &Path,
    port: u16,
    app_exe: &Path,
) -> Result<Identity, String> {
    let root = PathBuf::from(format!("/proc/{pid}"));
    if fs::metadata(&root)
        .map_err(|_| "Listener exited during verification")?
        .uid()
        != unsafe { libc::geteuid() }
    {
        return Err("Listener belongs to another user".into());
    }
    let before = identity(pid)?;
    if before.parent == std::process::id() {
        return Err("Listener already belongs to this running application".into());
    }
    if before.group != pid {
        return Err("Listener is not an isolated installed add-on process".into());
    }
    let parent_exe = fs::read_link(format!("/proc/{}/exe", before.parent));
    if parent_exe.as_deref().is_ok_and(|exe| {
        let bytes = exe.as_os_str().as_bytes();
        let clean = Path::new(std::ffi::OsStr::from_bytes(
            bytes.strip_suffix(b" (deleted)").unwrap_or(bytes),
        ));
        clean == app_exe || clean.file_name() == app_exe.file_name()
    }) {
        return Err("Listener belongs to another running DMC instance".into());
    }
    // An inaccessible live parent cannot be safely identified as a reaper.
    if parent_exe.is_err()
        && Path::new(&format!("/proc/{}", before.parent)).exists()
        && !is_system_reaper(before.parent)
    {
        return Err("Cannot verify the listener's live parent".into());
    }
    let expected_python = fs::canonicalize(python).map_err(|_| "Cannot verify installed Python")?;
    let expected_directory =
        fs::canonicalize(directory).map_err(|_| "Cannot verify add-on directory")?;
    if fs::read_link(root.join("exe")).map_err(|_| "Cannot verify listener executable")?
        != expected_python
        || fs::read_link(root.join("cwd")).map_err(|_| "Cannot verify listener directory")?
            != expected_directory
    {
        return Err("Listener executable or directory does not match this installed add-on".into());
    }
    let command = fs::read(root.join("cmdline")).map_err(|_| "Cannot verify listener command")?;
    let args: Vec<_> = command
        .split(|byte| *byte == 0)
        .filter(|arg| !arg.is_empty())
        .collect();
    let port_arg = port.to_string();
    let expected = [
        b"-m".as_slice(),
        b"uvicorn",
        b"app:app",
        b"--port",
        port_arg.as_bytes(),
        b"--host",
        b"127.0.0.1",
    ];
    if args.len() != 8
        || args[1..] != expected
        || fs::canonicalize(Path::new(std::ffi::OsStr::from_bytes(args[0])))
            .ok()
            .as_ref()
            != Some(&expected_python)
    {
        return Err("Listener command does not match this installed add-on".into());
    }
    if identity(pid)? != before {
        return Err("Listener identity changed during verification".into());
    }
    Ok(before)
}

fn open_verified_process(owner: &Identity) -> Result<OwnedFd, String> {
    let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, owner.pid, 0) };
    if descriptor < 0 {
        return Err("Cannot safely acquire listener process handle".into());
    }
    let handle = unsafe { OwnedFd::from_raw_fd(descriptor as i32) };
    if identity(owner.pid)? != *owner {
        return Err("Listener identity changed before recovery".into());
    }
    Ok(handle)
}

fn exited(handle: &OwnedFd) -> Result<bool, String> {
    let mut descriptor = libc::pollfd {
        fd: handle.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let result = unsafe { libc::poll(&mut descriptor, 1, 0) };
    if result < 0 {
        return Err("Cannot observe recovered listener exit".into());
    }
    Ok(result > 0)
}

fn signal(handle: &OwnedFd, signal: i32) -> Result<(), String> {
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            handle.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            0,
        )
    };
    if result < 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
        return Err("Cannot stop the verified installed listener".into());
    }
    Ok(())
}

fn port_free(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(100))
        .is_err_and(|error| error.kind() == std::io::ErrorKind::ConnectionRefused)
}

fn recover(
    python: &Path,
    directory: &Path,
    port: u16,
    app_exe: &Path,
    grace: Duration,
) -> Result<(), String> {
    if port_free(port) {
        return Ok(());
    }
    let pid = listener_pid(port)?;
    let owner = verify(pid, python, directory, port, app_exe)?;
    let handle = open_verified_process(&owner)?;
    // Start identity protects against PID reuse, but not exec/chdir within the
    // same process. Recheck the installation tuple after acquiring the pidfd.
    if verify(pid, python, directory, port, app_exe)? != owner {
        return Err("Listener installation changed before recovery".into());
    }
    signal(&handle, libc::SIGTERM)?;
    let deadline = Instant::now() + grace;
    while !exited(&handle)? && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(25));
    }
    if !exited(&handle)? {
        signal(&handle, libc::SIGKILL)?;
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if exited(&handle)? && port_free(port) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("Verified listener stopped but its port did not become available".into());
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

pub async fn recover_installed_listener(
    python: &Path,
    directory: &Path,
    port: u16,
) -> Result<(), String> {
    let python = python.to_owned();
    let directory = directory.to_owned();
    let app_exe =
        std::env::current_exe().map_err(|_| "Cannot verify current application executable")?;
    tokio::task::spawn_blocking(move || {
        recover(&python, &directory, port, &app_exe, Duration::from_secs(15))
    })
    .await
    .map_err(|_| "Installed listener recovery task failed")?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::process::{Child, Command, Stdio};

    struct Fixture {
        directory: tempfile::TempDir,
        port: u16,
        pid: u32,
        handle: OwnedFd,
        backend: Option<OwnedFd>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(backend) = &self.backend {
                let _ = signal(backend, libc::SIGKILL);
            }
            let _ = signal(&self.handle, libc::SIGKILL);
        }
    }
    struct ParentGuard(Child);
    impl Drop for ParentGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn process_handle(pid: u32) -> OwnedFd {
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
        assert!(raw >= 0);
        unsafe { OwnedFd::from_raw_fd(raw as i32) }
    }

    fn wait_ready(mut fixture: Fixture) -> Fixture {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Ok(value) = fs::read_to_string(fixture.directory.path().join("backend.pid")) {
                if fixture.backend.is_none() {
                    if let Ok(pid) = value.parse::<u32>() {
                        fixture.backend = Some(process_handle(pid));
                    }
                }
            }
            if fs::read_to_string(fixture.directory.path().join("ready.pid"))
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                == Some(fixture.pid)
            {
                return fixture;
            }
            assert!(
                Instant::now() < deadline,
                "synthetic listener did not start"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn free_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }
    fn python() -> PathBuf {
        PathBuf::from("/usr/bin/python3")
    }
    fn fixture_directory() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("uvicorn")).unwrap();
        fs::write(directory.path().join("uvicorn/__init__.py"), "").unwrap();
        fs::write(
            directory.path().join("uvicorn/__main__.py"),
            r#"
import os, signal, socket, subprocess, sys, time
from pathlib import Path
port = int(sys.argv[sys.argv.index('--port') + 1])
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', port))
listener.listen()
backend = None
if os.environ.get('SYNTHETIC_BACKEND'):
    backend = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    Path('backend.pid').write_text(str(backend.pid))
def stop(*_):
    if backend is not None:
        backend.terminate()
        backend.wait(timeout=2)
    listener.close()
    Path('graceful.done').write_text('yes')
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if os.environ.get('IGNORE_TERM') else stop)
Path('ready.pid').write_text(str(os.getpid()))
while True: time.sleep(0.05)
"#,
        )
        .unwrap();
        directory
    }

    fn orphan(backend: bool, ignore_term: bool, wrong_host: bool) -> Fixture {
        let directory = fixture_directory();
        let port = free_port();
        let output = Command::new(python()).args(["-c", r#"
import os, subprocess, sys
child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app', '--port', sys.argv[1], '--host', sys.argv[2]],
                         cwd=os.getcwd(), start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, flush=True)
"#, &port.to_string(), if wrong_host { "0.0.0.0" } else { "127.0.0.1" }])
            .current_dir(directory.path()).env_clear().env("HOME", directory.path())
            .env("SYNTHETIC_BACKEND", if backend { "1" } else { "" })
            .env("IGNORE_TERM", if ignore_term { "1" } else { "" }).output().unwrap();
        assert!(output.status.success());
        let pid = String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        wait_ready(Fixture {
            directory,
            port,
            pid,
            handle: process_handle(pid),
            backend: None,
        })
    }

    // AC: @owned-addon-port-recovery ac-verified-recovery
    // AC: @owned-addon-port-recovery ac-foreign-safety
    #[tokio::test]
    async fn owned_listener_recovery_releases_port_and_preserves_foreign_service() {
        let owned = orphan(true, false, false);
        let foreign = orphan(false, false, false);
        let backend_fd = owned.backend.as_ref().unwrap();
        let parent = identity(owned.pid).unwrap().parent;
        if fs::read_link(format!("/proc/{parent}/exe")).is_err() {
            assert!(
                is_system_reaper(parent),
                "an unreadable init or user-systemd reaper must be positively identified"
            );
        }
        recover_installed_listener(&python(), owned.directory.path(), owned.port)
            .await
            .unwrap();
        assert!(exited(&owned.handle).unwrap());
        assert!(
            exited(backend_fd).unwrap(),
            "graceful controller shutdown must stop its own child"
        );
        assert!(owned.directory.path().join("graceful.done").exists());
        assert!(!exited(&foreign.handle).unwrap());
        assert!(!port_free(foreign.port));
        // The replacement is spawned by the real production helper, not adopted.
        let mut replacement = super::super::sidecar::spawn_sidecar(
            "synthetic",
            &python(),
            owned.directory.path(),
            owned.directory.path(),
            owned.port,
            &[],
        )
        .await
        .unwrap();
        assert_ne!(replacement.id(), Some(owned.pid));
        let deadline = Instant::now() + Duration::from_secs(3);
        while port_free(owned.port) {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        replacement.kill().await.unwrap();
        replacement.wait().await.unwrap();
    }

    // AC: @owned-addon-port-recovery ac-foreign-safety
    #[test]
    fn wrong_installation_or_command_is_not_signalled() {
        let listener = orphan(false, false, false);
        let other = tempfile::tempdir().unwrap();
        let app_exe = std::env::current_exe().unwrap();
        assert!(recover(
            &python(),
            other.path(),
            listener.port,
            &app_exe,
            Duration::ZERO
        )
        .is_err());
        assert!(!exited(&listener.handle).unwrap());
        let wrong_command = orphan(false, false, true);
        assert!(recover(
            &python(),
            wrong_command.directory.path(),
            wrong_command.port,
            &app_exe,
            Duration::ZERO
        )
        .is_err());
        assert!(!exited(&wrong_command.handle).unwrap());
    }

    // AC: @owned-addon-port-recovery ac-foreign-safety
    #[tokio::test]
    async fn running_current_application_listener_is_not_reclaimed() {
        let directory = fixture_directory();
        let port = free_port();
        let mut child = super::super::sidecar::spawn_sidecar(
            "synthetic",
            &python(),
            directory.path(),
            directory.path(),
            port,
            &[],
        )
        .await
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while port_free(port) {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            recover_installed_listener(&python(), directory.path(), port)
                .await
                .unwrap_err()
                .contains("this running application")
        );
        assert!(child.try_wait().unwrap().is_none());
        child.kill().await.unwrap();
        child.wait().await.unwrap();
    }

    // AC: @owned-addon-port-recovery ac-foreign-safety
    #[test]
    fn changed_start_identity_is_rejected_before_signalling() {
        let listener = orphan(false, false, false);
        let mut expected = identity(listener.pid).unwrap();
        expected.started += 1;
        assert!(open_verified_process(&expected).is_err());
        assert!(!exited(&listener.handle).unwrap());
    }

    // AC: @owned-addon-port-recovery ac-foreign-safety
    #[test]
    fn other_live_application_owner_is_protected_across_install_paths() {
        let directory = fixture_directory();
        let port = free_port();
        let mut parent = ParentGuard(Command::new(python()).args(["-c", r#"
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app', '--port', sys.argv[1], '--host', '127.0.0.1'],
                         cwd=os.getcwd(), start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, flush=True)
time.sleep(60)
"#, &port.to_string()]).current_dir(directory.path()).env_clear().env("HOME", directory.path())
            .stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap());
        let mut pid_line = String::new();
        BufReader::new(parent.0.stdout.take().unwrap())
            .read_line(&mut pid_line)
            .unwrap();
        let pid = pid_line.trim().parse::<u32>().unwrap();
        let listener = wait_ready(Fixture {
            directory,
            port,
            pid,
            handle: process_handle(pid),
            backend: None,
        });
        // The Python fixture represents another application executable. Its
        // actual live /proc parent is matched by basename across install paths.
        let app_exe = Path::new("/another/installation")
            .join(fs::canonicalize(python()).unwrap().file_name().unwrap());
        let result = recover(
            &python(),
            listener.directory.path(),
            port,
            &app_exe,
            Duration::ZERO,
        );
        let alive = !exited(&listener.handle).unwrap();
        drop(parent);
        assert!(result.unwrap_err().contains("another running DMC"));
        assert!(alive);
    }

    // AC: @owned-addon-port-recovery ac-verified-recovery
    #[test]
    fn unresponsive_verified_controller_has_bounded_pid_scoped_escalation() {
        let listener = orphan(false, true, false);
        let start = Instant::now();
        recover(
            &python(),
            listener.directory.path(),
            listener.port,
            &std::env::current_exe().unwrap(),
            Duration::from_millis(100),
        )
        .unwrap();
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(exited(&listener.handle).unwrap());
        assert!(!listener.directory.path().join("graceful.done").exists());
    }

    // AC: @owned-addon-port-recovery ac-child-cleanup
    #[tokio::test]
    async fn actual_sidecar_handle_drop_stops_its_child_and_releases_port() {
        let directory = fixture_directory();
        let port = free_port();
        let child = super::super::sidecar::spawn_sidecar(
            "synthetic",
            &python(),
            directory.path(),
            directory.path(),
            port,
            &[],
        )
        .await
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while port_free(port) {
            assert!(Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        drop(child);
        while !port_free(port) {
            assert!(
                Instant::now() < deadline,
                "dropped production sidecar remains listening"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
