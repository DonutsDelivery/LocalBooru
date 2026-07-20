//! Sidecar Process Management Helpers
//!
//! Low-level utilities for Python virtual environment creation,
//! dependency installation, process spawning, and health checking.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command as TokioCommand;

/// Find a usable `python3` (or `python`) binary on the system PATH.
///
/// Checks `python3` first, then falls back to `python`, verifying each
/// is actually executable before returning it.
pub fn find_python() -> Option<PathBuf> {
    // Prefer python3 over python to avoid accidentally picking Python 2
    let candidates = if cfg!(target_os = "windows") {
        vec!["python3", "python"]
    } else {
        vec!["python3", "python"]
    };

    for name in candidates {
        let check = Command::new(name)
            .args(["--version"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();

        if let Ok(output) = check {
            if output.status.success() {
                let version_str = String::from_utf8_lossy(&output.stdout);
                let version_stderr = String::from_utf8_lossy(&output.stderr);
                // Python prints version to stdout (3.x) or stderr (2.x)
                let combined = format!("{}{}", version_str, version_stderr);
                if combined.contains("Python 3") {
                    // Return the resolved path via `which`/`where`
                    if let Some(resolved) = resolve_executable(name) {
                        return Some(resolved);
                    }
                    // If resolution failed, return the bare name and let the OS resolve it
                    return Some(PathBuf::from(name));
                }
            }
        }
    }

    None
}

fn parse_python_minor(version_output: &str) -> Option<u8> {
    let version = version_output.split_whitespace().find(|part| {
        part.chars()
            .next()
            .is_some_and(|character| character.is_ascii_digit())
    })?;
    let mut components = version.split('.');
    let major = components.next()?.parse::<u8>().ok()?;
    let minor = components.next()?.parse::<u8>().ok()?;
    (major == 3).then_some(minor)
}

pub fn validate_python_minor(python: &Path, minimum: u8, maximum: u8) -> Result<(), String> {
    let output = Command::new(python)
        .arg("--version")
        .output()
        .map_err(|error| format!("Failed to inspect Python version: {}", error))?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let minor = parse_python_minor(&combined)
        .ok_or_else(|| format!("Could not parse Python version from {:?}", combined.trim()))?;
    if (minimum..=maximum).contains(&minor) {
        Ok(())
    } else {
        Err(format!(
            "Auto Tagger requires Python 3.{} through 3.{}, but {} reports {}",
            minimum,
            maximum,
            python.display(),
            combined.trim()
        ))
    }
}

/// Resolve an executable name to its full path using `which` (Unix) or `where` (Windows).
fn resolve_executable(name: &str) -> Option<PathBuf> {
    let cmd = if cfg!(target_os = "windows") {
        "where"
    } else {
        "which"
    };

    Command::new(cmd)
        .arg(name)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout);
                let first_line = path_str.lines().next()?.trim();
                if !first_line.is_empty() {
                    Some(PathBuf::from(first_line))
                } else {
                    None
                }
            } else {
                None
            }
        })
}

/// Create a Python virtual environment at the given directory.
///
/// Runs: `python3 -m venv {venv_dir}`
pub fn create_venv(python: &Path, venv_dir: &Path) -> Result<(), String> {
    log::info!(
        "[Addon] Creating venv at {} using {}",
        venv_dir.display(),
        python.display()
    );

    let output = Command::new(python)
        .args(["-m", "venv", &venv_dir.to_string_lossy()])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run python -m venv: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("venv creation failed: {}", stderr.trim()));
    }

    log::info!(
        "[Addon] venv created successfully at {}",
        venv_dir.display()
    );
    Ok(())
}

/// Install Python dependencies into a virtual environment.
///
/// Runs: `{venv_python} -m pip install {deps...}`
/// Skips the step entirely if `deps` is empty.
pub fn install_deps(venv_dir: &Path, deps: &[&str]) -> Result<(), String> {
    if deps.is_empty() {
        log::info!("[Addon] No Python dependencies to install");
        return Ok(());
    }

    let python = get_venv_python(venv_dir);
    log::info!(
        "[Addon] Installing deps via {}: {:?}",
        python.display(),
        deps
    );

    let mut args = vec!["-m", "pip", "install", "--upgrade"];
    args.extend(deps.iter().copied());

    let output = Command::new(&python)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run pip install: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pip install failed: {}", stderr.trim()));
    }

    log::info!("[Addon] Dependencies installed successfully");
    Ok(())
}

/// Remove Python distributions from a virtual environment if present.
pub fn uninstall_deps(venv_dir: &Path, deps: &[&str]) -> Result<(), String> {
    if deps.is_empty() {
        return Ok(());
    }

    let python = get_venv_python(venv_dir);
    let mut args = vec!["-m", "pip", "uninstall", "-y"];
    args.extend(deps.iter().copied());
    let output = Command::new(&python)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to run pip uninstall: {}", error))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "pip uninstall failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

/// Probe the installed ONNX Runtime distribution through the managed Python.
pub fn probe_onnxruntime(
    venv_dir: &Path,
    expected_distribution: &str,
    expected_version: &str,
) -> Result<serde_json::Value, String> {
    let python = get_venv_python(venv_dir);
    let script = r#"import importlib.metadata as m, json, onnxruntime as ort
names = ['onnxruntime', 'onnxruntime-gpu', 'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12']
packages = {}
for name in names:
    try:
        packages[name] = m.version(name)
    except m.PackageNotFoundError:
        pass
preload = {'attempted': False, 'succeeded': None, 'error': None}
if hasattr(ort, 'preload_dlls'):
    preload['attempted'] = True
    try:
        ort.preload_dlls(directory='')
        preload['succeeded'] = True
    except Exception as error:
        preload['succeeded'] = False
        preload['error'] = str(error)
print(json.dumps({'onnxruntime': ort.__version__, 'available_providers': ort.get_available_providers(), 'packages': packages, 'preload': preload}))"#;
    let output = Command::new(&python)
        .args(["-c", script])
        .output()
        .map_err(|error| format!("Failed to probe ONNX Runtime: {}", error))?;
    if !output.status.success() {
        return Err(format!(
            "ONNX Runtime probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let probe: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("ONNX Runtime probe returned invalid JSON: {}", error))?;
    let packages = probe
        .get("packages")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "ONNX Runtime probe omitted package versions".to_string())?;
    if probe.get("onnxruntime").and_then(serde_json::Value::as_str) != Some(expected_version) {
        return Err(format!(
            "ONNX Runtime probe imported version {:?}, expected {}",
            probe.get("onnxruntime"),
            expected_version
        ));
    }
    if packages
        .get(expected_distribution)
        .and_then(serde_json::Value::as_str)
        != Some(expected_version)
    {
        return Err(format!(
            "ONNX Runtime probe expected {} {}, got {:?}",
            expected_distribution,
            expected_version,
            packages.get(expected_distribution)
        ));
    }
    if expected_distribution == "onnxruntime" && packages.contains_key("onnxruntime-gpu") {
        return Err("ONNX Runtime CPU probe found a conflicting GPU distribution".to_string());
    }
    if expected_distribution == "onnxruntime-gpu" {
        if packages.contains_key("onnxruntime") {
            return Err("ONNX Runtime GPU probe found a conflicting CPU distribution".to_string());
        }
        let preload_failed = probe
            .get("preload")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|preload| {
                preload
                    .get("attempted")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                    && preload
                        .get("succeeded")
                        .and_then(serde_json::Value::as_bool)
                        == Some(false)
            });
        if preload_failed {
            return Err(format!(
                "ONNX Runtime GPU preload failed: {}",
                probe
                    .get("preload")
                    .and_then(|preload| preload.get("error"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown native library error")
            ));
        }
        for package in [
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
        ] {
            if !packages.contains_key(package) {
                return Err(format!("ONNX Runtime GPU probe is missing {}", package));
            }
        }
        let cuda_available = probe
            .get("available_providers")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|providers| {
                providers
                    .iter()
                    .any(|provider| provider.as_str() == Some("CUDAExecutionProvider"))
            });
        if !cuda_available {
            return Err("ONNX Runtime GPU probe could not load CUDAExecutionProvider".to_string());
        }
    }
    Ok(probe)
}

/// Get the path to the Python binary inside a virtual environment.
///
/// Returns `{venv}/bin/python` on Unix or `{venv}\Scripts\python.exe` on Windows.
pub fn get_venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

const MAX_OUTPUT_LINE_BYTES: usize = 64 * 1024;

async fn drain_output<R, F>(mut reader: R, mut emit: F) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    F: FnMut(String),
{
    let mut pending = Vec::with_capacity(8 * 1024);
    let mut chunk = [0_u8; 8 * 1024];
    let mut line_was_split = false;
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            if !pending.is_empty() {
                emit(String::from_utf8_lossy(&pending).into_owned());
            }
            return Ok(());
        }

        for &byte in &chunk[..count] {
            if byte == b'\n' {
                while pending.last() == Some(&b'\r') {
                    pending.pop();
                }
                if !pending.is_empty() || !line_was_split {
                    emit(String::from_utf8_lossy(&pending).into_owned());
                }
                pending.clear();
                line_was_split = false;
            } else {
                pending.push(byte);
                if pending.len() == MAX_OUTPUT_LINE_BYTES {
                    emit(String::from_utf8_lossy(&pending).into_owned());
                    pending.clear();
                    line_was_split = true;
                }
            }
        }
    }
}

fn format_output_line(addon_id: &str, stream: &str, line: &str) -> String {
    format!("[Addon:{}][{}] {}", addon_id, stream, line)
}

fn spawn_output_logger<R>(reader: R, addon_id: String, stream: &'static str)
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let result = drain_output(reader, |line| {
            log::info!("{}", format_output_line(&addon_id, stream, &line));
        })
        .await;
        if let Err(error) = result {
            log::warn!(
                "[Addon:{}][{}] Failed to read sidecar output: {}",
                addon_id,
                stream,
                error
            );
        }
    });
}

/// Spawn an addon sidecar process running a uvicorn FastAPI app.
///
/// Runs: `{venv_python} -m uvicorn app:app --port {port} --host 127.0.0.1`
/// with the working directory set to `app_dir`.
///
/// The child process is returned so the caller can track and kill it.
pub async fn spawn_sidecar(
    addon_id: &str,
    python: &Path,
    app_dir: &Path,
    port: u16,
    envs: &[(String, String)],
) -> Result<tokio::process::Child, String> {
    log::info!(
        "[Addon] Spawning sidecar: {} -m uvicorn app:app --port {} (cwd: {})",
        python.display(),
        port,
        app_dir.display()
    );

    let mut cmd = TokioCommand::new(python);
    cmd.args([
        "-m",
        "uvicorn",
        "app:app",
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
    ])
    .current_dir(app_dir)
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped())
    .env("PYTHONUNBUFFERED", "1");

    for (key, value) in envs {
        cmd.env(key, value);
    }

    // Unix: create a new process group for clean shutdown
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    // Windows: hide console window
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn addon sidecar: {}", e))?;

    if let Some(stdout) = child.stdout.take() {
        spawn_output_logger(stdout, addon_id.to_owned(), "stdout");
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_output_logger(stderr, addon_id.to_owned(), "stderr");
    }

    log::info!("[Addon] Sidecar spawned with PID {:?}", child.id());
    Ok(child)
}

#[derive(Debug)]
pub struct ManagedCommandOutput {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ManagedCommandError {
    Spawn(String),
    TimedOut(Duration),
    Wait(String),
}

struct ManagedProcessGuard {
    pid: Option<u32>,
    armed: bool,
}

impl Drop for ManagedProcessGuard {
    fn drop(&mut self) {
        if self.armed {
            if let Some(pid) = self.pid {
                kill_process(pid);
            }
        }
    }
}

impl fmt::Display for ManagedCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(error) => write!(formatter, "failed to spawn managed command: {}", error),
            Self::TimedOut(timeout) => {
                write!(
                    formatter,
                    "managed command timed out after {:.1} seconds",
                    timeout.as_secs_f64()
                )
            }
            Self::Wait(error) => write!(formatter, "failed to wait for managed command: {}", error),
        }
    }
}

async fn read_bounded_output<R>(mut reader: R, maximum: usize) -> std::io::Result<String>
where
    R: AsyncRead + Unpin,
{
    let mut output = Vec::with_capacity(maximum.min(8 * 1024));
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = reader.read(&mut chunk).await?;
        if count == 0 {
            break;
        }
        let remaining = maximum.saturating_sub(output.len());
        output.extend_from_slice(&chunk[..count.min(remaining)]);
    }
    Ok(String::from_utf8_lossy(&output).into_owned())
}

fn abort_managed_output(
    stdout: &mut Option<tokio::task::JoinHandle<std::io::Result<String>>>,
    stderr: &mut Option<tokio::task::JoinHandle<std::io::Result<String>>>,
) {
    for task in [stdout, stderr].into_iter().flatten() {
        task.abort();
    }
}

async fn terminate_managed_child(
    child: &mut tokio::process::Child,
    pid: Option<u32>,
    stdout: &mut Option<tokio::task::JoinHandle<std::io::Result<String>>>,
    stderr: &mut Option<tokio::task::JoinHandle<std::io::Result<String>>>,
) {
    if let Some(pid) = pid {
        let _ = tokio::time::timeout(
            Duration::from_secs(2),
            tokio::task::spawn_blocking(move || kill_process(pid)),
        )
        .await;
    }
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
    abort_managed_output(stdout, stderr);
}

/// Run a managed child under a hard parent-owned deadline.
///
/// Output is drained concurrently and capped so a noisy or malformed probe cannot
/// deadlock or consume unbounded memory. A timeout kills the process group/tree and
/// waits for the direct child to be reaped before returning.
pub async fn run_managed_command(
    executable: &Path,
    args: &[String],
    working_directory: &Path,
    timeout: Duration,
    maximum_output_bytes: usize,
) -> Result<ManagedCommandOutput, ManagedCommandError> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut command = TokioCommand::new(executable);
    command
        .args(args)
        .current_dir(working_directory)
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    #[cfg(unix)]
    command.process_group(0);

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }

    let mut child = command
        .spawn()
        .map_err(|error| ManagedCommandError::Spawn(error.to_string()))?;
    let pid = child.id();
    let mut process_guard = ManagedProcessGuard { pid, armed: true };
    let mut stdout = child
        .stdout
        .take()
        .map(|reader| tokio::spawn(read_bounded_output(reader, maximum_output_bytes)));
    let mut stderr = child
        .stderr
        .take()
        .map(|reader| tokio::spawn(read_bounded_output(reader, maximum_output_bytes)));

    let status = match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            terminate_managed_child(&mut child, pid, &mut stdout, &mut stderr).await;
            process_guard.armed = false;
            return Err(ManagedCommandError::Wait(error.to_string()));
        }
        Err(_) => {
            terminate_managed_child(&mut child, pid, &mut stdout, &mut stderr).await;
            process_guard.armed = false;
            return Err(ManagedCommandError::TimedOut(timeout));
        }
    };
    if let Some(pid) = pid {
        kill_process(pid);
    }
    process_guard.armed = false;

    async fn collect(
        task: &mut Option<tokio::task::JoinHandle<std::io::Result<String>>>,
    ) -> Result<String, ManagedCommandError> {
        match task {
            None => Ok(String::new()),
            Some(task) => (&mut *task)
                .await
                .map_err(|error| ManagedCommandError::Wait(error.to_string()))?
                .map_err(|error| ManagedCommandError::Wait(error.to_string())),
        }
    }

    let collected = tokio::time::timeout_at(deadline, async {
        tokio::try_join!(collect(&mut stdout), collect(&mut stderr))
    })
    .await;
    let (stdout_value, stderr_value) = match collected {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            abort_managed_output(&mut stdout, &mut stderr);
            return Err(error);
        }
        Err(_) => {
            abort_managed_output(&mut stdout, &mut stderr);
            return Err(ManagedCommandError::TimedOut(timeout));
        }
    };

    Ok(ManagedCommandOutput {
        success: status.success(),
        exit_code: status.code(),
        stdout: stdout_value,
        stderr: stderr_value,
    })
}

/// Perform a single health check against an addon's `/health` endpoint.
///
/// Returns `true` if the server responds with HTTP 200, `false` otherwise.
pub async fn check_health(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{}/health", port);

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };

    match client.get(&url).send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

/// Poll an addon's health endpoint until it responds or the timeout expires.
///
/// Checks every 250ms. Returns `true` if the addon became healthy within
/// the given `timeout`, `false` if it timed out.
pub async fn wait_for_healthy(port: u16, timeout: Duration) -> bool {
    let start = std::time::Instant::now();
    let poll_interval = Duration::from_millis(250);

    while start.elapsed() < timeout {
        if check_health(port).await {
            log::info!("[Addon] Port {} healthy after {:?}", port, start.elapsed());
            return true;
        }
        tokio::time::sleep(poll_interval).await;
    }

    log::warn!(
        "[Addon] Port {} did not become healthy within {:?}",
        port,
        timeout
    );
    false
}

/// Kill a sidecar process and its process group.
pub fn kill_process(pid: u32) {
    #[cfg(unix)]
    {
        // SAFETY: kill accepts process-group IDs as negative PIDs. The child was
        // spawned into a fresh group whose ID is its PID.
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }

    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/pid", &pid.to_string(), "/T", "/F"])
            .output();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::AsyncWriteExt;

    #[test]
    fn auto_tagger_python_version_range_is_bounded() {
        assert_eq!(parse_python_minor("Python 3.10.14"), Some(10));
        assert_eq!(parse_python_minor("Python 3.13.1"), Some(13));
        assert_eq!(parse_python_minor("Python 3.9.20"), Some(9));
        assert_eq!(parse_python_minor("Python 3.14.0"), Some(14));
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-2
    // AC: @auto-tagger-runtime-acceleration-deployment ac-4
    #[cfg(target_os = "linux")]
    #[test]
    fn gpu_probe_rejects_installed_wheels_when_cuda_provider_cannot_load() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-provider-probe-test-{}",
            uuid::Uuid::new_v4()
        ));
        let bin = root.join("bin");
        let python = bin.join("python");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            &python,
            r#"#!/bin/sh
echo '{"onnxruntime":"1.23.2","available_providers":["CPUExecutionProvider"],"packages":{"onnxruntime-gpu":"1.23.2","nvidia-cublas-cu12":"12.9","nvidia-cuda-runtime-cu12":"12.9","nvidia-cudnn-cu12":"9.24"}}'
"#,
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();

        let error = probe_onnxruntime(&root, "onnxruntime-gpu", "1.23.2").unwrap_err();

        assert!(error.contains("could not load CUDAExecutionProvider"));
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-4
    #[cfg(target_os = "linux")]
    #[test]
    fn gpu_probe_rejects_conflicting_cpu_distribution() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-conflicting-ort-test-{}",
            uuid::Uuid::new_v4()
        ));
        let bin = root.join("bin");
        let python = bin.join("python");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            &python,
            r#"#!/bin/sh
echo '{"onnxruntime":"1.23.2","available_providers":["CUDAExecutionProvider","CPUExecutionProvider"],"packages":{"onnxruntime":"1.23.2","onnxruntime-gpu":"1.23.2","nvidia-cublas-cu12":"12.9","nvidia-cuda-runtime-cu12":"12.9","nvidia-cudnn-cu12":"9.24"}}'
"#,
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();

        let error = probe_onnxruntime(&root, "onnxruntime-gpu", "1.23.2").unwrap_err();

        assert!(error.contains("conflicting CPU distribution"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_command_closes_inherited_output_pipes_after_parent_exit() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-managed-command-timeout-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("probe");
        std::fs::write(
            &script,
            r#"#!/bin/sh
sleep 30 &
exit 0
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let started = std::time::Instant::now();

        let result =
            run_managed_command(&script, &[], &root, Duration::from_millis(100), 1024).await;

        assert!(result.unwrap().success);
        assert!(started.elapsed() < Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_command_kills_background_descendants_after_success() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-managed-command-descendant-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("probe");
        std::fs::write(
            &script,
            r#"#!/bin/sh
sleep 30 >/dev/null 2>&1 &
printf '%s' "$!" > descendant.pid
printf '%s\n' '{"ready":true}'
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let result = run_managed_command(&script, &[], &root, Duration::from_secs(2), 1024)
            .await
            .unwrap();
        let descendant: i32 = std::fs::read_to_string(root.join("descendant.pid"))
            .unwrap()
            .parse()
            .unwrap();
        for _ in 0..20 {
            // SAFETY: signal zero only tests whether the recorded child PID still exists.
            if unsafe { libc::kill(descendant, 0) } != 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        assert!(result.success);
        // SAFETY: signal zero does not modify the process.
        assert_ne!(unsafe { libc::kill(descendant, 0) }, 0);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_command_cancellation_kills_process_group() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-managed-command-cancel-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("probe");
        std::fs::write(
            &script,
            r#"#!/bin/sh
printf '%s' "$$" > process.pid
sleep 30
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let task_root = root.clone();
        let task_script = script.clone();
        let task = tokio::spawn(async move {
            run_managed_command(&task_script, &[], &task_root, Duration::from_secs(30), 1024).await
        });
        let pid_path = root.join("process.pid");
        for _ in 0..40 {
            if pid_path.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let pid: i32 = std::fs::read_to_string(&pid_path).unwrap().parse().unwrap();

        task.abort();
        let _ = task.await;
        for _ in 0..20 {
            // SAFETY: signal zero only tests whether the recorded PID still exists.
            if unsafe { libc::kill(pid, 0) } != 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        // SAFETY: signal zero does not modify the process.
        assert_ne!(unsafe { libc::kill(pid, 0) }, 0);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @sidecar-diagnostics ac-1
    // AC: @sidecar-diagnostics ac-3
    #[tokio::test]
    async fn output_drain_preserves_lines_and_decodes_invalid_utf8_lossily() {
        assert_eq!(
            format_output_line("auto-tagger", "stderr", "CUDA unavailable"),
            "[Addon:auto-tagger][stderr] CUDA unavailable"
        );

        let (mut writer, reader) = tokio::io::duplex(64);
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = lines.clone();
        let drain = tokio::spawn(async move {
            drain_output(reader, |line| captured.lock().unwrap().push(line))
                .await
                .unwrap();
        });

        writer.write_all(b"first\ninvalid-\xff\r\n").await.unwrap();
        drop(writer);
        drain.await.unwrap();

        assert_eq!(
            *lines.lock().unwrap(),
            vec!["first".to_string(), "invalid-�".to_string()]
        );
    }

    // AC: @sidecar-diagnostics ac-2
    // AC: @sidecar-diagnostics ac-3
    #[tokio::test]
    async fn unterminated_output_is_split_into_bounded_log_lines() {
        let (mut writer, reader) = tokio::io::duplex(64);
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = lines.clone();
        let drain = tokio::spawn(async move {
            drain_output(reader, |line| captured.lock().unwrap().push(line))
                .await
                .unwrap();
        });
        let payload = vec![b'x'; MAX_OUTPUT_LINE_BYTES * 2 + 17];

        writer.write_all(&payload).await.unwrap();
        drop(writer);
        drain.await.unwrap();

        let lines = lines.lock().unwrap();
        assert_eq!(lines.iter().map(String::len).sum::<usize>(), payload.len());
        assert!(lines.iter().all(|line| line.len() <= MAX_OUTPUT_LINE_BYTES));
        assert_eq!(lines.len(), 3);
    }

    // AC: @sidecar-diagnostics ac-2
    #[tokio::test]
    async fn stdout_and_stderr_can_drain_concurrently_past_pipe_capacity() {
        let (mut stdout_writer, stdout_reader) = tokio::io::duplex(64);
        let (mut stderr_writer, stderr_reader) = tokio::io::duplex(64);
        let stdout_drain = tokio::spawn(drain_output(stdout_reader, |_| {}));
        let stderr_drain = tokio::spawn(drain_output(stderr_reader, |_| {}));
        let payload = vec![b'x'; 128 * 1024];
        let stderr_payload = payload.clone();

        let stdout_write = tokio::spawn(async move {
            stdout_writer.write_all(&payload).await.unwrap();
            stdout_writer.write_all(b"\n").await.unwrap();
        });
        let stderr_write = tokio::spawn(async move {
            stderr_writer.write_all(&stderr_payload).await.unwrap();
            stderr_writer.write_all(b"\n").await.unwrap();
        });

        tokio::time::timeout(Duration::from_secs(2), async {
            stdout_write.await.unwrap();
            stderr_write.await.unwrap();
            stdout_drain.await.unwrap().unwrap();
            stderr_drain.await.unwrap().unwrap();
        })
        .await
        .expect("both streams should drain without blocking");
    }
}
