//! Sidecar Process Management Helpers
//!
//! Low-level utilities for Python virtual environment creation,
//! dependency installation, process spawning, and health checking.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

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

/// Resolve an executable name to its full path using `which` (Unix) or `where` (Windows).
fn resolve_executable(name: &str) -> Option<PathBuf> {
    let cmd = if cfg!(target_os = "windows") { "where" } else { "which" };

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

    log::info!("[Addon] venv created successfully at {}", venv_dir.display());
    Ok(())
}

/// Install Python dependencies into a virtual environment.
///
/// Runs: `{venv}/bin/pip install {deps...}`
/// Skips the step entirely if `deps` is empty.
pub fn install_deps(venv_dir: &Path, deps: &[&str]) -> Result<(), String> {
    if deps.is_empty() {
        log::info!("[Addon] No Python dependencies to install");
        return Ok(());
    }

    let pip = get_venv_pip(venv_dir);
    log::info!(
        "[Addon] Installing deps via {}: {:?}",
        pip.display(),
        deps
    );

    let mut args = vec!["install", "--upgrade"];
    args.extend(deps.iter().copied());

    let output = Command::new(&pip)
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

/// Get the path to the pip binary inside a virtual environment.
fn get_venv_pip(venv_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("pip.exe")
    } else {
        venv_dir.join("bin").join("pip")
    }
}

/// Spawn an addon sidecar process running a uvicorn FastAPI app.
///
/// Runs: `{venv_python} -m uvicorn app:app --port {port} --host 127.0.0.1`
/// with the working directory set to `app_dir`.
///
/// The child process is returned so the caller can track and kill it.
pub async fn spawn_sidecar(
    python: &Path,
    app_dir: &Path,
    port: u16,
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

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn addon sidecar: {}", e))?;

    log::info!("[Addon] Sidecar spawned with PID {:?}", child.id());
    Ok(child)
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
        // Kill the entire process group (negative PID)
        let _ = Command::new("kill")
            .args(["-9", &format!("-{}", pid)])
            .output();
        // Also kill the specific PID in case it was not in a group
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }

    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/pid", &pid.to_string(), "/T", "/F"])
            .output();
    }
}
