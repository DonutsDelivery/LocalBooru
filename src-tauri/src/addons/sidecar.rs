//! Sidecar Process Management Helpers
//!
//! Low-level utilities for Python virtual environment creation,
//! dependency installation, process spawning, and health checking.

#[cfg(any(target_os = "windows", test))]
use std::collections::HashSet;
#[cfg(target_os = "windows")]
use std::ffi::OsStr;
#[cfg(any(target_os = "windows", test))]
use std::ffi::OsString;
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

#[cfg(any(target_os = "windows", test))]
fn managed_nvidia_bin_dirs(venv_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let nvidia_dir = venv_dir.join("Lib").join("site-packages").join("nvidia");
    if !nvidia_dir.is_dir() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(&nvidia_dir)
        .map_err(|error| format!("Failed to read '{}': {}", nvidia_dir.display(), error))?;
    let mut directories = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|error| format!("Failed to read '{}': {}", nvidia_dir.display(), error))?;
        if entry
            .file_type()
            .map_err(|error| format!("Failed to inspect '{}': {}", entry.path().display(), error))?
            .is_dir()
        {
            let bin_dir = entry.path().join("bin");
            if bin_dir.is_dir() {
                directories.push(bin_dir);
            }
        }
    }
    directories.sort();
    Ok(directories)
}

#[cfg(any(target_os = "windows", test))]
#[derive(Eq, Hash, PartialEq)]
enum WindowsPathKey {
    Text(String),
    Native(OsString),
}

#[cfg(any(target_os = "windows", test))]
fn windows_path_key(path: &Path) -> WindowsPathKey {
    match path.to_str() {
        Some(value) => WindowsPathKey::Text(value.to_lowercase()),
        None => WindowsPathKey::Native(path.as_os_str().to_os_string()),
    }
}

#[cfg(any(target_os = "windows", test))]
fn deduplicate_windows_path_entries(entries: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    entries
        .into_iter()
        .filter(|entry| seen.insert(windows_path_key(entry)))
        .collect()
}

#[cfg(target_os = "windows")]
fn compose_windows_search_path(
    managed: &[PathBuf],
    inherited: Option<&OsStr>,
) -> Result<OsString, String> {
    if managed.is_empty() {
        return Ok(inherited.unwrap_or_default().to_os_string());
    }

    let entries = managed.iter().cloned().chain(
        inherited
            .into_iter()
            .flat_map(|value| std::env::split_paths(value)),
    );
    std::env::join_paths(deduplicate_windows_path_entries(entries))
        .map_err(|error| format!("Failed to compose managed Windows PATH: {}", error))
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
    _venv_dir: &Path,
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
        if cfg!(target_os = "windows") && key.eq_ignore_ascii_case("PATH") {
            continue;
        }
        cmd.env(key, value);
    }

    #[cfg(target_os = "windows")]
    {
        let inherited_path = envs
            .iter()
            .rev()
            .find(|(key, _)| key.eq_ignore_ascii_case("PATH"))
            .map(|(_, value)| OsString::from(value))
            .or_else(|| std::env::var_os("PATH"));
        let managed = match managed_nvidia_bin_dirs(_venv_dir) {
            Ok(directories) => directories,
            Err(error) => {
                log::warn!("[Addon:{}] {}", addon_id, error);
                Vec::new()
            }
        };
        if !managed.is_empty() || inherited_path.is_some() {
            match compose_windows_search_path(&managed, inherited_path.as_deref()) {
                Ok(path) => {
                    cmd.env("PATH", path);
                }
                Err(error) => {
                    log::warn!("[Addon:{}] {}", addon_id, error);
                    if let Some(path) = inherited_path {
                        cmd.env("PATH", path);
                    }
                }
            }
        }
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
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).output();
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

    // AC: @auto-tagger-runtime-acceleration-deployment ac-managed-native-library-search
    #[test]
    fn managed_nvidia_bin_dirs_are_existing_and_deterministic() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-nvidia-bin-test-{}",
            uuid::Uuid::new_v4()
        ));
        let nvidia = root.join("Lib").join("site-packages").join("nvidia");
        let cudnn = nvidia.join("cudnn").join("bin");
        let cublas = nvidia.join("cublas").join("bin");
        std::fs::create_dir_all(&cudnn).unwrap();
        std::fs::create_dir_all(&cublas).unwrap();
        std::fs::create_dir_all(nvidia.join("cuda_runtime")).unwrap();
        std::fs::write(nvidia.join("not-a-package"), b"file").unwrap();

        let directories = managed_nvidia_bin_dirs(&root).unwrap();

        assert_eq!(directories, vec![cublas, cudnn]);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-managed-native-library-search
    #[test]
    fn managed_nvidia_bins_precede_and_deduplicate_inherited_windows_path() {
        let managed_with_semicolon = PathBuf::from(r"C:\Apps;Lab\NVIDIA\cuDNN\bin");
        let entries = vec![
            managed_with_semicolon.clone(),
            PathBuf::from(r"C:\Managed\CUDA\bin"),
            PathBuf::from(r"c:\apps;lab\nvidia\CUDNN\BIN"),
            PathBuf::from(r"C:\Windows\System32"),
            PathBuf::from(r"C:\Tools"),
            PathBuf::from(r"C:\WINDOWS\SYSTEM32"),
        ];

        assert_eq!(
            deduplicate_windows_path_entries(entries),
            vec![
                managed_with_semicolon,
                PathBuf::from(r"C:\Managed\CUDA\bin"),
                PathBuf::from(r"C:\Windows\System32"),
                PathBuf::from(r"C:\Tools"),
            ]
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_path_composition_quotes_semicolon_entries() {
        let managed = vec![PathBuf::from(r"C:\Apps;Lab\NVIDIA\cuDNN\bin")];
        let inherited = OsStr::new(r"C:\Windows\System32;C:\Tools");

        let composed = compose_windows_search_path(&managed, Some(inherited)).unwrap();
        let entries: Vec<PathBuf> = std::env::split_paths(&composed).collect();

        assert_eq!(
            entries,
            vec![
                managed[0].clone(),
                PathBuf::from(r"C:\Windows\System32"),
                PathBuf::from(r"C:\Tools"),
            ]
        );
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
