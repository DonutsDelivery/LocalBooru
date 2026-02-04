//! Backend Process Management
//!
//! Handles Python executable detection, environment setup, and process spawning.
//! Mirrors electron/backend/process.js

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Child, Stdio};
use std::collections::HashMap;

use super::config::get_packages_dir;

/// Get the Python executable path based on platform
pub fn get_python_path(is_packaged: bool) -> String {
    if cfg!(target_os = "windows") {
        if is_packaged {
            // Bundled Python in resources folder
            // For Tauri, this would be in the resource directory
            if let Ok(exe_path) = env::current_exe() {
                if let Some(parent) = exe_path.parent() {
                    let bundled = parent.join("python-embed").join("python.exe");
                    if bundled.exists() {
                        return bundled.to_string_lossy().to_string();
                    }
                }
            }
        }

        // Development: check for local venv
        let dev_paths = [
            PathBuf::from("python-embed/python.exe"),
            PathBuf::from(".venv/Scripts/python.exe"),
        ];

        for path in &dev_paths {
            if path.exists() {
                return path.to_string_lossy().to_string();
            }
        }

        // Fall back to system Python
        "python".to_string()
    } else {
        // Linux/macOS
        if is_packaged {
            // Check for bundled venv in resources folder
            if let Ok(exe_path) = env::current_exe() {
                if let Some(parent) = exe_path.parent() {
                    let bundled = parent.join("python-venv").join("bin").join("python");
                    if bundled.exists() {
                        return bundled.to_string_lossy().to_string();
                    }
                }
            }
        }

        // Development: check for local venv
        let dev_paths = [
            PathBuf::from("python-venv-linux/bin/python"),
            PathBuf::from(".venv/bin/python"),
        ];

        for path in &dev_paths {
            if path.exists() {
                return path.to_string_lossy().to_string();
            }
        }

        // Fall back to system Python
        "python".to_string()
    }
}

/// Get the working directory for the Python process
pub fn get_working_directory(is_packaged: bool) -> PathBuf {
    if is_packaged {
        // For packaged app, look for api folder relative to executable
        if let Ok(exe_path) = env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                let api_path = parent.join("api");
                if api_path.exists() {
                    return parent.to_path_buf();
                }
                // Fallback: check resources subdirectory
                let resources_api = parent.join("resources").join("api");
                if resources_api.exists() {
                    return parent.join("resources");
                }
            }
        }
    }

    // Development: use current directory or find project root
    if let Ok(cwd) = env::current_dir() {
        // Check if api folder exists in cwd
        if cwd.join("api").exists() {
            return cwd;
        }
        // Check parent directories
        let mut current = cwd.clone();
        for _ in 0..3 {
            if let Some(parent) = current.parent() {
                if parent.join("api").exists() {
                    return parent.to_path_buf();
                }
                current = parent.to_path_buf();
            }
        }
        return cwd;
    }

    PathBuf::from(".")
}

/// Build environment variables for the Python subprocess
pub fn get_python_env(
    is_packaged: bool,
    portable_data_dir: Option<&PathBuf>,
) -> HashMap<String, String> {
    let mut env_vars: HashMap<String, String> = env::vars().collect();

    // Always unbuffer Python output for real-time logging
    env_vars.insert("PYTHONUNBUFFERED".to_string(), "1".to_string());

    let packages_dir = get_packages_dir();

    // Ensure packages directory exists
    if !packages_dir.exists() {
        let _ = fs::create_dir_all(&packages_dir);
    }

    // Add portable data directory if in portable mode
    if let Some(data_dir) = portable_data_dir {
        env_vars.insert(
            "LOCALBOORU_PORTABLE_DATA".to_string(),
            data_dir.to_string_lossy().to_string(),
        );
    }

    let working_dir = get_working_directory(is_packaged);
    let python_path = get_python_path(is_packaged);

    if cfg!(target_os = "windows") {
        let python_dir = PathBuf::from(&python_path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();

        if is_packaged || python_dir.join("python.exe").exists() {
            // Using bundled Python
            let site_packages = python_dir.join("Lib").join("site-packages");
            let onnx_capi = packages_dir.join("onnxruntime").join("capi");
            let bundled_onnx_capi = site_packages.join("onnxruntime").join("capi");

            // Build PATH
            let current_path = env_vars.get("PATH").cloned().unwrap_or_default();
            let new_path = format!(
                "{};{};{};{};{}",
                onnx_capi.to_string_lossy(),
                bundled_onnx_capi.to_string_lossy(),
                python_dir.to_string_lossy(),
                python_dir.join("Scripts").to_string_lossy(),
                current_path
            );
            env_vars.insert("PATH".to_string(), new_path);
            env_vars.insert("PYTHONHOME".to_string(), python_dir.to_string_lossy().to_string());

            let python_path_val = format!(
                "{};{};{}",
                packages_dir.to_string_lossy(),
                working_dir.to_string_lossy(),
                site_packages.to_string_lossy()
            );
            env_vars.insert("PYTHONPATH".to_string(), python_path_val);
            env_vars.insert("LOCALBOORU_PACKAGED".to_string(), "1".to_string());
            env_vars.insert("LOCALBOORU_PACKAGES_DIR".to_string(), packages_dir.to_string_lossy().to_string());
        }
    } else {
        // Linux/macOS
        let python_path_buf = PathBuf::from(&python_path);
        let is_bundled_venv = python_path.contains("python-venv");

        if is_bundled_venv {
            // Using bundled venv
            let venv_root = python_path_buf
                .parent() // bin
                .and_then(|p| p.parent()) // venv root
                .unwrap_or(&python_path_buf);

            // Find site-packages
            let lib_dir = venv_root.join("lib");
            let mut site_packages_path = String::new();

            if lib_dir.exists() {
                if let Ok(entries) = fs::read_dir(&lib_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let name_str = name.to_string_lossy();
                        if name_str.starts_with("python3.") {
                            let sp_path = entry.path().join("site-packages");
                            if sp_path.exists() {
                                site_packages_path = sp_path.to_string_lossy().to_string();
                                break;
                            }
                        }
                    }
                }
            }

            // Update PATH
            let current_path = env_vars.get("PATH").cloned().unwrap_or_default();
            let bin_path = venv_root.join("bin");
            env_vars.insert("PATH".to_string(), format!("{}:{}", bin_path.to_string_lossy(), current_path));

            // Set PYTHONPATH
            let mut python_path_val = format!(
                "{}:{}",
                packages_dir.to_string_lossy(),
                working_dir.to_string_lossy()
            );
            if !site_packages_path.is_empty() {
                python_path_val = format!("{}:{}", python_path_val, site_packages_path);
            }
            env_vars.insert("PYTHONPATH".to_string(), python_path_val);
        } else {
            // System Python with pyenv support
            let home_dir = env::var("HOME").unwrap_or_default();
            let pyenv_path = format!("{}/.pyenv/shims:{}/.pyenv/bin", home_dir, home_dir);
            let current_path = env_vars.get("PATH").cloned().unwrap_or_default();
            env_vars.insert("PATH".to_string(), format!("{}:{}", pyenv_path, current_path));

            let python_path_val = format!(
                "{}:{}",
                packages_dir.to_string_lossy(),
                working_dir.to_string_lossy()
            );
            env_vars.insert("PYTHONPATH".to_string(), python_path_val);
        }

        env_vars.insert(
            "LOCALBOORU_PACKAGED".to_string(),
            if is_packaged { "1" } else { "" }.to_string(),
        );
        env_vars.insert(
            "LOCALBOORU_PACKAGES_DIR".to_string(),
            packages_dir.to_string_lossy().to_string(),
        );
    }

    env_vars
}

/// Spawn the uvicorn backend process
pub fn spawn_backend(
    bind_host: &str,
    port: u16,
    is_packaged: bool,
    portable_data_dir: Option<&PathBuf>,
) -> std::io::Result<Child> {
    let python_path = get_python_path(is_packaged);
    let cwd = get_working_directory(is_packaged);
    let env_vars = get_python_env(is_packaged, portable_data_dir);

    log::info!("[Backend] Python path: {}", python_path);
    log::info!("[Backend] Working directory: {:?}", cwd);
    log::info!("[Backend] Binding to: {}", bind_host);

    // Spawn uvicorn via python -m for better compatibility
    let mut cmd = Command::new(&python_path);
    cmd.args([
        "-m", "uvicorn",
        "api.main:app",
        "--host", bind_host,
        "--port", &port.to_string(),
    ])
    .current_dir(&cwd)
    .envs(env_vars)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    // Windows-specific: hide console window
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    cmd.spawn()
}

/// Kill uvicorn processes aggressively
/// Used for cleanup on startup and shutdown
#[cfg(not(target_os = "windows"))]
pub fn kill_uvicorn_processes() {
    // Kill any uvicorn process for api.main:app specifically
    let _ = Command::new("pkill")
        .args(["-9", "-f", "uvicorn api.main:app"])
        .output();
}

#[cfg(target_os = "windows")]
pub fn kill_uvicorn_processes() {
    // Windows: try to kill python processes running uvicorn
    let _ = Command::new("taskkill")
        .args(["/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq uvicorn*"])
        .output();
}

/// Force kill a specific process by PID
#[cfg(not(target_os = "windows"))]
pub fn force_kill_process(pid: u32) {
    let _ = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output();
}

#[cfg(target_os = "windows")]
pub fn force_kill_process(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/pid", &pid.to_string(), "/T", "/F"])
        .output();
}

/// Gracefully terminate a process
#[cfg(not(target_os = "windows"))]
pub fn graceful_kill_process(pid: u32) {
    // Send SIGINT first (Ctrl+C equivalent)
    let _ = Command::new("kill")
        .args(["-2", &pid.to_string()])
        .output();
}

#[cfg(target_os = "windows")]
pub fn graceful_kill_process(pid: u32) {
    // First try graceful tree kill (no /F flag)
    let _ = Command::new("taskkill")
        .args(["/pid", &pid.to_string(), "/T"])
        .output();
}
