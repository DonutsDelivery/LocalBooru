//! Backend Health Management
//!
//! Handles health checks, port management, and zombie process cleanup.
//! Mirrors electron/backend/health.js

use std::net::TcpListener;
use std::time::Duration;
use std::process::Command;
use serde::Deserialize;

/// Health check response from the backend
#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
}

/// Check if backend is healthy and responding with valid content
pub async fn health_check(port: u16) -> bool {
    health_check_host("127.0.0.1", port).await
}

/// Check if backend is healthy on a specific host address
pub async fn health_check_host(host: &str, port: u16) -> bool {
    let url = format!("http://{}:{}/health", host, port);

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };

    match client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<HealthResponse>().await {
                    Ok(health) => health.status == "healthy",
                    Err(_) => false,
                }
            } else {
                false
            }
        }
        Err(_) => false,
    }
}

/// Wait for the backend to respond to health checks
pub async fn wait_for_ready(port: u16, timeout_ms: u64) -> Result<(), String> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_millis(timeout_ms);

    loop {
        if start.elapsed() > timeout {
            return Err("Backend startup timeout".to_string());
        }

        if health_check(port).await {
            log::info!("[Backend] Server ready");
            return Ok(());
        }

        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Check if port is free
pub fn is_port_free(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

/// Information about a process using a port
#[derive(Debug)]
pub struct PortUser {
    pub pid: String,
    pub name: String,
}

/// Get information about what process is using a port
#[cfg(target_os = "linux")]
pub fn get_port_user(port: u16) -> Option<PortUser> {
    let output = Command::new("lsof")
        .args(["-ti", &format!(":{}", port)])
        .output()
        .ok()?;

    let pid = String::from_utf8_lossy(&output.stdout)
        .trim()
        .lines()
        .next()?
        .to_string();

    if pid.is_empty() || !pid.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    // Get process name
    let name_output = Command::new("ps")
        .args(["-p", &pid, "-o", "comm="])
        .output()
        .ok();

    let name = name_output
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Some(PortUser { pid, name })
}

#[cfg(target_os = "windows")]
pub fn get_port_user(port: u16) -> Option<PortUser> {
    let output = Command::new("netstat")
        .args(["-ano"])
        .output()
        .ok()?;

    let output_str = String::from_utf8_lossy(&output.stdout);
    let port_str = format!(":{}", port);

    for line in output_str.lines() {
        if line.contains(&port_str) && line.contains("LISTENING") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(pid) = parts.last() {
                if pid.chars().all(|c| c.is_ascii_digit()) {
                    // Get process name using tasklist
                    let task_output = Command::new("tasklist")
                        .args(["/FI", &format!("PID eq {}", pid), "/FO", "CSV", "/NH"])
                        .output()
                        .ok();

                    let name = task_output
                        .map(|o| {
                            String::from_utf8_lossy(&o.stdout)
                                .split(',')
                                .next()
                                .map(|s| s.trim_matches('"').to_string())
                                .unwrap_or_else(|| "unknown".to_string())
                        })
                        .unwrap_or_else(|| "unknown".to_string());

                    return Some(PortUser {
                        pid: pid.to_string(),
                        name,
                    });
                }
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
pub fn get_port_user(port: u16) -> Option<PortUser> {
    // macOS uses same lsof approach as Linux
    get_port_user_lsof(port)
}

#[cfg(target_os = "linux")]
fn get_port_user_lsof(port: u16) -> Option<PortUser> {
    get_port_user(port)
}

#[cfg(target_os = "macos")]
fn get_port_user_lsof(port: u16) -> Option<PortUser> {
    let output = Command::new("lsof")
        .args(["-ti", &format!(":{}", port)])
        .output()
        .ok()?;

    let pid = String::from_utf8_lossy(&output.stdout)
        .trim()
        .lines()
        .next()?
        .to_string();

    if pid.is_empty() || !pid.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let name_output = Command::new("ps")
        .args(["-p", &pid, "-o", "comm="])
        .output()
        .ok();

    let name = name_output
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Some(PortUser { pid, name })
}

/// Kill zombie processes on a port (Linux/macOS)
#[cfg(not(target_os = "windows"))]
pub async fn kill_zombie_processes(port: u16) -> Result<(), String> {
    log::info!("[Backend] Checking for zombie processes on port {}", port);

    for attempt in 1..=3 {
        // Try to find and kill processes using lsof
        if let Ok(output) = Command::new("lsof")
            .args(["-ti", &format!(":{}", port)])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout).to_string();
            let pids: Vec<String> = output_str
                .trim()
                .lines()
                .filter(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
                .map(|s| s.to_string())
                .collect();

            for pid in pids {
                log::info!("[Backend] Killing zombie process {}", pid);
                let _ = Command::new("kill")
                    .args(["-9", &pid])
                    .output();
            }
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        if is_port_free(port) {
            log::info!("[Backend] Port is free");
            return Ok(());
        }

        log::info!("[Backend] Port still in use after attempt {}, retrying...", attempt);
    }

    // Final check
    if !is_port_free(port) {
        let port_user = get_port_user(port);
        let msg = match port_user {
            Some(pu) => format!("Port {} is already in use by {} (PID: {})", port, pu.name, pu.pid),
            None => format!("Port {} is already in use", port),
        };
        log::error!("[Backend] CRITICAL: {}", msg);
        return Err(msg);
    }

    Ok(())
}

/// Kill zombie processes on a port (Windows)
#[cfg(target_os = "windows")]
pub async fn kill_zombie_processes(port: u16) -> Result<(), String> {
    log::info!("[Backend] Checking for zombie processes on port {}", port);

    for attempt in 1..=3 {
        if let Ok(output) = Command::new("netstat")
            .args(["-ano"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let port_str = format!(":{}", port);
            let mut pids = std::collections::HashSet::new();

            for line in output_str.lines() {
                if line.contains(&port_str) && line.contains("LISTENING") {
                    if let Some(pid) = line.split_whitespace().last() {
                        if pid != "0" && pid.chars().all(|c| c.is_ascii_digit()) {
                            pids.insert(pid.to_string());
                        }
                    }
                }
            }

            for pid in pids {
                log::info!("[Backend] Killing zombie process {}", pid);
                let _ = Command::new("taskkill")
                    .args(["/F", "/PID", &pid])
                    .output();
            }
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        if is_port_free(port) {
            log::info!("[Backend] Port is free");
            return Ok(());
        }

        log::info!("[Backend] Port still in use after attempt {}, retrying...", attempt);
    }

    // Final check
    if !is_port_free(port) {
        let port_user = get_port_user(port);
        let msg = match port_user {
            Some(pu) => format!("Port {} is already in use by {} (PID: {})", port, pu.name, pu.pid),
            None => format!("Port {} is already in use", port),
        };
        log::error!("[Backend] CRITICAL: {}", msg);
        return Err(msg);
    }

    Ok(())
}

/// Kill processes on a specific port
#[cfg(not(target_os = "windows"))]
pub fn kill_processes_on_port(port: u16) {
    if let Ok(output) = Command::new("lsof")
        .args(["-ti", &format!(":{}", port)])
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout).to_string();
        let pids: Vec<String> = output_str
            .trim()
            .lines()
            .filter(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
            .map(|s| s.to_string())
            .collect();

        for pid in pids {
            let _ = Command::new("kill")
                .args(["-9", &pid])
                .output();
        }
    }
}

#[cfg(target_os = "windows")]
pub fn kill_processes_on_port(port: u16) {
    if let Ok(output) = Command::new("netstat")
        .args(["-ano"])
        .output()
    {
        let output_str = String::from_utf8_lossy(&output.stdout);
        let port_str = format!(":{}", port);

        for line in output_str.lines() {
            if line.contains(&port_str) && line.contains("LISTENING") {
                if let Some(pid) = line.split_whitespace().last() {
                    if pid != "0" && pid.chars().all(|c| c.is_ascii_digit()) {
                        let _ = Command::new("taskkill")
                            .args(["/F", "/PID", pid])
                            .output();
                    }
                }
            }
        }
    }
}
