//! Backend Manager
//!
//! Manages the FastAPI backend as a subprocess.
//! Main coordinator that uses config, process, and health modules.
//! Mirrors electron/backend/manager.js

use std::path::PathBuf;
use std::process::Child;
use std::sync::Arc;
use std::io::{BufRead, BufReader};
use tokio::sync::Mutex;
use tokio::time::{Duration, interval};

use super::config::{
    detect_portable_mode, get_bind_host, get_data_dir, get_default_port,
    get_local_ip, get_network_settings, get_settings_path, NetworkSettings,
};
use super::health::{
    health_check, is_port_free, kill_zombie_processes, kill_processes_on_port,
    wait_for_ready,
};
use super::process::{
    force_kill_process, graceful_kill_process, kill_uvicorn_processes, spawn_backend,
};

/// Backend manager state
pub struct BackendManager {
    /// The backend process handle
    process: Arc<Mutex<Option<Child>>>,
    /// Portable data directory if in portable mode
    portable_data_dir: Option<PathBuf>,
    /// Port the backend is running on
    port: u16,
    /// Whether the app is packaged (production) or development
    is_packaged: bool,
    /// Restart attempt counter
    restart_attempts: Arc<Mutex<u32>>,
    /// Maximum restart attempts
    max_restart_attempts: u32,
    /// Whether the manager is shutting down
    shutting_down: Arc<Mutex<bool>>,
}

impl BackendManager {
    /// Create a new backend manager
    pub fn new(is_packaged: bool) -> Self {
        let portable_data_dir = detect_portable_mode();
        let is_portable = portable_data_dir.is_some();

        // Get port from settings or use default
        let network_settings = get_network_settings(portable_data_dir.as_ref());
        let default_port = get_default_port(is_portable);
        let port = network_settings.local_port.unwrap_or(default_port);

        log::info!(
            "[Backend] Mode: {} | Port: {}",
            if is_portable { "portable" } else { "system" },
            port
        );

        Self {
            process: Arc::new(Mutex::new(None)),
            portable_data_dir,
            port,
            is_packaged,
            restart_attempts: Arc::new(Mutex::new(0)),
            max_restart_attempts: 5,
            shutting_down: Arc::new(Mutex::new(false)),
        }
    }

    /// Check if running in portable mode
    pub fn is_portable(&self) -> bool {
        self.portable_data_dir.is_some()
    }

    /// Get the data directory
    pub fn get_data_dir(&self) -> PathBuf {
        get_data_dir(self.portable_data_dir.as_ref())
    }

    /// Get the settings path
    pub fn get_settings_path(&self) -> PathBuf {
        get_settings_path(self.portable_data_dir.as_ref())
    }

    /// Get network settings
    pub fn get_network_settings(&self) -> NetworkSettings {
        get_network_settings(self.portable_data_dir.as_ref())
    }

    /// Get the local IP address
    pub fn get_local_ip(&self) -> String {
        get_local_ip()
    }

    /// Get the port the backend is running on
    pub fn get_port(&self) -> u16 {
        self.port
    }

    /// Check if backend is running
    pub async fn is_running(&self) -> bool {
        self.process.lock().await.is_some()
    }

    /// Start the backend server
    pub async fn start(&self) -> Result<(), String> {
        // Check if already running
        if self.is_running().await {
            log::info!("[Backend] Already running");
            return Ok(());
        }

        // AGGRESSIVE cleanup - kill ALL uvicorn processes
        log::info!("[Backend] Cleaning up any existing uvicorn processes...");
        kill_uvicorn_processes();
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Standard zombie kill on port
        if let Err(e) = kill_zombie_processes(self.port).await {
            log::warn!("[Backend] Warning: Could not free port: {}", e);
        }

        log::info!("[Backend] Starting server on port {}", self.port);

        let bind_host = get_bind_host(self.portable_data_dir.as_ref());
        let network_settings = self.get_network_settings();

        log::info!("[Backend] Binding to: {}", bind_host);
        if bind_host == "0.0.0.0" {
            let local_ip = self.get_local_ip();
            if network_settings.local_network_enabled {
                log::info!(
                    "[Backend] Local network access: enabled (http://{}:{})",
                    local_ip, self.port
                );
            }
            log::info!(
                "[Backend] Public access: {}",
                if network_settings.public_network_enabled { "enabled" } else { "disabled" }
            );
        }

        // Spawn the backend process
        let mut child = spawn_backend(
            &bind_host,
            self.port,
            self.is_packaged,
            self.portable_data_dir.as_ref(),
        ).map_err(|e| format!("Failed to spawn backend: {}", e))?;

        // Set up stdout/stderr logging in background
        let pid = child.id();

        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            std::thread::spawn(move || {
                for line in reader.lines().map_while(Result::ok) {
                    if !line.is_empty() {
                        log::info!("[Backend] {}", line);
                    }
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            std::thread::spawn(move || {
                for line in reader.lines().map_while(Result::ok) {
                    if !line.is_empty() {
                        if line.contains("ERROR") || line.contains("Exception") {
                            log::error!("[Backend Error] {}", line);
                        } else {
                            log::info!("[Backend] {}", line);
                        }
                    }
                }
            });
        }

        // Store the process
        *self.process.lock().await = Some(child);

        // Wait for backend to be ready
        wait_for_ready(self.port, 30000).await?;

        // Start health check task
        self.start_health_check();

        // Reset restart counter on successful start
        *self.restart_attempts.lock().await = 0;

        log::info!("[Backend] Started successfully (PID: {:?})", pid);
        Ok(())
    }

    /// Start periodic health checks
    fn start_health_check(&self) {
        let port = self.port;
        let process = Arc::clone(&self.process);
        let shutting_down = Arc::clone(&self.shutting_down);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Stop if shutting down
                if *shutting_down.lock().await {
                    break;
                }

                // Check if process exists
                if process.lock().await.is_some() {
                    let healthy = health_check(port).await;
                    if !healthy {
                        log::warn!("[Backend] Health check failed");
                    }
                } else {
                    // Process is gone, stop checking
                    break;
                }
            }
        });
    }

    /// Stop the backend server gracefully
    pub async fn stop(&self) -> Result<(), String> {
        *self.shutting_down.lock().await = true;

        let mut process_guard = self.process.lock().await;

        if process_guard.is_none() {
            log::info!("[Backend] Not running");
            return Ok(());
        }

        log::info!("[Backend] Initiating graceful shutdown...");

        let child = process_guard.as_mut().unwrap();
        let pid = child.id();

        // Send SIGINT first for clean uvicorn shutdown
        #[cfg(not(target_os = "windows"))]
        {
            log::info!("[Backend] Sending SIGINT...");
            graceful_kill_process(pid);
        }

        #[cfg(target_os = "windows")]
        {
            graceful_kill_process(pid);
        }

        // Wait for graceful shutdown with timeout
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(10);

        loop {
            match child.try_wait() {
                Ok(Some(_)) => {
                    log::info!("[Backend] Stopped gracefully");
                    *process_guard = None;
                    return Ok(());
                }
                Ok(None) => {
                    if start.elapsed() > timeout {
                        break;
                    }
                    // Still running, escalate after 5 seconds
                    #[cfg(not(target_os = "windows"))]
                    if start.elapsed() > Duration::from_secs(5) {
                        log::info!("[Backend] Escalating to SIGTERM...");
                        let _ = child.kill();
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(e) => {
                    log::error!("[Backend] Error checking process status: {}", e);
                    break;
                }
            }
        }

        // Force kill if still running
        log::info!("[Backend] Graceful shutdown timeout, force killing...");
        self.force_kill_internal(&mut process_guard);

        Ok(())
    }

    /// Force kill the backend process
    fn force_kill_internal(&self, process_guard: &mut Option<Child>) {
        if let Some(child) = process_guard.as_mut() {
            let pid = child.id();
            force_kill_process(pid);
            let _ = child.kill();
        }
        *process_guard = None;

        // Also kill any zombie uvicorn processes
        kill_uvicorn_processes();
        kill_processes_on_port(self.port);
    }

    /// Force kill the backend (public)
    pub async fn force_kill(&self) {
        let mut process_guard = self.process.lock().await;
        self.force_kill_internal(&mut process_guard);
    }

    /// Restart the backend server
    pub async fn restart(&self) -> Result<(), String> {
        self.stop().await?;
        tokio::time::sleep(Duration::from_secs(1)).await;
        self.start().await
    }

    /// Perform health check
    pub async fn health_check(&self) -> bool {
        health_check(self.port).await
    }

    /// Check if port is free
    pub fn is_port_free(&self) -> bool {
        is_port_free(self.port)
    }
}

impl Drop for BackendManager {
    fn drop(&mut self) {
        // Synchronous cleanup on drop
        if let Ok(mut guard) = self.process.try_lock() {
            if guard.is_some() {
                self.force_kill_internal(&mut guard);
            }
        }
    }
}
