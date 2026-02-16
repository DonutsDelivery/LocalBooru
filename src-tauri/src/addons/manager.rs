//! Addon Lifecycle Manager
//!
//! Manages installation, startup, shutdown, and status tracking for all addons.
//! Each addon runs as a separate Python sidecar process with its own virtual
//! environment under `{data_dir}/addons/{addon_id}/`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as TokioMutex;

use super::manifest::{get_addon_manifest, get_addon_registry};
use super::sidecar;

/// Runtime state for a single addon, including its process handle when running.
struct AddonState {
    status: AddonStatus,
    /// The child process handle, present only while the addon is running.
    process: Option<Arc<TokioMutex<tokio::process::Child>>>,
}

/// Public-facing information about an addon, suitable for JSON serialization.
#[derive(Clone, Serialize)]
pub struct AddonInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub port: u16,
    pub status: AddonStatus,
    pub installed: bool,
}

/// The lifecycle status of an addon.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AddonStatus {
    NotInstalled,
    Installed,
    Starting,
    Running,
    Stopped,
    Error(String),
}

/// Persisted on-disk state for an addon (stored in `{addon_dir}/state.json`).
#[derive(Serialize, Deserialize)]
struct AddonPersistedState {
    enabled: bool,
}

/// Manages the full lifecycle of all addon sidecar processes.
pub struct AddonManager {
    addons: DashMap<String, AddonState>,
    data_dir: PathBuf,
}

impl AddonManager {
    /// Create a new manager and scan for already-installed addons on disk.
    pub fn new(data_dir: &Path) -> Self {
        let addons = DashMap::new();
        let addons_base = data_dir.join("addons");

        // Initialize state for every known addon
        for manifest in get_addon_registry() {
            let addon_dir = addons_base.join(manifest.id);
            let venv_dir = addon_dir.join("venv");

            let status = if venv_dir.exists() {
                AddonStatus::Installed
            } else {
                AddonStatus::NotInstalled
            };

            addons.insert(
                manifest.id.to_string(),
                AddonState {
                    status,
                    process: None,
                },
            );
        }

        log::info!(
            "[AddonManager] Initialized with {} addons (data_dir: {})",
            addons.len(),
            data_dir.display()
        );

        Self {
            addons,
            data_dir: data_dir.to_path_buf(),
        }
    }

    /// The base directory where all addon data lives: `{data_dir}/addons/`.
    fn addons_base(&self) -> PathBuf {
        self.data_dir.join("addons")
    }

    /// The directory for a specific addon: `{data_dir}/addons/{id}/`.
    fn addon_dir(&self, id: &str) -> PathBuf {
        self.addons_base().join(id)
    }

    /// The venv directory for a specific addon.
    fn venv_dir(&self, id: &str) -> PathBuf {
        self.addon_dir(id).join("venv")
    }

    /// Path to the persisted state file for an addon: `{addon_dir}/state.json`.
    fn state_file_path(&self, id: &str) -> PathBuf {
        self.addon_dir(id).join("state.json")
    }

    /// Persist the enabled/disabled state for an addon to disk.
    fn save_enabled_state(&self, id: &str, enabled: bool) {
        let path = self.state_file_path(id);
        let state = AddonPersistedState { enabled };
        match serde_json::to_string_pretty(&state) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    log::warn!(
                        "[AddonManager] Failed to write state file for '{}': {}",
                        id, e
                    );
                }
            }
            Err(e) => {
                log::warn!(
                    "[AddonManager] Failed to serialize state for '{}': {}",
                    id, e
                );
            }
        }
    }

    /// Read the persisted state for an addon, returning `None` if missing or invalid.
    fn load_enabled_state(&self, id: &str) -> Option<bool> {
        let path = self.state_file_path(id);
        let contents = std::fs::read_to_string(&path).ok()?;
        let state: AddonPersistedState = serde_json::from_str(&contents).ok()?;
        Some(state.enabled)
    }

    /// Resume all addons that were previously enabled.
    ///
    /// Called once during app startup. Each addon that has a persisted
    /// `{"enabled": true}` state and is installed will be started.
    /// Failures are logged and do not prevent other addons from starting.
    pub async fn resume_addons(&self) {
        log::info!("[AddonManager] Checking for previously-enabled addons to resume...");

        let mut resumed = 0u32;
        for manifest in get_addon_registry() {
            let id = manifest.id;

            // Only attempt to resume installed addons that were previously enabled
            let is_installed = self.venv_dir(id).exists();
            let was_enabled = self.load_enabled_state(id).unwrap_or(false);

            if is_installed && was_enabled {
                log::info!("[AddonManager] Resuming previously-enabled addon '{}'", id);
                match self.start_addon(id).await {
                    Ok(()) => {
                        resumed += 1;
                        log::info!("[AddonManager] Successfully resumed addon '{}'", id);
                    }
                    Err(e) => {
                        log::error!(
                            "[AddonManager] Failed to resume addon '{}': {}",
                            id, e
                        );
                    }
                }
            }
        }

        log::info!(
            "[AddonManager] Addon resume complete ({} addon(s) started)",
            resumed
        );
    }

    /// List all addons with their current status and installation state.
    pub fn list_addons(&self) -> Vec<AddonInfo> {
        get_addon_registry()
            .iter()
            .map(|manifest| {
                let status = self
                    .addons
                    .get(manifest.id)
                    .map(|s| s.status.clone())
                    .unwrap_or(AddonStatus::NotInstalled);

                let installed = self.venv_dir(manifest.id).exists();

                AddonInfo {
                    id: manifest.id.to_string(),
                    name: manifest.name.to_string(),
                    description: manifest.description.to_string(),
                    port: manifest.port,
                    status,
                    installed,
                }
            })
            .collect()
    }

    /// Get information about a single addon.
    pub fn get_addon(&self, id: &str) -> Option<AddonInfo> {
        let manifest = get_addon_manifest(id)?;
        let status = self
            .addons
            .get(id)
            .map(|s| s.status.clone())
            .unwrap_or(AddonStatus::NotInstalled);

        let installed = self.venv_dir(id).exists();

        Some(AddonInfo {
            id: manifest.id.to_string(),
            name: manifest.name.to_string(),
            description: manifest.description.to_string(),
            port: manifest.port,
            status,
            installed,
        })
    }

    /// Quick status check for a single addon.
    pub fn get_addon_status(&self, id: &str) -> AddonStatus {
        self.addons
            .get(id)
            .map(|s| s.status.clone())
            .unwrap_or(AddonStatus::NotInstalled)
    }

    /// Install an addon: create its directory, virtual environment, and install dependencies.
    ///
    /// This is a blocking operation (venv creation + pip install) and should be called
    /// from a context where blocking is acceptable (e.g. `tokio::task::spawn_blocking`).
    pub fn install_addon(&self, id: &str) -> Result<(), String> {
        let manifest =
            get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;

        let addon_dir = self.addon_dir(id);
        let venv_dir = self.venv_dir(id);

        if venv_dir.exists() {
            log::info!("[AddonManager] Addon '{}' already installed", id);
            return Ok(());
        }

        // Find a usable Python interpreter
        let python = sidecar::find_python()
            .ok_or_else(|| "Could not find Python 3 on PATH".to_string())?;

        // Ensure addon directory exists
        std::fs::create_dir_all(&addon_dir)
            .map_err(|e| format!("Failed to create addon directory: {}", e))?;

        // Create virtual environment
        sidecar::create_venv(&python, &venv_dir)?;

        // Install uvicorn (always needed) plus addon-specific dependencies
        let mut deps: Vec<&str> = vec!["uvicorn[standard]", "fastapi"];
        deps.extend_from_slice(manifest.python_deps);
        sidecar::install_deps(&venv_dir, &deps)?;

        // Deploy embedded app.py if available
        if let Some(source) = super::sources::get_addon_source(id) {
            std::fs::write(addon_dir.join("app.py"), source)
                .map_err(|e| format!("Failed to write app.py: {}", e))?;
            log::info!("[AddonManager] Deployed app.py for addon '{}'", id);
        }

        // Update state
        self.set_status(id, AddonStatus::Installed);

        log::info!("[AddonManager] Addon '{}' installed successfully", id);
        Ok(())
    }

    /// Uninstall an addon by removing its entire directory from disk.
    ///
    /// If the addon is currently running, it will be stopped first.
    pub fn uninstall_addon(&self, id: &str) -> Result<(), String> {
        let _ = get_addon_manifest(id)
            .ok_or_else(|| format!("Unknown addon: {}", id))?;

        // Stop if running
        let current_status = self.get_addon_status(id);
        if current_status == AddonStatus::Running || current_status == AddonStatus::Starting {
            // Attempt to kill the process synchronously
            if let Some(mut state) = self.addons.get_mut(id) {
                if let Some(proc) = state.process.take() {
                    // Try to get the PID and kill it
                    if let Ok(mut child) = proc.try_lock() {
                        if let Some(pid) = child.id() {
                            sidecar::kill_process(pid);
                        }
                        // start_kill is non-blocking
                        let _ = child.start_kill();
                    }
                }
            }
        }

        let addon_dir = self.addon_dir(id);
        if addon_dir.exists() {
            std::fs::remove_dir_all(&addon_dir)
                .map_err(|e| format!("Failed to remove addon directory: {}", e))?;
        }

        self.set_status(id, AddonStatus::NotInstalled);
        log::info!("[AddonManager] Addon '{}' uninstalled", id);
        Ok(())
    }

    /// Start an addon sidecar process.
    ///
    /// The addon must be installed. The process is spawned asynchronously
    /// and health-checked before marking it as running.
    pub async fn start_addon(&self, id: &str) -> Result<(), String> {
        let manifest =
            get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;

        let current = self.get_addon_status(id);
        if current == AddonStatus::Running {
            return Ok(());
        }
        if current == AddonStatus::NotInstalled {
            return Err(format!("Addon '{}' is not installed", id));
        }

        self.set_status(id, AddonStatus::Starting);

        let venv_dir = self.venv_dir(id);
        let python = sidecar::get_venv_python(&venv_dir);
        let app_dir = self.addon_dir(id);
        let port = manifest.port;

        if !python.exists() {
            self.set_status(id, AddonStatus::Error("venv python not found".into()));
            return Err("Virtual environment python binary not found".into());
        }

        // Spawn the sidecar
        let child = match sidecar::spawn_sidecar(&python, &app_dir, port).await {
            Ok(c) => c,
            Err(e) => {
                self.set_status(id, AddonStatus::Error(e.clone()));
                return Err(e);
            }
        };

        let process = Arc::new(TokioMutex::new(child));

        // Store process handle immediately so it can be killed if needed
        if let Some(mut state) = self.addons.get_mut(id) {
            state.process = Some(process.clone());
        }

        // Wait for the addon to become healthy
        let healthy = sidecar::wait_for_healthy(port, Duration::from_secs(30)).await;

        if healthy {
            self.set_status(id, AddonStatus::Running);
            self.save_enabled_state(id, true);
            log::info!("[AddonManager] Addon '{}' is running on port {}", id, port);
            Ok(())
        } else {
            // Kill the unhealthy process
            {
                let mut child = process.lock().await;
                if let Some(pid) = child.id() {
                    sidecar::kill_process(pid);
                }
                let _ = child.start_kill();
            }
            if let Some(mut state) = self.addons.get_mut(id) {
                state.process = None;
            }

            let msg = format!("Addon '{}' failed to become healthy within 30s", id);
            self.set_status(id, AddonStatus::Error(msg.clone()));
            Err(msg)
        }
    }

    /// Stop a running addon by killing its sidecar process.
    pub async fn stop_addon(&self, id: &str) -> Result<(), String> {
        let _ = get_addon_manifest(id)
            .ok_or_else(|| format!("Unknown addon: {}", id))?;

        let process = {
            let mut state = self
                .addons
                .get_mut(id)
                .ok_or_else(|| format!("No state for addon '{}'", id))?;
            state.process.take()
        };

        if let Some(proc) = process {
            let mut child = proc.lock().await;
            if let Some(pid) = child.id() {
                log::info!("[AddonManager] Stopping addon '{}' (PID {})", id, pid);
                sidecar::kill_process(pid);
            }
            let _ = child.start_kill();
            // Wait briefly for the process to exit
            let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
        }

        // Reset status to Installed (the venv is still on disk)
        let venv_dir = self.venv_dir(id);
        if venv_dir.exists() {
            self.set_status(id, AddonStatus::Installed);
        } else {
            self.set_status(id, AddonStatus::NotInstalled);
        }

        self.save_enabled_state(id, false);
        log::info!("[AddonManager] Addon '{}' stopped", id);
        Ok(())
    }

    /// Get the base URL for a running addon, or `None` if it is not running.
    pub fn addon_url(&self, id: &str) -> Option<String> {
        let manifest = get_addon_manifest(id)?;
        let status = self.get_addon_status(id);

        if status == AddonStatus::Running {
            Some(format!("http://127.0.0.1:{}", manifest.port))
        } else {
            None
        }
    }

    /// Shut down all running addon sidecar processes.
    ///
    /// Called during application exit to ensure no orphan processes remain.
    pub fn stop_all(&self) {
        log::info!("[AddonManager] Stopping all running addons...");

        for mut entry in self.addons.iter_mut() {
            let id = entry.key().clone();
            if let Some(proc) = entry.value_mut().process.take() {
                if let Ok(mut child) = proc.try_lock() {
                    if let Some(pid) = child.id() {
                        log::info!("[AddonManager] Killing addon '{}' (PID {})", id, pid);
                        sidecar::kill_process(pid);
                    }
                    let _ = child.start_kill();
                }
            }
            // Mark stopped unless uninstalled
            if entry.value().status != AddonStatus::NotInstalled {
                entry.value_mut().status = AddonStatus::Stopped;
            }
        }

        log::info!("[AddonManager] All addons stopped");
    }

    /// Internal: update the status field for an addon.
    fn set_status(&self, id: &str, status: AddonStatus) {
        if let Some(mut state) = self.addons.get_mut(id) {
            state.status = status;
        }
    }
}
