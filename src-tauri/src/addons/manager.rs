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

use super::manifest::{get_addon_manifest, get_addon_registry, AddonRuntime};
use super::sidecar;
use crate::routes::settings::get_config_section;

const ONNXRUNTIME_CPU: &str = "onnxruntime==1.23.2";
const ONNXRUNTIME_GPU: &str = "onnxruntime-gpu[cuda,cudnn]==1.23.2";

fn dependency_list_for_platform(
    id: &str,
    python_deps: &[&'static str],
    target_os: &str,
) -> Vec<&'static str> {
    let mut deps = vec!["uvicorn[standard]", "fastapi"];
    deps.extend_from_slice(python_deps);

    if id == "auto-tagger" {
        deps.retain(|dependency| {
            *dependency != "onnxruntime" && !dependency.starts_with("onnxruntime==")
        });
        if matches!(target_os, "linux" | "windows") {
            deps.push(ONNXRUNTIME_GPU);
        } else {
            deps.push(ONNXRUNTIME_CPU);
        }
    }

    deps
}

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
    pub runtime: AddonRuntime,
    pub port: Option<u16>,
    pub requires_start: bool,
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
            let installed = match manifest.runtime {
                AddonRuntime::Sidecar => addon_dir.join("venv").exists(),
                AddonRuntime::Builtin => addon_dir.join("installed.json").exists(),
            };

            let status = if installed {
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

    fn install_marker(&self, id: &str) -> PathBuf {
        self.addon_dir(id).join("installed.json")
    }

    fn is_installed(&self, manifest: &super::manifest::AddonManifest) -> bool {
        match manifest.runtime {
            AddonRuntime::Sidecar => self.venv_dir(manifest.id).exists(),
            AddonRuntime::Builtin => self.install_marker(manifest.id).exists(),
        }
    }

    fn dependency_list(
        &self,
        id: &str,
        manifest: &super::manifest::AddonManifest,
    ) -> Vec<&'static str> {
        dependency_list_for_platform(id, manifest.python_deps, std::env::consts::OS)
    }

    fn install_dependencies(
        &self,
        id: &str,
        manifest: &super::manifest::AddonManifest,
        venv_dir: &Path,
    ) -> Result<(), String> {
        let deps = self.dependency_list(id, manifest);
        if !deps.contains(&ONNXRUNTIME_GPU) {
            return sidecar::install_deps(venv_dir, &deps);
        }

        sidecar::uninstall_deps(venv_dir, &["onnxruntime"])?;
        if let Err(install_error) = sidecar::install_deps(venv_dir, &deps) {
            log::warn!(
                "[Addon:{}] GPU runtime installation failed; restoring CPU runtime",
                id
            );
            let cleanup_error = sidecar::uninstall_deps(venv_dir, &["onnxruntime-gpu"]).err();
            let restore_result = sidecar::install_deps(venv_dir, &[ONNXRUNTIME_CPU]);
            return match (cleanup_error, restore_result) {
                (None, Ok(())) => Err(format!(
                    "{}; restored the CPU inference runtime",
                    install_error
                )),
                (Some(cleanup_error), Ok(())) => Err(format!(
                    "{}; restored the CPU inference runtime after GPU cleanup failed: {}",
                    install_error, cleanup_error
                )),
                (_, Err(restore_error)) => Err(format!(
                    "{}; CPU runtime restoration also failed: {}",
                    install_error, restore_error
                )),
            };
        }
        Ok(())
    }

    fn sidecar_env(&self, id: &str) -> Vec<(String, String)> {
        let mut envs = vec![(
            "LOCALBOORU_DATA_DIR".into(),
            self.data_dir.to_string_lossy().into_owned(),
        )];
        if id != "auto-tagger" {
            return envs;
        }

        let config = get_config_section(&self.data_dir, "auto_tagger");
        let model = config
            .get("model")
            .and_then(|value| value.as_str())
            .unwrap_or("vit-v3");
        let device = config
            .get("device")
            .and_then(|value| value.as_str())
            .unwrap_or("auto");
        let general_threshold = config
            .get("general_threshold")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.35);
        let character_threshold = config
            .get("character_threshold")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.75);

        envs.extend([
            (
                "TAGGER_MODEL_DIR".into(),
                self.data_dir
                    .join("models")
                    .join("tagger")
                    .join(model)
                    .to_string_lossy()
                    .into_owned(),
            ),
            ("TAGGER_MODEL".into(), model.into()),
            ("TAGGER_REQUESTED_DEVICE".into(), device.into()),
            ("TAGGER_THRESHOLD".into(), general_threshold.to_string()),
            (
                "TAGGER_CHARACTER_THRESHOLD".into(),
                character_threshold.to_string(),
            ),
        ]);
        envs
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
                        id,
                        e
                    );
                }
            }
            Err(e) => {
                log::warn!(
                    "[AddonManager] Failed to serialize state for '{}': {}",
                    id,
                    e
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
            if manifest.runtime == AddonRuntime::Builtin {
                continue;
            }
            let id = manifest.id;

            // Only attempt to resume installed addons that were previously enabled
            let is_installed = self.is_installed(manifest);
            let was_enabled = self.load_enabled_state(id).unwrap_or(false);

            if is_installed && was_enabled {
                log::info!("[AddonManager] Resuming previously-enabled addon '{}'", id);
                match self.start_addon(id).await {
                    Ok(()) => {
                        resumed += 1;
                        log::info!("[AddonManager] Successfully resumed addon '{}'", id);
                    }
                    Err(e) => {
                        log::error!("[AddonManager] Failed to resume addon '{}': {}", id, e);
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

                let installed = self.is_installed(manifest);

                AddonInfo {
                    id: manifest.id.to_string(),
                    name: manifest.name.to_string(),
                    description: manifest.description.to_string(),
                    runtime: manifest.runtime,
                    port: manifest.port,
                    requires_start: manifest.runtime == AddonRuntime::Sidecar,
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

        let installed = self.is_installed(manifest);

        Some(AddonInfo {
            id: manifest.id.to_string(),
            name: manifest.name.to_string(),
            description: manifest.description.to_string(),
            runtime: manifest.runtime,
            port: manifest.port,
            requires_start: manifest.runtime == AddonRuntime::Sidecar,
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
        let manifest = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;

        let addon_dir = self.addon_dir(id);
        let venv_dir = self.venv_dir(id);

        if self.is_installed(manifest) {
            log::info!("[AddonManager] Addon '{}' already installed", id);
            return Ok(());
        }

        std::fs::create_dir_all(&addon_dir)
            .map_err(|e| format!("Failed to create addon directory: {}", e))?;

        if manifest.runtime == AddonRuntime::Builtin {
            std::fs::write(self.install_marker(id), "{\"installed\":true}\n")
                .map_err(|e| format!("Failed to write install marker: {}", e))?;
            self.set_status(id, AddonStatus::Installed);
            log::info!("[AddonManager] Built-in addon '{}' installed", id);
            return Ok(());
        }

        // Find a usable Python interpreter
        let python =
            sidecar::find_python().ok_or_else(|| "Could not find Python 3 on PATH".to_string())?;
        if id == "auto-tagger" {
            sidecar::validate_python_minor(&python, 10, 13)?;
        }

        // Create virtual environment
        sidecar::create_venv(&python, &venv_dir)?;

        // Install uvicorn plus add-on dependencies. CUDA-capable platforms use
        // the ONNX GPU wheel, which retains a CPU execution-provider fallback.
        self.install_dependencies(id, manifest, &venv_dir)?;

        // Deploy embedded addon sources if available.
        if let Some(sources) = super::sources::get_addon_sources(id) {
            for (filename, source) in sources {
                std::fs::write(addon_dir.join(filename), source)
                    .map_err(|e| format!("Failed to write {}: {}", filename, e))?;
            }
            log::info!("[AddonManager] Deployed sources for addon '{}'", id);
        }

        // Update state
        self.set_status(id, AddonStatus::Installed);

        log::info!("[AddonManager] Addon '{}' installed successfully", id);
        Ok(())
    }

    /// Reinstall an existing addon's dependencies and redeploy its embedded source.
    pub fn repair_addon(&self, id: &str) -> Result<(), String> {
        let manifest = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;
        if manifest.runtime == AddonRuntime::Builtin {
            return Err(format!("Addon '{}' does not require a sidecar", id));
        }
        let venv_dir = self.venv_dir(id);
        if !venv_dir.exists() {
            return self.install_addon(id);
        }
        if id == "auto-tagger" {
            sidecar::validate_python_minor(&sidecar::get_venv_python(&venv_dir), 10, 13)?;
        }
        self.install_dependencies(id, manifest, &venv_dir)?;
        if let Some(sources) = super::sources::get_addon_sources(id) {
            for (filename, source) in sources {
                std::fs::write(self.addon_dir(id).join(filename), source)
                    .map_err(|error| format!("Failed to deploy {}: {}", filename, error))?;
            }
        }
        self.set_status(id, AddonStatus::Installed);
        Ok(())
    }

    /// Uninstall an addon by removing its entire directory from disk.
    ///
    /// If the addon is currently running, it will be stopped first.
    pub fn uninstall_addon(&self, id: &str) -> Result<(), String> {
        let _ = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;

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
        let manifest = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;
        if manifest.runtime == AddonRuntime::Builtin {
            return Err(format!("Addon '{}' does not require a sidecar", id));
        }

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
        let port = manifest.port.expect("sidecar addons must have a port");
        let sidecar_env = self.sidecar_env(id);

        // Always deploy the latest embedded sources before starting.
        // This ensures installed addons pick up source updates on restart.
        if let Some(sources) = super::sources::get_addon_sources(id) {
            for (filename, source) in sources {
                if let Err(e) = std::fs::write(app_dir.join(filename), source) {
                    log::warn!(
                        "[AddonManager] Failed to deploy {} for '{}': {}",
                        filename,
                        id,
                        e
                    );
                }
            }
        }

        if !python.exists() {
            self.set_status(id, AddonStatus::Error("venv python not found".into()));
            return Err("Virtual environment python binary not found".into());
        }

        // Spawn the sidecar
        let child = match sidecar::spawn_sidecar(id, &python, &app_dir, port, &sidecar_env).await {
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
        let manifest = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;
        if manifest.runtime == AddonRuntime::Builtin {
            return Err(format!("Addon '{}' does not require a sidecar", id));
        }

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
            manifest
                .port
                .map(|port| format!("http://127.0.0.1:{}", port))
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

#[cfg(test)]
mod tests {
    use super::*;

    // AC: @addon-platform-dependencies ac-1
    #[test]
    fn auto_tagger_gpu_platforms_include_managed_cuda_dependencies() {
        for target_os in ["windows", "linux"] {
            let deps = dependency_list_for_platform("auto-tagger", &["onnxruntime"], target_os);
            assert!(deps.contains(&ONNXRUNTIME_GPU));
            assert!(!deps.iter().any(|dependency| *dependency == "onnxruntime"));
            assert_eq!(
                deps.iter()
                    .filter(|dependency| dependency.starts_with("onnxruntime"))
                    .count(),
                1
            );
        }
    }

    // AC: @addon-platform-dependencies ac-4
    #[test]
    fn auto_tagger_non_gpu_platforms_use_pinned_cpu_runtime() {
        let deps = dependency_list_for_platform("auto-tagger", &["onnxruntime"], "macos");
        assert!(deps.contains(&ONNXRUNTIME_CPU));
        assert!(!deps.iter().any(|dependency| *dependency == ONNXRUNTIME_GPU));
    }

    // AC: @addon-platform-dependencies ac-3
    #[test]
    fn install_and_repair_resolve_the_same_auto_tagger_dependencies() {
        let manifest = get_addon_manifest("auto-tagger").unwrap();
        let manager = AddonManager::new(Path::new("unused"));
        assert_eq!(
            manager.dependency_list("auto-tagger", manifest),
            dependency_list_for_platform("auto-tagger", manifest.python_deps, std::env::consts::OS)
        );
    }

    // AC: @addon-platform-dependencies ac-3
    #[cfg(unix)]
    #[test]
    fn repair_updates_dependencies_without_removing_persisted_resources() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-repair-test-{}",
            uuid::Uuid::new_v4()
        ));
        let addon_dir = root.join("addons/auto-tagger");
        let venv_bin = addon_dir.join("venv/bin");
        let model = root.join("models/tagger/vit-v3/model.onnx");
        std::fs::create_dir_all(&venv_bin).unwrap();
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"model-data").unwrap();
        std::fs::write(addon_dir.join("state.json"), br#"{"enabled":true}"#).unwrap();
        std::fs::write(
            root.join("settings.json"),
            br#"{"auto_tagger":{"device":"cuda"}}"#,
        )
        .unwrap();
        let python = venv_bin.join("python");
        let invocations = root.join("pip-invocations");
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'Python 3.11.9'; else printf '%s\\n' \"$*\" >> '{}'; fi\nexit 0\n",
                invocations.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let manager = AddonManager::new(&root);

        manager.repair_addon("auto-tagger").unwrap();

        let invocations = std::fs::read_to_string(invocations).unwrap();
        let commands: Vec<_> = invocations.lines().collect();
        assert_eq!(commands[0], "-m pip uninstall -y onnxruntime");
        assert!(commands[1].starts_with("-m pip install --upgrade "));
        assert!(commands[1].contains(ONNXRUNTIME_GPU));
        assert!(!commands[1]
            .split_whitespace()
            .any(|arg| arg == "onnxruntime"));
        assert_eq!(std::fs::read(&model).unwrap(), b"model-data");
        assert_eq!(
            std::fs::read(addon_dir.join("state.json")).unwrap(),
            br#"{"enabled":true}"#
        );
        assert_eq!(
            std::fs::read(root.join("settings.json")).unwrap(),
            br#"{"auto_tagger":{"device":"cuda"}}"#
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @addon-platform-dependencies ac-3
    #[cfg(unix)]
    #[test]
    fn failed_gpu_repair_restores_cpu_inference_runtime() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-rollback-test-{}",
            uuid::Uuid::new_v4()
        ));
        let venv = root.join("venv");
        let bin = venv.join("bin");
        let python = bin.join("python");
        let invocations = root.join("pip-invocations");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nfor arg in \"$@\"; do [ \"$arg\" = '{}' ] && exit 1; done\nexit 0\n",
                invocations.display(),
                ONNXRUNTIME_GPU
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let manager = AddonManager::new(&root);
        let manifest = get_addon_manifest("auto-tagger").unwrap();

        let error = manager
            .install_dependencies("auto-tagger", manifest, &venv)
            .unwrap_err();

        assert!(error.contains("restored the CPU inference runtime"));
        let commands = std::fs::read_to_string(invocations).unwrap();
        let commands: Vec<_> = commands.lines().collect();
        assert_eq!(commands[0], "-m pip uninstall -y onnxruntime");
        assert!(commands[1].contains(ONNXRUNTIME_GPU));
        assert_eq!(commands[2], "-m pip uninstall -y onnxruntime-gpu");
        assert_eq!(
            commands[3],
            format!("-m pip install --upgrade {}", ONNXRUNTIME_CPU)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn other_addon_dependencies_are_unchanged() {
        let deps = dependency_list_for_platform("age-detector", &["torch"], "windows");
        assert_eq!(deps, vec!["uvicorn[standard]", "fastapi", "torch"]);
    }

    // AC: @curation-game ac-1
    #[test]
    fn builtin_install_does_not_create_a_venv() {
        let root =
            std::env::temp_dir().join(format!("localbooru-addon-test-{}", uuid::Uuid::new_v4()));
        let manager = AddonManager::new(&root);
        manager.install_addon("curation-game").unwrap();
        assert!(root.join("addons/curation-game/installed.json").exists());
        assert!(!root.join("addons/curation-game/venv").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
