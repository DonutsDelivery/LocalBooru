//! Addon Lifecycle Manager
//!
//! Manages installation, startup, shutdown, and status tracking for all addons.
//! Each addon runs as a separate Python sidecar process with its own virtual
//! environment under `{data_dir}/addons/{addon_id}/`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock};

use super::manifest::{get_addon_manifest, get_addon_registry, AddonRuntime};
use super::sidecar;
use crate::routes::settings::get_config_section;

const ONNXRUNTIME_CPU: &str = "onnxruntime==1.23.2";
const ONNXRUNTIME_GPU: &str = "onnxruntime-gpu[cuda,cudnn]==1.23.2";
const ONNXRUNTIME_VERSION: &str = "1.23.2";
const NVIDIA_CUDNN: &str = "nvidia-cudnn-cu12==9.24.0.43";
const NVIDIA_CUDNN_VERSION: &str = "9.24.0.43";
const AUTO_TAGGER_SOURCE_REVISION: &str = "2026-07-21-cudnn-9.24.0.43";
const AUTO_TAGGER_DEPLOYMENT_FILE: &str = "dependency-deployment.json";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AddonDependencyDeployment {
    desired_revision: String,
    installed_revision: String,
    runtime: String,
    probe: serde_json::Value,
    warning: Option<String>,
}

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
            deps.push(NVIDIA_CUDNN);
        } else {
            deps.push(ONNXRUNTIME_CPU);
        }
    }

    deps
}

fn auto_tagger_dependency_revision(dependencies: &[&str], target_os: &str) -> String {
    format!(
        "{}:{}:{}",
        AUTO_TAGGER_SOURCE_REVISION,
        target_os,
        dependencies.join("|")
    )
}

fn cpu_fallback_dependencies<'a>(dependencies: &[&'a str]) -> Vec<&'a str> {
    dependencies
        .iter()
        .filter_map(|dependency| {
            if *dependency == ONNXRUNTIME_GPU {
                Some(ONNXRUNTIME_CPU)
            } else if *dependency == NVIDIA_CUDNN {
                None
            } else {
                Some(*dependency)
            }
        })
        .collect()
}

fn dependency_deployment_path(addon_dir: &Path) -> PathBuf {
    addon_dir.join(AUTO_TAGGER_DEPLOYMENT_FILE)
}

fn read_dependency_deployment(addon_dir: &Path) -> Option<AddonDependencyDeployment> {
    let contents = std::fs::read(dependency_deployment_path(addon_dir)).ok()?;
    serde_json::from_slice(&contents).ok()
}

fn probe_matches_runtime(runtime: &str, probe: &serde_json::Value) -> bool {
    let expected_distribution = match runtime {
        "cuda" => "onnxruntime-gpu",
        "cpu" => "onnxruntime",
        _ => return false,
    };
    let packages = probe.get("packages");
    let runtime_matches = probe.get("onnxruntime").and_then(|value| value.as_str())
        == Some(ONNXRUNTIME_VERSION)
        && packages
            .and_then(|packages| packages.get(expected_distribution))
            .and_then(|value| value.as_str())
            == Some(ONNXRUNTIME_VERSION);
    runtime_matches
        && (runtime != "cuda"
            || packages
                .and_then(|packages| packages.get("nvidia-cudnn-cu12"))
                .and_then(|value| value.as_str())
                == Some(NVIDIA_CUDNN_VERSION))
}

fn deployment_probe_matches_runtime(deployment: &AddonDependencyDeployment) -> bool {
    probe_matches_runtime(&deployment.runtime, &deployment.probe)
}

fn needs_dependency_reconciliation(
    addon_dir: &Path,
    desired_revision: &str,
    cpu_fallback_revision: &str,
) -> bool {
    match read_dependency_deployment(addon_dir) {
        None => true,
        Some(deployment) if deployment.desired_revision != desired_revision => true,
        Some(deployment) if !deployment_probe_matches_runtime(&deployment) => true,
        Some(deployment) if deployment.runtime == "cpu" => {
            deployment.warning.is_none() || deployment.installed_revision != cpu_fallback_revision
        }
        Some(deployment) => deployment.installed_revision != desired_revision,
    }
}

fn probe_committed_deployment(
    addon_dir: &Path,
    venv_dir: &Path,
) -> Result<serde_json::Value, String> {
    let deployment = read_dependency_deployment(addon_dir)
        .ok_or_else(|| "Auto Tagger dependency deployment is missing".to_string())?;
    let distribution = match deployment.runtime.as_str() {
        "cuda" => "onnxruntime-gpu",
        "cpu" => "onnxruntime",
        runtime => return Err(format!("Unknown deployed inference runtime: {}", runtime)),
    };
    let probe = sidecar::probe_onnxruntime(venv_dir, distribution, ONNXRUNTIME_VERSION)?;
    let live_deployment = AddonDependencyDeployment {
        probe: probe.clone(),
        ..deployment
    };
    if !deployment_probe_matches_runtime(&live_deployment) {
        return Err(format!(
            "Committed {} runtime no longer matches the exact managed dependency set",
            live_deployment.runtime
        ));
    }
    Ok(probe)
}

fn write_dependency_deployment(
    addon_dir: &Path,
    deployment: &AddonDependencyDeployment,
) -> Result<(), String> {
    let path = dependency_deployment_path(addon_dir);
    let temporary = path.with_extension("json.tmp");
    let document = serde_json::to_vec_pretty(deployment)
        .map_err(|error| format!("Failed to serialize dependency deployment: {}", error))?;
    std::fs::write(&temporary, document)
        .map_err(|error| format!("Failed to write dependency deployment: {}", error))?;
    if path.exists() {
        std::fs::remove_file(&path)
            .map_err(|error| format!("Failed to replace dependency deployment: {}", error))?;
    }
    std::fs::rename(&temporary, &path)
        .map_err(|error| format!("Failed to commit dependency deployment: {}", error))
}

fn install_dependency_list(id: &str, dependencies: &[&str], venv_dir: &Path) -> Result<(), String> {
    if !dependencies.contains(&ONNXRUNTIME_GPU) {
        return sidecar::install_deps(venv_dir, dependencies);
    }

    sidecar::uninstall_deps(
        venv_dir,
        &["onnxruntime", "onnxruntime-gpu", "nvidia-cudnn-cu12"],
    )?;
    if let Err(install_error) = sidecar::install_deps(venv_dir, dependencies) {
        log::warn!(
            "[Addon:{}] GPU runtime installation failed; restoring CPU runtime",
            id
        );
        let cleanup_error =
            sidecar::uninstall_deps(venv_dir, &["onnxruntime-gpu", "nvidia-cudnn-cu12"]).err();
        let fallback_dependencies = cpu_fallback_dependencies(dependencies);
        let restore_result = sidecar::install_deps(venv_dir, &fallback_dependencies);
        return match (cleanup_error, restore_result) {
            (None, Ok(())) => Err(format!(
                "{}; restored the CPU inference runtime; use Repair to retry CUDA",
                install_error
            )),
            (Some(cleanup_error), Ok(())) => Err(format!(
                "{}; restored the CPU inference runtime after GPU cleanup failed: {}; use Repair to retry CUDA",
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

fn reconcile_auto_tagger_dependencies(
    addon_dir: &Path,
    venv_dir: &Path,
    dependencies: &[&str],
    target_os: &str,
    force: bool,
) -> Result<AddonDependencyDeployment, String> {
    let desired_revision = auto_tagger_dependency_revision(dependencies, target_os);
    let cpu_fallback_revision =
        auto_tagger_dependency_revision(&cpu_fallback_dependencies(dependencies), target_os);
    if !force
        && !needs_dependency_reconciliation(addon_dir, &desired_revision, &cpu_fallback_revision)
    {
        return read_dependency_deployment(addon_dir)
            .ok_or_else(|| "Auto Tagger dependency deployment disappeared".to_string());
    }
    if force {
        let deployment_path = dependency_deployment_path(addon_dir);
        if deployment_path.exists() {
            std::fs::remove_file(&deployment_path).map_err(|error| {
                format!("Failed to invalidate dependency deployment: {}", error)
            })?;
        }
    }

    let (mut runtime, mut warning) =
        match install_dependency_list("auto-tagger", dependencies, venv_dir) {
            Ok(()) => (
                if dependencies.contains(&ONNXRUNTIME_GPU) {
                    "cuda"
                } else {
                    "cpu"
                },
                None,
            ),
            Err(error) if error.contains("restored the CPU inference runtime") => {
                ("cpu", Some(error))
            }
            Err(error) => return Err(error),
        };
    let probe = match sidecar::probe_onnxruntime(
        venv_dir,
        if runtime == "cuda" {
            "onnxruntime-gpu"
        } else {
            "onnxruntime"
        },
        ONNXRUNTIME_VERSION,
    )
    .and_then(|probe| {
        if probe_matches_runtime(runtime, &probe) {
            Ok(probe)
        } else {
            Err(format!(
                "Installed {} runtime does not match the exact managed dependency set",
                runtime
            ))
        }
    }) {
        Ok(probe) => probe,
        Err(probe_error) if runtime == "cuda" => {
            sidecar::uninstall_deps(venv_dir, &["onnxruntime-gpu", "nvidia-cudnn-cu12"]).map_err(
                |cleanup_error| {
                    format!(
                        "GPU runtime probe failed: {}; GPU cleanup also failed: {}",
                        probe_error, cleanup_error
                    )
                },
            )?;
            let fallback_dependencies = cpu_fallback_dependencies(dependencies);
            sidecar::install_deps(venv_dir, &fallback_dependencies).map_err(|restore_error| {
                format!(
                    "GPU runtime probe failed: {}; CPU runtime restoration also failed: {}",
                    probe_error, restore_error
                )
            })?;
            let cpu_probe =
                sidecar::probe_onnxruntime(venv_dir, "onnxruntime", ONNXRUNTIME_VERSION)?;
            runtime = "cpu";
            warning = Some(format!(
                "GPU runtime probe failed: {}; restored the verified CPU inference runtime; use Repair to retry CUDA",
                probe_error
            ));
            cpu_probe
        }
        Err(error) => return Err(error),
    };
    let installed_revision = if runtime == "cuda" {
        desired_revision.clone()
    } else {
        let installed_dependencies = cpu_fallback_dependencies(dependencies);
        auto_tagger_dependency_revision(&installed_dependencies, target_os)
    };
    let deployment = AddonDependencyDeployment {
        desired_revision,
        installed_revision,
        runtime: runtime.to_string(),
        probe,
        warning,
    };
    write_dependency_deployment(addon_dir, &deployment)?;
    Ok(deployment)
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
    Stopping,
    Repairing,
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
    lifecycle: TokioRwLock<()>,
    shutting_down: AtomicBool,
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
            lifecycle: TokioRwLock::new(()),
            shutting_down: AtomicBool::new(false),
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
        install_dependency_list(id, &self.dependency_list(id, manifest), venv_dir)
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
        let manifest = get_addon_manifest(id).expect("registered Auto Tagger manifest");
        let dependencies = self.dependency_list(id, manifest);
        let desired_revision = auto_tagger_dependency_revision(&dependencies, std::env::consts::OS);
        let deployment = read_dependency_deployment(&self.addon_dir(id));
        envs.extend([
            ("TAGGER_DEPLOYMENT_DESIRED".into(), desired_revision),
            (
                "TAGGER_DEPLOYMENT_INSTALLED".into(),
                deployment
                    .as_ref()
                    .map(|state| state.installed_revision.clone())
                    .unwrap_or_default(),
            ),
            (
                "TAGGER_DEPLOYMENT_RUNTIME".into(),
                deployment
                    .as_ref()
                    .map(|state| state.runtime.clone())
                    .unwrap_or_else(|| "unknown".into()),
            ),
            (
                "TAGGER_DEPLOYMENT_WARNING".into(),
                deployment
                    .and_then(|state| state.warning)
                    .unwrap_or_default(),
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

    pub fn begin_repair(&self, id: &str) -> Result<bool, String> {
        let mut state = self
            .addons
            .get_mut(id)
            .ok_or_else(|| format!("No state for addon '{}'", id))?;
        match state.status {
            AddonStatus::Starting | AddonStatus::Stopping | AddonStatus::Repairing => {
                Err(format!("Addon '{}' is already changing state", id))
            }
            AddonStatus::Running => {
                state.status = AddonStatus::Repairing;
                Ok(true)
            }
            _ => {
                state.status = AddonStatus::Repairing;
                Ok(false)
            }
        }
    }

    pub fn finish_failed_repair(&self, id: &str, error: String) {
        self.set_status(id, AddonStatus::Error(error));
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
            if id == "auto-tagger" {
                let dependencies = self.dependency_list(id, manifest);
                let desired_revision =
                    auto_tagger_dependency_revision(&dependencies, std::env::consts::OS);
                let cpu_fallback_revision = auto_tagger_dependency_revision(
                    &cpu_fallback_dependencies(&dependencies),
                    std::env::consts::OS,
                );
                if needs_dependency_reconciliation(
                    &addon_dir,
                    &desired_revision,
                    &cpu_fallback_revision,
                ) || probe_committed_deployment(&addon_dir, &venv_dir).is_err()
                {
                    return self.repair_addon(id);
                }
            }
            if let Some(sources) = super::sources::get_addon_sources(id) {
                for (filename, source) in sources {
                    std::fs::write(addon_dir.join(filename), source)
                        .map_err(|error| format!("Failed to write {}: {}", filename, error))?;
                }
            }
            self.set_status(id, AddonStatus::Installed);
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
        if id == "auto-tagger" {
            let dependencies = self.dependency_list(id, manifest);
            reconcile_auto_tagger_dependencies(
                &addon_dir,
                &venv_dir,
                &dependencies,
                std::env::consts::OS,
                true,
            )?;
        } else {
            self.install_dependencies(id, manifest, &venv_dir)?;
        }

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
            let managed_python = sidecar::get_venv_python(&venv_dir);
            if sidecar::validate_python_minor(&managed_python, 10, 13).is_err() {
                let python = sidecar::find_python()
                    .ok_or_else(|| "Could not find Python 3 on PATH".to_string())?;
                sidecar::validate_python_minor(&python, 10, 13)?;
                std::fs::remove_dir_all(&venv_dir).map_err(|error| {
                    format!(
                        "Failed to replace unsupported managed environment: {}",
                        error
                    )
                })?;
                sidecar::create_venv(&python, &venv_dir)?;
            }
        }
        if id == "auto-tagger" {
            let dependencies = self.dependency_list(id, manifest);
            reconcile_auto_tagger_dependencies(
                &self.addon_dir(id),
                &venv_dir,
                &dependencies,
                std::env::consts::OS,
                true,
            )?;
        } else {
            self.install_dependencies(id, manifest, &venv_dir)?;
        }
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
    /// The caller must claim the lifecycle transition and stop a running sidecar first.
    pub fn uninstall_addon(&self, id: &str) -> Result<(), String> {
        let _ = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;

        {
            let state = self
                .addons
                .get(id)
                .ok_or_else(|| format!("No state for addon '{}'", id))?;
            if matches!(
                state.status,
                AddonStatus::Running | AddonStatus::Starting | AddonStatus::Stopping
            ) || state.process.is_some()
            {
                return Err(format!(
                    "Addon '{}' must be stopped before uninstalling",
                    id
                ));
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
        let lifecycle = self.lifecycle.read().await;
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err("Application shutdown is in progress".into());
        }

        let manifest = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;
        if manifest.runtime == AddonRuntime::Builtin {
            return Err(format!("Addon '{}' does not require a sidecar", id));
        }

        {
            let mut state = self
                .addons
                .get_mut(id)
                .ok_or_else(|| format!("No state for addon '{}'", id))?;
            match state.status {
                AddonStatus::Running => return Ok(()),
                AddonStatus::Starting | AddonStatus::Stopping | AddonStatus::Repairing => {
                    return Err(format!("Addon '{}' is already changing state", id));
                }
                AddonStatus::NotInstalled => {
                    return Err(format!("Addon '{}' is not installed", id));
                }
                _ => state.status = AddonStatus::Starting,
            }
        }

        let venv_dir = self.venv_dir(id);
        let python = sidecar::get_venv_python(&venv_dir);
        let app_dir = self.addon_dir(id);
        let port = manifest.port.expect("sidecar addons must have a port");

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

        if id == "auto-tagger" {
            if let Err(error) = sidecar::validate_python_minor(&python, 10, 13) {
                self.set_status(id, AddonStatus::Error(error.clone()));
                return Err(format!(
                    "{}; use Repair to recreate the managed environment",
                    error
                ));
            }
            let dependencies = self.dependency_list(id, manifest);
            let reconcile_app_dir = app_dir.clone();
            let reconcile_venv_dir = venv_dir.clone();
            let result = tokio::task::spawn_blocking(move || {
                let desired_revision =
                    auto_tagger_dependency_revision(&dependencies, std::env::consts::OS);
                let cpu_fallback_revision = auto_tagger_dependency_revision(
                    &cpu_fallback_dependencies(&dependencies),
                    std::env::consts::OS,
                );
                let needs_reconciliation = needs_dependency_reconciliation(
                    &reconcile_app_dir,
                    &desired_revision,
                    &cpu_fallback_revision,
                );
                let current_probe_failed = !needs_reconciliation
                    && probe_committed_deployment(&reconcile_app_dir, &reconcile_venv_dir).is_err();
                if needs_reconciliation || current_probe_failed {
                    reconcile_auto_tagger_dependencies(
                        &reconcile_app_dir,
                        &reconcile_venv_dir,
                        &dependencies,
                        std::env::consts::OS,
                        current_probe_failed,
                    )?;
                }
                Ok::<(), String>(())
            })
            .await
            .map_err(|error| format!("Dependency reconciliation task failed: {}", error))?;
            if let Err(error) = result {
                self.set_status(id, AddonStatus::Error(error.clone()));
                return Err(format!(
                    "Auto Tagger dependency reconciliation failed: {}",
                    error
                ));
            }
        }

        let sidecar_env = self.sidecar_env(id);

        // Spawn the sidecar
        let child = match sidecar::spawn_sidecar(id, &python, &app_dir, port, &sidecar_env).await {
            Ok(c) => c,
            Err(e) => {
                self.set_status(id, AddonStatus::Error(e.clone()));
                return Err(e);
            }
        };

        let process = Arc::new(TokioMutex::new(child));

        // Store the process only if this start still owns the Starting state.
        let accepted = if let Some(mut state) = self.addons.get_mut(id) {
            if state.status == AddonStatus::Starting {
                state.process = Some(process.clone());
                true
            } else {
                false
            }
        } else {
            false
        };
        if !accepted {
            let mut child = process.lock().await;
            if let Some(pid) = child.id() {
                sidecar::kill_process(pid);
            }
            let _ = child.start_kill();
            return Err(format!("Addon '{}' start was cancelled", id));
        }
        drop(lifecycle);

        // Wait for the addon to become healthy
        let healthy = sidecar::wait_for_healthy(port, Duration::from_secs(30)).await;

        if healthy {
            let accepted = if let Some(mut state) = self.addons.get_mut(id) {
                let owns_process = state
                    .process
                    .as_ref()
                    .is_some_and(|tracked| Arc::ptr_eq(tracked, &process));
                if state.status == AddonStatus::Starting && owns_process {
                    state.status = AddonStatus::Running;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !accepted {
                {
                    let mut child = process.lock().await;
                    if let Some(pid) = child.id() {
                        sidecar::kill_process(pid);
                    }
                    let _ = child.start_kill();
                }
                if let Some(mut state) = self.addons.get_mut(id) {
                    let owns_process = state
                        .process
                        .as_ref()
                        .is_some_and(|tracked| Arc::ptr_eq(tracked, &process));
                    if owns_process {
                        state.process = None;
                    }
                }
                return Err(format!("Addon '{}' start was cancelled", id));
            }
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
            let msg = format!("Addon '{}' failed to become healthy within 30s", id);
            let accepted = if let Some(mut state) = self.addons.get_mut(id) {
                let owns_process = state
                    .process
                    .as_ref()
                    .is_some_and(|tracked| Arc::ptr_eq(tracked, &process));
                if state.status == AddonStatus::Starting && owns_process {
                    state.process = None;
                    state.status = AddonStatus::Error(msg.clone());
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if accepted {
                Err(msg)
            } else {
                Err(format!("Addon '{}' start was cancelled", id))
            }
        }
    }

    /// Stop a running addon by killing its sidecar process.
    pub async fn stop_addon(&self, id: &str) -> Result<(), String> {
        self.stop_addon_inner(id, false).await
    }

    pub async fn stop_addon_for_repair(&self, id: &str) -> Result<(), String> {
        self.stop_addon_inner(id, true).await
    }

    async fn stop_addon_inner(&self, id: &str, repairing_operation: bool) -> Result<(), String> {
        let _lifecycle = self.lifecycle.read().await;
        let manifest = get_addon_manifest(id).ok_or_else(|| format!("Unknown addon: {}", id))?;
        if manifest.runtime == AddonRuntime::Builtin {
            return Err(format!("Addon '{}' does not require a sidecar", id));
        }

        let (process, repairing) = {
            let mut state = self
                .addons
                .get_mut(id)
                .ok_or_else(|| format!("No state for addon '{}'", id))?;
            let repairing = state.status == AddonStatus::Repairing;
            if repairing {
                if !repairing_operation {
                    return Err(format!("Addon '{}' is being repaired", id));
                }
            } else {
                if state.status == AddonStatus::Stopping {
                    return Err(format!("Addon '{}' is already stopping", id));
                }
                state.status = AddonStatus::Stopping;
            }
            (state.process.take(), repairing)
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

        if repairing {
            if self.get_addon_status(id) != AddonStatus::Repairing {
                return Err(format!("Addon '{}' repair stop was cancelled", id));
            }
        } else {
            let status = if self.venv_dir(id).exists() {
                AddonStatus::Installed
            } else {
                AddonStatus::NotInstalled
            };
            let accepted = if let Some(mut state) = self.addons.get_mut(id) {
                if state.status == AddonStatus::Stopping {
                    state.status = status;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !accepted {
                return Err(format!("Addon '{}' stop was cancelled", id));
            }
            self.save_enabled_state(id, false);
        }
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
    pub async fn stop_all(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        let _lifecycle = self.lifecycle.write().await;
        log::info!("[AddonManager] Stopping all running addons...");

        let mut processes = Vec::new();
        for mut entry in self.addons.iter_mut() {
            let id = entry.key().clone();
            if let Some(process) = entry.value_mut().process.take() {
                processes.push((id, process));
            }
            // Mark stopped unless uninstalled
            if entry.value().status != AddonStatus::NotInstalled {
                entry.value_mut().status = AddonStatus::Stopped;
            }
        }

        for (id, process) in processes {
            let mut child = process.lock().await;
            if let Some(pid) = child.id() {
                log::info!("[AddonManager] Killing addon '{}' (PID {})", id, pid);
                sidecar::kill_process(pid);
            }
            if let Err(error) = child.start_kill() {
                log::warn!("[AddonManager] Failed to kill addon '{}': {}", id, error);
            }
            if tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .is_err()
            {
                log::warn!(
                    "[AddonManager] Timed out waiting for addon '{}' to exit",
                    id
                );
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
    // AC: @auto-tagger-runtime-acceleration-deployment ac-compatible-cudnn
    #[test]
    fn auto_tagger_gpu_platforms_include_managed_cuda_dependencies() {
        for target_os in ["windows", "linux"] {
            let deps = dependency_list_for_platform("auto-tagger", &["onnxruntime"], target_os);
            assert!(deps.contains(&ONNXRUNTIME_GPU));
            assert!(deps.contains(&"nvidia-cudnn-cu12==9.24.0.43"));
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
        assert!(!deps
            .iter()
            .any(|dependency| dependency.starts_with("nvidia-cudnn-cu12")));
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-3
    // AC: @auto-tagger-runtime-acceleration-deployment ac-4
    // AC: @auto-tagger-runtime-acceleration-deployment ac-compatible-cudnn
    #[test]
    fn auto_tagger_revision_detects_missing_stale_and_current_deployments() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-revision-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let dependencies =
            dependency_list_for_platform("auto-tagger", &["onnxruntime"], std::env::consts::OS);
        let desired = auto_tagger_dependency_revision(&dependencies, std::env::consts::OS);
        let fallback = auto_tagger_dependency_revision(
            &cpu_fallback_dependencies(&dependencies),
            std::env::consts::OS,
        );
        assert!(desired.contains(AUTO_TAGGER_SOURCE_REVISION));
        assert!(needs_dependency_reconciliation(&root, &desired, &fallback));

        let stale = AddonDependencyDeployment {
            desired_revision: "old".into(),
            installed_revision: "old".into(),
            runtime: "cpu".into(),
            probe: serde_json::json!({}),
            warning: None,
        };
        write_dependency_deployment(&root, &stale).unwrap();
        assert!(needs_dependency_reconciliation(&root, &desired, &fallback));

        let current = AddonDependencyDeployment {
            desired_revision: desired.clone(),
            installed_revision: desired.clone(),
            runtime: "cuda".into(),
            probe: serde_json::json!({
                "onnxruntime": ONNXRUNTIME_VERSION,
                "packages": {
                    "onnxruntime-gpu": ONNXRUNTIME_VERSION,
                    "nvidia-cudnn-cu12": "9.24.0.43"
                }
            }),
            warning: None,
        };
        write_dependency_deployment(&root, &current).unwrap();
        assert!(!needs_dependency_reconciliation(&root, &desired, &fallback));

        let incompatible_cudnn = AddonDependencyDeployment {
            probe: serde_json::json!({
                "onnxruntime": ONNXRUNTIME_VERSION,
                "packages": {
                    "onnxruntime-gpu": ONNXRUNTIME_VERSION,
                    "nvidia-cudnn-cu12": "9.25.0.15"
                }
            }),
            ..current.clone()
        };
        write_dependency_deployment(&root, &incompatible_cudnn).unwrap();
        assert!(needs_dependency_reconciliation(&root, &desired, &fallback));

        let missing_cudnn = AddonDependencyDeployment {
            probe: serde_json::json!({
                "onnxruntime": ONNXRUNTIME_VERSION,
                "packages": {"onnxruntime-gpu": ONNXRUNTIME_VERSION}
            }),
            ..current
        };
        write_dependency_deployment(&root, &missing_cudnn).unwrap();
        assert!(needs_dependency_reconciliation(&root, &desired, &fallback));

        let accepted_cpu_fallback = AddonDependencyDeployment {
            desired_revision: desired.clone(),
            installed_revision: fallback.clone(),
            runtime: "cpu".into(),
            probe: serde_json::json!({
                "onnxruntime": ONNXRUNTIME_VERSION,
                "packages": {"onnxruntime": ONNXRUNTIME_VERSION}
            }),
            warning: Some("GPU install failed; restored CPU".into()),
        };
        write_dependency_deployment(&root, &accepted_cpu_fallback).unwrap();
        assert!(!needs_dependency_reconciliation(&root, &desired, &fallback));
        assert!(probe_committed_deployment(&root, &root.join("missing-venv")).is_err());

        let stale_cpu_fallback = AddonDependencyDeployment {
            installed_revision: "unknown-cpu-runtime".into(),
            ..accepted_cpu_fallback
        };
        write_dependency_deployment(&root, &stale_cpu_fallback).unwrap();
        assert!(needs_dependency_reconciliation(&root, &desired, &fallback));
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-compatible-cudnn
    #[cfg(target_os = "linux")]
    #[test]
    fn committed_cuda_probe_rejects_live_cudnn_version_drift() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-live-cudnn-test-{}",
            uuid::Uuid::new_v4()
        ));
        let venv_bin = root.join("venv/bin");
        std::fs::create_dir_all(&venv_bin).unwrap();
        let deployment = AddonDependencyDeployment {
            desired_revision: "desired".into(),
            installed_revision: "desired".into(),
            runtime: "cuda".into(),
            probe: serde_json::json!({
                "onnxruntime": ONNXRUNTIME_VERSION,
                "packages": {
                    "onnxruntime-gpu": ONNXRUNTIME_VERSION,
                    "nvidia-cudnn-cu12": NVIDIA_CUDNN_VERSION
                }
            }),
            warning: None,
        };
        write_dependency_deployment(&root, &deployment).unwrap();
        let python = venv_bin.join("python");
        std::fs::write(
            &python,
            "#!/bin/sh\necho '{\"onnxruntime\":\"1.23.2\",\"available_providers\":[\"CUDAExecutionProvider\",\"CPUExecutionProvider\"],\"packages\":{\"onnxruntime-gpu\":\"1.23.2\",\"nvidia-cublas-cu12\":\"12.9\",\"nvidia-cuda-runtime-cu12\":\"12.9\",\"nvidia-cudnn-cu12\":\"9.25.0.15\"}}'\n",
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();

        let error = probe_committed_deployment(&root, &root.join("venv")).unwrap_err();

        assert!(error.contains("exact managed dependency set"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-2
    #[test]
    fn sidecar_environment_exposes_committed_deployment_evidence() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-environment-test-{}",
            uuid::Uuid::new_v4()
        ));
        let addon_dir = root.join("addons/auto-tagger");
        std::fs::create_dir_all(&addon_dir).unwrap();
        let deployment = AddonDependencyDeployment {
            desired_revision: "desired".into(),
            installed_revision: "installed-cpu".into(),
            runtime: "cpu".into(),
            probe: serde_json::json!({}),
            warning: Some("verified fallback".into()),
        };
        write_dependency_deployment(&addon_dir, &deployment).unwrap();
        let manager = AddonManager::new(&root);

        let environment: std::collections::HashMap<_, _> =
            manager.sidecar_env("auto-tagger").into_iter().collect();

        assert_eq!(
            environment.get("TAGGER_DEPLOYMENT_INSTALLED"),
            Some(&"installed-cpu".to_string())
        );
        assert_eq!(
            environment.get("TAGGER_DEPLOYMENT_RUNTIME"),
            Some(&"cpu".to_string())
        );
        assert_eq!(
            environment.get("TAGGER_DEPLOYMENT_WARNING"),
            Some(&"verified fallback".to_string())
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-3
    #[test]
    fn repair_claim_rejects_concurrent_repair() {
        let manager = AddonManager::new(Path::new("unused"));
        manager.set_status("auto-tagger", AddonStatus::Running);

        assert_eq!(manager.begin_repair("auto-tagger"), Ok(true));
        assert_eq!(
            manager.get_addon_status("auto-tagger"),
            AddonStatus::Repairing
        );
        assert!(manager.begin_repair("auto-tagger").is_err());
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-3
    #[tokio::test]
    async fn repair_ownership_rejects_external_stop_and_uninstall() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-lifecycle-test-{}",
            uuid::Uuid::new_v4()
        ));
        let manager = AddonManager::new(&root);
        manager.set_status("auto-tagger", AddonStatus::Repairing);

        assert!(manager.stop_addon("auto-tagger").await.is_err());
        manager.set_status("auto-tagger", AddonStatus::Starting);
        assert!(manager.uninstall_addon("auto-tagger").is_err());
        manager.set_status("auto-tagger", AddonStatus::Stopping);
        assert!(manager.stop_addon("auto-tagger").await.is_err());
        assert!(manager.begin_repair("auto-tagger").is_err());
        let _ = std::fs::remove_dir_all(root);
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

    // AC: @auto-tagger-runtime-acceleration-deployment ac-4
    #[cfg(target_os = "linux")]
    #[test]
    fn failed_runtime_probe_does_not_commit_dependency_revision() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-probe-failure-test-{}",
            uuid::Uuid::new_v4()
        ));
        let addon_dir = root.join("addons/auto-tagger");
        let venv = addon_dir.join("venv");
        let bin = venv.join("bin");
        let python = bin.join("python");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            &python,
            "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then echo 'probe failed' >&2; exit 1; fi\nexit 0\n",
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let dependencies =
            dependency_list_for_platform("auto-tagger", &["onnxruntime"], std::env::consts::OS);
        let prior = AddonDependencyDeployment {
            desired_revision: auto_tagger_dependency_revision(&dependencies, std::env::consts::OS),
            installed_revision: "previous-success".into(),
            runtime: "cuda".into(),
            probe: serde_json::json!({}),
            warning: None,
        };
        write_dependency_deployment(&addon_dir, &prior).unwrap();

        let error = reconcile_auto_tagger_dependencies(
            &addon_dir,
            &venv,
            &dependencies,
            std::env::consts::OS,
            true,
        )
        .unwrap_err();

        assert!(error.contains("probe failed"));
        assert!(read_dependency_deployment(&addon_dir).is_none());
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @auto-tagger-runtime-acceleration-deployment ac-4
    // AC: @auto-tagger-runtime-acceleration-deployment ac-5
    #[cfg(target_os = "linux")]
    #[test]
    fn unavailable_cuda_provider_restores_and_records_verified_cpu_runtime() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "localbooru-addon-provider-fallback-test-{}",
            uuid::Uuid::new_v4()
        ));
        let addon_dir = root.join("addons/auto-tagger");
        let venv = addon_dir.join("venv");
        let bin = venv.join("bin");
        let python = bin.join("python");
        let cpu_mode = root.join("cpu-mode");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then\n  if [ -f '{}' ]; then echo '{{\"onnxruntime\":\"1.23.2\",\"available_providers\":[\"CPUExecutionProvider\"],\"packages\":{{\"onnxruntime\":\"1.23.2\"}}}}';\n  else echo '{{\"onnxruntime\":\"1.23.2\",\"available_providers\":[\"CPUExecutionProvider\"],\"packages\":{{\"onnxruntime-gpu\":\"1.23.2\",\"nvidia-cublas-cu12\":\"12.9\",\"nvidia-cuda-runtime-cu12\":\"12.9\",\"nvidia-cudnn-cu12\":\"9.24.0.43\"}}}}'; fi\n  exit 0\nfi\nfor arg in \"$@\"; do [ \"$arg\" = '{}' ] && touch '{}'; done\nexit 0\n",
                cpu_mode.display(),
                ONNXRUNTIME_CPU,
                cpu_mode.display(),
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let dependencies =
            dependency_list_for_platform("auto-tagger", &["onnxruntime"], std::env::consts::OS);

        let deployment = reconcile_auto_tagger_dependencies(
            &addon_dir,
            &venv,
            &dependencies,
            std::env::consts::OS,
            true,
        )
        .unwrap();

        assert_eq!(deployment.runtime, "cpu");
        assert_ne!(deployment.installed_revision, deployment.desired_revision);
        assert!(deployment
            .warning
            .as_deref()
            .unwrap()
            .contains("could not load CUDAExecutionProvider"));
        assert_eq!(
            deployment.probe["available_providers"],
            serde_json::json!(["CPUExecutionProvider"])
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @addon-platform-dependencies ac-3
    // AC: @auto-tagger-runtime-acceleration-deployment ac-3
    // AC: @auto-tagger-runtime-acceleration-deployment ac-4
    #[cfg(target_os = "linux")]
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
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'Python 3.11.9'; elif [ \"$1\" = \"-c\" ]; then echo '{{\"onnxruntime\":\"1.23.2\",\"available_providers\":[\"CUDAExecutionProvider\",\"CPUExecutionProvider\"],\"packages\":{{\"onnxruntime-gpu\":\"1.23.2\",\"nvidia-cublas-cu12\":\"12.9\",\"nvidia-cuda-runtime-cu12\":\"12.9\",\"nvidia-cudnn-cu12\":\"9.24.0.43\"}}}}'; else printf '%s\\n' \"$*\" >> '{}'; fi\nexit 0\n",
                invocations.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let manager = AddonManager::new(&root);

        manager.repair_addon("auto-tagger").unwrap();

        let invocations = std::fs::read_to_string(invocations).unwrap();
        let commands: Vec<_> = invocations.lines().collect();
        assert_eq!(
            commands[0],
            "-m pip uninstall -y onnxruntime onnxruntime-gpu nvidia-cudnn-cu12"
        );
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
        let deployment = read_dependency_deployment(&addon_dir).unwrap();
        assert_eq!(deployment.runtime, "cuda");
        assert_eq!(deployment.installed_revision, deployment.desired_revision);
        assert!(!needs_dependency_reconciliation(
            &addon_dir,
            &deployment.desired_revision,
            "",
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @addon-platform-dependencies ac-3
    // AC: @auto-tagger-runtime-acceleration-deployment ac-5
    #[cfg(target_os = "linux")]
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
                "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then echo '{{\"onnxruntime\":\"1.23.2\",\"available_providers\":[\"CPUExecutionProvider\"],\"packages\":{{\"onnxruntime\":\"1.23.2\"}}}}'; exit 0; fi\nprintf '%s\\n' \"$*\" >> '{}'\nfor arg in \"$@\"; do [ \"$arg\" = '{}' ] && exit 1; done\nexit 0\n",
                invocations.display(),
                ONNXRUNTIME_GPU
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();
        let dependencies =
            dependency_list_for_platform("auto-tagger", &["onnxruntime"], std::env::consts::OS);

        let deployment = reconcile_auto_tagger_dependencies(
            &root,
            &venv,
            &dependencies,
            std::env::consts::OS,
            true,
        )
        .unwrap();

        assert_eq!(deployment.runtime, "cpu");
        assert_ne!(deployment.installed_revision, deployment.desired_revision);
        assert!(!needs_dependency_reconciliation(
            &root,
            &deployment.desired_revision,
            &deployment.installed_revision,
        ));
        assert!(deployment
            .warning
            .as_deref()
            .unwrap()
            .contains("restored the CPU inference runtime"));
        let commands = std::fs::read_to_string(invocations).unwrap();
        let commands: Vec<_> = commands.lines().collect();
        assert_eq!(
            commands[0],
            "-m pip uninstall -y onnxruntime onnxruntime-gpu nvidia-cudnn-cu12"
        );
        assert!(commands[1].contains(ONNXRUNTIME_GPU));
        assert_eq!(
            commands[2],
            "-m pip uninstall -y onnxruntime-gpu nvidia-cudnn-cu12"
        );
        assert!(commands[3].starts_with("-m pip install --upgrade "));
        assert!(commands[3].contains(ONNXRUNTIME_CPU));
        assert!(commands[3].contains("uvicorn[standard]"));
        assert!(commands[3].contains("fastapi"));
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @explicit-exit-process-cleanup ac-1
    // AC: @explicit-exit-process-cleanup ac-2
    #[tokio::test]
    async fn stop_all_waits_for_contended_process_and_reaps_it() {
        #[cfg(unix)]
        let mut command = {
            let mut command = tokio::process::Command::new("sh");
            command.args(["-c", "sleep 60"]);
            command
        };
        #[cfg(windows)]
        let mut command = {
            let mut command = tokio::process::Command::new("cmd");
            command.args(["/C", "ping -n 60 127.0.0.1 >NUL"]);
            command
        };
        command.kill_on_drop(true);
        let child = command.spawn().unwrap();

        let process = Arc::new(TokioMutex::new(child));
        let manager = Arc::new(AddonManager::new(Path::new("unused")));
        manager.addons.insert(
            "auto-tagger".to_string(),
            AddonState {
                status: AddonStatus::Running,
                process: Some(process.clone()),
            },
        );

        let guard = process.lock().await;
        let shutdown_manager = manager.clone();
        let shutdown = tokio::spawn(async move {
            shutdown_manager.stop_all().await;
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!shutdown.is_finished());

        drop(guard);
        tokio::time::timeout(Duration::from_secs(10), shutdown)
            .await
            .expect("shutdown timed out")
            .expect("shutdown task failed");

        let mut child = process.lock().await;
        assert!(child.try_wait().unwrap().is_some());
    }

    // AC: @explicit-exit-process-cleanup ac-1
    // AC: @explicit-exit-process-cleanup ac-2
    #[tokio::test]
    async fn stop_all_rejects_new_addon_starts() {
        let manager = AddonManager::new(Path::new("unused"));

        manager.stop_all().await;

        assert_eq!(
            manager.start_addon("auto-tagger").await.unwrap_err(),
            "Application shutdown is in progress"
        );
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
