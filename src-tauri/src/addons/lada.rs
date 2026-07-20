use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::sidecar::{self, ManagedCommandError};

pub const LADA_ADDON_VERSION: &str = "0.1.0";
pub const LADA_PROTOCOL_VERSION: u32 = 1;
pub const LADA_UPSTREAM_REVISION: &str = "20cb34a20a83c72c87a991d2c949032c70085b16";
pub const LADA_MODEL_REVISION: &str = "bcf461d46d9a98981fc64b815df5178f42215cdf";
pub const LADA_DEPLOYMENT_FILE: &str = "deployment.json";
const LADA_READINESS_FILE: &str = "readiness.json";
const MAX_PROBE_OUTPUT_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LadaBackendPreference {
    Auto,
    Cuda,
    Xpu,
}

impl LadaBackendPreference {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::Xpu => "xpu",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LadaBackend {
    Cuda,
    Xpu,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct CudaCompatibility {
    pub package: String,
    pub variant: String,
    pub minimum_driver_major: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct XpuCompatibility {
    pub package: String,
    pub kernel_drivers: Vec<String>,
    pub requires_render_node: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaBackendCompatibility {
    pub cuda: CudaCompatibility,
    pub xpu: XpuCompatibility,
}

impl Default for LadaBackendCompatibility {
    fn default() -> Self {
        Self {
            cuda: CudaCompatibility {
                package: "linux_x86_64_cuda".into(),
                variant: "cu128".into(),
                minimum_driver_major: 570,
            },
            xpu: XpuCompatibility {
                package: "linux_x86_64_xpu".into(),
                kernel_drivers: vec!["i915".into(), "xe".into()],
                requires_render_node: true,
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AcceleratorCandidate {
    pub hardware_present: bool,
    pub driver_available: bool,
    pub driver_version: Option<String>,
    pub reason: Option<String>,
}

impl AcceleratorCandidate {
    fn absent() -> Self {
        Self {
            hardware_present: false,
            driver_available: false,
            driver_version: None,
            reason: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AcceleratorDetection {
    pub cuda: AcceleratorCandidate,
    pub xpu: AcceleratorCandidate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaBackendSelection {
    pub backend: LadaBackend,
    pub package: String,
    pub variant: Option<String>,
    pub driver_version: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadaSelectionError {
    pub status: LadaReadinessStatus,
    pub reason: String,
}

fn drm_vendor_paths(sys_root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(sys_root.join("class/drm")) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("card") && !name.contains('-')
        })
        .map(|entry| entry.path().join("device"))
        .collect()
}

fn nvidia_driver_major(version: &str) -> Option<u32> {
    version
        .split(|character: char| !character.is_ascii_digit() && character != '.')
        .filter(|part| part.contains('.'))
        .find_map(|part| part.split('.').next()?.parse().ok())
}

pub fn detect_linux_accelerators(
    sys_root: &Path,
    dev_root: &Path,
    proc_root: &Path,
) -> AcceleratorDetection {
    let devices = drm_vendor_paths(sys_root);
    let nvidia_hardware = devices.iter().any(|device| {
        std::fs::read_to_string(device.join("vendor"))
            .map(|vendor| vendor.trim().eq_ignore_ascii_case("0x10de"))
            .unwrap_or(false)
    });
    let intel_devices: Vec<_> = devices
        .iter()
        .filter(|device| {
            std::fs::read_to_string(device.join("vendor"))
                .map(|vendor| vendor.trim().eq_ignore_ascii_case("0x8086"))
                .unwrap_or(false)
        })
        .collect();

    let mut cuda = AcceleratorCandidate::absent();
    cuda.hardware_present = nvidia_hardware;
    if nvidia_hardware {
        let version = std::fs::read_to_string(proc_root.join("driver/nvidia/version")).ok();
        let major = version.as_deref().and_then(nvidia_driver_major);
        cuda.driver_available = dev_root.join("nvidiactl").exists() && major.is_some();
        cuda.driver_version = version.as_deref().and_then(|value| {
            value
                .split_whitespace()
                .find(|part| {
                    part.chars().next().is_some_and(|c| c.is_ascii_digit()) && part.contains('.')
                })
                .map(|part| {
                    part.trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
                        .to_string()
                })
        });
        if !cuda.driver_available {
            cuda.reason = Some(
                "An NVIDIA GPU was detected, but a loaded driver and /dev/nvidiactl are required"
                    .into(),
            );
        }
    }

    let mut xpu = AcceleratorCandidate::absent();
    xpu.hardware_present = !intel_devices.is_empty();
    if xpu.hardware_present {
        let kernel_driver = intel_devices.iter().find_map(|device| {
            let uevent = std::fs::read_to_string(device.join("uevent")).ok()?;
            let driver = uevent.lines().find_map(|line| {
                line.strip_prefix("DRIVER=")
                    .filter(|driver| matches!(*driver, "i915" | "xe"))
            })?;
            let has_render_node = std::fs::read_dir(device.join("drm"))
                .ok()?
                .filter_map(Result::ok)
                .map(|entry| entry.file_name())
                .filter(|name| name.to_string_lossy().starts_with("renderD"))
                .any(|name| dev_root.join("dri").join(name).exists());
            has_render_node.then(|| driver.to_owned())
        });
        xpu.driver_available = kernel_driver.is_some();
        xpu.driver_version = kernel_driver;
        if !xpu.driver_available {
            xpu.reason = Some(
                "An Intel GPU was detected, but a usable i915/xe render device is unavailable"
                    .into(),
            );
        }
    }

    AcceleratorDetection { cuda, xpu }
}

pub fn detect_host_accelerators() -> AcceleratorDetection {
    #[cfg(target_os = "linux")]
    {
        detect_linux_accelerators(Path::new("/sys"), Path::new("/dev"), Path::new("/proc"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        AcceleratorDetection {
            cuda: AcceleratorCandidate::absent(),
            xpu: AcceleratorCandidate::absent(),
        }
    }
}

pub fn select_backend(
    detection: &AcceleratorDetection,
    preference: LadaBackendPreference,
) -> Result<LadaBackendSelection, LadaSelectionError> {
    select_backend_for_release(detection, preference, &LadaBackendCompatibility::default())
}

pub fn select_backend_for_release(
    detection: &AcceleratorDetection,
    preference: LadaBackendPreference,
    compatibility: &LadaBackendCompatibility,
) -> Result<LadaBackendSelection, LadaSelectionError> {
    let cuda_major = detection
        .cuda
        .driver_version
        .as_deref()
        .and_then(nvidia_driver_major);
    let cuda = (detection.cuda.driver_available
        && cuda_major.is_some_and(|major| major >= compatibility.cuda.minimum_driver_major))
    .then(|| LadaBackendSelection {
        backend: LadaBackend::Cuda,
        package: compatibility.cuda.package.clone(),
        variant: Some(compatibility.cuda.variant.clone()),
        driver_version: detection.cuda.driver_version.clone(),
    });
    let xpu_driver_compatible = detection
        .xpu
        .driver_version
        .as_ref()
        .is_some_and(|driver| compatibility.xpu.kernel_drivers.contains(driver));
    let xpu =
        (detection.xpu.driver_available && xpu_driver_compatible).then(|| LadaBackendSelection {
            backend: LadaBackend::Xpu,
            package: compatibility.xpu.package.clone(),
            variant: None,
            driver_version: detection.xpu.driver_version.clone(),
        });

    let selected = match preference {
        LadaBackendPreference::Auto => cuda.or(xpu),
        LadaBackendPreference::Cuda => cuda,
        LadaBackendPreference::Xpu => xpu,
    };
    if let Some(selected) = selected {
        return Ok(selected);
    }

    let requested_hardware_present = match preference {
        LadaBackendPreference::Auto => {
            detection.cuda.hardware_present || detection.xpu.hardware_present
        }
        LadaBackendPreference::Cuda => detection.cuda.hardware_present,
        LadaBackendPreference::Xpu => detection.xpu.hardware_present,
    };
    let reason = match preference {
        LadaBackendPreference::Cuda if detection.cuda.driver_available => Some(format!(
            "NVIDIA driver {} is installed, but this {} bundle requires driver {} or newer",
            detection
                .cuda
                .driver_version
                .as_deref()
                .unwrap_or("unknown"),
            compatibility.cuda.variant,
            compatibility.cuda.minimum_driver_major
        )),
        LadaBackendPreference::Cuda => detection.cuda.reason.clone(),
        LadaBackendPreference::Xpu => detection.xpu.reason.clone(),
        LadaBackendPreference::Auto => detection
            .cuda
            .reason
            .clone()
            .or_else(|| detection.xpu.reason.clone())
            .or_else(|| {
                detection.cuda.driver_available.then(|| {
                    format!(
                        "NVIDIA driver {} is incompatible with the available {} bundle",
                        detection
                            .cuda
                            .driver_version
                            .as_deref()
                            .unwrap_or("unknown"),
                        compatibility.cuda.variant
                    )
                })
            }),
    };
    if requested_hardware_present {
        Err(LadaSelectionError {
            status: LadaReadinessStatus::IncompatibleDriver,
            reason: reason
                .unwrap_or_else(|| "A detected accelerator has no compatible driver".into()),
        })
    } else {
        Err(LadaSelectionError {
            status: LadaReadinessStatus::AcceleratorUnavailable,
            reason: "No supported NVIDIA CUDA or Intel XPU accelerator was detected".into(),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LadaReadinessStatus {
    UnsupportedPlatform,
    NotInstalled,
    AcceleratorUnavailable,
    IncompatibleDriver,
    Downloading,
    Installing,
    Probing,
    RepairRequired,
    UpdateAvailable,
    Ready,
    RuntimeFailure,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LadaReadiness {
    pub status: LadaReadinessStatus,
    pub reason: Option<String>,
    pub configured_backend: LadaBackendPreference,
    pub active_backend: Option<LadaBackend>,
    pub probe_evidence: Option<Value>,
}

impl LadaReadiness {
    pub fn new(
        status: LadaReadinessStatus,
        configured_backend: LadaBackendPreference,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            status,
            reason: Some(reason.into()),
            configured_backend,
            active_backend: None,
            probe_evidence: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaDeployment {
    pub addon_version: String,
    pub protocol_version: u32,
    pub upstream_revision: String,
    pub model_revision: String,
    pub backend_compatibility: LadaBackendCompatibility,
    pub selected_backend: LadaBackend,
    pub selected_package: String,
    pub executable: PathBuf,
    pub probe_config: PathBuf,
}

impl LadaDeployment {
    pub fn load(addon_dir: &Path) -> Result<Self, String> {
        let path = addon_dir.join(LADA_DEPLOYMENT_FILE);
        let contents = std::fs::read(&path)
            .map_err(|error| format!("Failed to read LADA deployment: {}", error))?;
        serde_json::from_slice(&contents)
            .map_err(|error| format!("Failed to parse LADA deployment: {}", error))
    }

    fn identity_error(&self) -> Option<(LadaReadinessStatus, String)> {
        if self.addon_version != LADA_ADDON_VERSION {
            return Some((
                LadaReadinessStatus::UpdateAvailable,
                format!(
                    "LADA add-on {} is installed; {} is required",
                    self.addon_version, LADA_ADDON_VERSION
                ),
            ));
        }
        if self.protocol_version != LADA_PROTOCOL_VERSION {
            return Some((
                LadaReadinessStatus::RepairRequired,
                "The installed LADA protocol revision is incompatible; repair the add-on".into(),
            ));
        }
        if self.upstream_revision != LADA_UPSTREAM_REVISION
            || self.model_revision != LADA_MODEL_REVISION
        {
            return Some((
                LadaReadinessStatus::RepairRequired,
                "The installed LADA source or model revision is incompatible; repair the add-on"
                    .into(),
            ));
        }
        let expected_package = match self.selected_backend {
            LadaBackend::Cuda => &self.backend_compatibility.cuda.package,
            LadaBackend::Xpu => &self.backend_compatibility.xpu.package,
        };
        if &self.selected_package != expected_package {
            return Some((
                LadaReadinessStatus::RepairRequired,
                "The installed LADA backend package does not match its release manifest; repair the add-on"
                    .into(),
            ));
        }
        None
    }
}

fn deployment_path_error(addon_dir: &Path, deployment: &LadaDeployment) -> Option<String> {
    let root = match std::fs::canonicalize(addon_dir) {
        Ok(root) => root,
        Err(error) => return Some(format!("The LADA deployment is unavailable: {}", error)),
    };
    for (label, path, executable) in [
        ("executable", &deployment.executable, true),
        ("probe configuration", &deployment.probe_config, false),
    ] {
        let canonical = match std::fs::canonicalize(path) {
            Ok(canonical) => canonical,
            Err(error) => return Some(format!("The LADA {} is unavailable: {}", label, error)),
        };
        if !canonical.starts_with(&root) {
            return Some(format!(
                "The LADA {} is outside the managed deployment; repair the add-on",
                label
            ));
        }
        let metadata = match std::fs::metadata(&canonical) {
            Ok(metadata) => metadata,
            Err(error) => return Some(format!("The LADA {} is unavailable: {}", label, error)),
        };
        if !metadata.is_file() {
            return Some(format!(
                "The LADA {} is not a regular file; repair the add-on",
                label
            ));
        }
        #[cfg(unix)]
        if executable {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                return Some("The LADA executable cannot be run; repair the add-on".to_string());
            }
        }
        #[cfg(not(unix))]
        let _ = executable;
    }
    None
}

#[derive(Debug, Deserialize, Serialize)]
struct SidecarProbeReport {
    #[serde(default)]
    addon_version: String,
    protocol_version: u32,
    upstream_revision: String,
    revision_compatible: bool,
    model_revision: String,
    model_revision_compatible: bool,
    weights_ready: bool,
    requested_backend: String,
    active_backend: Option<LadaBackend>,
    ready: bool,
    reason: Option<String>,
    #[serde(default)]
    issues: Vec<String>,
    #[serde(default)]
    backend_evidence: Value,
    #[serde(default)]
    model_evidence: Value,
    model_error: Option<String>,
    backend_error: Option<String>,
}

fn concise(value: &str) -> String {
    let value = value.trim();
    if value.chars().count() <= 512 {
        return value.to_string();
    }
    value.chars().take(512).collect::<String>() + "…"
}

fn readiness_from_report(
    report: SidecarProbeReport,
    preference: LadaBackendPreference,
    expected_backend: LadaBackend,
    compatibility: &LadaBackendCompatibility,
) -> LadaReadiness {
    let identity_matches = report.addon_version == LADA_ADDON_VERSION
        && report.protocol_version == LADA_PROTOCOL_VERSION
        && report.upstream_revision == LADA_UPSTREAM_REVISION
        && report.revision_compatible
        && report.model_revision == LADA_MODEL_REVISION
        && report.model_revision_compatible;
    if !identity_matches || !report.weights_ready {
        return LadaReadiness::new(
            LadaReadinessStatus::RepairRequired,
            preference,
            "The installed LADA protocol, source, or model files failed verification; repair the add-on",
        );
    }

    let tensor_proven = report
        .backend_evidence
        .get("tensor_operation")
        .and_then(Value::as_bool)
        == Some(true);
    let model_proven = report
        .model_evidence
        .get("model_path_operation")
        .and_then(Value::as_bool)
        == Some(true);
    let requested_matches = report.requested_backend == preference.as_str();
    let active_matches = report.active_backend == Some(expected_backend)
        && matches!(
            (preference, report.active_backend),
            (LadaBackendPreference::Auto, Some(_))
                | (LadaBackendPreference::Cuda, Some(LadaBackend::Cuda))
                | (LadaBackendPreference::Xpu, Some(LadaBackend::Xpu))
        );
    let evidence_active = report
        .backend_evidence
        .get("active")
        .and_then(Value::as_str);
    let evidence_matches = match report.active_backend {
        Some(LadaBackend::Cuda) => evidence_active == Some("cuda"),
        Some(LadaBackend::Xpu) => evidence_active == Some("xpu"),
        None => false,
    };
    if report.ready
        && tensor_proven
        && model_proven
        && requested_matches
        && active_matches
        && evidence_matches
    {
        return LadaReadiness {
            status: LadaReadinessStatus::Ready,
            reason: None,
            configured_backend: preference,
            active_backend: report.active_backend,
            probe_evidence: serde_json::to_value(&report).ok(),
        };
    }

    let accelerator_issue = report
        .issues
        .iter()
        .any(|issue| issue == "accelerator_unavailable");
    let accelerator_diagnosis = accelerator_issue.then(|| {
        select_backend_for_release(&detect_host_accelerators(), preference, compatibility)
            .err()
            .unwrap_or(
            LadaSelectionError {
                status: LadaReadinessStatus::IncompatibleDriver,
                reason: "The selected accelerator is present, but the installed runtime could not use its driver"
                    .into(),
            },
        )
    });
    let status = if report.backend_error.is_some() {
        LadaReadinessStatus::RuntimeFailure
    } else if let Some(diagnosis) = accelerator_diagnosis.as_ref() {
        diagnosis.status.clone()
    } else if report.issues.iter().any(|issue| {
        matches!(
            issue.as_str(),
            "incompatible_protocol" | "incompatible_revision" | "weights_invalid"
        )
    }) {
        LadaReadinessStatus::RepairRequired
    } else {
        LadaReadinessStatus::RuntimeFailure
    };
    let reason = report
        .backend_error
        .as_deref()
        .map(concise)
        .or_else(|| {
            accelerator_diagnosis
                .as_ref()
                .map(|diagnosis| diagnosis.reason.clone())
        })
        .or_else(|| report.model_error.as_deref().map(concise))
        .or_else(|| report.reason.as_deref().map(concise))
        .unwrap_or_else(|| "The LADA runtime probe did not prove an active accelerator".into());
    LadaReadiness::new(status, preference, reason)
}

#[cfg(not(target_os = "windows"))]
fn replace_readiness_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[cfg(target_os = "windows")]
fn replace_readiness_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub(crate) fn persist_readiness(addon_dir: &Path, readiness: &LadaReadiness) -> Result<(), String> {
    use std::io::Write;

    std::fs::create_dir_all(addon_dir)
        .map_err(|error| format!("Failed to create LADA state directory: {}", error))?;
    let path = addon_dir.join(LADA_READINESS_FILE);
    let temporary = addon_dir.join(format!(
        ".{}.{}.tmp",
        LADA_READINESS_FILE,
        uuid::Uuid::new_v4()
    ));
    let document = serde_json::to_vec_pretty(readiness)
        .map_err(|error| format!("Failed to serialize LADA readiness: {}", error))?;
    let write_result = (|| -> Result<(), String> {
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| format!("Failed to create LADA readiness state: {}", error))?;
        file.write_all(&document)
            .map_err(|error| format!("Failed to write LADA readiness: {}", error))?;
        file.sync_all()
            .map_err(|error| format!("Failed to sync LADA readiness: {}", error))?;
        replace_readiness_file(&temporary, &path)
            .map_err(|error| format!("Failed to commit LADA readiness: {}", error))?;
        #[cfg(unix)]
        std::fs::File::open(addon_dir)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("Failed to sync LADA state directory: {}", error))?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result
}

pub fn load_readiness(addon_dir: &Path) -> Option<LadaReadiness> {
    let contents = std::fs::read(addon_dir.join(LADA_READINESS_FILE)).ok()?;
    serde_json::from_slice(&contents).ok()
}

async fn persist_readiness_async(
    addon_dir: &Path,
    readiness: &LadaReadiness,
) -> Result<(), String> {
    let addon_dir = addon_dir.to_path_buf();
    let readiness = readiness.clone();
    tokio::task::spawn_blocking(move || persist_readiness(&addon_dir, &readiness))
        .await
        .map_err(|error| format!("Failed to persist LADA readiness: {}", error))?
}

pub fn current_readiness(addon_dir: &Path, preference: LadaBackendPreference) -> LadaReadiness {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return LadaReadiness::new(
            LadaReadinessStatus::UnsupportedPlatform,
            preference,
            "LADA video restoration currently requires Linux x86_64",
        );
    }
    let deployment_path = addon_dir.join(LADA_DEPLOYMENT_FILE);
    if !deployment_path.exists() {
        return LadaReadiness::new(
            LadaReadinessStatus::NotInstalled,
            preference,
            "Install the LADA add-on to enable video restoration",
        );
    }
    let deployment = match LadaDeployment::load(addon_dir) {
        Ok(deployment) => deployment,
        Err(error) => {
            return LadaReadiness::new(
                LadaReadinessStatus::RepairRequired,
                preference,
                format!("{}; repair the add-on", concise(&error)),
            );
        }
    };
    if let Some((status, reason)) = deployment.identity_error() {
        return LadaReadiness::new(status, preference, reason);
    }
    if let Some(error) = deployment_path_error(addon_dir, &deployment) {
        return LadaReadiness::new(LadaReadinessStatus::RepairRequired, preference, error);
    }
    match load_readiness(addon_dir) {
        Some(readiness) if readiness.configured_backend != preference => LadaReadiness::new(
            LadaReadinessStatus::RepairRequired,
            preference,
            "The selected accelerator has changed and must be verified before playback",
        ),
        Some(readiness)
            if matches!(
                readiness.status,
                LadaReadinessStatus::Downloading
                    | LadaReadinessStatus::Installing
                    | LadaReadinessStatus::Probing
            ) =>
        {
            LadaReadiness::new(
                LadaReadinessStatus::RuntimeFailure,
                preference,
                "The previous LADA operation was interrupted; retry or repair the add-on",
            )
        }
        Some(readiness) if readiness.status == LadaReadinessStatus::Ready => LadaReadiness::new(
            LadaReadinessStatus::RepairRequired,
            preference,
            "The installed LADA runtime must be verified for this app session",
        ),
        Some(readiness) => readiness,
        None => LadaReadiness::new(
            LadaReadinessStatus::RepairRequired,
            preference,
            "The installed LADA runtime has no verified probe evidence; repair the add-on",
        ),
    }
}

pub async fn probe_and_persist(
    addon_dir: &Path,
    deployment: &LadaDeployment,
    preference: LadaBackendPreference,
    timeout: Duration,
) -> LadaReadiness {
    let initial = if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some(LadaReadiness::new(
            LadaReadinessStatus::UnsupportedPlatform,
            preference,
            "LADA video restoration currently requires Linux x86_64",
        ))
    } else if let Some(error) = deployment_path_error(addon_dir, deployment) {
        Some(LadaReadiness::new(
            LadaReadinessStatus::RepairRequired,
            preference,
            error,
        ))
    } else if matches!(
        (preference, deployment.selected_backend),
        (LadaBackendPreference::Cuda, LadaBackend::Xpu)
            | (LadaBackendPreference::Xpu, LadaBackend::Cuda)
    ) {
        Some(LadaReadiness::new(
            LadaReadinessStatus::RepairRequired,
            preference,
            "The selected accelerator does not match the installed LADA backend; update or repair the add-on",
        ))
    } else {
        deployment
            .identity_error()
            .map(|(status, reason)| LadaReadiness::new(status, preference, reason))
    };
    if let Some(readiness) = initial {
        let _ = persist_readiness_async(addon_dir, &readiness).await;
        return readiness;
    }

    let probing = LadaReadiness::new(
        LadaReadinessStatus::Probing,
        preference,
        "Verifying the selected accelerator with the LADA models",
    );
    if let Err(error) = persist_readiness_async(addon_dir, &probing).await {
        return LadaReadiness::new(LadaReadinessStatus::RuntimeFailure, preference, error);
    }

    let arguments = vec![
        "probe".to_string(),
        "--config".to_string(),
        deployment.probe_config.to_string_lossy().into_owned(),
        "--backend".to_string(),
        preference.as_str().to_string(),
    ];
    let output = sidecar::run_managed_command(
        &deployment.executable,
        &arguments,
        addon_dir,
        timeout,
        MAX_PROBE_OUTPUT_BYTES,
    )
    .await;
    let readiness = match output {
        Err(ManagedCommandError::TimedOut(_)) => LadaReadiness::new(
            LadaReadinessStatus::RuntimeFailure,
            preference,
            format!(
                "LADA runtime probe timed out after {:.1} seconds",
                timeout.as_secs_f64()
            ),
        ),
        Err(error) => LadaReadiness::new(
            LadaReadinessStatus::RuntimeFailure,
            preference,
            concise(&error.to_string()),
        ),
        Ok(output) => match serde_json::from_str::<SidecarProbeReport>(output.stdout.trim()) {
            Ok(report) if !output.success && report.ready => LadaReadiness::new(
                LadaReadinessStatus::RuntimeFailure,
                preference,
                format!(
                    "LADA runtime probe reported ready but exited with status {}",
                    output
                        .exit_code
                        .map(|code| code.to_string())
                        .unwrap_or_else(|| "unknown".into())
                ),
            ),
            Ok(report) => readiness_from_report(
                report,
                preference,
                deployment.selected_backend,
                &deployment.backend_compatibility,
            ),
            Err(error) => {
                let detail = if output.stderr.trim().is_empty() {
                    format!("invalid probe response: {}", error)
                } else {
                    concise(output.stderr.trim())
                };
                LadaReadiness::new(
                    LadaReadinessStatus::RuntimeFailure,
                    preference,
                    format!(
                        "LADA runtime probe failed{}: {}",
                        output
                            .exit_code
                            .map(|code| format!(" with exit code {}", code))
                            .unwrap_or_default(),
                        detail
                    ),
                )
            }
        },
    };
    if let Err(error) = persist_readiness_async(addon_dir, &readiness).await {
        return LadaReadiness::new(LadaReadinessStatus::RuntimeFailure, preference, error);
    }
    readiness
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn test_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "localbooru-lada-probe-test-{}",
            uuid::Uuid::new_v4()
        ))
    }

    fn pinned_deployment(root: &Path) -> LadaDeployment {
        LadaDeployment {
            addon_version: LADA_ADDON_VERSION.into(),
            protocol_version: LADA_PROTOCOL_VERSION,
            upstream_revision: LADA_UPSTREAM_REVISION.into(),
            model_revision: LADA_MODEL_REVISION.into(),
            backend_compatibility: LadaBackendCompatibility::default(),
            selected_backend: LadaBackend::Cuda,
            selected_package: "linux_x86_64_cuda".into(),
            executable: root.join("probe"),
            probe_config: root.join("probe.json"),
        }
    }

    fn write_executable(path: &Path, source: &str) {
        std::fs::write(path, source).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    // AC: @lada-runtime-readiness ac-ready-only-after-proof
    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn readiness_requires_pinned_identity_and_real_probe_evidence() {
        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        let deployment = pinned_deployment(&root);
        std::fs::write(&deployment.probe_config, "{}").unwrap();
        write_executable(
            &deployment.executable,
            r#"#!/bin/sh
printf '%s\n' '{"addon_version":"0.1.0","protocol_version":1,"upstream_revision":"20cb34a20a83c72c87a991d2c949032c70085b16","revision_compatible":true,"model_revision":"bcf461d46d9a98981fc64b815df5178f42215cdf","model_revision_compatible":true,"weights_ready":true,"requested_backend":"auto","available_backends":["cuda"],"active_backend":"cuda","device":"Test GPU","ready":true,"reason":null,"issues":[],"backend_evidence":{"active":"cuda","tensor_operation":true},"model_evidence":{"model_path_operation":true},"model_error":null}'
"#,
        );

        let result = probe_and_persist(
            &root,
            &deployment,
            LadaBackendPreference::Auto,
            Duration::from_secs(2),
        )
        .await;

        assert_eq!(result.status, LadaReadinessStatus::Ready);
        assert_eq!(result.configured_backend, LadaBackendPreference::Auto);
        assert_eq!(result.active_backend, Some(LadaBackend::Cuda));
        assert!(result.probe_evidence.is_some());
        assert_eq!(load_readiness(&root).unwrap(), result);
        std::fs::write(
            root.join(LADA_DEPLOYMENT_FILE),
            serde_json::to_vec(&deployment).unwrap(),
        )
        .unwrap();
        assert_eq!(
            current_readiness(&root, LadaBackendPreference::Auto).status,
            LadaReadinessStatus::RepairRequired
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @lada-runtime-readiness ac-active-backend-evidence
    #[test]
    #[cfg(target_os = "linux")]
    fn persisted_evidence_cannot_claim_a_different_configured_backend() {
        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        let deployment = pinned_deployment(&root);
        std::fs::write(
            root.join(LADA_DEPLOYMENT_FILE),
            serde_json::to_vec(&deployment).unwrap(),
        )
        .unwrap();
        persist_readiness(
            &root,
            &LadaReadiness {
                status: LadaReadinessStatus::Ready,
                reason: None,
                configured_backend: LadaBackendPreference::Auto,
                active_backend: Some(LadaBackend::Cuda),
                probe_evidence: Some(serde_json::json!({"model_path_operation": true})),
            },
        )
        .unwrap();

        let current = current_readiness(&root, LadaBackendPreference::Xpu);

        assert_eq!(current.status, LadaReadinessStatus::RepairRequired);
        assert_eq!(current.configured_backend, LadaBackendPreference::Xpu);
        assert_eq!(current.active_backend, None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn malformed_deployment_requires_repair() {
        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(LADA_DEPLOYMENT_FILE), b"not json").unwrap();

        let readiness = current_readiness(&root, LadaBackendPreference::Auto);

        assert_eq!(readiness.status, LadaReadinessStatus::RepairRequired);
        assert!(readiness.reason.as_deref().unwrap().contains("repair"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(unix)]
    fn deployment_requires_regular_executable_and_config_files() {
        use std::os::unix::fs::PermissionsExt;

        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        let deployment = pinned_deployment(&root);
        std::fs::write(&deployment.executable, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(
            &deployment.executable,
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        std::fs::create_dir_all(&deployment.probe_config).unwrap();

        let config_error = deployment_path_error(&root, &deployment).unwrap();
        assert!(config_error.contains("not a regular file"));

        std::fs::remove_dir(&deployment.probe_config).unwrap();
        std::fs::write(&deployment.probe_config, "{}").unwrap();
        std::fs::set_permissions(
            &deployment.executable,
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        let executable_error = deployment_path_error(&root, &deployment).unwrap();
        assert!(executable_error.contains("cannot be run"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn interrupted_probe_becomes_runtime_failure() {
        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        let deployment = pinned_deployment(&root);
        std::fs::write(&deployment.probe_config, "{}").unwrap();
        write_executable(&deployment.executable, "#!/bin/sh\nexit 0\n");
        std::fs::write(
            root.join(LADA_DEPLOYMENT_FILE),
            serde_json::to_vec(&deployment).unwrap(),
        )
        .unwrap();
        persist_readiness(
            &root,
            &LadaReadiness {
                status: LadaReadinessStatus::Probing,
                reason: None,
                configured_backend: LadaBackendPreference::Cuda,
                active_backend: None,
                probe_evidence: None,
            },
        )
        .unwrap();

        let readiness = current_readiness(&root, LadaBackendPreference::Cuda);

        assert_eq!(readiness.status, LadaReadinessStatus::RuntimeFailure);
        assert!(readiness.reason.as_deref().unwrap().contains("interrupted"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_cuda_preference_rejects_xpu_active_evidence() {
        let report = SidecarProbeReport {
            addon_version: LADA_ADDON_VERSION.into(),
            protocol_version: LADA_PROTOCOL_VERSION,
            upstream_revision: LADA_UPSTREAM_REVISION.into(),
            revision_compatible: true,
            model_revision: LADA_MODEL_REVISION.into(),
            model_revision_compatible: true,
            weights_ready: true,
            requested_backend: "cuda".into(),
            active_backend: Some(LadaBackend::Xpu),
            ready: true,
            reason: None,
            issues: vec![],
            backend_evidence: serde_json::json!({
                "active": "xpu",
                "tensor_operation": true,
            }),
            model_evidence: serde_json::json!({"model_path_operation": true}),
            model_error: None,
            backend_error: None,
        };

        let readiness = readiness_from_report(
            report,
            LadaBackendPreference::Cuda,
            LadaBackend::Cuda,
            &LadaBackendCompatibility::default(),
        );

        assert_eq!(readiness.status, LadaReadinessStatus::RuntimeFailure);
        assert_eq!(readiness.active_backend, None);
    }

    // AC: @lada-runtime-readiness ac-ready-only-after-proof
    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn mismatched_deployment_is_repair_required_without_running_probe() {
        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        let mut deployment = pinned_deployment(&root);
        deployment.model_revision = "wrong".into();
        write_executable(
            &deployment.executable,
            &format!("#!/bin/sh\ntouch '{}'\n", root.join("ran").display()),
        );

        let result = probe_and_persist(
            &root,
            &deployment,
            LadaBackendPreference::Cuda,
            Duration::from_secs(2),
        )
        .await;

        assert_eq!(result.status, LadaReadinessStatus::RepairRequired);
        assert_eq!(result.active_backend, None);
        assert!(!root.join("ran").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @lada-runtime-readiness ac-actionable-status
    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn hung_probe_is_killed_at_parent_deadline_and_persisted() {
        let root = test_root();
        std::fs::create_dir_all(&root).unwrap();
        let deployment = pinned_deployment(&root);
        std::fs::write(&deployment.probe_config, "{}").unwrap();
        write_executable(&deployment.executable, "#!/bin/sh\nsleep 30\n");
        let started = Instant::now();

        let result = probe_and_persist(
            &root,
            &deployment,
            LadaBackendPreference::Cuda,
            Duration::from_millis(100),
        )
        .await;

        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(result.status, LadaReadinessStatus::RuntimeFailure);
        assert!(result.reason.as_deref().unwrap().contains("timed out"));
        assert_eq!(result.active_backend, None);
        assert_eq!(load_readiness(&root).unwrap(), result);
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @lada-runtime-readiness ac-actionable-status
    #[test]
    fn readiness_statuses_are_stable_and_actionable() {
        let statuses = [
            LadaReadinessStatus::UnsupportedPlatform,
            LadaReadinessStatus::AcceleratorUnavailable,
            LadaReadinessStatus::IncompatibleDriver,
            LadaReadinessStatus::Downloading,
            LadaReadinessStatus::Installing,
            LadaReadinessStatus::Probing,
            LadaReadinessStatus::RepairRequired,
            LadaReadinessStatus::UpdateAvailable,
            LadaReadinessStatus::RuntimeFailure,
        ];
        let encoded = serde_json::to_value(statuses).unwrap();
        assert_eq!(encoded[0], "unsupported_platform");
        assert_eq!(encoded[2], "incompatible_driver");
        assert_eq!(encoded[8], "runtime_failure");
    }

    // AC: @lada-runtime-readiness ac-active-backend-evidence
    #[test]
    fn configured_preference_is_not_reported_as_active_backend() {
        let readiness = LadaReadiness::new(
            LadaReadinessStatus::AcceleratorUnavailable,
            LadaBackendPreference::Cuda,
            "No compatible accelerator was detected",
        );
        assert_eq!(readiness.configured_backend, LadaBackendPreference::Cuda);
        assert_eq!(readiness.active_backend, None);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn hardware_and_driver_detection_selects_compatible_packages() {
        let root = test_root();
        let sys = root.join("sys");
        let dev = root.join("dev");
        let proc = root.join("proc");
        std::fs::create_dir_all(sys.join("class/drm/card0/device")).unwrap();
        std::fs::create_dir_all(dev.join("dri")).unwrap();
        std::fs::create_dir_all(proc.join("driver/nvidia")).unwrap();
        std::fs::write(sys.join("class/drm/card0/device/vendor"), "0x10de\n").unwrap();
        std::fs::write(
            proc.join("driver/nvidia/version"),
            "NVRM version: 570.86.16\n",
        )
        .unwrap();
        std::fs::write(dev.join("nvidiactl"), "").unwrap();

        let detection = detect_linux_accelerators(&sys, &dev, &proc);
        let selection = select_backend(&detection, LadaBackendPreference::Auto).unwrap();

        assert_eq!(selection.backend, LadaBackend::Cuda);
        assert_eq!(selection.package, "linux_x86_64_cuda");
        assert_eq!(selection.variant.as_deref(), Some("cu128"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn intel_render_node_must_belong_to_the_detected_device() {
        let root = test_root();
        let sys = root.join("sys");
        let dev = root.join("dev");
        let proc = root.join("proc");
        let intel_device = sys.join("class/drm/card0/device");
        std::fs::create_dir_all(intel_device.join("drm")).unwrap();
        std::fs::create_dir_all(dev.join("dri")).unwrap();
        std::fs::create_dir_all(&proc).unwrap();
        std::fs::write(intel_device.join("vendor"), "0x8086\n").unwrap();
        std::fs::write(intel_device.join("uevent"), "DRIVER=i915\n").unwrap();
        std::fs::write(dev.join("dri/renderD129"), "").unwrap();

        let unrelated = detect_linux_accelerators(&sys, &dev, &proc);
        assert!(unrelated.xpu.hardware_present);
        assert!(!unrelated.xpu.driver_available);

        std::fs::create_dir_all(intel_device.join("drm/renderD129")).unwrap();
        let associated = detect_linux_accelerators(&sys, &dev, &proc);
        assert!(associated.xpu.driver_available);
        assert_eq!(associated.xpu.driver_version.as_deref(), Some("i915"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn release_compatibility_selects_only_the_matching_cuda_variant() {
        let detection = AcceleratorDetection {
            cuda: AcceleratorCandidate {
                hardware_present: true,
                driver_available: true,
                driver_version: Some("560.35.03".into()),
                reason: None,
            },
            xpu: AcceleratorCandidate::absent(),
        };
        let current_error = select_backend(&detection, LadaBackendPreference::Cuda).unwrap_err();
        assert_eq!(
            current_error.status,
            LadaReadinessStatus::IncompatibleDriver
        );

        let mut compatibility = LadaBackendCompatibility::default();
        compatibility.cuda.variant = "cu126".into();
        compatibility.cuda.minimum_driver_major = 560;
        let selection =
            select_backend_for_release(&detection, LadaBackendPreference::Cuda, &compatibility)
                .unwrap();
        assert_eq!(selection.variant.as_deref(), Some("cu126"));
        assert_eq!(selection.package, "linux_x86_64_cuda");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn present_gpu_without_compatible_driver_is_distinguished() {
        let root = test_root();
        let sys = root.join("sys");
        let dev = root.join("dev");
        let proc = root.join("proc");
        std::fs::create_dir_all(sys.join("class/drm/card0/device")).unwrap();
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::create_dir_all(&proc).unwrap();
        std::fs::write(sys.join("class/drm/card0/device/vendor"), "0x10de\n").unwrap();

        let detection = detect_linux_accelerators(&sys, &dev, &proc);
        let error = select_backend(&detection, LadaBackendPreference::Auto).unwrap_err();

        assert_eq!(error.status, LadaReadinessStatus::IncompatibleDriver);
        let _ = std::fs::remove_dir_all(root);
    }
}
