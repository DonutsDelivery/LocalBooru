//! Interpolation backend detection and capabilities
//!
//! This module handles detection of available interpolation backends
//! and provides information about their capabilities.

use std::path::Path;
use std::process::Command;
use serde::{Deserialize, Serialize};
use super::config::InterpolationBackend;

/// Capabilities and status of an interpolation backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// The backend type
    pub backend: InterpolationBackend,

    /// Whether the backend is available on this system
    pub available: bool,

    /// Human-readable status message
    pub status: String,

    /// Path to the executable/library (if applicable)
    pub path: Option<String>,

    /// Version string (if detectable)
    pub version: Option<String>,

    /// Whether GPU acceleration is available
    pub gpu_available: bool,

    /// Whether NVIDIA Optical Flow is available (SVP only)
    pub nvof_available: bool,

    /// Maximum recommended resolution for real-time
    pub max_realtime_resolution: Option<(u32, u32)>,

    /// Estimated performance tier (1-5, higher = faster)
    pub performance_tier: u8,
}

impl BackendCapabilities {
    fn unavailable(backend: InterpolationBackend, reason: &str) -> Self {
        Self {
            backend,
            available: false,
            status: reason.to_string(),
            path: None,
            version: None,
            gpu_available: false,
            nvof_available: false,
            max_realtime_resolution: None,
            performance_tier: 0,
        }
    }
}

/// SVP installation paths to check
const SVP_PATHS: &[&str] = &[
    "/opt/svp",
    "/usr/share/svp",
    "/usr/local/svp",
];

/// SVP plugin library names
const SVP_PLUGINS: &[&str] = &[
    "libsvpflow1.so",
    "libsvpflow2.so",
];

/// RIFE-NCNN binary names to check
const RIFE_BINARIES: &[&str] = &[
    "rife-ncnn-vulkan",
    "rife-ncnn",
];

/// Detect SVP installation and capabilities
fn detect_svp() -> BackendCapabilities {
    // Find SVP installation
    let svp_path = SVP_PATHS.iter()
        .map(Path::new)
        .find(|p| p.exists());

    let Some(svp_dir) = svp_path else {
        return BackendCapabilities::unavailable(
            InterpolationBackend::Svp,
            "SVP not found. Install from https://www.svp-team.com/"
        );
    };

    // Check for required plugins
    let plugins_dir = svp_dir.join("plugins");
    let has_plugins = SVP_PLUGINS.iter().all(|plugin| {
        plugins_dir.join(plugin).exists()
    });

    if !has_plugins {
        return BackendCapabilities::unavailable(
            InterpolationBackend::Svp,
            "SVP plugins missing (libsvpflow1.so, libsvpflow2.so)"
        );
    }

    // Check for mpv with vapoursynth support
    let mpv_check = Command::new("mpv")
        .args(["--version"])
        .output();

    let has_mpv = mpv_check.is_ok();

    // Check for vspipe (VapourSynth)
    let vspipe_check = Command::new("vspipe")
        .args(["--version"])
        .output();

    let has_vspipe = vspipe_check.is_ok();

    // Detect NVIDIA GPU for NVOF
    let nvof_available = detect_nvidia_gpu();

    let status = if has_mpv && has_vspipe {
        if nvof_available {
            "SVP ready with NVIDIA Optical Flow"
        } else {
            "SVP ready (CPU motion estimation)"
        }
    } else if !has_mpv {
        "SVP found but mpv missing"
    } else {
        "SVP found but vspipe missing"
    };

    BackendCapabilities {
        backend: InterpolationBackend::Svp,
        available: has_mpv && has_vspipe,
        status: status.to_string(),
        path: Some(svp_dir.to_string_lossy().to_string()),
        version: None, // SVP doesn't have a simple version query
        gpu_available: true, // SVP always uses GPU for rendering
        nvof_available,
        max_realtime_resolution: if nvof_available {
            Some((3840, 2160)) // 4K with NVOF
        } else {
            Some((1920, 1080)) // 1080p without
        },
        performance_tier: if nvof_available { 5 } else { 3 },
    }
}

/// Detect RIFE-NCNN installation
fn detect_rife_ncnn() -> BackendCapabilities {
    // Check for rife-ncnn-vulkan binary
    let rife_path = RIFE_BINARIES.iter()
        .find_map(|name| {
            Command::new("which")
                .arg(name)
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        });

    // Also check SVP's bundled RIFE
    let svp_rife = SVP_PATHS.iter()
        .map(|p| Path::new(p).join("rife/libRIFE.so"))
        .find(|p| p.exists());

    let (path, source) = if let Some(ref p) = rife_path {
        (Some(p.clone()), "system")
    } else if let Some(ref p) = svp_rife {
        (Some(p.to_string_lossy().to_string()), "SVP bundled")
    } else {
        (None, "")
    };

    if path.is_none() {
        return BackendCapabilities::unavailable(
            InterpolationBackend::RifeNcnn,
            "RIFE-NCNN not found. Install rife-ncnn-vulkan or use SVP's bundled RIFE."
        );
    }

    // Check Vulkan availability
    let vulkan_check = Command::new("vulkaninfo")
        .args(["--summary"])
        .output();

    let has_vulkan = vulkan_check
        .map(|o| o.status.success())
        .unwrap_or(false);

    BackendCapabilities {
        backend: InterpolationBackend::RifeNcnn,
        available: has_vulkan,
        status: if has_vulkan {
            format!("RIFE-NCNN ready ({})", source)
        } else {
            "RIFE-NCNN found but Vulkan unavailable".to_string()
        },
        path,
        version: None,
        gpu_available: has_vulkan,
        nvof_available: false,
        max_realtime_resolution: Some((2560, 1440)), // 1440p
        performance_tier: 4,
    }
}

/// Detect FFmpeg minterpolate availability
fn detect_minterpolate() -> BackendCapabilities {
    let ffmpeg_check = Command::new("ffmpeg")
        .args(["-filters"])
        .output();

    let has_minterpolate = ffmpeg_check
        .map(|o| {
            o.status.success() &&
            String::from_utf8_lossy(&o.stdout).contains("minterpolate")
        })
        .unwrap_or(false);

    // Get FFmpeg version
    let version = Command::new("ffmpeg")
        .args(["-version"])
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|l| l.to_string())
        })
        .flatten();

    if !has_minterpolate {
        return BackendCapabilities::unavailable(
            InterpolationBackend::Minterpolate,
            "FFmpeg minterpolate filter not available"
        );
    }

    BackendCapabilities {
        backend: InterpolationBackend::Minterpolate,
        available: true,
        status: "FFmpeg minterpolate ready (CPU-based)".to_string(),
        path: Some("/usr/bin/ffmpeg".to_string()),
        version,
        gpu_available: false,
        nvof_available: false,
        max_realtime_resolution: Some((1280, 720)), // 720p max for real-time
        performance_tier: 1, // Slowest
    }
}

/// Detect NVIDIA GPU presence (for NVOF support)
fn detect_nvidia_gpu() -> bool {
    // Check nvidia-smi
    let nvidia_smi = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output();

    if nvidia_smi.map(|o| o.status.success()).unwrap_or(false) {
        return true;
    }

    // Check for nvidia kernel module
    let lsmod = Command::new("lsmod")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("nvidia"))
        .unwrap_or(false);

    lsmod
}

/// Detect all available interpolation backends
pub fn detect_available_backends() -> Vec<BackendCapabilities> {
    vec![
        detect_svp(),
        detect_rife_ncnn(),
        detect_minterpolate(),
    ]
}

/// Get the best available backend for the given resolution
pub fn recommend_backend(
    width: u32,
    height: u32,
    prefer_quality: bool,
) -> Option<InterpolationBackend> {
    let backends = detect_available_backends();

    // Filter to available backends that can handle the resolution in real-time
    let viable: Vec<_> = backends.iter()
        .filter(|b| b.available)
        .filter(|b| {
            b.max_realtime_resolution
                .map(|(max_w, max_h)| width <= max_w && height <= max_h)
                .unwrap_or(true)
        })
        .collect();

    if viable.is_empty() {
        return None;
    }

    // Sort by preference
    if prefer_quality {
        // Prefer SVP > RIFE > minterpolate
        viable.iter()
            .max_by_key(|b| match b.backend {
                InterpolationBackend::Svp => 3,
                InterpolationBackend::RifeNcnn => 2,
                InterpolationBackend::Minterpolate => 1,
                InterpolationBackend::None => 0,
            })
            .map(|b| b.backend)
    } else {
        // Prefer by performance tier
        viable.iter()
            .max_by_key(|b| b.performance_tier)
            .map(|b| b.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_backends() {
        let backends = detect_available_backends();
        assert_eq!(backends.len(), 3);

        // Should have SVP, RIFE, and minterpolate entries
        assert!(backends.iter().any(|b| b.backend == InterpolationBackend::Svp));
        assert!(backends.iter().any(|b| b.backend == InterpolationBackend::RifeNcnn));
        assert!(backends.iter().any(|b| b.backend == InterpolationBackend::Minterpolate));
    }
}
