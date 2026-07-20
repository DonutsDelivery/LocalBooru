//! Addon Manifest Definitions and Registry
//!
//! Static registry of all available addons with their metadata,
//! port assignments, and Python dependency lists.

use std::sync::LazyLock;

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AddonRuntime {
    Sidecar,
    Builtin,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AddonInstallation {
    PythonVenv,
    ManagedBundle,
    Builtin,
}

/// Static manifest describing an addon's identity and requirements.
pub struct AddonManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub runtime: AddonRuntime,
    pub installation: AddonInstallation,
    pub port: Option<u16>,
    pub python_deps: &'static [&'static str],
}

/// The complete registry of all known addons.
static ADDON_REGISTRY: LazyLock<Vec<AddonManifest>> = LazyLock::new(|| {
    vec![
        AddonManifest {
            id: "auto-tagger",
            name: "Auto Tagger",
            description: "Automatically tag images using ONNX-based classification models",
            runtime: AddonRuntime::Sidecar,
            installation: AddonInstallation::PythonVenv,
            port: Some(18001),
            // huggingface-hub pinned (M1, supply-chain). onnxruntime/numpy/Pillow
            // left unpinned: native/heavy, version constrained by the resolver.
            python_deps: &["onnxruntime", "numpy", "Pillow", "huggingface-hub==1.18.0"],
        },
        AddonManifest {
            id: "age-detector",
            name: "Age Detector",
            description: "Detect and classify age ratings in images using deep learning",
            runtime: AddonRuntime::Sidecar,
            installation: AddonInstallation::PythonVenv,
            port: Some(18002),
            python_deps: &[
                "torch",
                "torchvision",
                "transformers",
                "ultralytics",
                "insightface",
                "onnxruntime",
                "numpy",
                "Pillow",
                "opencv-python-headless",
            ],
        },
        AddonManifest {
            id: "whisper-subtitles",
            name: "Whisper Subtitles",
            description: "Generate subtitles from video audio using Whisper speech recognition",
            runtime: AddonRuntime::Sidecar,
            installation: AddonInstallation::PythonVenv,
            port: Some(18003),
            // faster-whisper pinned (M1). numpy left to faster-whisper's resolver.
            python_deps: &["faster-whisper==1.2.1", "numpy"],
        },
        AddonManifest {
            id: "cast",
            name: "Chromecast/DLNA",
            description:
                "Cast media to Chromecast and DLNA-compatible devices on the local network",
            runtime: AddonRuntime::Sidecar,
            installation: AddonInstallation::PythonVenv,
            port: Some(18006),
            // pychromecast pinned (M1, top-level lib); it resolves compatible
            // async-upnp-client/aiohttp, so those stay unpinned to avoid a cap conflict.
            python_deps: &["pychromecast==14.0.10", "async-upnp-client", "aiohttp"],
        },
        AddonManifest {
            id: "svp",
            name: "SVP (SmoothVideo Project)",
            description: "High quality frame interpolation using VapourSynth and SVPflow",
            runtime: AddonRuntime::Sidecar,
            installation: AddonInstallation::PythonVenv,
            port: Some(18008),
            python_deps: &["vapoursynth"],
        },
        AddonManifest {
            id: "lada",
            name: "LADA Video Restoration",
            description: "Restore mosaic-obscured video using a managed accelerated sidecar",
            runtime: AddonRuntime::Sidecar,
            installation: AddonInstallation::ManagedBundle,
            port: None,
            python_deps: &[],
        },
        AddonManifest {
            id: "curation-game",
            name: "Curation Game",
            description: "Rapidly keep or discard media from the current gallery view",
            runtime: AddonRuntime::Builtin,
            installation: AddonInstallation::Builtin,
            port: None,
            python_deps: &[],
        },
    ]
});

/// Get the full addon registry as a slice.
pub fn get_addon_registry() -> &'static [AddonManifest] {
    &ADDON_REGISTRY
}

/// Look up a single addon manifest by its ID.
pub fn get_addon_manifest(id: &str) -> Option<&'static AddonManifest> {
    ADDON_REGISTRY.iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    // AC: @curation-game ac-1
    #[test]
    fn curation_game_is_a_builtin_addon() {
        let addon = get_addon_manifest("curation-game").unwrap();
        assert_eq!(addon.runtime, AddonRuntime::Builtin);
        assert_eq!(addon.installation, AddonInstallation::Builtin);
        assert_eq!(addon.port, None);
        assert!(addon.python_deps.is_empty());
    }

    #[test]
    fn lada_runtime_and_installation_strategy_are_independent() {
        let addon = get_addon_manifest("lada").unwrap();
        assert_eq!(addon.runtime, AddonRuntime::Sidecar);
        assert_eq!(addon.installation, AddonInstallation::ManagedBundle);
        assert_eq!(addon.port, None);
        assert!(addon.python_deps.is_empty());
    }
}
