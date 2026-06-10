//! Addon Manifest Definitions and Registry
//!
//! Static registry of all available addons with their metadata,
//! port assignments, and Python dependency lists.

use std::sync::LazyLock;

/// Static manifest describing an addon's identity and requirements.
pub struct AddonManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub port: u16,
    pub python_deps: &'static [&'static str],
}

/// The complete registry of all known addons.
static ADDON_REGISTRY: LazyLock<Vec<AddonManifest>> = LazyLock::new(|| {
    vec![
        AddonManifest {
            id: "auto-tagger",
            name: "Auto Tagger",
            description: "Automatically tag images using ONNX-based classification models",
            port: 18001,
            // huggingface-hub pinned (M1, supply-chain). onnxruntime/numpy/Pillow
            // left unpinned: native/heavy, version constrained by the resolver.
            python_deps: &["onnxruntime", "numpy", "Pillow", "huggingface-hub==1.18.0"],
        },
        AddonManifest {
            id: "age-detector",
            name: "Age Detector",
            description: "Detect and classify age ratings in images using deep learning",
            port: 18002,
            python_deps: &["torch", "torchvision", "transformers", "ultralytics", "insightface", "onnxruntime", "numpy", "Pillow", "opencv-python-headless"],
        },
        AddonManifest {
            id: "whisper-subtitles",
            name: "Whisper Subtitles",
            description: "Generate subtitles from video audio using Whisper speech recognition",
            port: 18003,
            // faster-whisper pinned (M1). numpy left to faster-whisper's resolver.
            python_deps: &["faster-whisper==1.2.1", "numpy"],
        },
        AddonManifest {
            id: "frame-interpolation",
            name: "Frame Interpolation",
            description: "Increase video frame rate using optical flow and neural network interpolation",
            port: 18004,
            python_deps: &["numpy", "opencv-python", "rife-ncnn-vulkan-python-tntwise"],
        },
        AddonManifest {
            id: "cast",
            name: "Chromecast/DLNA",
            description: "Cast media to Chromecast and DLNA-compatible devices on the local network",
            port: 18006,
            // pychromecast pinned (M1, top-level lib); it resolves compatible
            // async-upnp-client/aiohttp, so those stay unpinned to avoid a cap conflict.
            python_deps: &["pychromecast==14.0.10", "async-upnp-client", "aiohttp"],
        },
        AddonManifest {
            id: "svp",
            name: "SVP (SmoothVideo Project)",
            description: "High quality frame interpolation using VapourSynth and SVPflow",
            port: 18008,
            python_deps: &["vapoursynth"],
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
