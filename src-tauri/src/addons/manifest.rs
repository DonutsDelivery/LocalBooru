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
            python_deps: &["onnxruntime"],
        },
        AddonManifest {
            id: "age-detector",
            name: "Age Detector",
            description: "Detect and classify age ratings in images using deep learning",
            port: 18002,
            python_deps: &["torch", "transformers", "ultralytics"],
        },
        AddonManifest {
            id: "whisper-subtitles",
            name: "Whisper Subtitles",
            description: "Generate subtitles from video audio using Whisper speech recognition",
            port: 18003,
            python_deps: &["faster-whisper"],
        },
        AddonManifest {
            id: "frame-interpolation",
            name: "Frame Interpolation",
            description: "Increase video frame rate using optical flow and neural network interpolation",
            port: 18004,
            python_deps: &["numpy", "opencv-python", "rife-ncnn-vulkan-python-tntwise"],
        },
        AddonManifest {
            id: "video-transcode",
            name: "Video Transcode",
            description: "Transcode videos between formats using FFmpeg",
            port: 18005,
            python_deps: &[],
        },
        AddonManifest {
            id: "cast",
            name: "Chromecast/DLNA",
            description: "Cast media to Chromecast and DLNA-compatible devices on the local network",
            port: 18006,
            python_deps: &["pychromecast", "async-upnp-client"],
        },
        AddonManifest {
            id: "share-streaming",
            name: "Share Streaming",
            description: "Share media streams with other devices over the network",
            port: 18007,
            python_deps: &[],
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
