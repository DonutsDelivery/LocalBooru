//! Embedded Python sources for addon sidecars.
//!
//! Each addon's `app.py` is embedded at compile time via `include_str!()`,
//! so the binary is self-contained and can deploy addons without external files.

/// Get the embedded Python source for an addon, if available.
pub fn get_addon_source(id: &str) -> Option<&'static str> {
    match id {
        "auto-tagger" => Some(include_str!("../../../addons/auto-tagger/app.py")),
        "age-detector" => Some(include_str!("../../../addons/age-detector/app.py")),
        "whisper-subtitles" => Some(include_str!("../../../addons/whisper-subtitles/app.py")),
        "frame-interpolation" => Some(include_str!("../../../addons/frame-interpolation/app.py")),
        "cast" => Some(include_str!("../../../addons/cast/app.py")),
        "svp" => Some(include_str!("../../../addons/svp/app.py")),
        _ => None,
    }
}
