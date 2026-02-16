//! Embedded Python sources for addon sidecars.
//!
//! Each addon's `app.py` is embedded at compile time via `include_str!()`,
//! so the binary is self-contained and can deploy addons without external files.

/// Get the embedded Python source for an addon, if available.
pub fn get_addon_source(id: &str) -> Option<&'static str> {
    match id {
        "frame-interpolation" => Some(include_str!("../../../addons/frame-interpolation/app.py")),
        "svp" => Some(include_str!("../../../addons/svp/app.py")),
        _ => None,
    }
}
