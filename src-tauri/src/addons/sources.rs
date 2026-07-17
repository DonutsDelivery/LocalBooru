//! Embedded Python sources for addon sidecars.
//!
//! Addon sources are embedded at compile time via `include_str!()`, so the
//! binary is self-contained and can deploy addons without external files.

pub type AddonSource = (&'static str, &'static str);

/// Get the embedded Python sources for an addon, if available.
pub fn get_addon_sources(id: &str) -> Option<&'static [AddonSource]> {
    match id {
        "auto-tagger" => Some(&[("app.py", include_str!("../../../addons/auto-tagger/app.py"))]),
        "age-detector" => Some(&[(
            "app.py",
            include_str!("../../../addons/age-detector/app.py"),
        )]),
        "whisper-subtitles" => Some(&[(
            "app.py",
            include_str!("../../../addons/whisper-subtitles/app.py"),
        )]),
        "cast" => Some(&[("app.py", include_str!("../../../addons/cast/app.py"))]),
        "svp" => Some(&[
            ("app.py", include_str!("../../../addons/svp/app.py")),
            (
                "session_protocol.py",
                include_str!("../../../addons/svp/session_protocol.py"),
            ),
            (
                "processing_session.py",
                include_str!("../../../addons/svp/processing_session.py"),
            ),
            (
                "session_api.py",
                include_str!("../../../addons/svp/session_api.py"),
            ),
            (
                "manager_graph.py",
                include_str!("../../../addons/svp/manager_graph.py"),
            ),
            (
                "fmp4_stream.py",
                include_str!("../../../addons/svp/fmp4_stream.py"),
            ),
            (
                "fmp4_processor.py",
                include_str!("../../../addons/svp/fmp4_processor.py"),
            ),
        ]),
        _ => None,
    }
}
