//! Embedded Python sources for addon sidecars.
//!
//! Addon sources are embedded at compile time via `include_str!()`, so the
//! binary is self-contained and can deploy addons without external files.

pub type AddonSource = (&'static str, &'static str);

pub fn write_source(root: &std::path::Path, name: &str, source: &str) -> std::io::Result<()> {
    let path = root.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, source)
}

/// Get the embedded Python sources for an addon, if available.
pub fn get_addon_sources(id: &str) -> Option<&'static [AddonSource]> {
    match id {
        "donut-create" => Some(&[
            (
                "app.py",
                include_str!("../../../addons/donut-create/app.py"),
            ),
            (
                "installer.py",
                include_str!("../../../addons/donut-create/installer.py"),
            ),
            (
                "runtime.json",
                include_str!("../../../addons/donut-create/runtime.json"),
            ),
            (
                "workflow.json",
                include_str!("../../../addons/donut-create/workflow.json"),
            ),
            (
                "model_sources.json",
                include_str!("../../../addons/donut-create/model_sources.json"),
            ),
            (
                "assets/donut-create.js",
                include_str!("../../../addons/donut-create/assets/donut-create.js"),
            ),
            (
                "assets/donut-create.css",
                include_str!("../../../addons/donut-create/assets/donut-create.css"),
            ),
            (
                "assets/base-workflow.json",
                include_str!("../../../addons/donut-create/assets/base-workflow.json"),
            ),
        ]),
        "auto-tagger" => Some(&[
            ("app.py", include_str!("../../../addons/auto-tagger/app.py")),
            (
                "runtime_probe.py",
                include_str!("../../../addons/auto-tagger/runtime_probe.py"),
            ),
        ]),
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

#[cfg(test)]
mod tests {
    use super::get_addon_sources;

    #[test]
    fn auto_tagger_deploys_the_real_model_runtime_probe() {
        // AC: @auto-tagger-runtime-acceleration-deployment ac-1
        let source_names: Vec<_> = get_addon_sources("auto-tagger")
            .unwrap()
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(source_names, ["app.py", "runtime_probe.py"]);
    }

    // AC: @donut-create-plugin ac-managed-setup
    #[test]
    fn creation_sources_deploy_with_nested_assets_and_matching_base_preset() {
        let temporary = tempfile::tempdir().unwrap();
        let sources = get_addon_sources("donut-create").unwrap();
        for (name, source) in sources {
            super::write_source(temporary.path(), name, source).unwrap();
            assert_eq!(
                std::fs::read_to_string(temporary.path().join(name)).unwrap(),
                *source
            );
        }
        let preset: serde_json::Value = serde_json::from_slice(
            &std::fs::read(temporary.path().join("assets/base-workflow.json")).unwrap(),
        )
        .unwrap();
        assert!(!preset["nodes"].as_array().unwrap().is_empty());
        assert!(!preset["definitions"]["subgraphs"]
            .as_array()
            .unwrap()
            .is_empty());
    }
}
