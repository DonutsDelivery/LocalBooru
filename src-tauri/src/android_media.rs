//! Native permission-aware selection of Android's on-device media folders.
use serde_json::Value;
use tauri::{
    plugin::{PluginHandle, TauriPlugin},
    Wry,
};

static MEDIA: std::sync::OnceLock<PluginHandle<Wry>> = std::sync::OnceLock::new();

pub fn metadata(path: &str) -> Option<(i32, i32, f64)> {
    let data: Value = MEDIA
        .get()?
        .run_mobile_plugin("mediaInfo", serde_json::json!({ "path": path }))
        .ok()?;
    Some((
        data.get("width")?.as_i64()? as i32,
        data.get("height")?.as_i64()? as i32,
        data.get("duration")?.as_f64()?,
    ))
}

pub fn thumbnail(path: &str, output: &str, size: u32) -> bool {
    MEDIA
        .get()
        .and_then(|handle| {
            handle
                .run_mobile_plugin::<Value>(
                    "videoThumbnail",
                    serde_json::json!({ "path": path, "output": output, "size": size }),
                )
                .ok()
        })
        .is_some_and(|value| value.get("saved").and_then(Value::as_bool) == Some(true))
}

pub fn previews(
    path: &str,
    output_dir: &std::path::Path,
    count: usize,
    size: u32,
) -> Vec<std::path::PathBuf> {
    let Some(handle) = MEDIA.get() else {
        return Vec::new();
    };
    let result: Result<Value, _> = handle.run_mobile_plugin(
        "videoPreviews",
        serde_json::json!({ "path": path, "output": output_dir, "count": count, "size": size }),
    );
    match result
        .ok()
        .and_then(|value| value.get("files").and_then(Value::as_array).cloned())
    {
        Some(files) => files
            .into_iter()
            .filter_map(|path| path.as_str().map(std::path::PathBuf::from))
            .collect(),
        None => Vec::new(),
    }
}

pub fn init() -> TauriPlugin<Wry> {
    tauri::plugin::Builder::new("local-media")
        .setup(|_app, api| {
            let handle = api.register_android_plugin("com.localbooru.app", "LocalMediaPlugin")?;
            MEDIA
                .set(handle)
                .map_err(|_| "Android media bridge already initialized")?;
            Ok(())
        })
        .build()
}

#[tauri::command]
pub async fn android_pick_media_directory(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, crate::server::state::AppState>,
) -> Result<Value, String> {
    if window.label() != "main" || state.get_remote_proxy().await.is_some() {
        return Err("Switch to This Device to add folders from this phone.".into());
    }
    MEDIA
        .get()
        .ok_or("Android media access is unavailable")?
        .run_mobile_plugin_async("pickDirectory", serde_json::json!({}))
        .await
        .map_err(|error| error.to_string())
}
