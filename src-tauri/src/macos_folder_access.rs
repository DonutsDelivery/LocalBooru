//! Native picker boundary for the macOS Lite folder grant store.
#[cfg(all(target_os = "macos", feature = "store-lite"))]
use crate::macos_grants::{apple, Grants};
#[cfg(all(target_os = "macos", feature = "store-lite"))]
use std::path::Path;

#[derive(serde::Serialize)]
pub struct AccessStatus {
    pub required: bool,
    pub failed: usize,
    pub storage_error: bool,
    pub failed_grants: Vec<FailedGrant>,
}

#[derive(serde::Serialize)]
pub struct FailedGrant {
    id: String,
    label: String,
}

#[cfg(all(target_os = "macos", feature = "store-lite"))]
fn grant_id(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    use std::os::unix::ffi::OsStrExt;
    format!("{:x}", Sha256::digest(path.as_os_str().as_bytes()))
}

#[cfg(all(target_os = "macos", feature = "store-lite"))]
struct NativeGrants {
    grants: Grants<apple::AppleBackend>,
    storage_error: bool,
}

#[cfg(all(target_os = "macos", feature = "store-lite"))]
static HELPER_BOOKMARK_DIRECTORY: std::sync::OnceLock<std::path::PathBuf> =
    std::sync::OnceLock::new();

/// Only native initialization supplies the child's bookmark store. A web request
/// or inherited environment variable cannot select a different grant directory.
#[cfg(all(target_os = "macos", feature = "store-lite"))]
pub(crate) fn helper_bookmark_directory() -> Option<&'static Path> {
    HELPER_BOOKMARK_DIRECTORY
        .get()
        .map(std::path::PathBuf::as_path)
}

/// Called before opening auxiliary libraries or starting any directory watcher.
#[cfg(all(target_os = "macos", feature = "store-lite"))]
pub fn initialize(app: &tauri::AppHandle, data: &Path) {
    use tauri::Manager;
    let directory = data.join("folder-grants");
    let mut grants = Grants::new(directory.clone(), apple::AppleBackend);
    let storage_error = grants.restore().is_err();
    let _ = HELPER_BOOKMARK_DIRECTORY.set(directory);
    app.manage(std::sync::Mutex::new(NativeGrants {
        grants,
        storage_error,
    }));
}

#[tauri::command]
pub fn folder_access_status(
    app: tauri::AppHandle,
    library: tauri::State<'_, crate::server::state::AppState>,
) -> Result<AccessStatus, String> {
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    {
        use tauri::Manager;
        let state = app.state::<std::sync::Mutex<NativeGrants>>();
        let state = state
            .lock()
            .map_err(|_| "Folder access is busy; try again")?;
        let mut labels = std::collections::HashMap::new();
        for lib in library.library_manager().all_mounted() {
            if let Ok(connection) = lib.main_pool.get() {
                if let Ok(mut query) = connection.prepare("SELECT path FROM watch_directories") {
                    if let Ok(paths) = query.query_map([], |row| row.get::<_, String>(0)) {
                        for path in paths.flatten() {
                            labels.insert(grant_id(Path::new(&path)), path);
                        }
                    }
                }
            }
        }
        let failed_grants = state
            .grants
            .failed_ids()
            .into_iter()
            .map(|id| FailedGrant {
                label: labels
                    .remove(&id)
                    .unwrap_or_else(|| "Unavailable folder".into()),
                id,
            })
            .collect();
        return Ok(AccessStatus {
            required: true,
            failed: state.grants.failed_count(),
            storage_error: state.storage_error,
            failed_grants,
        });
    }
    #[cfg(not(all(target_os = "macos", feature = "store-lite")))]
    {
        let _ = (app, library);
        Ok(AccessStatus {
            required: false,
            failed: 0,
            storage_error: false,
            failed_grants: vec![],
        })
    }
}

/// No path argument: only a native user selection can authorize persistent access.
#[tauri::command]
pub async fn pick_persistent_media_folder(
    app: tauri::AppHandle,
    state: tauri::State<'_, crate::server::state::AppState>,
    replacing_grant: Option<String>,
    previous_path: Option<String>,
) -> Result<Option<String>, String> {
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    {
        use tauri::Manager;
        use tauri_plugin_dialog::{DialogExt, FilePath};
        let selected = app
            .dialog()
            .file()
            .set_title("Select folder to allow access")
            .blocking_pick_folder();
        let path = match selected {
            None => return Ok(None),
            Some(FilePath::Path(path)) => path,
            _ => return Err("Select a local folder".into()),
        };
        let bookmark = apple::bookmark_from_selection(&path)?;
        {
            let grants = app.state::<std::sync::Mutex<NativeGrants>>();
            let mut grants = grants
                .lock()
                .map_err(|_| "Folder access is busy; try again")?;
            let id = if let Some(id) = replacing_grant {
                if !grants.grants.failed_ids().contains(&id) {
                    return Err("Folder recovery changed. Refresh and try again.".into());
                }
                id
            } else {
                previous_path
                    .as_deref()
                    .map(Path::new)
                    .map(grant_id)
                    .filter(|id| grants.grants.knows(id))
                    .unwrap_or_else(|| grant_id(&path))
            };
            grants.grants.persist_and_activate(&id, &bookmark)
                .map_err(|_| "Folder access could not be saved. Check available storage and select the folder again.")?;
            // A successful save says nothing about unread grants. Reconcile the
            // complete store before clearing a startup enumeration error.
            grants.storage_error = grants.grants.restore().is_err();
        }
        let path = path
            .to_str()
            .ok_or("This folder name is not supported")?
            .to_string();
        state.allow_asset_dir(&path);
        if let Some(watcher) = state.directory_watcher() {
            watcher.refresh();
        }
        return Ok(Some(path));
    }
    #[cfg(not(all(target_os = "macos", feature = "store-lite")))]
    {
        let _ = (app, state, replacing_grant, previous_path);
        Err("Persistent folder grants are only used by macOS Lite".into())
    }
}
