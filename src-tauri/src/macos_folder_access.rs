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
}

#[cfg(all(target_os = "macos", feature = "store-lite"))]
struct NativeGrants {
    grants: Grants<apple::AppleBackend>,
    storage_error: bool,
}

/// Called before opening auxiliary libraries or starting any directory watcher.
#[cfg(all(target_os = "macos", feature = "store-lite"))]
pub fn initialize(app: &tauri::AppHandle, data: &Path) {
    use tauri::Manager;
    let mut grants = Grants::new(data.join("folder-grants"), apple::AppleBackend);
    let storage_error = grants.restore().is_err();
    app.manage(std::sync::Mutex::new(NativeGrants {
        grants,
        storage_error,
    }));
}

#[tauri::command]
pub fn folder_access_status(app: tauri::AppHandle) -> Result<AccessStatus, String> {
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    {
        use tauri::Manager;
        let state = app.state::<std::sync::Mutex<NativeGrants>>();
        let state = state
            .lock()
            .map_err(|_| "Folder access is busy; try again")?;
        return Ok(AccessStatus {
            required: true,
            failed: state.grants.failed_count(),
            storage_error: state.storage_error,
        });
    }
    #[cfg(not(all(target_os = "macos", feature = "store-lite")))]
    {
        let _ = app;
        Ok(AccessStatus {
            required: false,
            failed: 0,
            storage_error: false,
        })
    }
}

/// No path argument: only a native user selection can authorize persistent access.
#[tauri::command]
pub async fn pick_persistent_media_folder(
    app: tauri::AppHandle,
    state: tauri::State<'_, crate::server::state::AppState>,
) -> Result<Option<String>, String> {
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    {
        use sha2::{Digest, Sha256};
        use std::os::unix::ffi::OsStrExt;
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
        let id = format!("{:x}", Sha256::digest(path.as_os_str().as_bytes()));
        {
            let grants = app.state::<std::sync::Mutex<NativeGrants>>();
            let mut grants = grants
                .lock()
                .map_err(|_| "Folder access is busy; try again")?;
            grants.grants.persist_and_activate(&id, &bookmark)
                .map_err(|_| "Folder access could not be saved. Check available storage and select the folder again.")?;
            grants.storage_error = false;
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
        let _ = (app, state);
        Err("Persistent folder grants are only used by macOS Lite".into())
    }
}
