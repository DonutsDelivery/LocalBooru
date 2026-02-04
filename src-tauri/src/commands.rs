//! Tauri Commands
//!
//! Exposes backend management functionality to the frontend via Tauri commands.

use tauri::{AppHandle, State};
use std::sync::Arc;
use std::process::Command;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

use crate::backend::BackendManager;

/// State wrapper for the backend manager
pub struct BackendState(pub Arc<Mutex<BackendManager>>);

/// Backend status response
#[derive(Debug, Serialize, Deserialize)]
pub struct BackendStatus {
    pub running: bool,
    pub port: u16,
    pub healthy: bool,
    pub mode: String, // "portable" or "system"
    pub data_dir: String,
}

/// Start the backend server
#[tauri::command]
pub async fn backend_start(state: State<'_, BackendState>) -> Result<(), String> {
    let manager = state.0.lock().await;
    manager.start().await
}

/// Stop the backend server
#[tauri::command]
pub async fn backend_stop(state: State<'_, BackendState>) -> Result<(), String> {
    let manager = state.0.lock().await;
    manager.stop().await
}

/// Restart the backend server
#[tauri::command]
pub async fn backend_restart(state: State<'_, BackendState>) -> Result<(), String> {
    let manager = state.0.lock().await;
    manager.restart().await
}

/// Get backend status
#[tauri::command]
pub async fn backend_status(state: State<'_, BackendState>) -> Result<BackendStatus, String> {
    let manager = state.0.lock().await;
    let running = manager.is_running().await;
    let healthy = if running {
        manager.health_check().await
    } else {
        false
    };

    Ok(BackendStatus {
        running,
        port: manager.get_port(),
        healthy,
        mode: if manager.is_portable() { "portable".to_string() } else { "system".to_string() },
        data_dir: manager.get_data_dir().to_string_lossy().to_string(),
    })
}

/// Check if backend is healthy
#[tauri::command]
pub async fn backend_health_check(state: State<'_, BackendState>) -> Result<bool, String> {
    let manager = state.0.lock().await;
    Ok(manager.health_check().await)
}

/// Get the backend port
#[tauri::command]
pub async fn backend_get_port(state: State<'_, BackendState>) -> Result<u16, String> {
    let manager = state.0.lock().await;
    Ok(manager.get_port())
}

/// Get local IP address
#[tauri::command]
pub async fn backend_get_local_ip(state: State<'_, BackendState>) -> Result<String, String> {
    let manager = state.0.lock().await;
    Ok(manager.get_local_ip())
}

/// Get network settings
#[tauri::command]
pub async fn backend_get_network_settings(
    state: State<'_, BackendState>,
) -> Result<serde_json::Value, String> {
    let manager = state.0.lock().await;
    let settings = manager.get_network_settings();
    serde_json::to_value(settings).map_err(|e| e.to_string())
}

// ============================================================================
// IPC Commands (ported from Electron)
// ============================================================================

/// Show file in native file explorer
#[tauri::command]
pub async fn show_in_folder(path: String) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        // Try xdg-open on the parent directory, then select file if possible
        let path_buf = std::path::Path::new(&path);
        if let Some(parent) = path_buf.parent() {
            Command::new("xdg-open")
                .arg(parent)
                .spawn()
                .map_err(|e| e.to_string())?;
        }
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .args(["-R", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "windows")]
    {
        Command::new("explorer")
            .args(["/select,", &path])
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Get app version
#[tauri::command]
pub fn get_app_version(app: AppHandle) -> String {
    app.config().version.clone().unwrap_or_else(|| "0.0.0".to_string())
}

/// Quit the application
#[tauri::command]
pub async fn quit_app(app: AppHandle, state: State<'_, BackendState>) -> Result<(), String> {
    // Stop the backend gracefully
    let manager = state.0.lock().await;
    if let Err(e) = manager.stop().await {
        log::error!("Error stopping backend during quit: {}", e);
    }
    drop(manager);

    // Exit the app
    app.exit(0);
    Ok(())
}

/// Copy image to clipboard response
#[derive(Debug, Serialize)]
pub struct CopyImageResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Copy image to clipboard
#[tauri::command]
pub async fn copy_image_to_clipboard(image_url: String) -> Result<CopyImageResult, String> {
    // Fetch the image data
    let response = reqwest::get(&image_url)
        .await
        .map_err(|e| format!("Failed to fetch image: {}", e))?;

    if !response.status().is_success() {
        return Ok(CopyImageResult {
            success: false,
            error: Some(format!("HTTP error: {}", response.status())),
        });
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read image data: {}", e))?;

    // For now, we'll use a platform-specific approach
    // On Linux, we can use wl-copy or xclip
    #[cfg(target_os = "linux")]
    {
        use std::io::Write;
        use std::process::{Command, Stdio};

        // Detect image type from magic bytes
        let mime_type = if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            "image/png"
        } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            "image/jpeg"
        } else if bytes.starts_with(&[0x47, 0x49, 0x46]) {
            "image/gif"
        } else if bytes.starts_with(&[0x52, 0x49, 0x46, 0x46]) {
            "image/webp"
        } else {
            "image/png" // Default
        };

        // Try wl-copy first (Wayland)
        let wl_result = Command::new("wl-copy")
            .arg("--type")
            .arg(mime_type)
            .stdin(Stdio::piped())
            .spawn();

        if let Ok(mut child) = wl_result {
            if let Some(mut stdin) = child.stdin.take() {
                if stdin.write_all(&bytes).is_ok() {
                    drop(stdin);
                    if child.wait().map(|s| s.success()).unwrap_or(false) {
                        return Ok(CopyImageResult {
                            success: true,
                            error: None,
                        });
                    }
                }
            }
        }

        // Try xclip (X11)
        let xclip_result = Command::new("xclip")
            .args(["-selection", "clipboard", "-t", mime_type])
            .stdin(Stdio::piped())
            .spawn();

        if let Ok(mut child) = xclip_result {
            if let Some(mut stdin) = child.stdin.take() {
                if stdin.write_all(&bytes).is_ok() {
                    drop(stdin);
                    if child.wait().map(|s| s.success()).unwrap_or(false) {
                        return Ok(CopyImageResult {
                            success: true,
                            error: None,
                        });
                    }
                }
            }
        }

        return Ok(CopyImageResult {
            success: false,
            error: Some("No clipboard tool available (install wl-copy or xclip)".to_string()),
        });
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, we need to use NSPasteboard through a helper or convert to PNG
        // For now, return an error suggesting to use browser copy
        return Ok(CopyImageResult {
            success: false,
            error: Some("macOS clipboard not yet implemented".to_string()),
        });
    }

    #[cfg(target_os = "windows")]
    {
        // On Windows, we'd need to use the Windows clipboard API
        return Ok(CopyImageResult {
            success: false,
            error: Some("Windows clipboard not yet implemented".to_string()),
        });
    }

    #[allow(unreachable_code)]
    Ok(CopyImageResult {
        success: false,
        error: Some("Unsupported platform".to_string()),
    })
}

/// Context menu options
#[derive(Debug, Deserialize)]
pub struct ImageContextMenuOptions {
    #[serde(rename = "imageUrl")]
    pub image_url: Option<String>,
    #[serde(rename = "filePath")]
    pub file_path: Option<String>,
    #[serde(rename = "isVideo")]
    pub is_video: Option<bool>,
}

/// Show image context menu
/// Note: Tauri 2 handles context menus differently - for now this is a no-op
/// The frontend can implement its own context menu using HTML/CSS
#[tauri::command]
pub async fn show_image_context_menu(_options: ImageContextMenuOptions) -> Result<(), String> {
    // Context menus in Tauri 2 require the menu plugin and more complex setup
    // For now, the frontend will handle context menus via HTML
    log::info!("Context menu requested - frontend should handle this");
    Ok(())
}
