//! Tauri Commands (v2)
//!
//! The axum backend is embedded in the Tauri process — no separate process management needed.

use tauri::{AppHandle, Manager, State};
use serde::{Deserialize, Serialize};
use std::process::Command;

use crate::server::state::AppState;

/// Backend status response (kept for frontend compatibility)
#[derive(Debug, Serialize, Deserialize)]
pub struct BackendStatus {
    pub running: bool,
    pub port: u16,
    pub healthy: bool,
    pub mode: String,
    pub data_dir: String,
}

/// Backend is always running (embedded) — returns true immediately.
#[tauri::command]
pub async fn backend_start() -> Result<(), String> {
    Ok(())
}

/// No-op — can't stop embedded server without quitting.
#[tauri::command]
pub async fn backend_stop() -> Result<(), String> {
    Ok(())
}

/// No-op — embedded server doesn't need restart.
#[tauri::command]
pub async fn backend_restart() -> Result<(), String> {
    Ok(())
}

/// Get backend status — always running.
#[tauri::command]
pub async fn backend_status(state: State<'_, AppState>) -> Result<BackendStatus, String> {
    Ok(BackendStatus {
        running: true,
        port: state.port(),
        healthy: true,
        mode: "embedded".to_string(),
        data_dir: state.data_dir().to_string_lossy().to_string(),
    })
}

/// Check if the embedded HTTP server is ready (listening on port).
#[tauri::command]
pub async fn backend_health_check(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(state.is_server_ready())
}

/// Get the backend port.
#[tauri::command]
pub async fn backend_get_port(state: State<'_, AppState>) -> Result<u16, String> {
    Ok(state.port())
}

/// Get local IP address.
#[tauri::command]
pub async fn backend_get_local_ip() -> Result<String, String> {
    // Simple local IP detection
    Ok("127.0.0.1".to_string())
}

/// Get network settings.
#[tauri::command]
pub async fn backend_get_network_settings() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "local_network_enabled": false,
        "public_network_enabled": false,
    }))
}

// ============================================================================
// IPC Commands
// ============================================================================

/// Show file in native file explorer
#[tauri::command]
pub async fn show_in_folder(path: String) -> Result<(), String> {
    // Mobile: no-op (no native file explorer integration)
    #[cfg(mobile)]
    {
        let _ = path;
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
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

    #[cfg(desktop)]
    Ok(())
}

/// Get app version
#[tauri::command]
pub fn get_app_version(app: AppHandle) -> String {
    app.config()
        .version
        .clone()
        .unwrap_or_else(|| "2.0.0".to_string())
}

/// Quit the application
#[tauri::command]
pub async fn quit_app(app: AppHandle) -> Result<(), String> {
    // Kill all FFmpeg transcode processes before exiting — process::exit() skips destructors
    if let Some(state) = app.try_state::<AppState>() {
        state.transcode_manager().stop_all();
    }
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
    // Mobile: not supported yet (could use Android/iOS share sheet in the future)
    #[cfg(mobile)]
    {
        let _ = image_url;
        return Ok(CopyImageResult {
            success: false,
            error: Some("Clipboard copy not supported on mobile yet".to_string()),
        });
    }

    #[cfg(desktop)]
    {
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

        #[cfg(target_os = "linux")]
        {
            use std::io::Write;
            use std::process::Stdio;

            let mime_type = if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
                "image/png"
            } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
                "image/jpeg"
            } else if bytes.starts_with(&[0x47, 0x49, 0x46]) {
                "image/gif"
            } else if bytes.starts_with(&[0x52, 0x49, 0x46, 0x46]) {
                "image/webp"
            } else {
                "image/png"
            };

            // Try wl-copy (Wayland)
            if let Ok(mut child) = Command::new("wl-copy")
                .arg("--type")
                .arg(mime_type)
                .stdin(Stdio::piped())
                .spawn()
            {
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
            if let Ok(mut child) = Command::new("xclip")
                .args(["-selection", "clipboard", "-t", mime_type])
                .stdin(Stdio::piped())
                .spawn()
            {
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
            return Ok(CopyImageResult {
                success: false,
                error: Some("macOS clipboard not yet implemented".to_string()),
            });
        }

        #[cfg(target_os = "windows")]
        {
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
}

/// Test connection to a remote server (used by mobile QR pairing).
/// Routes through Rust to avoid mixed-content blocks in the WebView.
#[derive(Debug, Serialize)]
pub struct TestServerResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[tauri::command]
pub async fn test_remote_server(url: String) -> Result<TestServerResult, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| e.to_string())?;

    match client.get(format!("{}/api", url)).send().await {
        Ok(resp) => {
            if resp.status() == 401 {
                Ok(TestServerResult { success: false, error: Some("Authentication required".into()) })
            } else if !resp.status().is_success() {
                Ok(TestServerResult { success: false, error: Some(format!("Server returned {}", resp.status())) })
            } else {
                Ok(TestServerResult { success: true, error: None })
            }
        }
        Err(e) => {
            Ok(TestServerResult { success: false, error: Some(e.to_string()) })
        }
    }
}

/// Verify handshake with a remote server and get JWT token.
#[derive(Debug, Serialize)]
pub struct HandshakeResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[tauri::command]
pub async fn verify_remote_handshake(url: String, nonce: String) -> Result<HandshakeResult, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| e.to_string())?;

    let body = serde_json::json!({ "nonce": nonce });
    match client.post(format!("{}/api/network/verify-handshake", url)).json(&body).send().await {
        Ok(resp) => {
            if !resp.status().is_success() {
                return Ok(HandshakeResult { success: false, token: None, error: Some(format!("HTTP {}", resp.status())) });
            }
            match resp.json::<serde_json::Value>().await {
                Ok(body) => {
                    let token = body.get("token").and_then(|t| t.as_str()).map(|s| s.to_string());
                    let success = body.get("success").and_then(|s| s.as_bool()).unwrap_or(token.is_some());
                    Ok(HandshakeResult { success, token, error: None })
                }
                Err(e) => Ok(HandshakeResult { success: false, token: None, error: Some(e.to_string()) })
            }
        }
        Err(e) => {
            Ok(HandshakeResult { success: false, token: None, error: Some(e.to_string()) })
        }
    }
}

/// Configure the remote server proxy on the embedded HTTP server.
/// When set, requests to /remote/* are forwarded to the target URL.
/// Call with url=null to disable.
#[tauri::command]
pub async fn set_remote_proxy(
    state: State<'_, AppState>,
    url: Option<String>,
    token: Option<String>,
) -> Result<(), String> {
    log::info!("[Proxy] Setting remote proxy to: {:?}", url);
    state.set_remote_proxy(url, token).await;
    Ok(())
}

/// Context menu options
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ImageContextMenuOptions {
    #[serde(rename = "imageUrl")]
    pub image_url: Option<String>,
    #[serde(rename = "filePath")]
    pub file_path: Option<String>,
    #[serde(rename = "isVideo")]
    pub is_video: Option<bool>,
}

/// Show image context menu (handled by frontend)
#[tauri::command]
pub async fn show_image_context_menu(_options: ImageContextMenuOptions) -> Result<(), String> {
    log::info!("Context menu requested - frontend should handle this");
    Ok(())
}
