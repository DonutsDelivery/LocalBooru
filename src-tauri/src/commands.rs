//! Tauri Commands (v2)
//!
//! The axum backend is embedded in the Tauri process — no separate process management needed.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};

use std::sync::{Mutex, OnceLock};

use crate::server::state::AppState;

const PAIRED_SERVER_CREDENTIAL_SERVICE: &str = "com.localbooru.app";
const PAIRED_SERVER_CREDENTIAL_ACCOUNT: &str = "paired-server-credentials";
const CREDENTIAL_DIR: &str = ".credentials";
const PAIRED_CREDENTIAL_FILE: &str = "paired-server-credentials.json";

fn paired_credential_cache() -> &'static Mutex<Option<serde_json::Value>> {
    static CACHE: OnceLock<Mutex<Option<serde_json::Value>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Path of the file-backed credential store. Mirrors `get_data_dir` in lib.rs
/// so credentials live in the same app-private directory on every platform —
/// including Android, where the OS keyring is unavailable and credentials were
/// previously lost on every app restart (the "pairing invalid the next day" bug).
fn paired_credential_file_path(#[allow(unused)] app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let data_dir = if let Ok(portable_data) = std::env::var("LOCALBOORU_PORTABLE_DATA") {
        PathBuf::from(portable_data)
    } else {
        #[cfg(mobile)]
        {
            app.path()
                .app_data_dir()
                .map_err(|error| format!("App data directory is unavailable: {error}"))?
        }
        #[cfg(desktop)]
        {
            #[cfg(target_os = "windows")]
            let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
            #[cfg(not(target_os = "windows"))]
            let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            base.join(".localbooru")
        }
    };
    Ok(data_dir.join(CREDENTIAL_DIR).join(PAIRED_CREDENTIAL_FILE))
}

fn read_credential_file(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "Could not read stored paired-server credentials: {error}"
        )),
    }
}

fn write_credential_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path
        .parent()
        .ok_or_else(|| "Paired-server credential path has no parent".to_string())?;
    fs::create_dir_all(dir)
        .map_err(|error| format!("Could not create the credential directory: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("Could not protect the credential directory: {error}"))?;
    }
    let temp_path = path.with_extension("json.tmp");
    fs::write(&temp_path, bytes)
        .map_err(|error| format!("Could not write stored paired-server credentials: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&temp_path, fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("Could not protect stored credentials: {error}"))?;
    }
    fs::rename(&temp_path, path)
        .map_err(|error| format!("Could not finalize stored credentials: {error}"))?;
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn load_desktop_keyring_credentials() -> Result<Option<Vec<u8>>, String> {
    let entry = keyring::Entry::new(
        PAIRED_SERVER_CREDENTIAL_SERVICE,
        PAIRED_SERVER_CREDENTIAL_ACCOUNT,
    )
    .map_err(|error| format!("OS credential store is unavailable: {error}"))?;
    match entry.get_secret() {
        Ok(secret) => Ok(Some(secret)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(error) => Err(format!("Could not load OS-protected credentials: {error}")),
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn store_desktop_keyring_credentials(bytes: &[u8]) -> Result<(), String> {
    keyring::Entry::new(
        PAIRED_SERVER_CREDENTIAL_SERVICE,
        PAIRED_SERVER_CREDENTIAL_ACCOUNT,
    )
    .map_err(|error| format!("OS credential store is unavailable: {error}"))?
    .set_secret(bytes)
    .map_err(|error| format!("Could not store OS-protected credentials: {error}"))
}

/// Load remote server tokens from the app's protected credential store.
///
/// Primary source: the durable file-backed store in the app data directory
/// (the only store available on Android). On desktop the OS keyring is also
/// kept in sync and preferred when it holds newer data, so tokens survive
/// restarts everywhere while desktop keeps keyring-grade protection.
#[tauri::command]
pub fn load_paired_server_credentials(app: AppHandle) -> Result<serde_json::Value, String> {
    if let Ok(cache) = paired_credential_cache().lock() {
        if let Some(value) = cache.as_ref() {
            return Ok(value.clone());
        }
    }
    let file_path = paired_credential_file_path(&app)?;
    let file_value = match read_credential_file(&file_path)? {
        Some(bytes) if !bytes.is_empty() => Some(
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .map_err(|_| "Stored paired-server credentials are corrupt".to_string())?,
        ),
        _ => None,
    };
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let keyring_value = match load_desktop_keyring_credentials()? {
        Some(bytes) => Some(
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .map_err(|_| "OS keyring credentials are corrupt".to_string())?,
        ),
        None => None,
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let keyring_value: Option<serde_json::Value> = None;
    // Prefer whichever store actually has credentials; if both do and they
    // disagree, the file wins because it is written on every platform while
    // the keyring is desktop-only. They are kept in sync on every store.
    let loaded = file_value
        .or(keyring_value)
        .unwrap_or(serde_json::json!({}));
    if let Ok(mut cache) = paired_credential_cache().lock() {
        *cache = Some(loaded.clone());
    }
    Ok(loaded)
}

/// Atomically replace the protected map of remote server credentials.
///
/// Writes the durable file-backed store on every platform and additionally
/// mirrors into the OS keyring where one exists (desktop). The file is the
/// source of truth so Android keeps pairings across restarts.
#[tauri::command]
pub fn store_paired_server_credentials(
    app: AppHandle,
    credentials: serde_json::Value,
) -> Result<(), String> {
    let object = credentials
        .as_object()
        .ok_or_else(|| "Paired-server credentials must be an object".to_string())?;
    if object.len() > 128 {
        return Err("Too many paired-server credentials".into());
    }
    let bytes = serde_json::to_vec(&credentials).map_err(|error| error.to_string())?;
    if bytes.len() > 256 * 1024 {
        return Err("Paired-server credential store is too large".into());
    }
    if let Ok(cache) = paired_credential_cache().lock() {
        if cache.as_ref() == Some(&credentials) {
            return Ok(());
        }
    }
    let file_path = paired_credential_file_path(&app)?;
    write_credential_file(&file_path, &bytes)?;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        // Best effort: the file above is the durable source of truth, the
        // keyring is a desktop convenience mirror. A keyring failure must not
        // lose the credentials the file already saved.
        let _ = store_desktop_keyring_credentials(&bytes);
    }
    if let Ok(mut cache) = paired_credential_cache().lock() {
        *cache = Some(credentials);
    }
    Ok(())
}

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
        .unwrap_or_else(|| "2.0.4".to_string())
}

/// Quit the application
#[tauri::command]
pub async fn quit_app(app: AppHandle) {
    // Explicit exit skips destructors, so finish managed process cleanup first.
    if let Some(state) = app.try_state::<AppState>() {
        state.shutdown_managed_processes().await;
    }
    if let Some(quit_flag) = app.try_state::<Arc<AtomicBool>>() {
        quit_flag.store(true, Ordering::SeqCst);
    }
    app.exit(0);
}

/// Copy image to clipboard response
#[derive(Debug, Serialize)]
pub struct CopyImageResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(any(test, target_os = "windows"))]
fn decode_clipboard_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| format!("Failed to decode image for clipboard: {error}"))?;
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok((rgba.into_raw(), width, height))
}

/// Copy image to clipboard
#[tauri::command]
pub async fn copy_image_to_clipboard(
    app: AppHandle,
    image_url: String,
) -> Result<CopyImageResult, String> {
    // Mobile: not supported yet (could use Android/iOS share sheet in the future)
    #[cfg(mobile)]
    {
        let _ = (app, image_url);
        return Ok(CopyImageResult {
            success: false,
            error: Some("Clipboard copy not supported on mobile yet".to_string()),
        });
    }

    #[cfg(desktop)]
    {
        #[cfg(not(target_os = "windows"))]
        let _ = &app;

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
            use tauri_plugin_clipboard_manager::ClipboardExt;

            let (rgba, width, height) = match decode_clipboard_image(&bytes) {
                Ok(image) => image,
                Err(error) => {
                    return Ok(CopyImageResult {
                        success: false,
                        error: Some(error),
                    });
                }
            };
            let image = tauri::image::Image::new_owned(rgba, width, height);
            return match app.clipboard().write_image(&image) {
                Ok(()) => Ok(CopyImageResult {
                    success: true,
                    error: None,
                }),
                Err(error) => Ok(CopyImageResult {
                    success: false,
                    error: Some(format!("Failed to write image to clipboard: {error}")),
                }),
            };
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
                Ok(TestServerResult {
                    success: false,
                    error: Some("Authentication required".into()),
                })
            } else if !resp.status().is_success() {
                Ok(TestServerResult {
                    success: false,
                    error: Some(format!("Server returned {}", resp.status())),
                })
            } else {
                Ok(TestServerResult {
                    success: true,
                    error: None,
                })
            }
        }
        Err(e) => Ok(TestServerResult {
            success: false,
            error: Some(e.to_string()),
        }),
    }
}

/// Verify handshake with a remote server and get JWT token.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HandshakeResult {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_name: Option<String>,
}

#[tauri::command]
pub async fn verify_remote_handshake(
    url: String,
    nonce: String,
) -> Result<HandshakeResult, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| e.to_string())?;

    let body = serde_json::json!({ "nonce": nonce });
    match client
        .post(format!("{}/api/network/verify-handshake", url))
        .json(&body)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let body = match resp.json::<serde_json::Value>().await {
                Ok(body) => body,
                Err(e) => {
                    return Ok(HandshakeResult {
                        success: false,
                        token: None,
                        error: Some(e.to_string()),
                        server_id: None,
                        server_name: None,
                    });
                }
            };
            if !status.is_success() {
                return Ok(HandshakeResult {
                    success: false,
                    token: None,
                    error: Some(
                        body.get("detail")
                            .or_else(|| body.get("error"))
                            .and_then(|value| value.as_str())
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("HTTP {status}")),
                    ),
                    server_id: None,
                    server_name: None,
                });
            }
            let token = body
                .get("token")
                .and_then(|value| value.as_str())
                .map(str::to_owned);
            let success = body
                .get("success")
                .and_then(|value| value.as_bool())
                .unwrap_or(token.is_some());
            Ok(HandshakeResult {
                success,
                token,
                error: None,
                server_id: body
                    .get("serverId")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
                server_name: body
                    .get("serverName")
                    .and_then(|value| value.as_str())
                    .map(str::to_owned),
            })
        }
        Err(e) => Ok(HandshakeResult {
            success: false,
            token: None,
            error: Some(e.to_string()),
            server_id: None,
            server_name: None,
        }),
    }
}

/// Configure the remote server proxy on the embedded HTTP server.
/// When set, requests to /remote/* are forwarded to the target URL.
/// `fallback_url` is tried on network errors from the primary (e.g. Tailscale fallback).
/// Call with url=null to disable.
#[tauri::command(rename_all = "camelCase")]
pub async fn set_remote_proxy(
    state: State<'_, AppState>,
    url: Option<String>,
    fallback_url: Option<String>,
    token: Option<String>,
) -> Result<(), String> {
    log::info!(
        "[Proxy] Setting remote proxy to: {:?} (fallback: {:?})",
        url,
        fallback_url
    );
    state.set_remote_proxy(url, fallback_url, token).await;
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

#[cfg(test)]
mod tests {
    use super::decode_clipboard_image;
    use image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};
    use std::io::Cursor;

    fn encoded_test_image(format: ImageFormat) -> Vec<u8> {
        let pixels = ImageBuffer::from_fn(3, 2, |x, y| {
            Rgba([
                (x * 40) as u8,
                (y * 80) as u8,
                160,
                if x == 0 && y == 0 { 64 } else { 255 },
            ])
        });
        let mut encoded = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(pixels)
            .write_to(&mut encoded, format)
            .unwrap();
        encoded.into_inner()
    }

    #[test]
    fn clipboard_payload_decodes_supported_image_formats() {
        // AC: @windows-copy-image ac-1
        // AC: @windows-copy-image ac-3
        for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
            let encoded = encoded_test_image(format);
            let (rgba, width, height) = decode_clipboard_image(&encoded).unwrap();
            assert_eq!((width, height), (3, 2));
            assert_eq!(rgba.len(), 3 * 2 * 4);
            if format != ImageFormat::Jpeg {
                assert_eq!(rgba[3], 64);
            }
        }
    }

    #[test]
    fn clipboard_payload_reports_decode_failure() {
        // AC: @windows-copy-image ac-2
        let error = decode_clipboard_image(b"not an image").unwrap_err();
        assert!(error.contains("Failed to decode image for clipboard"));
    }
}
