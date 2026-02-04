//! Backend Configuration
//!
//! Handles portable mode detection, settings, and path resolution.
//! Mirrors electron/backend/config.js

use std::env;
use std::fs;
use std::path::PathBuf;
use std::net::UdpSocket;
use serde::{Deserialize, Serialize};

/// Network settings from settings.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkSettings {
    #[serde(default)]
    pub local_network_enabled: bool,
    #[serde(default)]
    pub public_network_enabled: bool,
    #[serde(default)]
    pub local_port: Option<u16>,
    #[serde(default = "default_public_port")]
    pub public_port: u16,
    #[serde(default)]
    pub auth_required_level: String,
    #[serde(default)]
    pub upnp_enabled: bool,
}

fn default_public_port() -> u16 {
    8791
}

/// Settings file structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    #[serde(default)]
    pub network: NetworkSettings,
}

/// Detect if running in portable mode
///
/// Portable mode is DEFAULT for packaged apps, unless:
/// - Running from Program Files (Windows installer location)
/// - A '.use-appdata' marker file exists next to the executable
///
/// Returns the portable data directory path or None
pub fn detect_portable_mode() -> Option<PathBuf> {
    // In development mode, don't use portable mode
    if cfg!(debug_assertions) {
        log::info!("[Backend] Development mode - not using portable mode");
        return None;
    }

    // Get the directory containing the executable
    let exe_path = env::current_exe().ok()?;
    let app_dir = exe_path.parent()?;

    let portable_data_path = app_dir.join("data");
    let use_appdata_marker = app_dir.join(".use-appdata");

    // Check if we should use AppData instead of portable mode
    if use_appdata_marker.exists() {
        log::info!("[Backend] Found .use-appdata marker, using AppData");
        return None;
    }

    // Check if installed in Program Files (Windows)
    #[cfg(target_os = "windows")]
    {
        let app_dir_str = app_dir.to_string_lossy().to_lowercase();
        if app_dir_str.contains("program files") {
            log::info!("[Backend] Running from Program Files, using AppData");
            return None;
        }
    }

    // Default: use portable mode - create data folder next to exe
    if !portable_data_path.exists() {
        if let Err(e) = fs::create_dir_all(&portable_data_path) {
            log::error!("[Backend] Failed to create portable data dir: {}", e);
            return None;
        }
    }

    log::info!("[Backend] Portable mode enabled, data: {:?}", portable_data_path);
    Some(portable_data_path)
}

/// Get LocalBooru data directory (matches Python API location)
pub fn get_data_dir(portable_data_dir: Option<&PathBuf>) -> PathBuf {
    // Use portable data directory if in portable mode
    if let Some(dir) = portable_data_dir {
        return dir.clone();
    }

    // Default: platform-specific app data location
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = env::var_os("APPDATA") {
            return PathBuf::from(appdata).join(".localbooru");
        }
    }

    // Linux/Mac: ~/.localbooru
    dirs::home_dir()
        .map(|h| h.join(".localbooru"))
        .unwrap_or_else(|| PathBuf::from(".localbooru"))
}

/// Get settings.json path
pub fn get_settings_path(portable_data_dir: Option<&PathBuf>) -> PathBuf {
    get_data_dir(portable_data_dir).join("settings.json")
}

/// Load network settings from settings.json
pub fn get_network_settings(portable_data_dir: Option<&PathBuf>) -> NetworkSettings {
    let settings_path = get_settings_path(portable_data_dir);

    if settings_path.exists() {
        match fs::read_to_string(&settings_path) {
            Ok(content) => {
                match serde_json::from_str::<Settings>(&content) {
                    Ok(settings) => return settings.network,
                    Err(e) => log::warn!("[Backend] Failed to parse settings: {}", e),
                }
            }
            Err(e) => log::warn!("[Backend] Failed to read settings: {}", e),
        }
    }

    NetworkSettings::default()
}

/// Get the local IP address
pub fn get_local_ip() -> String {
    // Connect to a public DNS to determine local IP (doesn't actually send data)
    if let Ok(socket) = UdpSocket::bind("0.0.0.0:0") {
        if socket.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = socket.local_addr() {
                return addr.ip().to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

/// Determine the host to bind to based on network settings
pub fn get_bind_host(portable_data_dir: Option<&PathBuf>) -> String {
    let network_settings = get_network_settings(portable_data_dir);

    // If local network or public is enabled, bind to all interfaces
    if network_settings.local_network_enabled || network_settings.public_network_enabled {
        return "0.0.0.0".to_string();
    }

    // Default: localhost only
    "127.0.0.1".to_string()
}

/// Get the default port based on portable mode
pub fn get_default_port(is_portable: bool) -> u16 {
    // Different default ports: portable=8791, system=8790
    // This allows running both simultaneously without conflicts
    if is_portable { 8791 } else { 8790 }
}

/// Get the packages directory for pip packages (persistent storage)
pub fn get_packages_dir() -> PathBuf {
    dirs::data_local_dir()
        .map(|d| d.join("com.localbooru.app").join("packages"))
        .unwrap_or_else(|| {
            dirs::home_dir()
                .map(|h| h.join(".localbooru").join("packages"))
                .unwrap_or_else(|| PathBuf::from("packages"))
        })
}
