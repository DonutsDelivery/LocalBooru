use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::Manager;
use tauri::tray::{TrayIconBuilder, MouseButton, MouseButtonState};
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};

pub mod addons;
mod commands;
pub mod db;
pub mod routes;
pub mod server;
pub mod services;

use commands::{
    backend_start, backend_stop, backend_restart, backend_status,
    backend_health_check, backend_get_port, backend_get_local_ip,
    backend_get_network_settings,
    show_in_folder, get_app_version, quit_app,
    copy_image_to_clipboard, show_image_context_menu,
};
use server::state::AppState;

/// Default port for the embedded HTTP server.
const DEFAULT_PORT: u16 = 8790;

/// Get the data directory path (same logic as Python config.py).
fn get_data_dir() -> PathBuf {
    // Check for portable mode
    if let Ok(portable_data) = std::env::var("LOCALBOORU_PORTABLE_DATA") {
        let path = PathBuf::from(portable_data);
        std::fs::create_dir_all(&path).ok();
        return path;
    }

    // Default: ~/.localbooru (Linux/Mac) or %APPDATA%\.localbooru (Windows)
    #[cfg(target_os = "windows")]
    let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    #[cfg(not(target_os = "windows"))]
    let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

    let data_dir = base.join(".localbooru");
    std::fs::create_dir_all(&data_dir).ok();
    data_dir
}

/// Resolve the frontend dist directory.
fn get_frontend_dir() -> Option<PathBuf> {
    // In dev mode, Tauri serves the frontend via devUrl.
    // In production, the frontend is bundled at a known relative path.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    if let Some(ref dir) = exe_dir {
        // Production: bundled next to executable
        let dist = dir.join("frontend").join("dist");
        if dist.exists() {
            return Some(dist);
        }
        // Also check one level up (dev builds)
        let dist_alt = dir.parent().and_then(|p| {
            let d = p.join("frontend").join("dist");
            if d.exists() { Some(d) } else { None }
        });
        if dist_alt.is_some() {
            return dist_alt;
        }
    }

    // Workspace-relative fallback
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("frontend").join("dist"));
    if let Some(ref ws) = workspace {
        if ws.exists() {
            return workspace;
        }
    }

    None
}

/// Show a hidden window and force WebKit to recomposite (Linux workaround).
fn show_window(window: &tauri::WebviewWindow) {
    let _ = window.show();
    let _ = window.unminimize();

    #[cfg(target_os = "linux")]
    {
        if let Ok(size) = window.inner_size() {
            let _ = window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                width: size.width + 1,
                height: size.height,
            }));
            let _ = window.set_size(tauri::Size::Physical(size));
        }
    }

    let _ = window.set_focus();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // ── Linux: Fully hardware-accelerated video pipeline in WebKitGTK ──
    // The zero-copy path: VA-API decode → DMA-BUF → EGL texture import → GPU compositor
    // All components must be enabled for this to work end-to-end.
    #[cfg(target_os = "linux")]
    {
        // 1. VA-API decoders preferred (produce DMA-BUF output for zero-copy)
        //    NVDEC as fallback, software decoders disabled.
        if std::env::var("GST_PLUGIN_FEATURE_RANK").is_err() {
            std::env::set_var(
                "GST_PLUGIN_FEATURE_RANK",
                "vah264dec:MAX,vah265dec:MAX,vaav1dec:MAX,vavp9dec:MAX,\
                 nvh264dec:PRIMARY+1,nvh265dec:PRIMARY+1,nvav1dec:PRIMARY+1,nvvp9dec:PRIMARY+1,\
                 nvh264sldec:PRIMARY,nvh265sldec:PRIMARY,\
                 avdec_h264:NONE,avdec_h265:NONE",
            );
        }

        // 2. Enable nvidia-vaapi-driver's direct backend for GStreamer VA decoders
        if std::env::var("GST_VA_ALL_DRIVERS").is_err() {
            std::env::set_var("GST_VA_ALL_DRIVERS", "1");
        }
        if std::env::var("LIBVA_DRIVER_NAME").is_err() {
            std::env::set_var("LIBVA_DRIVER_NAME", "nvidia");
        }

        // 3. DMA-BUF renderer for zero-copy GPU textures.
        std::env::remove_var("WEBKIT_DISABLE_DMABUF_RENDERER");
        std::env::remove_var("WEBKIT_DISABLE_COMPOSITING_MODE");
        // Native Wayland with explicit sync disabled to avoid Error 71.
        // WebKitGTK's GTK3 backend doesn't support Wayland explicit sync,
        // causing a protocol error crash regardless of compositor (KDE/GNOME).
        std::env::set_var("__NV_DISABLE_EXPLICIT_SYNC", "1");
        std::env::remove_var("GDK_BACKEND");

        // 4. GStreamer GL: use EGL platform for DMA-BUF texture import
        if std::env::var("GST_GL_PLATFORM").is_err() {
            std::env::set_var("GST_GL_PLATFORM", "egl");
        }
    }

    let mut builder = tauri::Builder::default();

    // Single instance: focus existing window if already running
    #[cfg(desktop)]
    {
        builder = builder.plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(quit_flag) = app.try_state::<Arc<AtomicBool>>() {
                if quit_flag.load(Ordering::SeqCst) {
                    return;
                }
            }
            if let Some(window) = app.get_webview_window("main") {
                show_window(&window);
            }
        }));
    }

    builder
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_window_state::Builder::new().build())
        .setup(|app| {
            // Logging
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // ── Initialize AppState (database + config) ──
            let data_dir = get_data_dir();
            let port = DEFAULT_PORT;
            let app_state = AppState::new(&data_dir, port)
                .map_err(|e| format!("Failed to initialize app state: {}", e))?;

            // Store AppState for both Tauri commands and axum server
            app.manage(app_state.clone());

            // ── Start embedded axum server ──
            let frontend_dir = get_frontend_dir();
            log::info!("[Startup] Data dir: {}", data_dir.display());
            log::info!("[Startup] Frontend dir: {:?}", frontend_dir);

            tauri::async_runtime::spawn(async move {
                // Start directory watcher (needs tokio runtime for internal spawns)
                let mut watcher = services::directory_watcher::DirectoryWatcher::new(app_state.clone());
                watcher.start();

                // Store watcher in AppState so route handlers can register/unregister directories
                let watcher_arc = std::sync::Arc::new(watcher);
                app_state.set_directory_watcher(watcher_arc);

                // Start task queue worker (needs tokio runtime)
                app_state.task_queue_arc().start(app_state.clone());

                // Resume addons that were previously enabled (non-blocking per addon)
                app_state.addon_manager().resume_addons().await;

                if let Err(e) = server::start_server(app_state, frontend_dir).await {
                    log::error!("Axum server error: {}", e);
                }

                // Watcher stays alive via AppState Arc until server shuts down
            });

            // ── Window setup ──
            let window = app.get_webview_window("main").unwrap();

            // Enable GPU-accelerated compositing in WebKitGTK.
            // Combined with VA-API decoders + DMA-BUF renderer, this gives
            // a zero-copy video path: NVDEC → DMA-BUF → GL texture → compositor.
            #[cfg(target_os = "linux")]
            {
                use webkit2gtk::{WebViewExt, SettingsExt};
                window.with_webview(|webview| {
                    let wv = &webview.inner();
                    if let Some(settings) = wv.settings() {
                        settings.set_hardware_acceleration_policy(
                            webkit2gtk::HardwareAccelerationPolicy::Always,
                        );
                        log::info!("[WebView] Hardware acceleration policy set to Always");
                    }
                }).ok();
            }

            log::info!("LocalBooru v2 started (embedded Rust backend)");

            // ── Tray icon ──
            let quit_flag = Arc::new(AtomicBool::new(false));
            app.manage(quit_flag.clone());

            let open_item = MenuItem::with_id(app, "open", "Open LocalBooru", true, None::<&str>)?;
            let browser_item = MenuItem::with_id(app, "browser", "Open in Browser", true, None::<&str>)?;
            let separator = PredefinedMenuItem::separator(app)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_item, &browser_item, &separator, &quit_item])?;

            let quit_flag_menu = quit_flag.clone();
            let server_port = port;
            let tray_icon = tauri::image::Image::from_bytes(include_bytes!("../icons/32x32.png"))?;
            let _tray = TrayIconBuilder::new()
                .icon(tray_icon)
                .menu(&menu)
                .tooltip("LocalBooru")
                .on_menu_event(move |app_handle, event| {
                    match event.id.as_ref() {
                        "open" => {
                            if let Some(window) = app_handle.get_webview_window("main") {
                                show_window(&window);
                            }
                        }
                        "browser" => {
                            let url = format!("http://127.0.0.1:{}", server_port);
                            #[cfg(target_os = "linux")]
                            let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
                            #[cfg(target_os = "macos")]
                            let _ = std::process::Command::new("open").arg(&url).spawn();
                            #[cfg(target_os = "windows")]
                            let _ = std::process::Command::new("cmd")
                                .args(["/c", "start", &url])
                                .spawn();
                        }
                        "quit" => {
                            // Kill FFmpeg transcode processes before exiting
                            if let Some(state) = app_handle.try_state::<AppState>() {
                                state.transcode_manager().stop_all();
                            }
                            quit_flag_menu.store(true, Ordering::SeqCst);
                            app_handle.exit(0);
                        }
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        if let Some(window) = tray.app_handle().get_webview_window("main") {
                            show_window(&window);
                        }
                    }
                })
                .build(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let quit_flag: tauri::State<Arc<AtomicBool>> = window.app_handle().state();
                if quit_flag.load(Ordering::SeqCst) {
                    log::info!("Quit requested, closing window...");
                } else {
                    log::info!("Window close requested, hiding to tray...");
                    // Pause all media before hiding so audio doesn't keep playing
                    if let Some(wv) = window.app_handle().get_webview_window("main") {
                        let _ = wv.eval(
                            "document.querySelectorAll('video, audio').forEach(el => el.pause())"
                        );
                    }
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            // Backend commands (compatibility stubs + status)
            backend_start,
            backend_stop,
            backend_restart,
            backend_status,
            backend_health_check,
            backend_get_port,
            backend_get_local_ip,
            backend_get_network_settings,
            // IPC commands
            show_in_folder,
            get_app_version,
            quit_app,
            copy_image_to_clipboard,
            show_image_context_menu,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
