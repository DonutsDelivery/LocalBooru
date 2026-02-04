use std::sync::Arc;
use tauri::Manager;
use tokio::sync::Mutex;

mod backend;
mod commands;
pub mod video;

use backend::BackendManager;
use commands::{
    backend_get_local_ip, backend_get_network_settings, backend_get_port, backend_health_check,
    backend_restart, backend_start, backend_status, backend_stop, BackendState,
    // IPC commands (ported from Electron)
    show_in_folder, get_app_version, quit_app, copy_image_to_clipboard, show_image_context_menu,
};
use video::{
    VideoPlayerState, video_get_system_info, video_init, video_play, video_pause,
    video_resume, video_stop, video_seek, video_get_position, video_get_duration,
    video_set_volume, video_get_volume, video_set_muted, video_get_state, video_cleanup,
    // VFR (Variable Frame Rate) commands
    VfrVideoPlayerState, video_vfr_analyze, video_vfr_init, video_vfr_play, video_vfr_pause,
    video_vfr_resume, video_vfr_stop, video_vfr_seek, video_vfr_seek_with_mode,
    video_vfr_step_frame, video_vfr_get_position, video_vfr_get_duration, video_vfr_get_state,
    video_vfr_get_stream_info, video_vfr_get_frame_info, video_vfr_set_seek_mode,
    video_vfr_get_seek_mode, video_vfr_set_rate, video_vfr_set_volume, video_vfr_get_volume,
    video_vfr_set_muted, video_vfr_is_muted, video_vfr_cleanup,
    // Transcoding commands
    TranscodeManagerState, transcode_get_capabilities, transcode_check_needed, transcode_start,
    transcode_get_progress, transcode_is_ready, transcode_get_playlist, transcode_stop,
    transcode_stop_all, transcode_cleanup_cache, transcode_set_cache_limit,
    // Interpolation commands
    InterpolatedPlayerState, InterpolationConfigState, InterpolationConfig,
    interpolation_detect_backends, interpolation_get_config, interpolation_set_config,
    interpolation_set_enabled, interpolation_set_backend, interpolation_set_target_fps,
    interpolation_set_preset, interpolation_get_presets, interpolation_init,
    interpolation_play, interpolation_stop, interpolation_pause, interpolation_resume,
    interpolation_seek, interpolation_set_volume, interpolation_get_state,
    interpolation_get_stats, interpolation_cleanup, interpolation_is_available,
    interpolation_recommend_backend,
    // Event streaming commands
    VideoEventState, VideoEventStreamer, video_events_start, video_events_stop, video_events_is_active,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .setup(|app| {
            // Set up logging in debug builds
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Determine if we're in a packaged (production) build
            let is_packaged = !cfg!(debug_assertions);

            // Create the backend manager
            let backend_manager = BackendManager::new(is_packaged);
            let backend_state = BackendState(Arc::new(Mutex::new(backend_manager)));

            // Store state in app
            app.manage(backend_state);

            // Initialize video player state
            let video_state = VideoPlayerState(Arc::new(Mutex::new(None)));
            app.manage(video_state);

            // Initialize VFR video player state
            let vfr_video_state = VfrVideoPlayerState(Arc::new(Mutex::new(None)));
            app.manage(vfr_video_state);

            // Initialize transcoding manager state
            let transcode_state = TranscodeManagerState(Arc::new(Mutex::new(None)));
            app.manage(transcode_state);

            // Initialize interpolation states
            let interpolation_config = InterpolationConfigState(Arc::new(Mutex::new(InterpolationConfig::default())));
            app.manage(interpolation_config);
            let interpolated_player_state = InterpolatedPlayerState(Arc::new(Mutex::new(None)));
            app.manage(interpolated_player_state);

            // Initialize video event streaming state
            let video_event_state = VideoEventState(Arc::new(Mutex::new(VideoEventStreamer::new())));
            app.manage(video_event_state);

            // Get the main window and configure it
            let window = app.get_webview_window("main").unwrap();

            // Open devtools in debug mode
            #[cfg(debug_assertions)]
            window.open_devtools();

            log::info!("LocalBooru Tauri app started");

            // Start the backend automatically
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state: tauri::State<BackendState> = app_handle.state();
                let manager = state.0.lock().await;
                if let Err(e) = manager.start().await {
                    log::error!("Failed to start backend: {}", e);
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Handle app close - stop backend gracefully
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                log::info!("Window close requested, stopping backend...");
                let app_handle = window.app_handle().clone();

                // Stop the backend when window closes
                tauri::async_runtime::spawn(async move {
                    let state: tauri::State<BackendState> = app_handle.state();
                    let manager = state.0.lock().await;
                    if let Err(e) = manager.stop().await {
                        log::error!("Error stopping backend: {}", e);
                    }
                });
            }
        })
        .invoke_handler(tauri::generate_handler![
            // Backend commands
            backend_start,
            backend_stop,
            backend_restart,
            backend_status,
            backend_health_check,
            backend_get_port,
            backend_get_local_ip,
            backend_get_network_settings,
            // Video commands (GStreamer prototype)
            video_get_system_info,
            video_init,
            video_play,
            video_pause,
            video_resume,
            video_stop,
            video_seek,
            video_get_position,
            video_get_duration,
            video_set_volume,
            video_get_volume,
            video_set_muted,
            video_get_state,
            video_cleanup,
            // VFR video commands
            video_vfr_analyze,
            video_vfr_init,
            video_vfr_play,
            video_vfr_pause,
            video_vfr_resume,
            video_vfr_stop,
            video_vfr_seek,
            video_vfr_seek_with_mode,
            video_vfr_step_frame,
            video_vfr_get_position,
            video_vfr_get_duration,
            video_vfr_get_state,
            video_vfr_get_stream_info,
            video_vfr_get_frame_info,
            video_vfr_set_seek_mode,
            video_vfr_get_seek_mode,
            video_vfr_set_rate,
            video_vfr_set_volume,
            video_vfr_get_volume,
            video_vfr_set_muted,
            video_vfr_is_muted,
            video_vfr_cleanup,
            // Transcoding commands
            transcode_get_capabilities,
            transcode_check_needed,
            transcode_start,
            transcode_get_progress,
            transcode_is_ready,
            transcode_get_playlist,
            transcode_stop,
            transcode_stop_all,
            transcode_cleanup_cache,
            transcode_set_cache_limit,
            // IPC commands (ported from Electron)
            show_in_folder,
            get_app_version,
            quit_app,
            copy_image_to_clipboard,
            show_image_context_menu,
            // Interpolation commands
            interpolation_detect_backends,
            interpolation_get_config,
            interpolation_set_config,
            interpolation_set_enabled,
            interpolation_set_backend,
            interpolation_set_target_fps,
            interpolation_set_preset,
            interpolation_get_presets,
            interpolation_init,
            interpolation_play,
            interpolation_stop,
            interpolation_pause,
            interpolation_resume,
            interpolation_seek,
            interpolation_set_volume,
            interpolation_get_state,
            interpolation_get_stats,
            interpolation_cleanup,
            interpolation_is_available,
            interpolation_recommend_backend,
            // Video event streaming commands
            video_events_start,
            video_events_stop,
            video_events_is_active,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
