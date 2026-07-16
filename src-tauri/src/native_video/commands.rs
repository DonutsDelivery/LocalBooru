use std::sync::Mutex;
#[cfg(target_os = "linux")]
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager, State};

use super::coordinator::{
    CanonicalPlaybackState, DesktopPlayerMode, NativePlaybackCoordinator, PresentationState,
    RuntimeCapabilities,
};
use super::display_geometry::DisplayMode;
#[cfg(target_os = "linux")]
use super::dmabuf_surface_consumer::DmabufSurfaceConsumer;
#[cfg(target_os = "linux")]
use super::egl_dmabuf_import::build_nv12_attributes;
#[cfg(target_os = "linux")]
use super::helper_process::HelperProcessOptions;
#[cfg(target_os = "linux")]
use super::protocol::{NativeVideoCommand, NativeVideoEvent};
#[cfg(target_os = "linux")]
use super::runtime::{NativeVideoRuntime, NativeVideoRuntimeHandle, RuntimeNotification};
#[cfg(target_os = "linux")]
use super::surface_channel::SurfaceChannelMessage;
#[cfg(target_os = "linux")]
use super::surface_protocol::{SurfaceDescriptor, SurfaceFrameRelease, SurfaceHandleKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationTarget {
    ReactImage,
    PreparingNative,
    NativeVideo,
    WebFallback,
    CastReceiver,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NativePlaybackSnapshot {
    pub generation: u64,
    pub presentation: PresentationTarget,
    pub position: f64,
    pub item_id: Option<i64>,
    pub playback: CanonicalPlaybackState,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeViewportRequest {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub visible: bool,
}

pub struct NativeVideoState {
    capabilities: Mutex<RuntimeCapabilities>,
    coordinator: Mutex<NativePlaybackCoordinator>,
    transition: Mutex<()>,
    #[cfg(target_os = "linux")]
    runtime: Mutex<Option<NativeVideoRuntime>>,
    #[cfg(target_os = "linux")]
    runtime_epoch: AtomicU64,
    #[cfg(target_os = "linux")]
    runtime_owner: Mutex<Option<(u64, u64)>>,
    current_media: Mutex<Option<CurrentNativeMedia>>,
    recovery_attempted_generation: Mutex<Option<u64>>,
    playback_paused: AtomicBool,
    force_copy: bool,
    diagnostics_enabled: bool,
    opened_at: Mutex<Option<(u64, std::time::Instant)>>,
    first_frame_latency_bits: AtomicU64,
}

#[derive(Clone)]
struct CurrentNativeMedia {
    generation: u64,
    item_id: i64,
    path: String,
}

impl NativeVideoState {
    pub fn new(capabilities: RuntimeCapabilities) -> Self {
        Self::new_with_options(capabilities, false, false)
    }

    pub fn new_with_options(
        capabilities: RuntimeCapabilities,
        force_copy: bool,
        diagnostics_enabled: bool,
    ) -> Self {
        Self {
            capabilities: Mutex::new(capabilities),
            coordinator: Mutex::new(NativePlaybackCoordinator::new(capabilities)),
            transition: Mutex::new(()),
            #[cfg(target_os = "linux")]
            runtime: Mutex::new(None),
            #[cfg(target_os = "linux")]
            runtime_epoch: AtomicU64::new(0),
            #[cfg(target_os = "linux")]
            runtime_owner: Mutex::new(None),
            current_media: Mutex::new(None),
            recovery_attempted_generation: Mutex::new(None),
            playback_paused: AtomicBool::new(false),
            force_copy,
            diagnostics_enabled,
            opened_at: Mutex::new(None),
            first_frame_latency_bits: AtomicU64::new(0),
        }
    }

    pub fn capabilities(&self) -> RuntimeCapabilities {
        *self
            .capabilities
            .lock()
            .expect("native video capabilities poisoned")
    }

    pub fn set_desktop_player_mode(&self, mode: DesktopPlayerMode) {
        self.capabilities
            .lock()
            .expect("native video capabilities poisoned")
            .desktop_player_mode = mode;
        self.coordinator
            .lock()
            .expect("native video state poisoned")
            .set_desktop_player_mode(mode);
    }

    pub fn set_display_mode(&self, mode: DisplayMode) {
        let mode = match mode {
            DisplayMode::Fit => "fit",
            DisplayMode::Fill => "fill",
            DisplayMode::Original => "original",
        };
        self.coordinator
            .lock()
            .expect("native video state poisoned")
            .set_display_mode(mode);
    }

    pub fn open_video(&self, item_id: i64, resume_position: f64) -> NativePlaybackSnapshot {
        let mut coordinator = self
            .coordinator
            .lock()
            .expect("native video state poisoned");
        coordinator.open_video(item_id, resume_position);
        snapshot(&coordinator)
    }

    pub fn show_image(&self, item_id: i64) -> NativePlaybackSnapshot {
        let mut coordinator = self
            .coordinator
            .lock()
            .expect("native video state poisoned");
        coordinator.show_image(item_id);
        snapshot(&coordinator)
    }

    pub fn first_frame(&self, generation: u64) -> Option<NativePlaybackSnapshot> {
        let mut coordinator = self
            .coordinator
            .lock()
            .expect("native video state poisoned");
        coordinator
            .handle_first_frame(generation)
            .then(|| snapshot(&coordinator))
    }

    pub fn renderer_failed(&self, generation: u64) -> Option<NativePlaybackSnapshot> {
        let mut coordinator = self
            .coordinator
            .lock()
            .expect("native video state poisoned");
        coordinator
            .handle_renderer_failure(generation)
            .then(|| snapshot(&coordinator))
    }

    pub fn update_position(&self, generation: u64, position: f64) -> bool {
        self.coordinator
            .lock()
            .expect("native video state poisoned")
            .update_position(generation, position)
    }

    #[cfg(target_os = "linux")]
    #[allow(clippy::too_many_arguments)]
    fn update_playback_state(
        &self,
        generation: u64,
        position: f64,
        duration: f64,
        paused: bool,
        volume: f64,
        muted: bool,
        speed: f64,
        selected_audio_track: Option<String>,
        selected_subtitle_track: Option<String>,
        subtitle_delay: f64,
        interpolation_engine: &str,
        interpolation_preset: Option<String>,
        interpolation_target_fps: u32,
    ) -> bool {
        self.coordinator
            .lock()
            .expect("native video state poisoned")
            .update_playback_state(
                generation,
                position,
                duration,
                paused,
                volume,
                muted,
                speed,
                selected_audio_track,
                selected_subtitle_track,
                subtitle_delay,
                interpolation_engine,
                interpolation_preset,
                interpolation_target_fps,
            )
    }

    fn record_first_frame_latency(&self, generation: u64) {
        let started = {
            let mut opened_at = self
                .opened_at
                .lock()
                .expect("native video latency state poisoned");
            if opened_at.as_ref().map(|(owner, _)| *owner) != Some(generation) {
                return;
            }
            opened_at.take().map(|(_, started)| started)
        };
        if let Some(started) = started {
            let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
            self.first_frame_latency_bits
                .store(milliseconds.to_bits(), Ordering::Release);
        }
    }

    fn first_frame_latency_ms(&self) -> Option<f64> {
        let bits = self.first_frame_latency_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    #[cfg(target_os = "linux")]
    fn runtime_handle(
        &self,
        app: &tauri::AppHandle,
    ) -> Result<(NativeVideoRuntimeHandle, u64), String> {
        let mut slot = self.runtime.lock().expect("native video runtime poisoned");
        if let Some(runtime) = slot.as_ref() {
            return Ok((runtime.handle(), self.runtime_epoch.load(Ordering::Acquire)));
        }
        let executable = resolve_helper_path(app)?;
        let mut runtime = NativeVideoRuntime::start_with_options(
            &executable,
            HelperProcessOptions {
                force_copy: self.force_copy,
            },
        )?;
        let notifications = runtime.take_notifications()?;
        let handle = runtime.handle();
        let epoch = self.runtime_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        std::thread::Builder::new()
            .name("native-video-notifications".to_string())
            .spawn({
                let app = app.clone();
                let handle = handle.clone();
                move || notification_loop(app, handle, notifications, epoch)
            })
            .map_err(|error| format!("failed to start native video notification pump: {error}"))?;
        *slot = Some(runtime);
        Ok((handle, epoch))
    }

    #[cfg(target_os = "linux")]
    fn running_handle(&self) -> Option<NativeVideoRuntimeHandle> {
        self.runtime
            .lock()
            .expect("native video runtime poisoned")
            .as_ref()
            .map(NativeVideoRuntime::handle)
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn send_runtime_control(&self, command: NativeVideoCommand) -> Result<(), String> {
        if matches!(
            command,
            NativeVideoCommand::OpenMedia { .. } | NativeVideoCommand::CloseMedia { .. }
        ) {
            return Err("media lifecycle commands must use the coordinator".to_string());
        }
        self.running_handle()
            .ok_or_else(|| "native video runtime is not running".to_string())?
            .send(command)
    }

    #[cfg(target_os = "linux")]
    fn clear_runtime(&self, epoch: u64) {
        if self.runtime_epoch.load(Ordering::Acquire) == epoch {
            self.runtime
                .lock()
                .expect("native video runtime poisoned")
                .take();
        }
    }

    #[cfg(target_os = "linux")]
    fn set_runtime_owner(&self, epoch: u64, generation: u64) {
        *self
            .runtime_owner
            .lock()
            .expect("native video runtime owner poisoned") = Some((epoch, generation));
    }

    #[cfg(target_os = "linux")]
    fn clear_runtime_owner(&self) {
        self.runtime_owner
            .lock()
            .expect("native video runtime owner poisoned")
            .take();
    }

    #[cfg(target_os = "linux")]
    fn runtime_owner_generation(&self, epoch: u64) -> Option<u64> {
        match *self
            .runtime_owner
            .lock()
            .expect("native video runtime owner poisoned")
        {
            Some((owner_epoch, generation)) if owner_epoch == epoch => Some(generation),
            _ => None,
        }
    }

    #[cfg(target_os = "linux")]
    fn fail_runtime_owner(&self, epoch: u64) -> Option<NativePlaybackSnapshot> {
        let generation = {
            let mut owner = self
                .runtime_owner
                .lock()
                .expect("native video runtime owner poisoned");
            match *owner {
                Some((owner_epoch, generation)) if owner_epoch == epoch => {
                    owner.take();
                    Some(generation)
                }
                _ => None,
            }
        }?;
        (self.current_generation() == generation)
            .then(|| self.renderer_failed(generation))
            .flatten()
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn current_generation(&self) -> u64 {
        self.coordinator
            .lock()
            .expect("native video state poisoned")
            .generation()
    }

    fn whisper_media(&self, generation: u64) -> Option<(CurrentNativeMedia, f64)> {
        let media = self
            .current_media
            .lock()
            .expect("native video media state poisoned")
            .clone()?;
        let coordinator = self
            .coordinator
            .lock()
            .expect("native video state poisoned");
        (media.generation == generation && coordinator.generation() == generation)
            .then(|| (media, coordinator.position()))
    }

    #[cfg(target_os = "linux")]
    fn register_whisper_track(
        &self,
        app: &tauri::AppHandle,
        generation: u64,
        path: String,
        language: Option<String>,
    ) -> Result<(), String> {
        if self.current_generation() != generation {
            return Err("Whisper result belongs to a stale native generation".to_string());
        }
        let (runtime, _) = self.runtime_handle(app)?;
        let label = language
            .as_deref()
            .map(|language| format!("Whisper {language}"))
            .unwrap_or_else(|| "Whisper subtitles".to_string());
        runtime.send(NativeVideoCommand::RegisterSubtitleTrack {
            generation,
            id: format!("whisper:{generation}"),
            path,
            label,
            language,
            select: true,
        })
    }

    #[cfg(target_os = "linux")]
    fn claim_recovery_attempt(&self, generation: u64) -> bool {
        let mut attempted = self
            .recovery_attempted_generation
            .lock()
            .expect("native video recovery state poisoned");
        if *attempted == Some(generation) {
            false
        } else {
            *attempted = Some(generation);
            true
        }
    }

    #[cfg(target_os = "linux")]
    fn restart_runtime_once(&self, app: &tauri::AppHandle, generation: u64) -> Result<(), String> {
        let (media, position) = self
            .whisper_media(generation)
            .ok_or_else(|| "native recovery generation is no longer active".to_string())?;
        if !self.claim_recovery_attempt(generation) {
            return Err("native helper restart was already attempted".to_string());
        }
        let autoplay = !self.playback_paused.load(Ordering::Acquire);
        log::info!(
            "[NativeVideo] restarting generation {generation} at position={position:.3} autoplay={autoplay}"
        );
        let (runtime, runtime_epoch) = self.runtime_handle(app)?;
        runtime.send(NativeVideoCommand::OpenMedia {
            generation,
            item_id: media.item_id,
            path: media.path,
            resume_position: position,
            autoplay,
        })?;
        self.set_runtime_owner(runtime_epoch, generation);
        Ok(())
    }

    pub(crate) fn open_runtime_media(
        &self,
        app: &tauri::AppHandle,
        item_id: i64,
        path: String,
        resume_position: f64,
    ) -> Result<NativePlaybackSnapshot, String> {
        let _transition = self
            .transition
            .lock()
            .expect("native video transition poisoned");
        let opened = self.open_video(item_id, resume_position);
        #[cfg(target_os = "linux")]
        {
            if opened.presentation != PresentationTarget::PreparingNative {
                return Ok(opened);
            }
            *self
                .opened_at
                .lock()
                .expect("native video latency state poisoned") =
                Some((opened.generation, std::time::Instant::now()));
            self.first_frame_latency_bits.store(0, Ordering::Release);
            if path.is_empty() {
                self.renderer_failed(opened.generation);
                return Err("native video path is empty".to_string());
            }
            let window = match app.get_webview_window("main") {
                Some(window) => window,
                None => {
                    self.renderer_failed(opened.generation);
                    return Err("main WebView window is unavailable".to_string());
                }
            };
            if let Err(error) = super::platform::linux::attach(&window) {
                self.renderer_failed(opened.generation);
                return Err(format!("native viewport attach failed: {error}"));
            }
            let canonical_path = match std::fs::canonicalize(&path) {
                Ok(path) => path,
                Err(error) => {
                    self.renderer_failed(opened.generation);
                    return Err(format!("native video path is unavailable: {error}"));
                }
            };
            if !canonical_path.is_file() {
                self.renderer_failed(opened.generation);
                return Err("native video path is not a regular file".to_string());
            }
            let (runtime, runtime_epoch) = match self.runtime_handle(app) {
                Ok(runtime) => runtime,
                Err(error) => {
                    self.renderer_failed(opened.generation);
                    return Err(error);
                }
            };
            if self.current_generation() != opened.generation {
                return Err("native video open was superseded by a newer selection".to_string());
            }
            let _ = app.run_on_main_thread(|| {
                let _ = super::platform::linux::set_visible(false);
            });
            self.set_runtime_owner(runtime_epoch, opened.generation);
            let canonical_path = canonical_path.to_string_lossy().into_owned();
            if let Err(error) = runtime.send(NativeVideoCommand::OpenMedia {
                generation: opened.generation,
                item_id,
                path: canonical_path.clone(),
                resume_position: opened.position,
                autoplay: true,
            }) {
                self.clear_runtime_owner();
                self.renderer_failed(opened.generation);
                return Err(error);
            }
            *self
                .current_media
                .lock()
                .expect("native video media state poisoned") = Some(CurrentNativeMedia {
                generation: opened.generation,
                item_id,
                path: canonical_path,
            });
            self.recovery_attempted_generation
                .lock()
                .expect("native video recovery state poisoned")
                .take();
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (app, path);
        Ok(opened)
    }

    pub(crate) fn show_runtime_image(
        &self,
        app: &tauri::AppHandle,
        item_id: i64,
    ) -> NativePlaybackSnapshot {
        let _transition = self
            .transition
            .lock()
            .expect("native video transition poisoned");
        #[cfg(target_os = "linux")]
        let previous_generation = self.current_generation();
        let shown = self.show_image(item_id);
        self.current_media
            .lock()
            .expect("native video media state poisoned")
            .take();
        #[cfg(target_os = "linux")]
        {
            self.clear_runtime_owner();
            if let Some(runtime) = self.running_handle() {
                let _ = runtime.send(NativeVideoCommand::CloseMedia {
                    generation: previous_generation,
                });
            }
            let _ = app.run_on_main_thread(|| {
                let _ = super::platform::linux::set_visible(false);
            });
        }
        #[cfg(not(target_os = "linux"))]
        let _ = app;
        shown
    }
}

#[cfg(target_os = "linux")]
fn resolve_helper_path(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    if let Some(configured) = std::env::var_os("LOCALBOORU_NATIVE_VIDEO_HELPER") {
        let path = std::path::PathBuf::from(configured);
        return path.is_file().then_some(path.clone()).ok_or_else(|| {
            format!(
                "configured native video helper is missing: {}",
                path.display()
            )
        });
    }
    let mut candidates = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("native-video/localbooru-native-video"));
        candidates.push(resource_dir.join("localbooru-native-video"));
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(directory) = current_exe.parent() {
            candidates.push(directory.join("localbooru-native-video"));
        }
    }
    candidates.push(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../native-video/build/localbooru-native-video"),
    );
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| "native video helper executable was not found".to_string())
}

#[cfg(target_os = "linux")]
fn notification_loop(
    app: tauri::AppHandle,
    runtime: NativeVideoRuntimeHandle,
    notifications: std::sync::mpsc::Receiver<RuntimeNotification>,
    runtime_epoch: u64,
) {
    let mut dmabuf_consumer = DmabufSurfaceConsumer::default();
    let mut dmabuf_registry_frames = 0_u64;
    while let Ok(notification) = notifications.recv() {
        match notification {
            RuntimeNotification::Control(mut event) => {
                super::platform::linux::enrich_diagnostics(&mut event);
                let state = app.state::<NativeVideoState>();
                if event
                    .generation()
                    .is_some_and(|generation| generation != state.current_generation())
                {
                    log::debug!("[NativeVideo] discarded stale control event: {event:?}");
                    continue;
                }
                match &event {
                    NativeVideoEvent::CapabilitiesChanged {
                        interpolation_engines,
                        svp_status,
                        ..
                    } => {
                        let interpolation_engines = interpolation_engines.clone();
                        let svp_status = svp_status.clone();
                        let _ = app.run_on_main_thread(move || {
                            let _ = super::platform::linux::update_interpolation_capabilities(
                                &interpolation_engines,
                                svp_status.as_deref(),
                            );
                        });
                    }
                    NativeVideoEvent::PlaybackState {
                        generation,
                        position,
                        duration,
                        paused,
                        volume,
                        muted,
                        speed,
                        selected_audio_track,
                        selected_subtitle_track,
                        subtitle_delay,
                        interpolation_engine,
                        interpolation_preset,
                        interpolation_target_fps,
                    } => {
                        state.update_playback_state(
                            *generation,
                            *position,
                            *duration,
                            *paused,
                            *volume,
                            *muted,
                            *speed,
                            selected_audio_track.clone(),
                            selected_subtitle_track.clone(),
                            *subtitle_delay,
                            interpolation_engine,
                            interpolation_preset.clone(),
                            *interpolation_target_fps,
                        );
                        let playback_snapshot = {
                            let coordinator = state
                                .coordinator
                                .lock()
                                .expect("native video state poisoned");
                            snapshot(&coordinator)
                        };
                        let _ = app.emit("native-video-snapshot", playback_snapshot);
                        state.playback_paused.store(*paused, Ordering::Release);
                        let position = *position;
                        let duration = *duration;
                        let paused = *paused;
                        let volume = *volume;
                        let muted = *muted;
                        let speed = *speed;
                        let selected_audio_track = selected_audio_track.clone();
                        let selected_subtitle_track = selected_subtitle_track.clone();
                        let subtitle_delay = *subtitle_delay;
                        let interpolation_engine = interpolation_engine.clone();
                        let generation = *generation;
                        let scheduled_app = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            if scheduled_app
                                .state::<NativeVideoState>()
                                .current_generation()
                                != generation
                            {
                                return;
                            }
                            let _ = super::platform::linux::update_playback(
                                position,
                                duration,
                                paused,
                                volume,
                                muted,
                                speed,
                                selected_audio_track.as_deref(),
                                selected_subtitle_track.as_deref(),
                                subtitle_delay,
                                &interpolation_engine,
                            );
                        });
                    }
                    NativeVideoEvent::FirstFrameReady { generation } => {
                        state.record_first_frame_latency(*generation);
                        if let Some(snapshot) = state.first_frame(*generation) {
                            log::info!(
                                "[NativeVideo] helper first frame ready for generation {generation}"
                            );
                            let _ = app.emit("native-video-snapshot", snapshot);
                        }
                    }
                    NativeVideoEvent::TrackList {
                        generation,
                        audio,
                        subtitles,
                    } => {
                        let generation = *generation;
                        let audio = audio.clone();
                        let subtitles = subtitles.clone();
                        let scheduled_app = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            if scheduled_app
                                .state::<NativeVideoState>()
                                .current_generation()
                                != generation
                            {
                                return;
                            }
                            let _ = super::platform::linux::update_tracks(&audio, &subtitles);
                        });
                    }
                    NativeVideoEvent::SubtitleTrackAdded { generation, track } => {
                        let generation = *generation;
                        let track = track.clone();
                        let scheduled_app = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            if scheduled_app
                                .state::<NativeVideoState>()
                                .current_generation()
                                != generation
                            {
                                return;
                            }
                            let _ = super::platform::linux::add_subtitle_track(&track, true);
                        });
                    }
                    NativeVideoEvent::SubtitleText { generation, lines } => {
                        let generation = *generation;
                        let lines = lines.clone();
                        let scheduled_app = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            if scheduled_app
                                .state::<NativeVideoState>()
                                .current_generation()
                                != generation
                            {
                                return;
                            }
                            let _ = super::platform::linux::update_subtitle_text(&lines);
                        });
                    }
                    NativeVideoEvent::Diagnostics {
                        generation,
                        produced_fps,
                        presented_fps,
                        dropped_frames,
                        zero_cpu_copy,
                        fallback_reason,
                        decoder,
                        hardware_device,
                        source_fps,
                        queue_depth,
                        queue_latency_ms,
                        accepted_frames,
                        draw_completed_frames,
                        interpolation_engine,
                        surface_mode,
                        width,
                        height,
                        av_drift_ms,
                        seek_latency_ms,
                        ..
                    } => {
                        log::info!(
                            "[NativeVideo] diagnostics generation={} produced_fps={:.1} presented_fps={:.1} accepted={} draw_completed={} dropped={} queue_depth={} queue_latency_ms={:.3} av_drift_ms={:.3} zero_cpu_copy={} copy_mode={} fallback_reason={}",
                            generation,
                            produced_fps,
                            presented_fps,
                            accepted_frames.unwrap_or(0),
                            draw_completed_frames.unwrap_or(0),
                            dropped_frames,
                            queue_depth.unwrap_or(0),
                            queue_latency_ms.unwrap_or(0.0),
                            av_drift_ms.unwrap_or(0.0),
                            zero_cpu_copy,
                            surface_mode.as_deref().unwrap_or("unknown"),
                            fallback_reason.as_deref().unwrap_or("none")
                        );
                        let produced_fps = *produced_fps;
                        let presented_fps = *presented_fps;
                        let dropped_frames = *dropped_frames;
                        let zero_cpu_copy = *zero_cpu_copy;
                        let decoder = decoder.clone();
                        let hardware_device = hardware_device.clone();
                        let source_fps = *source_fps;
                        let queue_depth = *queue_depth;
                        let interpolation_engine = interpolation_engine.clone();
                        let surface_mode = surface_mode.clone();
                        let width = *width;
                        let height = *height;
                        let av_drift_ms = *av_drift_ms;
                        let seek_latency_ms = *seek_latency_ms;
                        let first_frame_latency_ms = state.first_frame_latency_ms();
                        let generation = *generation;
                        let diagnostics_enabled = state.diagnostics_enabled;
                        let scheduled_app = app.clone();
                        let _ = app.run_on_main_thread(move || {
                            if !diagnostics_enabled
                                || scheduled_app
                                    .state::<NativeVideoState>()
                                    .current_generation()
                                    != generation
                            {
                                return;
                            }
                            let _ = super::platform::linux::update_diagnostics(
                                produced_fps,
                                presented_fps,
                                dropped_frames,
                                zero_cpu_copy,
                                decoder.as_deref(),
                                hardware_device.as_deref(),
                                source_fps,
                                queue_depth,
                                interpolation_engine.as_deref(),
                                surface_mode.as_deref(),
                                width,
                                height,
                                av_drift_ms,
                                first_frame_latency_ms,
                                seek_latency_ms,
                            );
                        });
                    }
                    NativeVideoEvent::RecoverableError { message, .. } => {
                        let message = message.clone();
                        let _ = app.run_on_main_thread(move || {
                            let _ = super::platform::linux::update_status(Some(&message), false);
                        });
                    }
                    NativeVideoEvent::FatalError { message, .. } => {
                        log::warn!(
                            "[NativeVideo] helper reported a fatal error before exit: {message}"
                        );
                        let message = message.clone();
                        let _ = app.run_on_main_thread(move || {
                            let _ = super::platform::linux::update_status(Some(&message), true);
                        });
                    }
                    NativeVideoEvent::PlaybackEnded { generation, .. }
                        if std::env::var_os("LOCALBOORU_NATIVE_RUNTIME_SPIKE_LOOP").is_some()
                            && app.state::<NativeVideoState>().current_generation()
                                == *generation =>
                    {
                        log::info!(
                            "[NativeVideo] managed runtime spike looping generation {generation}"
                        );
                        if let Err(error) = runtime.send(NativeVideoCommand::Seek { position: 0.0 })
                        {
                            let _ = app.emit("native-video-runtime-error", error);
                        } else if let Err(error) =
                            runtime.send(NativeVideoCommand::SetPaused { paused: false })
                        {
                            let _ = app.emit("native-video-runtime-error", error);
                        }
                    }
                    _ => {}
                }
                let _ = app.emit("native-video-event", event);
            }
            RuntimeNotification::Surface(received) => {
                let state = app.state::<NativeVideoState>();
                if surface_generation(&received.message)
                    .is_some_and(|generation| generation != state.current_generation())
                {
                    if let Some(release) = surface_release(&received.message) {
                        let _ = runtime.release(release);
                    }
                    continue;
                }
                if matches!(
                    &received.message,
                    SurfaceChannelMessage::SurfaceCreated { descriptor }
                        if descriptor.handle_kind == SurfaceHandleKind::DmaBuf
                ) {
                    if let Err(error) = dmabuf_consumer.register(received) {
                        let _ = app.emit("native-video-runtime-error", error);
                    }
                    continue;
                }
                if matches!(
                    &received.message,
                    SurfaceChannelMessage::SurfaceCreated { descriptor }
                        if descriptor.handle_kind != SurfaceHandleKind::DmaBuf
                ) {
                    // Buffer ids are reused when switching transport within
                    // one playback generation. Remove stale DMA ownership so
                    // SHM frame-ready messages cannot be mistaken for DMA.
                    dmabuf_consumer.reset();
                    if let SurfaceChannelMessage::SurfaceCreated { descriptor } = &received.message
                    {
                        log::info!(
                            "[NativeVideo] registering SHM surface buffer={} fourcc={:#x} planes={}",
                            descriptor.buffer_id,
                            descriptor.fourcc,
                            descriptor.planes.len()
                        );
                    }
                }
                if let SurfaceChannelMessage::FrameReady { frame } = &received.message {
                    if dmabuf_consumer.contains_surface(frame.generation, frame.buffer_id) {
                        let release = SurfaceFrameRelease {
                            generation: frame.generation,
                            buffer_id: frame.buffer_id,
                            sequence: frame.sequence,
                        };
                        let registry_result = match dmabuf_consumer.begin_frame(received) {
                            Ok(view) => {
                                let has_objects = !view.object_fds.is_empty();
                                let import_attributes =
                                    build_nv12_attributes(view.descriptor, &view.object_fds);
                                let probe_frame =
                                    (view.descriptor.clone(), view.object_fds.clone());
                                drop(view);
                                if !has_objects {
                                    Err("DMA-BUF registry frame has no object descriptors"
                                        .to_string())
                                } else if let Err(error) = import_attributes {
                                    Err(error)
                                } else if super::platform::linux::is_attached() {
                                    import_dmabuf_on_main_thread(&app, probe_frame.0, probe_frame.1)
                                        .and_then(|()| {
                                            dmabuf_consumer.complete_frame(release.clone())
                                        })
                                } else {
                                    dmabuf_consumer.complete_frame(release.clone())
                                }
                            }
                            Err(error) => Err(error),
                        };
                        match registry_result {
                            Ok(()) => {
                                dmabuf_registry_frames += 1;
                                if super::platform::linux::is_attached() {
                                    let state = app.state::<NativeVideoState>();
                                    state.record_first_frame_latency(release.generation);
                                    if let Some(snapshot) = state.first_frame(release.generation) {
                                        log::info!(
                                            "[NativeVideo] GTK DMA-BUF first frame rendered for generation {}",
                                            release.generation
                                        );
                                        let visible_app = app.clone();
                                        let visible_generation = release.generation;
                                        let _ = app.run_on_main_thread(move || {
                                            if visible_app
                                                .state::<NativeVideoState>()
                                                .current_generation()
                                                != visible_generation
                                            {
                                                return;
                                            }
                                            if let Err(error) =
                                                super::platform::linux::set_visible(true)
                                            {
                                                let _ = visible_app
                                                    .emit("native-video-runtime-error", error);
                                            }
                                        });
                                        let _ = app.emit("native-video-snapshot", snapshot);
                                    }
                                }
                                if dmabuf_registry_frames == 1 || dmabuf_registry_frames % 120 == 0
                                {
                                    log::info!(
                                        "[NativeVideo] DMA-BUF registry consumed {} frames",
                                        dmabuf_registry_frames
                                    );
                                }
                                let _ = runtime.release(release);
                            }
                            Err(error) => {
                                let _ = runtime.release(release);
                                let _ = app.emit("native-video-runtime-error", error);
                            }
                        }
                        continue;
                    }
                }
                let generation = surface_generation(&received.message);
                let fallback_release = surface_release(&received.message);
                let scheduled_app = app.clone();
                let presented_runtime = runtime.clone();
                let discarded_runtime = runtime.clone();
                let error_runtime = runtime.clone();
                let fallback_for_schedule = fallback_release.clone();
                let stale_runtime = runtime.clone();
                let schedule_result = app.run_on_main_thread(move || {
                    if generation.is_some_and(|generation| {
                        scheduled_app
                            .state::<NativeVideoState>()
                            .current_generation()
                            != generation
                    }) {
                        if let Some(release) = fallback_for_schedule.clone() {
                            let _ = stale_runtime.release(release);
                        }
                        return;
                    }
                    let presented_app = scheduled_app.clone();
                    let result = super::platform::linux::handle_surface_message(
                        received,
                        move |release| {
                            let generation = release.generation;
                            let _ = presented_runtime.release(release);
                            let state = presented_app.state::<NativeVideoState>();
                            state.record_first_frame_latency(generation);
                            if let Some(snapshot) = state.first_frame(generation) {
                                log::info!(
                                    "[NativeVideo] GTK first frame presented for generation {generation}"
                                );
                                let _ = super::platform::linux::set_visible(true);
                                let _ = presented_app.emit("native-video-snapshot", snapshot);
                            }
                        },
                        move |release| {
                            let _ = discarded_runtime.release(release);
                        },
                    );
                    if let Err(error) = result {
                        log::error!("[NativeVideo] GTK surface message failed: {error}");
                        if let Some(release) = fallback_for_schedule {
                            let _ = error_runtime.release(release);
                        }
                        if let Some(generation) = generation {
                            let state = scheduled_app.state::<NativeVideoState>();
                            if let Some(snapshot) = state.renderer_failed(generation) {
                                let _ = scheduled_app.emit("native-video-snapshot", snapshot);
                            }
                        }
                        let _ = scheduled_app.emit("native-video-runtime-error", error);
                    }
                });
                if let Err(error) = schedule_result {
                    log::error!("[NativeVideo] GTK surface scheduling failed: {error}");
                    if let Some(release) = fallback_release {
                        let _ = runtime.release(release);
                    }
                    let _ = app.emit(
                        "native-video-runtime-error",
                        format!("failed to schedule GTK surface message: {error}"),
                    );
                }
            }
            RuntimeNotification::Exited { success } => {
                log::warn!(
                    "[NativeVideo] helper runtime epoch {runtime_epoch} exited success={success}"
                );
                let state = app.state::<NativeVideoState>();
                let owner_generation = state.runtime_owner_generation(runtime_epoch);
                state.clear_runtime(runtime_epoch);
                if let Some(generation) = owner_generation {
                    match state.restart_runtime_once(&app, generation) {
                        Ok(()) => {
                            let _ = app.emit(
                                "native-video-runtime-recovered",
                                serde_json::json!({"generation": generation, "reason": "helper_exit"}),
                            );
                            return;
                        }
                        Err(error) => log::warn!(
                            "[NativeVideo] one-shot helper recovery failed for generation {generation}: {error}"
                        ),
                    }
                }
                if let Some(snapshot) = state.fail_runtime_owner(runtime_epoch) {
                    let _ = app.emit("native-video-snapshot", snapshot);
                }
                let _ = app.emit("native-video-runtime-exited", success);
                return;
            }
            RuntimeNotification::Error(error) => {
                log::warn!(
                    "[NativeVideo] helper runtime epoch {runtime_epoch} error before recovery: {error}"
                );
                let state = app.state::<NativeVideoState>();
                let owner_generation = state.runtime_owner_generation(runtime_epoch);
                state.clear_runtime(runtime_epoch);
                if let Some(generation) = owner_generation {
                    match state.restart_runtime_once(&app, generation) {
                        Ok(()) => {
                            let _ = app.emit(
                                "native-video-runtime-recovered",
                                serde_json::json!({"generation": generation, "reason": "runtime_error"}),
                            );
                            return;
                        }
                        Err(restart_error) => log::warn!(
                            "[NativeVideo] one-shot helper recovery failed for generation {generation}: {restart_error}"
                        ),
                    }
                }
                if let Some(snapshot) = state.fail_runtime_owner(runtime_epoch) {
                    let _ = app.emit("native-video-snapshot", snapshot);
                }
                let _ = app.emit("native-video-runtime-error", error);
                return;
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn import_dmabuf_on_main_thread(
    app: &tauri::AppHandle,
    descriptor: SurfaceDescriptor,
    object_fds: Vec<i32>,
) -> Result<(), String> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let scheduled_app = app.clone();
    let generation = descriptor.generation;
    app.run_on_main_thread(move || {
        let error_sender = sender.clone();
        if scheduled_app
            .state::<NativeVideoState>()
            .current_generation()
            != generation
        {
            let _ = error_sender.send(Err(format!(
                "discarded stale DMA-BUF generation {generation}"
            )));
            return;
        }
        if let Err(error) =
            super::platform::linux::queue_dmabuf_render(descriptor, object_fds, sender)
        {
            let _ = error_sender.send(Err(error));
        }
    })
    .map_err(|error| format!("failed to schedule DMA-BUF import: {error}"))?;
    receiver
        .recv_timeout(Duration::from_secs(2))
        .map_err(|error| format!("DMA-BUF import timed out: {error}"))?
}

#[cfg(target_os = "linux")]
fn surface_generation(message: &SurfaceChannelMessage) -> Option<u64> {
    match message {
        SurfaceChannelMessage::SurfaceCreated { descriptor } => Some(descriptor.generation),
        SurfaceChannelMessage::FrameReady { frame } => Some(frame.generation),
        SurfaceChannelMessage::FrameRelease { release } => Some(release.generation),
    }
}

#[cfg(target_os = "linux")]
fn surface_release(message: &SurfaceChannelMessage) -> Option<SurfaceFrameRelease> {
    match message {
        SurfaceChannelMessage::FrameReady { frame } => Some(SurfaceFrameRelease {
            generation: frame.generation,
            buffer_id: frame.buffer_id,
            sequence: frame.sequence,
        }),
        _ => None,
    }
}

fn snapshot(coordinator: &NativePlaybackCoordinator) -> NativePlaybackSnapshot {
    let state = coordinator.state();
    let current_position = coordinator.position();
    let current_item_id = coordinator.item_id();
    let build = |generation, presentation, position, item_id| NativePlaybackSnapshot {
        generation,
        presentation,
        position,
        item_id,
        playback: coordinator.playback().clone(),
    };
    match *state {
        PresentationState::Image {
            item_id,
            generation,
        } => build(
            generation,
            PresentationTarget::ReactImage,
            0.0,
            Some(item_id),
        ),
        PresentationState::PreparingVideo { generation } => build(
            generation,
            PresentationTarget::PreparingNative,
            current_position,
            current_item_id,
        ),
        PresentationState::VisibleVideo { generation } => build(
            generation,
            PresentationTarget::NativeVideo,
            current_position,
            current_item_id,
        ),
        PresentationState::WebFallback {
            generation,
            position,
        } => build(
            generation,
            PresentationTarget::WebFallback,
            position,
            current_item_id,
        ),
        PresentationState::Casting {
            generation,
            position,
        } => build(
            generation,
            PresentationTarget::CastReceiver,
            position,
            current_item_id,
        ),
    }
}

#[tauri::command]
pub fn native_video_capabilities(state: State<'_, NativeVideoState>) -> RuntimeCapabilities {
    let capabilities = state.capabilities();
    log::info!(
        "[NativeVideo] capabilities desktop={} available={} safe_mode={}",
        capabilities.desktop_tauri,
        capabilities.native_renderer_available,
        capabilities.safe_mode
    );
    capabilities
}

#[tauri::command]
pub fn native_video_set_desktop_player_mode(
    mode: DesktopPlayerMode,
    state: State<'_, NativeVideoState>,
) -> RuntimeCapabilities {
    state.set_desktop_player_mode(mode);
    state.capabilities()
}

#[cfg(target_os = "linux")]
pub(crate) fn request_whisper_subtitles(
    app: &tauri::AppHandle,
    generation: u64,
) -> Result<(), String> {
    use crate::server::state::AppState;

    let native_state = app.state::<NativeVideoState>();
    let (media, position) = native_state
        .whisper_media(generation)
        .ok_or_else(|| "Whisper request belongs to a stale native generation".to_string())?;
    let server_state = app.state::<AppState>().inner().clone();
    let task_app = app.clone();
    tauri::async_runtime::spawn(async move {
        fn emit_status(
            app: &tauri::AppHandle,
            generation: u64,
            status: &str,
            message: Option<String>,
        ) {
            let _ = app.emit(
                "native-video-whisper-status",
                serde_json::json!({
                    "generation": generation,
                    "status": status,
                    "message": message,
                }),
            );
            let status = status.to_string();
            let scheduled_app = app.clone();
            let _ = app.run_on_main_thread(move || {
                if scheduled_app
                    .state::<NativeVideoState>()
                    .current_generation()
                    == generation
                {
                    let _ = super::platform::linux::update_whisper_status(&status);
                }
            });
        }

        emit_status(&task_app, generation, "starting", None);
        let response = match crate::routes::settings::generate_whisper_for_native(
            server_state.clone(),
            media.path,
            media.item_id,
            position,
        )
        .await
        {
            Ok(response) => response,
            Err(error) => {
                emit_status(&task_app, generation, "failed", Some(error.to_string()));
                return;
            }
        };
        let Some(stream_id) = response
            .get("stream_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
        else {
            emit_status(
                &task_app,
                generation,
                "failed",
                Some("Whisper returned no stream ID".to_string()),
            );
            return;
        };
        loop {
            if task_app.state::<NativeVideoState>().current_generation() != generation {
                return;
            }
            let status =
                match crate::routes::settings::whisper_status_for_native(&server_state, &stream_id)
                    .await
                {
                    Ok(status) => status,
                    Err(error) => {
                        emit_status(&task_app, generation, "failed", Some(error.to_string()));
                        return;
                    }
                };
            match status.get("state").and_then(serde_json::Value::as_str) {
                Some("completed") => {
                    let path = status
                        .get("cue_path")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                    let language = status
                        .get("language")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                    let Some(path) = path else {
                        emit_status(
                            &task_app,
                            generation,
                            "failed",
                            Some("Whisper completed without a durable track".to_string()),
                        );
                        return;
                    };
                    match task_app
                        .state::<NativeVideoState>()
                        .register_whisper_track(&task_app, generation, path, language)
                    {
                        Ok(()) => emit_status(&task_app, generation, "completed", None),
                        Err(error) => emit_status(&task_app, generation, "failed", Some(error)),
                    }
                    return;
                }
                Some("failed") => {
                    emit_status(
                        &task_app,
                        generation,
                        "failed",
                        status
                            .get("error")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string),
                    );
                    return;
                }
                Some(state) => emit_status(&task_app, generation, state, None),
                None => emit_status(&task_app, generation, "generating", None),
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });
    Ok(())
}

#[tauri::command]
pub fn native_video_open(
    app: tauri::AppHandle,
    state: State<'_, NativeVideoState>,
    item_id: i64,
    path: String,
    resume_position: f64,
) -> Result<NativePlaybackSnapshot, String> {
    state.open_runtime_media(&app, item_id, path, resume_position)
}

#[tauri::command]
pub fn native_video_show_image(
    app: tauri::AppHandle,
    state: State<'_, NativeVideoState>,
    item_id: i64,
) -> NativePlaybackSnapshot {
    state.show_runtime_image(&app, item_id)
}

#[tauri::command]
pub fn native_video_first_frame(
    state: State<'_, NativeVideoState>,
    generation: u64,
) -> Option<NativePlaybackSnapshot> {
    state.first_frame(generation)
}

#[tauri::command]
pub fn native_video_renderer_failed(
    state: State<'_, NativeVideoState>,
    generation: u64,
) -> Option<NativePlaybackSnapshot> {
    state.renderer_failed(generation)
}

#[tauri::command]
pub fn native_video_set_interpolation(
    state: State<'_, NativeVideoState>,
    engine: String,
    preset: Option<String>,
    target_fps: u32,
) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        if !matches!(engine.as_str(), "off" | "svp") {
            return Err(format!("unsupported native interpolation engine: {engine}"));
        }
        if !(24..=240).contains(&target_fps) {
            return Err("native interpolation target FPS must be between 24 and 240".to_string());
        }
        log::info!(
            "[NativeVideo] interpolation request engine={} preset={:?} target_fps={}",
            engine,
            preset,
            target_fps
        );
        return state.send_runtime_control(NativeVideoCommand::SetInterpolation {
            engine,
            preset,
            target_fps,
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (state, engine, preset, target_fps);
        Err("native interpolation is not implemented on this platform".to_string())
    }
}

#[tauri::command]
pub fn native_video_set_paused(
    state: State<'_, NativeVideoState>,
    paused: bool,
) -> Result<(), String> {
    state.send_runtime_control(NativeVideoCommand::SetPaused { paused })
}

#[tauri::command]
pub fn native_video_set_muted(
    state: State<'_, NativeVideoState>,
    muted: bool,
) -> Result<(), String> {
    state.send_runtime_control(NativeVideoCommand::SetMuted { muted })
}

#[tauri::command]
pub fn native_video_set_volume(
    state: State<'_, NativeVideoState>,
    volume: f64,
) -> Result<(), String> {
    if !volume.is_finite() {
        return Err("native video volume must be finite".to_string());
    }
    state.send_runtime_control(NativeVideoCommand::SetVolume {
        volume: volume.clamp(0.0, 1.0),
    })
}

#[tauri::command]
pub fn native_video_set_speed(
    state: State<'_, NativeVideoState>,
    speed: f64,
) -> Result<(), String> {
    if !speed.is_finite() {
        return Err("native video speed must be finite".to_string());
    }
    state.send_runtime_control(NativeVideoCommand::SetSpeed {
        speed: speed.clamp(0.5, 2.0),
    })
}

#[tauri::command]
pub fn native_video_set_viewport(
    app: tauri::AppHandle,
    request: NativeViewportRequest,
) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        use super::platform::linux::{self, ViewportBounds};
        let (sender, receiver) = mpsc::sync_channel(1);
        app.run_on_main_thread(move || {
            let result = linux::set_bounds(ViewportBounds {
                x: request.x,
                y: request.y,
                width: request.width,
                height: request.height,
            })
            .and_then(|_| linux::set_visible(request.visible));
            let _ = sender.send(result);
        })
        .map_err(|error| format!("failed to schedule native viewport update: {error}"))?;
        return receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| format!("native viewport update timed out: {error}"))?;
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (app, request);
        Err("native viewport host is not implemented on this platform".to_string())
    }
}

#[tauri::command]
pub fn native_video_release_viewport(
    app: tauri::AppHandle,
    state: State<'_, NativeVideoState>,
    generation: u64,
) -> Result<bool, String> {
    if state.current_generation() != generation {
        return Ok(false);
    }
    #[cfg(target_os = "linux")]
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        app.run_on_main_thread(move || {
            let _ = sender.send(super::platform::linux::set_visible(false));
        })
        .map_err(|error| format!("failed to schedule native viewport release: {error}"))?;
        receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| format!("native viewport release timed out: {error}"))??;
        return Ok(true);
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = app;
        Ok(false)
    }
}

#[tauri::command]
pub fn native_video_set_display_mode(
    app: tauri::AppHandle,
    mode: DisplayMode,
    state: State<'_, NativeVideoState>,
) -> Result<(), String> {
    state.set_display_mode(mode);
    #[cfg(target_os = "linux")]
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        app.run_on_main_thread(move || {
            let _ = sender.send(super::platform::linux::set_display_mode(mode));
        })
        .map_err(|error| format!("failed to schedule native display-mode update: {error}"))?;
        return receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| format!("native display-mode update timed out: {error}"))?;
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (app, mode);
        Err("native display modes are not implemented on this platform".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_state_opens_video_and_rejects_stale_updates() {
        let state = NativeVideoState::new(RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        });
        let opened = state.open_video(8, 1.25);
        assert_eq!(opened.generation, 1);
        assert_eq!(opened.presentation, PresentationTarget::PreparingNative);
        assert_eq!(opened.position, 1.25);
        assert_eq!(opened.item_id, Some(8));
        assert!(!state.update_position(opened.generation + 1, 9.0));
        assert!(state.update_position(opened.generation, 4.5));
    }

    #[test]
    fn showing_an_image_preserves_the_new_canonical_generation() {
        let state = NativeVideoState::new(RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        });
        let video = state.open_video(8, 0.0);
        let image = state.show_image(9);

        assert_eq!(image.generation, video.generation + 1);
        assert_eq!(image.presentation, PresentationTarget::ReactImage);
        assert_eq!(image.item_id, Some(9));
        assert!(state.first_frame(video.generation).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn stale_runtime_exit_cannot_fail_a_newer_selection() {
        let state = NativeVideoState::new(RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        });
        let video = state.open_video(8, 0.0);
        state.set_runtime_owner(4, video.generation);
        let image = state.show_image(9);

        assert!(state.fail_runtime_owner(4).is_none());
        assert_eq!(state.current_generation(), image.generation);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn first_frame_latency_is_generation_scoped() {
        let state = NativeVideoState::new(RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        });
        *state.opened_at.lock().unwrap() = Some((
            3,
            std::time::Instant::now() - std::time::Duration::from_millis(12),
        ));
        state.record_first_frame_latency(2);
        assert!(state.first_frame_latency_ms().is_none());
        state.record_first_frame_latency(3);
        assert!(state.first_frame_latency_ms().unwrap() >= 12.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn recovery_is_attempted_only_once_per_generation() {
        let state = NativeVideoState::new(RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        });
        assert!(state.claim_recovery_attempt(7));
        assert!(!state.claim_recovery_attempt(7));
        assert!(state.claim_recovery_attempt(8));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn owning_runtime_failure_falls_back_its_native_generation() {
        let state = NativeVideoState::new(RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        });
        let video = state.open_video(8, 2.5);
        state.set_runtime_owner(5, video.generation);

        let failed = state.fail_runtime_owner(5).unwrap();
        assert_eq!(failed.generation, video.generation);
        assert_eq!(failed.presentation, PresentationTarget::WebFallback);
        assert_eq!(failed.position, 2.5);
    }
}
