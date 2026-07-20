#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopPlayerMode {
    React,
    Native,
    NativeSvp,
}

impl DesktopPlayerMode {
    pub fn from_setting(value: &str) -> Self {
        match value {
            "native" => Self::Native,
            "native_svp" => Self::NativeSvp,
            _ => Self::React,
        }
    }

    pub fn uses_native(self) -> bool {
        !matches!(self, Self::React)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeCapabilities {
    pub desktop_tauri: bool,
    pub native_renderer_available: bool,
    pub safe_mode: bool,
    pub desktop_player_mode: DesktopPlayerMode,
}

impl RuntimeCapabilities {
    fn use_native_renderer(self) -> bool {
        self.desktop_tauri
            && self.native_renderer_available
            && !self.safe_mode
            && self.desktop_player_mode.uses_native()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CanonicalPlaybackState {
    pub item_id: Option<i64>,
    pub generation: u64,
    pub position: f64,
    pub duration: f64,
    pub paused: bool,
    pub volume: f64,
    pub muted: bool,
    pub speed: f64,
    pub selected_audio_track: Option<String>,
    pub selected_subtitle_track: Option<String>,
    pub subtitle_delay: f64,
    pub svp_enabled: bool,
    pub svp_preset: Option<String>,
    pub svp_target_fps: u32,
    pub display_mode: String,
}

impl Default for CanonicalPlaybackState {
    fn default() -> Self {
        Self {
            item_id: None,
            generation: 0,
            position: 0.0,
            duration: 0.0,
            paused: false,
            volume: 1.0,
            muted: false,
            speed: 1.0,
            selected_audio_track: None,
            selected_subtitle_track: None,
            subtitle_delay: 0.0,
            svp_enabled: false,
            svp_preset: None,
            svp_target_fps: 60,
            display_mode: "fit".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PresentationState {
    Image { item_id: i64, generation: u64 },
    PreparingVideo { generation: u64 },
    VisibleVideo { generation: u64 },
    WebFallback { generation: u64, position: f64 },
    Casting { generation: u64, position: f64 },
}

#[derive(Debug)]
pub struct NativePlaybackCoordinator {
    capabilities: RuntimeCapabilities,
    generation: u64,
    item_id: Option<i64>,
    position: f64,
    playback: CanonicalPlaybackState,
    state: PresentationState,
}

impl NativePlaybackCoordinator {
    pub fn new(capabilities: RuntimeCapabilities) -> Self {
        Self {
            capabilities,
            generation: 0,
            item_id: None,
            position: 0.0,
            playback: CanonicalPlaybackState::default(),
            state: PresentationState::Image {
                item_id: 0,
                generation: 0,
            },
        }
    }

    pub fn set_desktop_player_mode(&mut self, mode: DesktopPlayerMode) {
        self.capabilities.desktop_player_mode = mode;
    }

    pub fn state(&self) -> &PresentationState {
        &self.state
    }

    pub fn position(&self) -> f64 {
        self.position
    }

    pub fn item_id(&self) -> Option<i64> {
        self.item_id
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn playback(&self) -> &CanonicalPlaybackState {
        &self.playback
    }

    pub fn open_video(&mut self, item_id: i64, resume_position: f64) -> u64 {
        self.generation = self.generation.saturating_add(1);
        self.item_id = Some(item_id);
        self.position = sanitize_position(resume_position);
        self.playback = CanonicalPlaybackState {
            item_id: Some(item_id),
            generation: self.generation,
            position: self.position,
            ..CanonicalPlaybackState::default()
        };
        self.state = if self.capabilities.use_native_renderer() {
            PresentationState::PreparingVideo {
                generation: self.generation,
            }
        } else {
            PresentationState::WebFallback {
                generation: self.generation,
                position: self.position,
            }
        };
        self.generation
    }

    pub fn show_image(&mut self, item_id: i64) {
        self.generation = self.generation.saturating_add(1);
        self.item_id = Some(item_id);
        self.position = 0.0;
        self.playback = CanonicalPlaybackState {
            item_id: Some(item_id),
            generation: self.generation,
            ..CanonicalPlaybackState::default()
        };
        self.state = PresentationState::Image {
            item_id,
            generation: self.generation,
        };
    }

    pub fn handle_first_frame(&mut self, generation: u64) -> bool {
        if self.generation != generation
            || !matches!(
                self.state,
                PresentationState::PreparingVideo { generation: current } if current == generation
            )
        {
            return false;
        }
        self.state = PresentationState::VisibleVideo { generation };
        true
    }

    pub fn update_position(&mut self, generation: u64, position: f64) -> bool {
        if generation != self.generation {
            return false;
        }
        self.position = sanitize_position(position);
        self.playback.position = self.position;
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_playback_state(
        &mut self,
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
        if generation != self.generation {
            return false;
        }
        self.position = sanitize_position(position);
        self.playback.position = self.position;
        self.playback.duration = sanitize_non_negative(duration);
        self.playback.paused = paused;
        self.playback.volume = volume.clamp(0.0, 1.0);
        self.playback.muted = muted;
        self.playback.speed = sanitize_speed(speed);
        self.playback.selected_audio_track = selected_audio_track;
        self.playback.selected_subtitle_track = selected_subtitle_track;
        self.playback.subtitle_delay = subtitle_delay.clamp(-30.0, 30.0);
        self.playback.svp_enabled = interpolation_engine == "svp";
        self.playback.svp_preset = interpolation_preset;
        self.playback.svp_target_fps = interpolation_target_fps.clamp(24, 240);
        true
    }

    pub fn set_display_mode(&mut self, display_mode: &str) {
        self.playback.display_mode = match display_mode {
            "fill" | "original" => display_mode,
            _ => "fit",
        }
        .to_string();
    }

    pub fn handle_renderer_failure(&mut self, generation: u64) -> bool {
        if generation != self.generation {
            return false;
        }
        self.state = PresentationState::WebFallback {
            generation,
            position: self.position,
        };
        true
    }

    pub fn start_cast(&mut self) -> Option<(u64, f64)> {
        let generation = match self.state {
            PresentationState::PreparingVideo { generation }
            | PresentationState::VisibleVideo { generation }
            | PresentationState::WebFallback { generation, .. } => generation,
            PresentationState::Image { .. } | PresentationState::Casting { .. } => return None,
        };
        self.state = PresentationState::Casting {
            generation,
            position: self.position,
        };
        Some((generation, self.position))
    }
}

fn sanitize_position(position: f64) -> f64 {
    if position.is_finite() && position > 0.0 {
        position
    } else {
        0.0
    }
}

fn sanitize_non_negative(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

fn sanitize_speed(speed: f64) -> f64 {
    if speed.is_finite() {
        speed.clamp(0.25, 4.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desktop() -> RuntimeCapabilities {
        RuntimeCapabilities {
            desktop_tauri: true,
            native_renderer_available: true,
            safe_mode: false,
            desktop_player_mode: DesktopPlayerMode::Native,
        }
    }

    #[test]
    fn image_to_video_waits_for_matching_first_frame() {
        let mut coordinator = NativePlaybackCoordinator::new(desktop());
        let generation = coordinator.open_video(10, 2.5);
        assert_eq!(
            coordinator.state(),
            &PresentationState::PreparingVideo { generation }
        );
        assert!(!coordinator.handle_first_frame(generation + 1));
        assert_eq!(
            coordinator.state(),
            &PresentationState::PreparingVideo { generation }
        );
        assert!(coordinator.handle_first_frame(generation));
        assert_eq!(
            coordinator.state(),
            &PresentationState::VisibleVideo { generation }
        );
    }

    #[test]
    fn video_to_image_hides_native_view_and_ignores_late_frame() {
        let mut coordinator = NativePlaybackCoordinator::new(desktop());
        let generation = coordinator.open_video(11, 0.0);
        coordinator.show_image(12);
        assert_eq!(
            coordinator.state(),
            &PresentationState::Image {
                item_id: 12,
                generation: generation + 1,
            }
        );
        assert!(!coordinator.handle_first_frame(generation));
        assert_eq!(
            coordinator.state(),
            &PresentationState::Image {
                item_id: 12,
                generation: generation + 1,
            }
        );
    }

    #[test]
    fn renderer_failure_preserves_position_and_selects_web_fallback() {
        let mut coordinator = NativePlaybackCoordinator::new(desktop());
        let generation = coordinator.open_video(11, 4.0);
        coordinator.update_position(generation, 8.25);
        assert!(coordinator.handle_renderer_failure(generation));
        assert_eq!(
            coordinator.state(),
            &PresentationState::WebFallback {
                generation,
                position: 8.25
            }
        );
    }

    #[test]
    fn cast_hides_native_view_and_preserves_position() {
        let mut coordinator = NativePlaybackCoordinator::new(desktop());
        let generation = coordinator.open_video(11, 4.0);
        coordinator.update_position(generation, 9.5);
        assert_eq!(coordinator.start_cast(), Some((generation, 9.5)));
        assert_eq!(
            coordinator.state(),
            &PresentationState::Casting {
                generation,
                position: 9.5
            }
        );
    }

    #[test]
    fn canonical_handoff_tracks_helper_state_and_rejects_stale_generations() {
        let mut coordinator = NativePlaybackCoordinator::new(desktop());
        let generation = coordinator.open_video(11, 4.0);
        assert!(coordinator.update_playback_state(
            generation,
            8.25,
            120.0,
            true,
            0.4,
            true,
            1.5,
            Some("2".to_string()),
            Some("5".to_string()),
            0.75,
            "svp",
            Some("balanced".to_string()),
            60,
        ));
        coordinator.set_display_mode("fill");
        assert_eq!(
            coordinator.playback(),
            &CanonicalPlaybackState {
                item_id: Some(11),
                generation,
                position: 8.25,
                duration: 120.0,
                paused: true,
                volume: 0.4,
                muted: true,
                speed: 1.5,
                selected_audio_track: Some("2".to_string()),
                selected_subtitle_track: Some("5".to_string()),
                subtitle_delay: 0.75,
                svp_enabled: true,
                svp_preset: Some("balanced".to_string()),
                svp_target_fps: 60,
                display_mode: "fill".to_string(),
            }
        );
        assert!(!coordinator.update_playback_state(
            generation + 1,
            99.0,
            120.0,
            false,
            1.0,
            false,
            1.0,
            None,
            None,
            0.0,
            "off",
            None,
            60,
        ));
        assert_eq!(coordinator.playback().position, 8.25);
    }

    #[test]
    fn browser_or_mobile_never_selects_native_presentation() {
        for capabilities in [
            RuntimeCapabilities {
                desktop_tauri: false,
                native_renderer_available: true,
                safe_mode: false,
                desktop_player_mode: DesktopPlayerMode::Native,
            },
            RuntimeCapabilities {
                desktop_tauri: true,
                native_renderer_available: false,
                safe_mode: false,
                desktop_player_mode: DesktopPlayerMode::Native,
            },
            RuntimeCapabilities {
                desktop_tauri: true,
                native_renderer_available: true,
                safe_mode: true,
                desktop_player_mode: DesktopPlayerMode::Native,
            },
            RuntimeCapabilities {
                desktop_tauri: true,
                native_renderer_available: true,
                safe_mode: false,
                desktop_player_mode: DesktopPlayerMode::React,
            },
        ] {
            let mut coordinator = NativePlaybackCoordinator::new(capabilities);
            let generation = coordinator.open_video(1, 0.0);
            assert_eq!(
                coordinator.state(),
                &PresentationState::WebFallback {
                    generation,
                    position: 0.0
                }
            );
        }
    }
}
