use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1000;
const PROTOCOL_MAJOR_DIVISOR: u32 = 1000;

fn default_volume() -> f64 {
    1.0
}

fn default_speed() -> f64 {
    1.0
}

fn default_interpolation_engine() -> String {
    "off".to_string()
}

fn default_target_fps() -> u32 {
    60
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NativeVideoCommand {
    Hello {
        protocol_version: u32,
    },
    OpenMedia {
        generation: u64,
        item_id: i64,
        path: String,
        resume_position: f64,
        autoplay: bool,
    },
    CloseMedia {
        generation: u64,
    },
    SetViewport {
        width: u32,
        height: u32,
        scale_factor: f64,
    },
    SetVisible {
        visible: bool,
    },
    SetPaused {
        paused: bool,
    },
    Seek {
        position: f64,
    },
    SetVolume {
        volume: f64,
    },
    SetMuted {
        muted: bool,
    },
    SetSpeed {
        speed: f64,
    },
    SelectAudioTrack {
        track_id: String,
    },
    SelectSubtitleTrack {
        track_id: Option<String>,
    },
    SetSubtitleDelay {
        seconds: f64,
    },
    RegisterSubtitleTrack {
        generation: u64,
        id: String,
        path: String,
        label: String,
        language: Option<String>,
        select: bool,
    },
    SetInterpolation {
        engine: String,
        preset: Option<String>,
        target_fps: u32,
    },
    PointerMove {
        x: f64,
        y: f64,
    },
    PointerDown {
        x: f64,
        y: f64,
        button: u8,
    },
    PointerUp {
        x: f64,
        y: f64,
        button: u8,
    },
    Scroll {
        delta_x: f64,
        delta_y: f64,
    },
    Key {
        key: String,
        pressed: bool,
    },
    SetHudVisible {
        visible: bool,
    },
    SetFullscreen {
        fullscreen: bool,
    },
}

impl NativeVideoCommand {
    pub fn generation(&self) -> Option<u64> {
        match self {
            Self::OpenMedia { generation, .. }
            | Self::CloseMedia { generation }
            | Self::RegisterSubtitleTrack { generation, .. } => Some(*generation),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NativeVideoEvent {
    Ready {
        protocol_version: u32,
    },
    CapabilitiesChanged {
        zero_cpu_copy: bool,
        copy_mode: String,
        #[serde(default)]
        interpolation_engines: Vec<String>,
        #[serde(default)]
        svp_status: Option<String>,
    },
    MediaOpened {
        generation: u64,
        duration: f64,
    },
    FirstFrameReady {
        generation: u64,
    },
    PlaybackState {
        generation: u64,
        position: f64,
        duration: f64,
        paused: bool,
        #[serde(default = "default_volume")]
        volume: f64,
        #[serde(default)]
        muted: bool,
        #[serde(default = "default_speed")]
        speed: f64,
        #[serde(default)]
        selected_audio_track: Option<String>,
        #[serde(default)]
        selected_subtitle_track: Option<String>,
        #[serde(default)]
        subtitle_delay: f64,
        #[serde(default = "default_interpolation_engine")]
        interpolation_engine: String,
        #[serde(default)]
        interpolation_preset: Option<String>,
        #[serde(default = "default_target_fps")]
        interpolation_target_fps: u32,
    },
    PlaybackEnded {
        generation: u64,
        position: f64,
    },
    TrackList {
        generation: u64,
        audio: Vec<TrackInfo>,
        subtitles: Vec<TrackInfo>,
    },
    SubtitleTrackAdded {
        generation: u64,
        track: TrackInfo,
    },
    SubtitleText {
        generation: u64,
        lines: Vec<String>,
    },
    NavigatePrevious {
        generation: u64,
    },
    NavigateNext {
        generation: u64,
    },
    CloseRequested {
        generation: u64,
    },
    HudVisibilityChanged {
        generation: u64,
        visible: bool,
    },
    Diagnostics {
        generation: u64,
        produced_fps: f64,
        presented_fps: f64,
        dropped_frames: u64,
        zero_cpu_copy: bool,
        fallback_reason: Option<String>,
        #[serde(default)]
        decoder: Option<String>,
        #[serde(default)]
        hardware_device: Option<String>,
        #[serde(default)]
        source_fps: Option<f64>,
        #[serde(default)]
        queue_depth: Option<u64>,
        #[serde(default)]
        queue_latency_ms: Option<f64>,
        #[serde(default)]
        accepted_frames: Option<u64>,
        #[serde(default)]
        draw_completed_frames: Option<u64>,
        #[serde(default)]
        av_drift_ms: Option<f64>,
        #[serde(default)]
        interpolation_engine: Option<String>,
        #[serde(default)]
        surface_mode: Option<String>,
        #[serde(default)]
        width: Option<u32>,
        #[serde(default)]
        height: Option<u32>,
        #[serde(default)]
        first_frame_latency_ms: Option<f64>,
        #[serde(default)]
        seek_latency_ms: Option<f64>,
    },
    RecoverableError {
        generation: Option<u64>,
        message: String,
    },
    FatalError {
        generation: Option<u64>,
        message: String,
    },
    GpuPathChanged {
        generation: Option<u64>,
        zero_cpu_copy: bool,
        copy_mode: String,
        fallback_reason: Option<String>,
    },
}

impl NativeVideoEvent {
    pub fn generation(&self) -> Option<u64> {
        match self {
            Self::MediaOpened { generation, .. }
            | Self::FirstFrameReady { generation }
            | Self::PlaybackState { generation, .. }
            | Self::PlaybackEnded { generation, .. }
            | Self::TrackList { generation, .. }
            | Self::SubtitleTrackAdded { generation, .. }
            | Self::SubtitleText { generation, .. }
            | Self::NavigatePrevious { generation }
            | Self::NavigateNext { generation }
            | Self::CloseRequested { generation }
            | Self::HudVisibilityChanged { generation, .. }
            | Self::Diagnostics { generation, .. } => Some(*generation),
            Self::RecoverableError { generation, .. }
            | Self::FatalError { generation, .. }
            | Self::GpuPathChanged { generation, .. } => *generation,
            Self::Ready { .. } | Self::CapabilitiesChanged { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackInfo {
    pub id: String,
    pub language: Option<String>,
    pub label: String,
    pub is_default: bool,
    pub is_forced: bool,
    #[serde(default)]
    pub source_type: TrackSourceType,
    #[serde(default)]
    pub cue_format: Option<String>,
    #[serde(default)]
    pub cue_path: Option<String>,
    #[serde(default)]
    pub generation_status: TrackGenerationStatus,
    #[serde(default)]
    pub delay_seconds: f64,
    #[serde(default)]
    pub style: Option<SubtitleStyle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrackSourceType {
    #[default]
    Embedded,
    Sidecar,
    Whisper,
    Cast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrackGenerationStatus {
    #[default]
    Ready,
    Pending,
    Generating,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SubtitleStyle {
    pub font_family: Option<String>,
    pub font_size: Option<f64>,
    pub text_color: Option<String>,
    pub outline_color: Option<String>,
    pub background_color: Option<String>,
    pub safe_area_percent: Option<f64>,
}

pub fn validate_protocol_version(peer_version: u32) -> Result<(), String> {
    let local_major = PROTOCOL_VERSION / PROTOCOL_MAJOR_DIVISOR;
    let peer_major = peer_version / PROTOCOL_MAJOR_DIVISOR;
    if local_major == peer_major {
        Ok(())
    } else {
        Err(format!(
            "incompatible native-video protocol: local {}, peer {}",
            PROTOCOL_VERSION, peer_version
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_fixture_round_trips_with_generation_and_unknown_fields() {
        let fixture = r#"{"type":"open_media","generation":42,"item_id":7,"path":"/tmp/video.mp4","resume_position":3.5,"autoplay":true,"future":"ignored"}"#;
        let command: NativeVideoCommand = serde_json::from_str(fixture).unwrap();
        assert_eq!(command.generation(), Some(42));
        assert!(matches!(
            command,
            NativeVideoCommand::OpenMedia { item_id: 7, .. }
        ));
        let encoded = serde_json::to_value(command).unwrap();
        assert_eq!(encoded["type"], "open_media");
        assert_eq!(encoded["generation"], 42);
    }

    #[test]
    fn generated_subtitle_registration_is_generation_scoped() {
        let command = NativeVideoCommand::RegisterSubtitleTrack {
            generation: 44,
            id: "whisper:en".to_string(),
            path: "/library/video.whisper.en.vtt".to_string(),
            label: "Whisper en".to_string(),
            language: Some("en".to_string()),
            select: true,
        };
        assert_eq!(command.generation(), Some(44));
        let encoded = serde_json::to_value(command).unwrap();
        assert_eq!(encoded["type"], "register_subtitle_track");
        assert_eq!(encoded["select"], true);
    }

    #[test]
    fn playback_ended_event_preserves_generation_and_position() {
        let fixture = r#"{"type":"playback_ended","generation":12,"position":8.25}"#;
        let event: NativeVideoEvent = serde_json::from_str(fixture).unwrap();
        assert_eq!(event.generation(), Some(12));
        assert!(matches!(
            event,
            NativeVideoEvent::PlaybackEnded {
                generation: 12,
                position
            } if position == 8.25
        ));
    }

    #[test]
    fn event_fixture_round_trips_diagnostics_and_zero_copy_state() {
        let event = NativeVideoEvent::Diagnostics {
            generation: 9,
            produced_fps: 63.0,
            presented_fps: 59.0,
            dropped_frames: 0,
            zero_cpu_copy: true,
            fallback_reason: None,
            decoder: Some("ffmpeg_hw".to_string()),
            hardware_device: Some("nvdec".to_string()),
            source_fps: Some(60.0),
            queue_depth: Some(1),
            queue_latency_ms: Some(2.5),
            accepted_frames: Some(120),
            draw_completed_frames: Some(119),
            av_drift_ms: None,
            interpolation_engine: Some("off".to_string()),
            surface_mode: Some("dma_buf_external_oes".to_string()),
            width: Some(1920),
            height: Some(1080),
            first_frame_latency_ms: None,
            seek_latency_ms: Some(14.0),
        };
        let encoded = serde_json::to_string(&event).unwrap();
        let decoded: NativeVideoEvent = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.generation(), Some(9));
        assert_eq!(decoded, event);
    }

    #[test]
    fn capabilities_preserve_external_svp_availability() {
        let fixture = r#"{"type":"capabilities_changed","zero_cpu_copy":true,"copy_mode":"dma_buf_external_oes","interpolation_engines":["off","svp"],"svp_status":"available_external"}"#;
        let event: NativeVideoEvent = serde_json::from_str(fixture).unwrap();
        assert!(matches!(
            event,
            NativeVideoEvent::CapabilitiesChanged {
                interpolation_engines,
                svp_status: Some(status),
                ..
            } if interpolation_engines == ["off", "svp"] && status == "available_external"
        ));
    }

    #[test]
    fn track_model_defaults_embedded_tracks_and_preserves_whisper_metadata() {
        let embedded: TrackInfo = serde_json::from_str(
            r#"{"id":"2","language":"en","label":"English","is_default":true,"is_forced":false}"#,
        )
        .unwrap();
        assert_eq!(embedded.source_type, TrackSourceType::Embedded);
        assert_eq!(embedded.generation_status, TrackGenerationStatus::Ready);
        assert_eq!(embedded.delay_seconds, 0.0);

        let whisper: TrackInfo = serde_json::from_str(
            r#"{"id":"whisper-en","language":"en","label":"Generated English","is_default":false,"is_forced":false,"source_type":"whisper","cue_format":"vtt","cue_path":"/library/.localbooru/subtitles/item.vtt","generation_status":"generating","delay_seconds":-0.25,"style":{"font_family":"Inter","safe_area_percent":8.0}}"#,
        )
        .unwrap();
        assert_eq!(whisper.source_type, TrackSourceType::Whisper);
        assert_eq!(whisper.generation_status, TrackGenerationStatus::Generating);
        assert_eq!(whisper.cue_format.as_deref(), Some("vtt"));
        assert_eq!(whisper.delay_seconds, -0.25);
        assert_eq!(
            whisper.style.and_then(|style| style.safe_area_percent),
            Some(8.0)
        );
    }

    #[test]
    fn protocol_version_rejects_incompatible_major_versions() {
        assert!(validate_protocol_version(PROTOCOL_VERSION).is_ok());
        assert!(validate_protocol_version(PROTOCOL_VERSION + 1000).is_err());
    }
}
