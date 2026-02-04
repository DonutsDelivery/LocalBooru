//! Interpolation configuration and presets
//!
//! Defines configuration structures and quality presets for frame interpolation.

use serde::{Deserialize, Serialize};

/// Available interpolation backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationBackend {
    /// SVP via mpv pipe (best quality, requires SVP installation)
    Svp,
    /// RIFE neural network via ncnn-vulkan (good quality, GPU accelerated)
    RifeNcnn,
    /// FFmpeg minterpolate filter (always available, CPU intensive)
    Minterpolate,
    /// Disabled - pass through without interpolation
    None,
}

impl Default for InterpolationBackend {
    fn default() -> Self {
        Self::None
    }
}

impl InterpolationBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Svp => "svp",
            Self::RifeNcnn => "rife_ncnn",
            Self::Minterpolate => "minterpolate",
            Self::None => "none",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "svp" => Some(Self::Svp),
            "rife_ncnn" | "rife-ncnn" | "rife" => Some(Self::RifeNcnn),
            "minterpolate" | "ffmpeg" => Some(Self::Minterpolate),
            "none" | "disabled" | "off" => Some(Self::None),
            _ => None,
        }
    }
}

/// Quality presets for interpolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationPreset {
    /// Fastest processing, lower quality
    Fast,
    /// Balanced speed and quality (recommended)
    Balanced,
    /// Higher quality, slower
    Quality,
    /// Maximum quality, not for real-time
    Max,
    /// Optimized for anime/cartoons
    Animation,
    /// Optimized for live action film
    Film,
}

impl Default for InterpolationPreset {
    fn default() -> Self {
        Self::Balanced
    }
}

impl InterpolationPreset {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
            Self::Max => "max",
            Self::Animation => "animation",
            Self::Film => "film",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Fast => "Fastest processing, lower quality. Good for real-time on weaker hardware.",
            Self::Balanced => "Good balance of speed and quality. Recommended for most videos.",
            Self::Quality => "Higher quality motion estimation. Slower but smoother results.",
            Self::Max => "Maximum quality settings. Best for pre-rendering, not real-time.",
            Self::Animation => "Optimized for anime and cartoons with flat colors and sharp edges.",
            Self::Film => "Optimized for live action movies with natural motion blur.",
        }
    }

    /// Get SVP-compatible super params
    pub fn svp_super_params(&self) -> &'static str {
        match self {
            Self::Fast => "{gpu:1,pel:1,scale:{up:0,down:4}}",
            Self::Balanced => "{gpu:1,pel:2,scale:{up:0,down:2}}",
            Self::Quality => "{gpu:1,pel:2,scale:{up:0,down:2}}",
            Self::Max => "{gpu:1,pel:4,scale:{up:2,down:2}}",
            Self::Animation => "{gpu:1,pel:2,scale:{up:0,down:2}}",
            Self::Film => "{gpu:1,pel:2,scale:{up:0,down:2}}",
        }
    }

    /// Get SVP-compatible analyse params
    pub fn svp_analyse_params(&self) -> &'static str {
        match self {
            Self::Fast => "{gpu:1,block:{w:32,h:32,overlap:0},main:{search:{coarse:{type:2,distance:-6,bad:{sad:2000,range:24}},type:2,distance:6}},refine:[{thsad:200}]}",
            Self::Balanced => "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
            Self::Quality => "{gpu:1,block:{w:8,h:8,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
            Self::Max => "{gpu:1,block:{w:8,h:8,overlap:3},main:{search:{coarse:{type:4,distance:-12,bad:{sad:1000,range:24}},type:4,distance:12}},refine:[{thsad:200},{thsad:100},{thsad:50}]}",
            Self::Animation => "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:1500,range:24}},type:2,distance:10},penalty:{lambda:3.0}},refine:[{thsad:150}]}",
            Self::Film => "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        }
    }

    /// Get NVOF block size for this preset
    pub fn nvof_block_size(&self) -> u32 {
        match self {
            Self::Fast => 32,
            Self::Balanced => 16,
            Self::Quality => 8,
            Self::Max => 4,
            Self::Animation => 16,
            Self::Film => 16,
        }
    }

    /// Get RIFE model recommendation for this preset
    pub fn rife_model(&self) -> &'static str {
        match self {
            Self::Fast => "rife-v4",
            Self::Balanced => "rife-v4.6",
            Self::Quality => "rife-v4.6",
            Self::Max => "rife-v4.6",
            Self::Animation => "rife-v4.6", // v4.6 handles anime well
            Self::Film => "rife-v4.6",
        }
    }
}

/// Main interpolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolationConfig {
    /// Whether interpolation is enabled
    pub enabled: bool,

    /// Which backend to use
    pub backend: InterpolationBackend,

    /// Quality preset
    pub preset: InterpolationPreset,

    /// Target output FPS (typically 60, 120, or 144)
    pub target_fps: u32,

    /// Use NVIDIA Optical Flow (NVOF) when available (SVP only)
    pub use_nvof: bool,

    /// GPU device ID (0 = first GPU, -1 = CPU)
    pub gpu_id: i32,

    /// SVP algorithm/shader (13=uniform, 23=adaptive)
    pub svp_algorithm: u32,

    /// Artifact masking area (0-200, higher = fewer artifacts but less smoothing)
    pub artifact_masking: u32,

    /// Scene change detection sensitivity (0-100, higher = more sensitive)
    pub scene_sensitivity: u32,

    /// Custom SVP super params (overrides preset)
    pub custom_super: Option<String>,

    /// Custom SVP analyse params (overrides preset)
    pub custom_analyse: Option<String>,

    /// Custom SVP smooth params (overrides preset)
    pub custom_smooth: Option<String>,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: InterpolationBackend::None,
            preset: InterpolationPreset::Balanced,
            target_fps: 60,
            use_nvof: true,
            gpu_id: 0,
            svp_algorithm: 23,      // Adaptive (best quality)
            artifact_masking: 100,  // Balanced
            scene_sensitivity: 50,  // Medium
            custom_super: None,
            custom_analyse: None,
            custom_smooth: None,
        }
    }
}

impl InterpolationConfig {
    /// Create a config with a specific backend
    pub fn with_backend(backend: InterpolationBackend) -> Self {
        Self {
            enabled: backend != InterpolationBackend::None,
            backend,
            ..Default::default()
        }
    }

    /// Get effective SVP smooth params
    pub fn svp_smooth_params(&self) -> String {
        if let Some(ref custom) = self.custom_smooth {
            return custom.clone();
        }

        format!(
            "{{rate:{{num:{},den:1,abs:true}},gpuid:{},algo:{},mask:{{area:{}}},scene:{{}}}}",
            self.target_fps, self.gpu_id, self.svp_algorithm, self.artifact_masking
        )
    }

    /// Get effective SVP super params
    pub fn svp_super_params(&self) -> String {
        self.custom_super.clone().unwrap_or_else(|| self.preset.svp_super_params().to_string())
    }

    /// Get effective SVP analyse params
    pub fn svp_analyse_params(&self) -> String {
        self.custom_analyse.clone().unwrap_or_else(|| self.preset.svp_analyse_params().to_string())
    }

    /// Calculate interpolation multiplier (e.g., 24fps -> 60fps = 2.5x)
    pub fn multiplier_for_source(&self, source_fps: f64) -> f64 {
        self.target_fps as f64 / source_fps
    }

    /// Check if config is valid for real-time playback
    pub fn is_realtime_viable(&self, source_resolution: (u32, u32)) -> bool {
        let (width, height) = source_resolution;
        let pixels = width * height;

        // Rough heuristics based on typical GPU capabilities
        match self.backend {
            InterpolationBackend::Svp => {
                // SVP with NVOF can handle 4K in real-time on modern GPUs
                if self.use_nvof {
                    pixels <= 3840 * 2160 // Up to 4K
                } else {
                    pixels <= 1920 * 1080 // 1080p without NVOF
                }
            }
            InterpolationBackend::RifeNcnn => {
                // RIFE-NCNN is fast but still limited
                pixels <= 2560 * 1440 // Up to 1440p
            }
            InterpolationBackend::Minterpolate => {
                // CPU-based, very slow
                pixels <= 1280 * 720 // 720p max for real-time
            }
            InterpolationBackend::None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_from_str() {
        assert_eq!(InterpolationBackend::from_str("svp"), Some(InterpolationBackend::Svp));
        assert_eq!(InterpolationBackend::from_str("rife-ncnn"), Some(InterpolationBackend::RifeNcnn));
        assert_eq!(InterpolationBackend::from_str("none"), Some(InterpolationBackend::None));
    }

    #[test]
    fn test_config_defaults() {
        let config = InterpolationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.target_fps, 60);
        assert_eq!(config.backend, InterpolationBackend::None);
    }

    #[test]
    fn test_smooth_params_generation() {
        let config = InterpolationConfig {
            target_fps: 120,
            gpu_id: 0,
            svp_algorithm: 23,
            artifact_masking: 100,
            ..Default::default()
        };
        let params = config.svp_smooth_params();
        assert!(params.contains("num:120"));
        assert!(params.contains("algo:23"));
    }
}
