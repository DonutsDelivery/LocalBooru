"""
Shared Pydantic models and settings utilities for settings sub-routers.
"""
from pydantic import BaseModel
from typing import Optional
import json
import os
from pathlib import Path

from ...database import get_data_dir


# =============================================================================
# Settings File Management
# =============================================================================

def get_settings_file() -> Path:
    """Get the path to the settings JSON file."""
    return get_data_dir() / 'settings.json'


def load_settings() -> dict:
    """Load settings from JSON file"""
    settings_file = get_settings_file()
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_settings(settings: dict):
    """Save settings to JSON file"""
    settings_file = get_settings_file()
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)


def get_setting(key: str, default: str = None) -> Optional[str]:
    """Get a setting value"""
    settings = load_settings()
    return settings.get(key, default)


def set_setting(key: str, value: str):
    """Set a setting value"""
    settings = load_settings()
    settings[key] = value
    save_settings(settings)


# =============================================================================
# Settings Keys
# =============================================================================

AGE_DETECTION_ENABLED = "age_detection_enabled"
AGE_DETECTION_INSTALLED = "age_detection_installed"
AGE_DETECTION_INSTALLING = "age_detection_installing"
AGE_DETECTION_INSTALL_PROGRESS = "age_detection_install_progress"


# =============================================================================
# Network Settings
# =============================================================================

def get_default_local_port():
    """Get default local port based on install mode"""
    if os.environ.get('LOCALBOORU_PORTABLE_DATA'):
        return 8791  # Portable mode
    return 8790  # System install


DEFAULT_NETWORK_SETTINGS = {
    "local_network_enabled": False,
    "public_network_enabled": False,
    "local_port": get_default_local_port(),
    "public_port": 8791,
    "auth_required_level": "local_network",  # none, public, local_network, always
    "upnp_enabled": False,
    "allow_settings_local_network": False  # Allow settings/admin access from local network
}


def get_network_settings() -> dict:
    """Get network settings with defaults"""
    settings = load_settings()
    network = settings.get("network", {})
    # Merge with defaults
    return {**DEFAULT_NETWORK_SETTINGS, **network}


def save_network_settings(network_settings: dict):
    """Save network settings"""
    settings = load_settings()
    settings["network"] = network_settings
    save_settings(settings)


# =============================================================================
# Optical Flow Settings
# =============================================================================

DEFAULT_OPTICAL_FLOW_SETTINGS = {
    "enabled": False,
    "target_fps": 60,
    "use_gpu": True,
    "quality": "fast",  # svp, gpu_native, realtime, fast, balanced, quality
}


def get_optical_flow_settings() -> dict:
    """Get optical flow interpolation settings with defaults"""
    settings = load_settings()
    optical_flow = settings.get("optical_flow", {})
    # Merge with defaults
    return {**DEFAULT_OPTICAL_FLOW_SETTINGS, **optical_flow}


def save_optical_flow_settings(optical_flow_settings: dict):
    """Save optical flow interpolation settings"""
    settings = load_settings()
    settings["optical_flow"] = optical_flow_settings
    save_settings(settings)


# =============================================================================
# SVP Settings
# =============================================================================

DEFAULT_SVP_SETTINGS = {
    "enabled": False,
    "target_fps": 60,
    "preset": "balanced",  # fast, balanced, quality, max, animation, film
    # Key settings
    "use_nvof": True,           # Use NVIDIA Optical Flow
    "shader": 23,               # SVP shader/algo (1,2,11,13,21,23)
    "artifact_masking": 100,    # Artifact masking area (0-200)
    "frame_interpolation": 2,   # Frame interpolation mode (1=uniform, 2=adaptive)
    # Advanced settings (full override when set)
    "custom_super": None,
    "custom_analyse": None,
    "custom_smooth": None,
}


def get_svp_settings() -> dict:
    """Get SVP interpolation settings with defaults"""
    settings = load_settings()
    svp = settings.get("svp", {})
    # Merge with defaults
    return {**DEFAULT_SVP_SETTINGS, **svp}


def save_svp_settings(svp_settings: dict):
    """Save SVP interpolation settings"""
    settings = load_settings()
    settings["svp"] = svp_settings
    save_settings(settings)


# =============================================================================
# Quality Presets
# =============================================================================

QUALITY_PRESETS = {
    "original": {"bitrate": None, "resolution": None},
    "1440p": {"bitrate": "30M", "resolution": (2560, 1440)},
    "1080p_enhanced": {"bitrate": "20M", "resolution": (1920, 1080)},
    "1080p": {"bitrate": "12M", "resolution": (1920, 1080)},
    "720p": {"bitrate": "8M", "resolution": (1280, 720)},
    "480p": {"bitrate": "4M", "resolution": (854, 480)},
}


def parse_quality_preset(quality_preset, source_resolution=None):
    """Parse quality preset and return bitrate and resolution settings.

    Args:
        quality_preset: Preset name (e.g., '720p_3mbps') or None for original
        source_resolution: Tuple of (width, height) for source video

    Returns:
        Dict with 'bitrate' and 'resolution' keys (either can be None)
    """
    if not quality_preset or quality_preset == 'original' or quality_preset not in QUALITY_PRESETS:
        return {"bitrate": None, "resolution": None}

    preset = QUALITY_PRESETS[quality_preset]
    bitrate = preset["bitrate"]
    resolution = preset["resolution"]

    # Don't upscale - if source resolution is smaller than preset, skip
    if source_resolution and resolution:
        src_width, src_height = source_resolution
        preset_width, preset_height = resolution
        # Check if source is smaller than preset (upscaling would be needed)
        if src_width < preset_width or src_height < preset_height:
            # Return original quality for sources smaller than preset
            return {"bitrate": None, "resolution": None}

    return {"bitrate": bitrate, "resolution": resolution}


# =============================================================================
# Pydantic Models - General
# =============================================================================

class AgeDetectionToggle(BaseModel):
    enabled: bool


class ModelDownloadRequest(BaseModel):
    model_name: str


# =============================================================================
# Pydantic Models - Migration
# =============================================================================

class MigrationRequest(BaseModel):
    mode: str  # "system_to_portable" or "portable_to_system"
    directory_ids: Optional[list[int]] = None  # Selective migration: which watch directories to include


# =============================================================================
# Pydantic Models - Video/Streaming
# =============================================================================

class OpticalFlowConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    target_fps: Optional[int] = None
    use_gpu: Optional[bool] = None
    quality: Optional[str] = None  # svp, gpu_native, realtime, fast, balanced, quality


class InterpolationPlayRequest(BaseModel):
    file_path: str
    start_position: float = 0.0  # Seek position in seconds
    quality_preset: Optional[str] = None  # Quality preset name


class SVPConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    target_fps: Optional[int] = None
    preset: Optional[str] = None  # fast, balanced, quality, max, animation, film
    use_nvof: Optional[bool] = None  # Use NVIDIA Optical Flow
    shader: Optional[int] = None  # SVP shader/algo (1,2,11,13,21,23)
    artifact_masking: Optional[int] = None  # Artifact masking area (0-200)
    frame_interpolation: Optional[int] = None  # Frame interpolation mode (1=uniform, 2=adaptive)
    custom_super: Optional[str] = None
    custom_analyse: Optional[str] = None
    custom_smooth: Optional[str] = None


class SVPPlayRequest(BaseModel):
    file_path: str
    start_position: float = 0.0  # Seek position in seconds
    quality_preset: Optional[str] = None  # Quality preset name


class TranscodePlayRequest(BaseModel):
    file_path: str
    start_position: float = 0.0
    quality_preset: Optional[str] = None


# =============================================================================
# Saved Searches
# =============================================================================

def get_saved_searches() -> list:
    """Get saved searches from settings"""
    settings = load_settings()
    return settings.get("saved_searches", [])


def save_saved_searches(searches: list):
    """Save searches list to settings"""
    settings = load_settings()
    settings["saved_searches"] = searches
    save_settings(settings)


# =============================================================================
# Video Playback Settings (auto-advance, etc.)
# =============================================================================

DEFAULT_VIDEO_PLAYBACK_SETTINGS = {
    "auto_advance_enabled": False,
    "auto_advance_delay": 5,
}


def get_video_playback_settings() -> dict:
    """Get video playback settings with defaults"""
    settings = load_settings()
    playback = settings.get("video_playback", {})
    return {**DEFAULT_VIDEO_PLAYBACK_SETTINGS, **playback}


def save_video_playback_settings(playback_settings: dict):
    """Save video playback settings"""
    settings = load_settings()
    settings["video_playback"] = playback_settings
    save_settings(settings)


# =============================================================================
# Whisper Subtitle Settings
# =============================================================================

DEFAULT_WHISPER_SETTINGS = {
    "enabled": False,
    "auto_generate": False,
    "model_size": "medium",
    "language": "ja",
    "task": "translate",  # "transcribe" or "translate"
    "chunk_duration": 30,
    "beam_size": 8,
    "device": "auto",  # "auto", "cuda", "cpu"
    "compute_type": "auto",  # "auto", "float16", "int8_float16", "int8"
    "vad_filter": True,
    "suppress_nst": True,  # suppress non-speech tokens
    "cache_subtitles": True,
    "installing": False,  # transient: pip install in progress
    "install_progress": "",  # transient: install status message
    # Subtitle appearance
    "subtitle_font": "Trebuchet MS",
    "subtitle_font_size": 1.3,  # rem
    "subtitle_style": "outline",  # "outline", "background", "outline_background"
    "subtitle_color": "#ffffff",
    "subtitle_outline_color": "#000000",
    "subtitle_bg_opacity": 0.75,
}


def get_whisper_settings() -> dict:
    """Get whisper subtitle settings with defaults"""
    settings = load_settings()
    whisper = settings.get("whisper", {})
    return {**DEFAULT_WHISPER_SETTINGS, **whisper}


def save_whisper_settings(whisper_settings: dict):
    """Save whisper subtitle settings"""
    settings = load_settings()
    settings["whisper"] = whisper_settings
    save_settings(settings)


# =============================================================================
# Pydantic Models - Whisper Subtitles
# =============================================================================

class SavedSearchCreate(BaseModel):
    name: str
    filters: dict


class VideoPlaybackConfigUpdate(BaseModel):
    auto_advance_enabled: Optional[bool] = None
    auto_advance_delay: Optional[int] = None


class WhisperConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    auto_generate: Optional[bool] = None
    model_size: Optional[str] = None
    language: Optional[str] = None
    task: Optional[str] = None
    chunk_duration: Optional[int] = None
    beam_size: Optional[int] = None
    device: Optional[str] = None
    compute_type: Optional[str] = None
    vad_filter: Optional[bool] = None
    suppress_nst: Optional[bool] = None
    cache_subtitles: Optional[bool] = None
    subtitle_font: Optional[str] = None
    subtitle_font_size: Optional[float] = None
    subtitle_style: Optional[str] = None
    subtitle_color: Optional[str] = None
    subtitle_outline_color: Optional[str] = None
    subtitle_bg_opacity: Optional[float] = None


class WhisperSubtitleRequest(BaseModel):
    file_path: str
    language: Optional[str] = None  # override config language
    task: Optional[str] = None  # override config task
    start_position: float = 0.0  # start transcription from this timestamp
