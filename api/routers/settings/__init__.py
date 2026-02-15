"""
Settings router - aggregates all settings sub-routers

This module re-exports the combined router and key functions/constants
that are used by other parts of the application (e.g., main.py lifespan).
"""
from fastapi import APIRouter

# Import sub-routers
from .general import router as general_router
from .video import router as video_router
from .tagging import router as tagging_router
from .import_export import router as import_export_router
from .whisper import router as whisper_router
from .cast import router as cast_router
from .saved_searches import router as saved_searches_router

# Import functions and constants that main.py and other services need
from .tagging import (
    ensure_packages_in_path,
    get_packages_dir,
    patch_mivolo_for_timm_compat,
    are_required_deps_installed,
)

# Import settings utilities from models (used by main.py and elsewhere)
from .models import (
    load_settings,
    save_settings,
    get_setting,
    set_setting,
    get_settings_file,
    get_all_defaults,
    ensure_defaults_written,
    get_network_settings,
    save_network_settings,
    get_optical_flow_settings,
    save_optical_flow_settings,
    get_svp_settings,
    save_svp_settings,
    get_video_playback_settings,
    save_video_playback_settings,
    DEFAULT_VIDEO_PLAYBACK_SETTINGS,
    get_saved_searches,
    save_saved_searches,
    get_whisper_settings,
    save_whisper_settings,
    get_cast_settings,
    save_cast_settings,
    DEFAULT_CAST_SETTINGS,
    get_default_local_port,
    DEFAULT_NETWORK_SETTINGS,
    DEFAULT_OPTICAL_FLOW_SETTINGS,
    DEFAULT_SVP_SETTINGS,
    DEFAULT_WHISPER_SETTINGS,
    QUALITY_PRESETS,
    parse_quality_preset,
    AGE_DETECTION_ENABLED,
    AGE_DETECTION_INSTALLED,
    AGE_DETECTION_INSTALLING,
    AGE_DETECTION_INSTALL_PROGRESS,
)

# Create the combined router
router = APIRouter()

# Import the root endpoint handler for explicit route registration
from .general import get_all_settings

# Add root route explicitly to handle both with and without trailing slash
router.add_api_route("", get_all_settings, methods=["GET"])

# Include all sub-routers (no prefix - paths are already correct)
router.include_router(general_router)
router.include_router(video_router)
router.include_router(tagging_router)
router.include_router(import_export_router)
router.include_router(whisper_router)
router.include_router(cast_router)
router.include_router(saved_searches_router)
