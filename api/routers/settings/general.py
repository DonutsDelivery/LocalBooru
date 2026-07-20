"""
General app settings endpoints - main settings retrieval, utility endpoints.
"""
from fastapi import APIRouter

from .models import (
    get_setting,
    set_setting,
    AGE_DETECTION_ENABLED,
    AGE_DETECTION_INSTALL_PROGRESS,
)
from .tagging import check_age_detection_deps, are_required_deps_installed

router = APIRouter()


@router.get("/")
async def get_all_settings():
    """Get all app settings"""
    deps = check_age_detection_deps()
    installed = are_required_deps_installed()
    progress = get_setting(AGE_DETECTION_INSTALL_PROGRESS, "")

    # Clear stale "failed" progress message if required deps are actually installed
    if installed and "failed" in progress.lower():
        if deps.get("insightface", False):
            progress = "Installation complete!"
        else:
            progress = "Installation complete (using OpenCV fallback)"
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, progress)

    return {
        "age_detection": {
            "enabled": get_setting(AGE_DETECTION_ENABLED, "false") == "true",
            "installed": installed,
            "installing": get_setting("age_detection_installing", "false") == "true",
            "install_progress": progress,
            "dependencies": deps
        }
    }


@router.get("/util/dimensions")
async def get_file_dimensions(file_path: str):
    """
    Get dimensions for an image or video file.

    Used to fetch dimensions on-the-fly without storing in DB.
    """
    from ...services.importer import get_image_dimensions

    try:
        dims = get_image_dimensions(file_path)
        if dims:
            return {
                "success": True,
                "width": dims[0],
                "height": dims[1]
            }
        else:
            return {
                "success": False,
                "error": "Could not detect dimensions"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
