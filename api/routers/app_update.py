"""
App update endpoint — serves APK updates to mobile clients.
The APK is looked for in the project's updates directory.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
UPDATES_DIR = PROJECT_ROOT / "updates"
APK_FILENAMES = ("DonutMediaCenter.apk", "LocalBooru.apk")


def _apk_path():
    return next((UPDATES_DIR / name for name in APK_FILENAMES if (UPDATES_DIR / name).is_file()), None)


def _get_version() -> str:
    """Read project version from package.json."""
    pkg = PROJECT_ROOT / "package.json"
    if pkg.exists():
        data = json.loads(pkg.read_text())
        return data.get("version", "0.0.0")
    return "0.0.0"


@router.get("/check")
async def check_update(platform: str = "android", current_version: str = "0.0.0"):
    """Check if an update is available for the given platform."""
    server_version = _get_version()
    apk_path = _apk_path()

    return {
        "version": server_version,
        "apk_available": platform == "android" and apk_path is not None,
    }


@router.get("/download")
async def download_update():
    """Download the APK file."""
    apk_path = _apk_path()
    if apk_path is None:
        raise HTTPException(status_code=404, detail="No APK available")

    return FileResponse(
        path=str(apk_path),
        filename="DonutMediaCenter.apk",
        media_type="application/vnd.android.package-archive",
    )
