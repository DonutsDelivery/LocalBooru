"""
Chromecast & DLNA cast settings endpoints.
"""
from fastapi import APIRouter
import logging

from .models import (
    get_cast_settings,
    save_cast_settings,
    CastConfigUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_cast_deps() -> dict:
    """Check availability of all cast dependencies."""
    status = {
        "pychromecast_installed": False,
        "upnp_installed": False,
        "aiohttp_installed": False,
    }
    try:
        import pychromecast
        status["pychromecast_installed"] = True
    except Exception:
        pass
    try:
        import async_upnp_client
        status["upnp_installed"] = True
    except Exception:
        pass
    try:
        import aiohttp
        status["aiohttp_installed"] = True
    except Exception:
        pass
    return status


@router.get("/cast")
async def get_cast_config():
    """Get cast configuration and dependency availability."""
    config = get_cast_settings()
    return {
        **config,
        "status": _check_cast_deps(),
    }


@router.post("/cast")
async def update_cast_config(config: CastConfigUpdate):
    """Update cast configuration (partial update)."""
    current = get_cast_settings()

    if config.enabled is not None:
        current["enabled"] = config.enabled
    if config.cast_media_port is not None:
        current["cast_media_port"] = max(1024, min(65535, config.cast_media_port))

    save_cast_settings(current)
    return {"success": True, **current}


# =============================================================================
# Installation Endpoint
# =============================================================================

@router.post("/cast/install")
async def install_cast_deps():
    """Install cast dependencies (pychromecast, async-upnp-client, aiohttp) via pip."""
    import sys
    import threading

    # Check if already installed
    status = _check_cast_deps()
    if all(status.values()):
        return {"success": True, "message": "All cast dependencies are already installed"}

    # Check if already installing
    config = get_cast_settings()
    if config.get("installing"):
        return {"success": False, "error": "Installation already in progress"}

    def install_sync():
        import subprocess

        current = get_cast_settings()
        current["installing"] = True
        current["install_progress"] = "Installing cast dependencies..."
        save_cast_settings(current)

        try:
            packages = ["pychromecast", "async-upnp-client", "aiohttp"]
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + packages,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0 and "Permission denied" in result.stderr:
                current = get_cast_settings()
                current["install_progress"] = "Retrying with --user flag..."
                save_cast_settings(current)

                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user"] + packages,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

            if result.returncode == 0:
                current = get_cast_settings()
                current["installing"] = False
                current["install_progress"] = "Installation complete!"
                save_cast_settings(current)
                logger.info("[Cast] Dependencies installed successfully")
            else:
                current = get_cast_settings()
                current["installing"] = False
                current["install_progress"] = f"Installation failed: {result.stderr[:200]}"
                save_cast_settings(current)
                logger.error(f"[Cast] Install failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            current = get_cast_settings()
            current["installing"] = False
            current["install_progress"] = "Installation timed out"
            save_cast_settings(current)
        except Exception as e:
            current = get_cast_settings()
            current["installing"] = False
            current["install_progress"] = f"Installation error: {e}"
            save_cast_settings(current)

    thread = threading.Thread(target=install_sync, daemon=True)
    thread.start()

    return {"success": True, "message": "Installation started"}
