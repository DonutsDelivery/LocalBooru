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


@router.get("/cast")
async def get_cast_config():
    """Get cast configuration and dependency availability."""
    config = get_cast_settings()

    # Check pychromecast availability
    pychromecast_installed = False
    try:
        import pychromecast
        pychromecast_installed = True
    except ImportError:
        pass

    # Check async-upnp-client availability
    upnp_installed = False
    try:
        import async_upnp_client
        upnp_installed = True
    except ImportError:
        pass

    # Check aiohttp availability
    aiohttp_installed = False
    try:
        import aiohttp
        aiohttp_installed = True
    except ImportError:
        pass

    return {
        **config,
        "status": {
            "pychromecast_installed": pychromecast_installed,
            "upnp_installed": upnp_installed,
            "aiohttp_installed": aiohttp_installed,
        },
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
