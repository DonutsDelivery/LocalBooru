"""
Chromecast & DLNA casting API endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request Models
# =============================================================================

class CastPlayRequest(BaseModel):
    device_id: str
    file_path: str
    image_id: Optional[int] = None
    directory_id: Optional[int] = None


class CastControlRequest(BaseModel):
    action: str  # pause, resume, seek, volume
    value: Optional[float] = None  # seek position or volume level


# =============================================================================
# Device Discovery
# =============================================================================

@router.get("/devices")
async def list_devices():
    """List discovered cast devices."""
    from ..services.cast_discovery import get_devices
    devices = await get_devices()
    return {"devices": [d.to_dict() for d in devices]}


@router.post("/devices/refresh")
async def refresh_devices():
    """Force re-scan for cast devices."""
    from ..services.cast_discovery import refresh_devices
    devices = await refresh_devices()
    return {"devices": [d.to_dict() for d in devices]}


# =============================================================================
# Cast Control
# =============================================================================

@router.post("/play")
async def cast_play(request: CastPlayRequest):
    """Start casting media to a device."""
    from ..services.cast_session import start_cast

    try:
        result = await start_cast(
            device_id=request.device_id,
            file_path=request.file_path,
            image_id=request.image_id,
            directory_id=request.directory_id,
        )
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"[Cast] Failed to start casting: {e}")
        return {"success": False, "error": str(e)}


@router.post("/control")
async def cast_control_endpoint(request: CastControlRequest):
    """Control active cast session (pause/resume/seek/volume)."""
    from ..services.cast_session import cast_control

    try:
        result = await cast_control(request.action, request.value)
        return result
    except Exception as e:
        logger.error(f"[Cast] Control error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/status")
async def cast_status_stream():
    """SSE stream of cast playback status."""
    from ..services.events import cast_events

    async def event_generator():
        async for message in cast_events.subscribe():
            yield message

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )


@router.post("/stop")
async def cast_stop():
    """Stop casting and disconnect."""
    from ..services.cast_session import stop_cast

    try:
        await stop_cast()
        return {"success": True}
    except Exception as e:
        logger.error(f"[Cast] Stop error: {e}")
        return {"success": False, "error": str(e)}
