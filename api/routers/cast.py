"""
Chromecast & DLNA casting API endpoints.
"""
import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Cast Media Serving (replaces the old separate aiohttp server)
# =============================================================================

cast_media_router = APIRouter()


@cast_media_router.api_route("/{media_id}/file/{filename}", methods=["GET", "HEAD"])
async def serve_cast_file(media_id: str, filename: str):
    """Serve the original media file. Starlette FileResponse handles Range/206 and HEAD."""
    from ..services.cast_media_server import _registered_media

    entry = _registered_media.get(media_id)
    if entry is None:
        return Response(status_code=404, content="Media not found")

    file_path = entry["file_path"]
    if not os.path.isfile(file_path):
        return Response(status_code=404, content="File not found on disk")

    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return FileResponse(file_path, media_type=content_type)


@cast_media_router.api_route("/{media_id}/hls/{filename}", methods=["GET", "HEAD"])
async def serve_cast_hls(media_id: str, filename: str):
    """Serve HLS playlist and segment files."""
    from ..services.cast_media_server import _registered_media

    entry = _registered_media.get(media_id)
    if entry is None:
        return Response(status_code=404, content="Media not found")

    hls_dir = entry.get("hls_dir")
    if hls_dir is None:
        return Response(status_code=404, content="No HLS directory registered for this media")

    safe_name = Path(filename).name
    target = Path(hls_dir) / safe_name
    if not target.is_file():
        return Response(status_code=404, content=f"HLS file not found: {safe_name}")

    if safe_name.endswith(".m3u8"):
        content_type = "application/vnd.apple.mpegurl"
    elif safe_name.endswith(".ts"):
        content_type = "video/mp2t"
    else:
        content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

    return FileResponse(str(target), media_type=content_type)


@cast_media_router.api_route("/{media_id}/subs/{filename}", methods=["GET", "HEAD"])
async def serve_cast_subs(media_id: str, filename: str):
    """Serve VTT subtitle files."""
    from ..services.cast_media_server import _registered_media

    entry = _registered_media.get(media_id)
    if entry is None:
        return Response(status_code=404, content="Media not found")

    subtitle_paths = entry.get("subtitle_paths")
    if not subtitle_paths:
        return Response(status_code=404, content="No subtitles registered for this media")

    safe_name = Path(filename).name
    target = None
    for sub_path in subtitle_paths:
        if sub_path.name == safe_name:
            target = sub_path
            break

    if target is None or not target.is_file():
        return Response(status_code=404, content=f"Subtitle file not found: {safe_name}")

    return FileResponse(str(target), media_type="text/vtt; charset=utf-8")


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
