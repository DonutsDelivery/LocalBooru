"""
Share Stream router - creates shareable watch party links for videos.

Host creates a session, viewers connect via token, playback is synced via SSE.
"""

import json
import socket
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db, directory_db_manager
from ..models import Image, ImageFile, DirectoryImage, DirectoryImageFile
from ..services.share_session import (
    create_session, get_session, update_host_state, destroy_session,
)
from ..services.events import get_share_broadcaster
from ..services.tailscale import detect_tailscale, get_tailscale_url, get_os_name, is_tailscale_https, needs_operator_setup
from ..services.transcode_stream import TranscodeStream, get_active_transcode_stream
from .settings.models import QUALITY_PRESETS, parse_quality_preset

router = APIRouter()


class CreateShareRequest(BaseModel):
    image_id: int
    directory_id: Optional[int] = None


class SyncStateRequest(BaseModel):
    playing: Optional[bool] = None
    position: Optional[float] = None
    speed: Optional[float] = None


class QualityChangeRequest(BaseModel):
    quality: str  # Quality preset id: 'original', '1080p', '720p', etc.


def _get_local_ip() -> str:
    """Get local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@router.get("/network-info")
async def get_network_info(request: Request):
    """Get available network URLs for sharing."""
    # Determine the port from the request
    port = request.url.port or 8790

    local_ip = _get_local_ip()
    tailscale_info = await detect_tailscale()
    tailscale_url = await get_tailscale_url(port) if tailscale_info else None

    return {
        "local_url": f"http://{local_ip}:{port}",
        "tailscale_url": tailscale_url,
        "tailscale_installed": tailscale_info is not None,
        "tailscale_https": (await is_tailscale_https(port)) if tailscale_info else False,
        "tailscale_needs_operator": needs_operator_setup() if tailscale_info else False,
        "os": get_os_name(),
    }


@router.post("/create")
async def create_share(body: CreateShareRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """Create a share session for a video."""
    original_filename = None
    file_path = None

    # Try directory database first if directory_id provided
    if body.directory_id is not None and directory_db_manager.db_exists(body.directory_id):
        dir_db = await directory_db_manager.get_session(body.directory_id)
        try:
            # Get image metadata from directory DB
            img_result = await dir_db.execute(
                select(DirectoryImage).where(DirectoryImage.id == body.image_id)
            )
            dir_image = img_result.scalar_one_or_none()
            if dir_image:
                original_filename = dir_image.original_filename

            # Get file path from directory DB
            file_result = await dir_db.execute(
                select(DirectoryImageFile.original_path).where(
                    DirectoryImageFile.image_id == body.image_id
                ).limit(1)
            )
            file_path = file_result.scalar_one_or_none()
        finally:
            await dir_db.close()

    # Fallback to main database
    if not file_path:
        result = await db.execute(select(Image).where(Image.id == body.image_id))
        image = result.scalar_one_or_none()
        if image:
            original_filename = image.original_filename

        file_result = await db.execute(
            select(ImageFile.original_path).where(ImageFile.image_id == body.image_id).limit(1)
        )
        file_path = file_result.scalar_one_or_none()

    if not file_path:
        raise HTTPException(status_code=404, detail="Image not found")

    # Create session
    session = create_session(
        image_id=body.image_id,
        video_path=file_path,
        original_filename=original_filename or "video",
    )

    # Create a transcode stream for viewers
    stream = TranscodeStream(video_path=file_path)
    started = await stream.start()
    if not started:
        destroy_session(session.token)
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {stream.error}")

    # Store stream_id in session for later retrieval
    session.transcode_stream_id = stream.stream_id

    # Build share URL
    port = request.url.port or 8790
    tailscale_url = await get_tailscale_url(port)
    if tailscale_url:
        base_url = tailscale_url
    else:
        base_url = f"http://{_get_local_ip()}:{port}"

    share_url = f"{base_url}/watch/{session.token}"

    return {
        "token": session.token,
        "share_url": share_url,
        "stream_id": stream.stream_id,
    }


@router.post("/{token}/sync")
async def sync_state(token: str, body: SyncStateRequest):
    """Host sends playback state update, broadcast to viewers via SSE."""
    state = update_host_state(
        token,
        playing=body.playing,
        position=body.position,
        speed=body.speed,
    )
    if not state:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Broadcast to viewers
    broadcaster = get_share_broadcaster(token)
    await broadcaster.broadcast("sync", {
        "playing": state.playing,
        "position": state.position,
        "speed": state.speed,
        "updated_at": state.updated_at,
    })

    return {"ok": True}


@router.get("/{token}/events")
async def share_events(token: str):
    """SSE endpoint for viewers to receive host state updates."""
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    broadcaster = get_share_broadcaster(token)

    async def event_stream():
        # Send initial state
        initial = {
            "type": "sync",
            "data": {
                "playing": session.host_state.playing,
                "position": session.host_state.position,
                "speed": session.host_state.speed,
                "updated_at": session.host_state.updated_at,
            },
            "timestamp": str(session.host_state.updated_at),
        }
        yield f"data: {json.dumps(initial)}\n\n"

        async for message in broadcaster.subscribe():
            yield message

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/{token}/info")
async def get_share_info(token: str):
    """Get session metadata for the viewer page."""
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    return {
        "token": session.token,
        "image_id": session.image_id,
        "original_filename": session.original_filename,
        "host_state": {
            "playing": session.host_state.playing,
            "position": session.host_state.position,
            "speed": session.host_state.speed,
            "updated_at": session.host_state.updated_at,
        },
        "stream_id": getattr(session, 'transcode_stream_id', None),
    }


@router.delete("/{token}")
async def stop_share(token: str):
    """Host stops sharing."""
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Stop the transcode stream if it exists
    stream_id = getattr(session, 'transcode_stream_id', None)
    if stream_id:
        stream = get_active_transcode_stream(stream_id)
        if stream:
            stream.stop()

    # Broadcast disconnect to viewers
    broadcaster = get_share_broadcaster(token)
    await broadcaster.broadcast("disconnected", {"reason": "Host stopped sharing"})

    destroy_session(token)
    return {"ok": True}


@router.get("/{token}/hls/{filename:path}")
async def serve_share_hls(token: str, filename: str):
    """Proxy HLS segments for the share session (avoids auth on temp dirs)."""
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    stream_id = getattr(session, 'transcode_stream_id', None)
    if not stream_id:
        raise HTTPException(status_code=404, detail="No active stream")

    stream = get_active_transcode_stream(stream_id)
    if not stream or not stream.hls_dir:
        raise HTTPException(status_code=404, detail="Stream not found")

    file_path = stream.hls_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if filename.endswith('.m3u8'):
        media_type = 'application/vnd.apple.mpegurl'
    elif filename.endswith('.ts'):
        media_type = 'video/mp2t'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Access-Control-Allow-Origin': '*',
        },
    )


@router.post("/{token}/quality")
async def change_share_quality(token: str, body: QualityChangeRequest):
    """Change the transcode quality for a share session.

    Stops the current transcode stream and starts a new one with the requested quality.
    """
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Stop existing stream
    old_stream_id = getattr(session, 'transcode_stream_id', None)
    if old_stream_id:
        old_stream = get_active_transcode_stream(old_stream_id)
        if old_stream:
            old_stream.stop()

    # Parse quality preset
    quality_settings = parse_quality_preset(body.quality)
    bitrate = quality_settings.get("bitrate")
    resolution = quality_settings.get("resolution")

    # Start new transcode stream with requested quality
    stream = TranscodeStream(
        video_path=session.video_path,
        target_bitrate=bitrate,
        target_resolution=resolution,
    )
    started = await stream.start()
    if not started:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {stream.error}")

    session.transcode_stream_id = stream.stream_id

    return {"ok": True, "stream_id": stream.stream_id}


@router.get("/{token}/quality-presets")
async def get_quality_presets(token: str):
    """Return available quality presets for the viewer."""
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    presets = [
        {"id": "original", "label": "Original", "description": "No transcoding", "maxHeight": 99999},
        {"id": "1440p", "label": "1440p (QHD)", "description": "30 Mbps", "maxHeight": 1440},
        {"id": "1080p_enhanced", "label": "1080p Enhanced", "description": "20 Mbps", "maxHeight": 1080},
        {"id": "1080p", "label": "1080p", "description": "12 Mbps", "maxHeight": 1080},
        {"id": "720p", "label": "720p", "description": "8 Mbps", "maxHeight": 720},
        {"id": "480p", "label": "480p", "description": "4 Mbps", "maxHeight": 480},
    ]

    return {"presets": presets}
