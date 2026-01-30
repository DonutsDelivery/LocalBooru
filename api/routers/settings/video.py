"""
Video playback settings endpoints - SVP, optical flow, transcode streaming.
"""
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response
import os
import subprocess

from .models import (
    get_optical_flow_settings,
    save_optical_flow_settings,
    get_svp_settings,
    save_svp_settings,
    parse_quality_preset,
    OpticalFlowConfigUpdate,
    InterpolationPlayRequest,
    SVPConfigUpdate,
    SVPPlayRequest,
    TranscodePlayRequest,
)

router = APIRouter()


# =============================================================================
# Optical Flow Interpolation Endpoints
# =============================================================================

@router.get("/optical-flow")
async def get_optical_flow_config():
    """Get optical flow interpolation configuration and backend status"""
    import traceback
    import logging
    logger = logging.getLogger(__name__)

    config = get_optical_flow_settings()

    try:
        from ...services.optical_flow import get_backend_status
        backend = get_backend_status()
    except Exception as e:
        logger.error(f"Failed to get backend status: {e}\n{traceback.format_exc()}")
        backend = {"error": str(e), "any_backend_available": False}

    return {
        **config,
        "backend": backend
    }


@router.post("/optical-flow")
async def update_optical_flow_config(config: OpticalFlowConfigUpdate):
    """Update optical flow interpolation configuration"""
    current = get_optical_flow_settings()

    # Update only provided fields
    if config.enabled is not None:
        current["enabled"] = config.enabled
    if config.target_fps is not None:
        # Clamp to valid range
        current["target_fps"] = max(15, min(120, config.target_fps))
    if config.use_gpu is not None:
        current["use_gpu"] = config.use_gpu
    if config.quality is not None:
        # Validate quality preset
        if config.quality in ("svp", "gpu_native", "realtime", "fast", "balanced", "quality"):
            current["quality"] = config.quality

    save_optical_flow_settings(current)

    return {"success": True, **current}


@router.post("/optical-flow/play")
async def play_video_interpolated(request: InterpolationPlayRequest):
    """
    Start interpolated video stream via HLS.

    Returns the stream URL that can be used with hls.js.
    """
    from ...services.optical_flow_stream import create_interpolated_stream
    from ...services.optical_flow import get_backend_status

    config = get_optical_flow_settings()
    backend = get_backend_status()

    if not config["enabled"]:
        return {"success": False, "error": "Optical flow interpolation is not enabled"}

    if not backend["any_backend_available"]:
        return {"success": False, "error": "No interpolation backend available. Install OpenCV or PyTorch."}

    # Check if file exists
    if not os.path.exists(request.file_path):
        return {"success": False, "error": "File not found"}

    try:
        # Parse quality preset
        # Get source video resolution using ffprobe
        source_resolution = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                request.file_path
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    source_resolution = (int(parts[0]), int(parts[1]))
        except Exception as e:
            print(f"[OpticalFlow] Could not detect source resolution: {e}")

        quality_settings = parse_quality_preset(request.quality_preset, source_resolution)

        stream = await create_interpolated_stream(
            video_path=request.file_path,
            target_fps=config["target_fps"],
            use_gpu=config["use_gpu"] and backend["cuda_available"],
            quality=config.get("quality", "fast"),
            wait_for_buffer=True,
            min_segments=2,
            target_bitrate=quality_settings["bitrate"],
            target_resolution=quality_settings["resolution"],
            start_position=request.start_position,
        )

        if stream:
            return {
                "success": True,
                "stream_id": stream.stream_id,
                "stream_url": f"/api/settings/optical-flow/stream/{stream.stream_id}/stream.m3u8",
                "source_resolution": {"width": source_resolution[0], "height": source_resolution[1]} if source_resolution else None,
                "message": f"Interpolated stream started at {config['target_fps']} fps"
            }
        else:
            return {"success": False, "error": "Failed to start interpolated stream"}

    except Exception as e:
        return {"success": False, "error": f"Stream error: {str(e)}"}


@router.post("/optical-flow/stop")
async def stop_interpolated_stream():
    """Stop the active interpolated stream."""
    from ...services.optical_flow_stream import stop_all_streams

    stop_all_streams()
    return {"success": True, "message": "Stream stopped"}


@router.get("/optical-flow/stream/{stream_id}/{filename:path}")
async def serve_hls_file(stream_id: str, filename: str):
    """Serve HLS playlist or segment files for the interpolated stream."""
    from ...services.optical_flow_stream import get_active_stream

    stream = get_active_stream(stream_id)
    if not stream:
        return Response(content="Stream not found", status_code=404)

    file_path = stream.get_file_path(filename)
    if not file_path:
        return Response(content="File not found", status_code=404)

    # Determine content type
    if filename.endswith('.m3u8'):
        media_type = 'application/vnd.apple.mpegurl'
    elif filename.endswith('.ts'):
        media_type = 'video/mp2t'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Access-Control-Allow-Origin': '*'
        }
    )


# =============================================================================
# SVP (SmoothVideo Project) Interpolation Endpoints
# =============================================================================

@router.get("/svp")
async def get_svp_config():
    """Get SVP interpolation configuration and availability status"""
    from ...services.svp_stream import get_svp_status, SVP_PRESETS, SVP_ALGORITHMS, SVP_BLOCK_SIZES, SVP_PEL_OPTIONS, SVP_MASK_AREA

    config = get_svp_settings()
    status = get_svp_status()

    return {
        **config,
        "status": status,
        "presets": {
            name: {"name": preset["name"], "description": preset["description"]}
            for name, preset in SVP_PRESETS.items()
        },
        "options": {
            "algorithms": SVP_ALGORITHMS,
            "block_sizes": SVP_BLOCK_SIZES,
            "pel_options": SVP_PEL_OPTIONS,
            "mask_area": SVP_MASK_AREA,
        }
    }


@router.post("/svp")
async def update_svp_config(config: SVPConfigUpdate):
    """Update SVP interpolation configuration"""
    from ...services.svp_stream import SVP_PRESETS

    current = get_svp_settings()

    # Update only provided fields
    if config.enabled is not None:
        current["enabled"] = config.enabled
    if config.target_fps is not None:
        # Clamp to valid range
        current["target_fps"] = max(15, min(144, config.target_fps))
    if config.preset is not None:
        # Validate preset
        if config.preset in SVP_PRESETS:
            current["preset"] = config.preset
    if config.use_nvof is not None:
        current["use_nvof"] = config.use_nvof
    if config.shader is not None:
        # Validate shader value
        if config.shader in [1, 2, 11, 13, 21, 23]:
            current["shader"] = config.shader
    if config.artifact_masking is not None:
        # Clamp to valid range
        current["artifact_masking"] = max(0, min(200, config.artifact_masking))
    if config.frame_interpolation is not None:
        if config.frame_interpolation in [1, 2]:
            current["frame_interpolation"] = config.frame_interpolation
    if config.custom_super is not None:
        current["custom_super"] = config.custom_super if config.custom_super else None
    if config.custom_analyse is not None:
        current["custom_analyse"] = config.custom_analyse if config.custom_analyse else None
    if config.custom_smooth is not None:
        current["custom_smooth"] = config.custom_smooth if config.custom_smooth else None

    save_svp_settings(current)

    return {"success": True, **current}


@router.post("/svp/play")
async def play_video_svp(request: SVPPlayRequest):
    """
    Start SVP-interpolated video stream via HLS.

    SVP uses VapourSynth + SVPflow plugins for high-quality
    motion-compensated frame interpolation.
    """
    from ...services.svp_stream import SVPStream, get_svp_status, stop_all_svp_streams

    # Stop any existing SVP streams before starting a new one
    stop_all_svp_streams()

    config = get_svp_settings()
    status = get_svp_status()

    # Note: We don't check config["enabled"] here - the toggle button should work
    # regardless of the global setting. We allow users to try SVP even if detection
    # fails - they'll get a runtime error if it's actually missing. This prevents
    # detection from being a blocker.

    # Check if file exists
    if not os.path.exists(request.file_path):
        return {"success": False, "error": "File not found"}

    try:
        # Parse quality preset
        # Get source video resolution using ffprobe
        source_resolution = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                request.file_path
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    source_resolution = (int(parts[0]), int(parts[1]))
        except Exception as e:
            print(f"[SVP] Could not detect source resolution: {e}")

        quality_settings = parse_quality_preset(request.quality_preset, source_resolution)
        print(f"[SVP] Quality preset: {request.quality_preset}, source: {source_resolution}, settings: {quality_settings}")

        # Create SVP stream
        stream = SVPStream(
            video_path=request.file_path,
            target_fps=config["target_fps"],
            preset=config.get("preset", "balanced"),
            use_nvof=config.get("use_nvof", True),
            shader=config.get("shader", 23),
            artifact_masking=config.get("artifact_masking", 100),
            frame_interpolation=config.get("frame_interpolation", 2),
            custom_super=config.get("custom_super"),
            custom_analyse=config.get("custom_analyse"),
            custom_smooth=config.get("custom_smooth"),
            target_bitrate=quality_settings["bitrate"],
            target_resolution=quality_settings["resolution"],
            start_position=request.start_position,
        )

        # Start the stream
        success = await stream.start()

        if success:
            # Wait briefly for initial buffer, but return optimistically
            # Let the frontend HLS.js handle retry/buffering for large files
            import asyncio
            for _ in range(50):  # Wait up to 5 seconds
                if stream.playlist_ready:
                    break
                if stream.error:
                    return {"success": False, "error": stream.error}
                if not stream._running:
                    return {"success": False, "error": stream.error or "Pipeline failed to start"}
                await asyncio.sleep(0.1)

            # Return success - stream is running, frontend will handle buffering
            return {
                "success": True,
                "stream_id": stream.stream_id,
                "stream_url": f"/api/settings/svp/stream/{stream.stream_id}/stream.m3u8",
                "duration": stream._duration,
                "start_position": request.start_position,
                "source_resolution": {"width": source_resolution[0], "height": source_resolution[1]} if source_resolution else None,
                "message": f"SVP stream started at {config['target_fps']} fps with {config.get('preset', 'balanced')} preset"
            }
        else:
            return {"success": False, "error": stream.error or "Failed to start SVP stream"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"SVP stream error: {str(e)}"}


@router.post("/svp/stop")
async def stop_svp_stream():
    """Stop all active SVP streams."""
    from ...services.svp_stream import stop_all_svp_streams

    stop_all_svp_streams()
    return {"success": True, "message": "SVP streams stopped"}


@router.get("/svp/stream/{stream_id}/{filename:path}")
async def serve_svp_hls_file(stream_id: str, filename: str):
    """Serve HLS playlist or segment files for the SVP stream."""
    from ...services.svp_stream import get_active_svp_stream, _active_svp_streams

    stream = get_active_svp_stream(stream_id)
    if not stream:
        print(f"[SVP] Stream {stream_id} not found. Active streams: {list(_active_svp_streams.keys())}")
        return Response(content="Stream not found", status_code=404)

    if not stream.hls_dir:
        return Response(content="Stream not ready", status_code=404)

    file_path = stream.hls_dir / filename
    if not file_path.exists():
        return Response(content="File not found", status_code=404)

    # Determine content type
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
            'Access-Control-Allow-Origin': '*'
        }
    )


# =============================================================================
# Web Video SVP Endpoints (Browser Extension)
# =============================================================================

@router.post("/svp/web/play")
async def play_web_video_svp(url: str, quality: str = "best"):
    """
    Start SVP stream for a web video URL using yt-dlp.

    This endpoint:
    1. Checks if the URL is from a DRM-protected site
    2. Downloads the video via yt-dlp to a temp file
    3. Passes the temp file to the existing SVP pipeline
    4. Returns the HLS stream URL

    Args:
        url: Web video URL (YouTube, Vimeo, Twitch VOD, direct video, etc.)
        quality: Quality preference - "best", "1080p", "720p", "480p"

    Returns:
        On success: {"success": true, "stream_url": "...", "download_id": "..."}
        On pending: {"success": true, "status": "downloading", "download_id": "...", "progress": 0.5}
        On error: {"success": false, "error": "..."}
    """
    from ...services.web_video_downloader import (
        download_video,
        get_download,
        is_drm_site,
        is_live_stream,
    )
    from ...services.svp_stream import SVPStream, get_svp_status

    # Quick DRM check before starting download
    if is_drm_site(url):
        return {
            "success": False,
            "error": "This site uses DRM protection and cannot be processed",
            "drm_protected": True,
        }

    # Quick live stream check
    if is_live_stream(url):
        return {
            "success": False,
            "error": "Live streams are not supported yet (coming in v2)",
            "live_stream": True,
        }

    # Check SVP status before downloading
    config = get_svp_settings()
    status = get_svp_status()

    if not config["enabled"]:
        return {"success": False, "error": "SVP interpolation is not enabled"}

    if not status["ready"]:
        missing = []
        if not status["vapoursynth_available"]:
            missing.append("VapourSynth")
        if not status["svp_plugins_available"]:
            missing.append("SVPflow plugins")
        if not status["vspipe_available"]:
            missing.append("vspipe")
        return {"success": False, "error": f"SVP not ready. Missing: {', '.join(missing)}"}

    # Start or check download
    result = await download_video(url, quality)

    if not result.success:
        return {"success": False, "error": result.error}

    # Check download status
    download = get_download(result.download_id)
    if not download:
        return {"success": False, "error": "Download tracking error"}

    if download.status in ("pending", "downloading", "processing"):
        return {
            "success": True,
            "status": download.status,
            "download_id": download.download_id,
            "progress": download.progress,
        }

    if download.status == "error":
        return {"success": False, "error": download.error or "Download failed"}

    if download.status == "complete" and download.file_path:
        # Download complete, start SVP stream
        try:
            stream = SVPStream(
                video_path=download.file_path,
                target_fps=config["target_fps"],
                preset=config.get("preset", "balanced"),
                use_nvof=config.get("use_nvof", True),
                shader=config.get("shader", 23),
                artifact_masking=config.get("artifact_masking", 100),
                frame_interpolation=config.get("frame_interpolation", 2),
                custom_super=config.get("custom_super"),
                custom_analyse=config.get("custom_analyse"),
                custom_smooth=config.get("custom_smooth"),
            )

            success = await stream.start()

            if success:
                return {
                    "success": True,
                    "status": "streaming",
                    "download_id": download.download_id,
                    "stream_id": stream.stream_id,
                    "stream_url": f"/api/settings/svp/stream/{stream.stream_id}/stream.m3u8",
                    "duration": stream._duration,
                    "title": download.title,
                    "message": f"SVP stream started at {config['target_fps']} fps",
                }
            else:
                return {"success": False, "error": stream.error or "Failed to start SVP stream"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"SVP stream error: {str(e)}"}

    return {"success": False, "error": "Unexpected download state"}


@router.get("/svp/web/status/{download_id}")
async def get_web_download_status(download_id: str):
    """
    Get download progress for a web video.

    Args:
        download_id: The download ID returned from /svp/web/play

    Returns:
        Download status including progress (0.0 to 1.0), status, and any errors.
    """
    from ...services.web_video_downloader import get_download

    download = get_download(download_id)
    if not download:
        return {"success": False, "error": "Download not found"}

    return {
        "success": True,
        "download_id": download.download_id,
        "status": download.status,
        "progress": download.progress,
        "title": download.title,
        "file_path": download.file_path,
        "error": download.error,
    }


@router.get("/svp/web/drm-check")
async def check_drm_site(url: str):
    """
    Check if a URL is from a DRM-protected site.

    This is a quick check that the extension can use before attempting to play.

    Args:
        url: The URL to check

    Returns:
        {"drm_protected": true/false, "live_stream": true/false/null}
    """
    from ...services.web_video_downloader import is_drm_site, is_live_stream

    return {
        "drm_protected": is_drm_site(url),
        "live_stream": is_live_stream(url),
    }


# =============================================================================
# Transcoding Endpoints (fallback when SVP/OpticalFlow not available)
# =============================================================================

@router.post("/transcode/play")
async def play_video_transcode(request: TranscodePlayRequest):
    """
    Start simple HLS transcoding via FFmpeg only (no interpolation).

    This is a fallback when SVP or OpticalFlow aren't available but
    the user wants to change quality/bitrate.
    """
    from ...services.transcode_stream import TranscodeStream, stop_all_transcode_streams

    # Stop any existing transcode streams before starting a new one
    stop_all_transcode_streams()

    # Parse quality settings
    quality_settings = parse_quality_preset(request.quality_preset)

    try:
        # Detect source resolution
        source_resolution = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                request.file_path
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    source_resolution = (int(parts[0]), int(parts[1]))
        except Exception as e:
            print(f"[Transcode] Could not detect source resolution: {e}")

        quality_settings = parse_quality_preset(request.quality_preset, source_resolution)

        # Create transcode stream
        stream = TranscodeStream(
            video_path=request.file_path,
            target_bitrate=quality_settings["bitrate"],
            target_resolution=quality_settings["resolution"],
            start_position=request.start_position,
        )

        # Start the stream
        success = await stream.start()

        if success:
            # Wait briefly for initial buffer
            import asyncio
            for _ in range(50):  # Wait up to 5 seconds
                if stream.playlist_ready:
                    break
                if stream.error:
                    return {"success": False, "error": stream.error}
                if not stream._running:
                    return {"success": False, "error": stream.error or "Encoding failed"}
                await asyncio.sleep(0.1)

            # Return success
            return {
                "success": True,
                "stream_id": stream.stream_id,
                "stream_url": f"/api/settings/transcode/stream/{stream.stream_id}/playlist.m3u8",
                "duration": stream._duration,
                "start_position": request.start_position,
                "source_resolution": {"width": source_resolution[0], "height": source_resolution[1]} if source_resolution else None,
                "message": "Transcoding started"
            }
        else:
            return {"success": False, "error": stream.error or "Failed to start transcoding"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Transcoding error: {str(e)}"}


@router.post("/transcode/stop")
async def stop_transcode_stream():
    """Stop all active transcoding streams."""
    from ...services.transcode_stream import stop_all_transcode_streams

    stop_all_transcode_streams()
    return {"success": True, "message": "Transcode streams stopped"}


@router.get("/transcode/stream/{stream_id}/{filename:path}")
async def serve_transcode_hls_file(stream_id: str, filename: str):
    """Serve HLS playlist or segment files for transcoding stream."""
    from ...services.transcode_stream import get_active_transcode_stream

    stream = get_active_transcode_stream(stream_id)
    if not stream:
        return Response(content="Stream not found", status_code=404)

    if not stream.hls_dir:
        return Response(content="Stream not ready", status_code=404)

    file_path = stream.hls_dir / filename
    if not file_path.exists():
        return Response(content="File not found", status_code=404)

    # Determine content type
    if filename.endswith('.m3u8'):
        media_type = "application/vnd.apple.mpegurl"
    else:
        media_type = "video/mp2t"

    return FileResponse(file_path, media_type=media_type)
