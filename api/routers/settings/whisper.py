"""
Whisper subtitle generation endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response, StreamingResponse
import logging

from .models import (
    get_whisper_settings,
    save_whisper_settings,
    WhisperConfigUpdate,
    WhisperSubtitleRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Whisper Configuration Endpoints
# =============================================================================

@router.get("/whisper")
async def get_whisper_config():
    """Get whisper subtitle configuration and availability status."""
    config = get_whisper_settings()

    # Check faster-whisper availability
    faster_whisper_installed = False
    faster_whisper_error = None
    try:
        import faster_whisper
        faster_whisper_installed = True
    except ImportError:
        pass
    except Exception as e:
        # Installed but broken (e.g. dependency conflict)
        faster_whisper_error = str(e)

    # Check CUDA availability
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    status = {
        "faster_whisper_installed": faster_whisper_installed,
        "cuda_available": cuda_available,
    }
    if faster_whisper_error:
        status["error"] = faster_whisper_error

    return {
        **config,
        "status": status,
    }


@router.post("/whisper")
async def update_whisper_config(config: WhisperConfigUpdate):
    """Update whisper subtitle configuration (partial update)."""
    current = get_whisper_settings()

    if config.enabled is not None:
        current["enabled"] = config.enabled
    if config.auto_generate is not None:
        current["auto_generate"] = config.auto_generate
    if config.model_size is not None:
        if config.model_size in ("tiny", "base", "small", "medium", "large-v2", "large-v3"):
            current["model_size"] = config.model_size
    if config.language is not None:
        current["language"] = config.language
    if config.task is not None:
        if config.task in ("transcribe", "translate"):
            current["task"] = config.task
    if config.chunk_duration is not None:
        current["chunk_duration"] = max(10, min(120, config.chunk_duration))
    if config.beam_size is not None:
        current["beam_size"] = max(1, min(20, config.beam_size))
    if config.device is not None:
        if config.device in ("auto", "cuda", "cpu"):
            current["device"] = config.device
    if config.compute_type is not None:
        if config.compute_type in ("auto", "float16", "int8_float16", "int8"):
            current["compute_type"] = config.compute_type
    if config.vad_filter is not None:
        current["vad_filter"] = config.vad_filter
    if config.suppress_nst is not None:
        current["suppress_nst"] = config.suppress_nst
    if config.cache_subtitles is not None:
        current["cache_subtitles"] = config.cache_subtitles
    if config.subtitle_font is not None:
        current["subtitle_font"] = config.subtitle_font
    if config.subtitle_font_size is not None:
        current["subtitle_font_size"] = max(0.5, min(4.0, config.subtitle_font_size))
    if config.subtitle_style is not None:
        if config.subtitle_style in ("outline", "background", "outline_background"):
            current["subtitle_style"] = config.subtitle_style
    if config.subtitle_color is not None:
        current["subtitle_color"] = config.subtitle_color
    if config.subtitle_outline_color is not None:
        current["subtitle_outline_color"] = config.subtitle_outline_color
    if config.subtitle_bg_opacity is not None:
        current["subtitle_bg_opacity"] = max(0.0, min(1.0, config.subtitle_bg_opacity))

    save_whisper_settings(current)
    return {"success": True, **current}


# =============================================================================
# Installation Endpoint
# =============================================================================

@router.post("/whisper/install")
async def install_whisper_deps():
    """Install faster-whisper package via pip."""
    import sys
    import threading

    # Check if already installed and working
    try:
        import faster_whisper
        return {"success": True, "message": "faster-whisper is already installed"}
    except ImportError:
        pass
    except Exception:
        # Installed but broken (e.g. ctranslate2 version mismatch)
        # Proceed with reinstall
        pass

    # Check if already installing
    config = get_whisper_settings()
    if config.get("installing"):
        return {"success": False, "error": "Installation already in progress"}

    def install_sync():
        import subprocess

        current = get_whisper_settings()
        current["installing"] = True
        current["install_progress"] = "Installing faster-whisper..."
        save_whisper_settings(current)

        try:
            # Install faster-whisper (and its deps like ctranslate2)
            # Try normal install first, fall back to --user if permission denied
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "faster-whisper"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0 and "Permission denied" in result.stderr:
                # Retry with --user flag
                current = get_whisper_settings()
                current["install_progress"] = "Retrying with --user flag..."
                save_whisper_settings(current)

                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", "faster-whisper"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

            # Verify the import actually works
            verify = subprocess.run(
                [sys.executable, "-c", "import faster_whisper; print('OK')"],
                capture_output=True,
                text=True,
                timeout=30
            )

            current = get_whisper_settings()
            current["installing"] = False

            if verify.returncode == 0 and "OK" in verify.stdout:
                current["install_progress"] = "Installation complete!"
                current["enabled"] = True  # Auto-enable after successful install
            elif result.returncode == 0:
                # pip succeeded but import still fails - likely broken ctranslate2
                # Try upgrading ctranslate2 with --user flag
                current["install_progress"] = "Fixing ctranslate2 dependency..."
                save_whisper_settings(current)

                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--user", "--upgrade", "ctranslate2"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                # Re-verify
                verify2 = subprocess.run(
                    [sys.executable, "-c", "import faster_whisper; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                current = get_whisper_settings()
                current["installing"] = False

                if verify2.returncode == 0 and "OK" in verify2.stdout:
                    current["install_progress"] = "Installation complete!"
                    current["enabled"] = True
                else:
                    current["install_progress"] = f"Failed: {verify2.stderr[:200]}"
            else:
                current["install_progress"] = f"Failed: {result.stderr[:200]}"

            save_whisper_settings(current)

        except subprocess.TimeoutExpired:
            current = get_whisper_settings()
            current["installing"] = False
            current["install_progress"] = "Installation timed out"
            save_whisper_settings(current)
        except Exception as e:
            current = get_whisper_settings()
            current["installing"] = False
            current["install_progress"] = f"Error: {str(e)}"
            save_whisper_settings(current)

    thread = threading.Thread(target=install_sync, daemon=True)
    thread.start()

    return {
        "success": True,
        "installing": True,
        "message": "Installation started. This may take a few minutes."
    }


# =============================================================================
# Subtitle Generation Endpoints
# =============================================================================

@router.post("/whisper/generate")
async def generate_subtitles(request: WhisperSubtitleRequest):
    """Start subtitle generation for a video file.

    Returns stream_id and VTT URL for the frontend to use.
    """
    import os
    from ...services.whisper_subtitle_stream import (
        WhisperSubtitleStream,
        stop_all_subtitle_streams,
    )

    if not os.path.exists(request.file_path):
        return {"success": False, "error": "File not found"}

    # Stop any existing subtitle streams
    stop_all_subtitle_streams()

    config = get_whisper_settings()

    # Use request overrides or config defaults
    # Note: use `is not None` check because empty string = auto-detect
    language = request.language if request.language is not None else config["language"]
    task = request.task if request.task is not None else config["task"]

    stream = WhisperSubtitleStream(
        video_path=request.file_path,
        model_size=config["model_size"],
        language=language,
        task=task,
        chunk_duration=config["chunk_duration"],
        beam_size=config["beam_size"],
        device=config["device"],
        compute_type=config["compute_type"],
        vad_filter=config["vad_filter"],
        suppress_nst=config["suppress_nst"],
        cache_subtitles=config["cache_subtitles"],
        start_position=request.start_position,
    )

    success = await stream.start()

    if success:
        return {
            "success": True,
            "stream_id": stream.stream_id,
            "vtt_url": f"/api/settings/whisper/vtt/{stream.stream_id}/subtitles.vtt",
            "events_url": f"/api/settings/whisper/events/{stream.stream_id}",
            "cached": stream.cached_vtt_path is not None,
            "completed": stream.completed,
        }
    else:
        return {"success": False, "error": stream.error or "Failed to start subtitle generation"}


@router.post("/whisper/stop")
async def stop_subtitles():
    """Stop active subtitle generation."""
    from ...services.whisper_subtitle_stream import stop_all_subtitle_streams

    stop_all_subtitle_streams()
    return {"success": True, "message": "Subtitle generation stopped"}


# =============================================================================
# VTT File Serving
# =============================================================================

@router.get("/whisper/vtt/{stream_id}/subtitles.vtt")
async def serve_vtt_file(stream_id: str):
    """Serve the growing VTT file for a subtitle stream."""
    from ...services.whisper_subtitle_stream import get_active_subtitle_stream

    stream = get_active_subtitle_stream(stream_id)
    if not stream:
        return Response(content="Stream not found", status_code=404)

    if not stream.vtt_path or not stream.vtt_path.exists():
        return Response(content="VTT file not ready", status_code=404)

    return FileResponse(
        path=str(stream.vtt_path),
        media_type='text/vtt',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Access-Control-Allow-Origin': '*',
        }
    )


# =============================================================================
# SSE Events
# =============================================================================

@router.get("/whisper/events/{stream_id}")
async def subtitle_event_stream(stream_id: str):
    """SSE stream for subtitle generation progress and cue data."""
    from ...services.events import subtitle_events

    async def event_generator():
        async for message in subtitle_events.subscribe():
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
