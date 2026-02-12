"""
HTTP-only media server for casting to Chromecast/DLNA devices.

Runs on a separate port, only active during casting.
Chromecast rejects self-signed HTTPS, so this server is plain HTTP.
"""
import asyncio
import atexit
import logging
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from aiohttp import web

from .network import get_local_ip

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────

_registered_media: Dict[str, dict] = {}
_server_app: Optional[web.Application] = None
_server_runner: Optional[web.AppRunner] = None
_server_site: Optional[web.TCPSite] = None
_server_port: int = 8792

# CORS headers applied to every response
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "Range, Content-Type",
    "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
}


# ── Media registration ──────────────────────────────────────────────

def register_media(
    media_id: str,
    file_path: str,
    hls_dir: Optional[Path] = None,
    subtitle_paths: Optional[list] = None,
) -> None:
    """Register a media item so the cast server can serve it.

    Args:
        media_id: Unique identifier for the media item.
        file_path: Absolute path to the original media file.
        hls_dir: Optional path to directory containing HLS segments/playlist.
        subtitle_paths: Optional list of Path objects pointing to VTT subtitle files.
    """
    _registered_media[media_id] = {
        "file_path": file_path,
        "hls_dir": Path(hls_dir) if hls_dir else None,
        "subtitle_paths": [Path(p) for p in subtitle_paths] if subtitle_paths else None,
    }
    logger.info(f"[CastServer] Registered media {media_id}: {file_path}")


def unregister_media(media_id: str) -> None:
    """Remove a media item from the cast server registry."""
    if media_id in _registered_media:
        del _registered_media[media_id]
        logger.info(f"[CastServer] Unregistered media {media_id}")


# ── Chromecast compatibility check ──────────────────────────────────

def is_chromecast_compatible(path: str) -> bool:
    """Check whether a video file can be direct-played by Chromecast.

    Chromecast supports H.264 video + AAC audio inside an MP4/MOV container.
    Uses ffprobe to inspect the file.

    Returns True if the file is compatible, False otherwise.
    """
    try:
        # Check container format
        fmt_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=format_name",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if fmt_result.returncode != 0:
            return False

        format_name = fmt_result.stdout.strip().lower()
        # format_name can be comma-separated list like "mov,mp4,m4a,3gp,3g2,mj2"
        compatible_formats = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
        format_parts = {f.strip() for f in format_name.split(",")}
        if not format_parts & compatible_formats:
            return False

        # Check video codec
        video_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if video_result.returncode != 0:
            return False

        video_codec = video_result.stdout.strip().lower()
        if video_codec != "h264":
            return False

        # Check audio codec (no audio is acceptable)
        audio_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        audio_codec = audio_result.stdout.strip().lower()
        if audio_codec and audio_codec != "aac":
            return False

        return True

    except Exception as e:
        logger.warning(f"[CastServer] ffprobe check failed for {path}: {e}")
        return False


# ── URL construction ────────────────────────────────────────────────

def get_media_url(media_id: str, media_type: str = "file") -> Optional[str]:
    """Construct a LAN-accessible URL for a registered media item.

    Args:
        media_id: The registered media identifier.
        media_type: One of "file", "hls", "subs".

    Returns:
        Full URL string like ``http://192.168.1.100:8792/media/{id}/file``,
        or None if the server is not running or the IP cannot be determined.
    """
    ip = get_local_ip()
    if ip is None:
        return None

    base = f"http://{ip}:{_server_port}/media/{media_id}"

    if media_type == "file":
        return f"{base}/file"
    elif media_type == "hls":
        return f"{base}/hls/playlist.m3u8"
    elif media_type == "subs":
        return f"{base}/subs/"
    else:
        return f"{base}/{media_type}"


# ── Route handlers ──────────────────────────────────────────────────

def _cors_response(status: int = 200, **kwargs) -> web.Response:
    """Create a response with CORS headers already applied."""
    resp = web.Response(status=status, **kwargs)
    resp.headers.update(_CORS_HEADERS)
    return resp


def _cors_stream_response(status: int = 200, **kwargs) -> web.StreamResponse:
    """Create a StreamResponse with CORS headers already applied."""
    resp = web.StreamResponse(status=status, **kwargs)
    resp.headers.update(_CORS_HEADERS)
    return resp


async def _handle_options(request: web.Request) -> web.Response:
    """Handle CORS preflight OPTIONS requests."""
    return _cors_response(status=204)


async def _handle_media_file(request: web.Request) -> web.Response:
    """Serve the original media file with HTTP range-request support.

    Route: /media/{media_id}/file
    """
    media_id = request.match_info["media_id"]
    entry = _registered_media.get(media_id)
    if entry is None:
        return _cors_response(status=404, text="Media not found")

    file_path = entry["file_path"]
    if not os.path.isfile(file_path):
        return _cors_response(status=404, text="File not found on disk")

    file_size = os.path.getsize(file_path)
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    range_header = request.headers.get("Range")

    if range_header is None:
        # Full file response
        if request.method == "HEAD":
            resp = _cors_response(
                status=200,
                headers={
                    "Content-Type": content_type,
                    "Content-Length": str(file_size),
                    "Accept-Ranges": "bytes",
                },
            )
            return resp

        resp = _cors_response(
            status=200,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            },
            body=_read_file_bytes(file_path, 0, file_size - 1),
        )
        return resp

    # Parse Range header — only support "bytes=START-END" / "bytes=START-"
    try:
        range_spec = range_header.replace("bytes=", "").strip()
        parts = range_spec.split("-", 1)
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
    except (ValueError, IndexError):
        return _cors_response(
            status=416,
            headers={"Content-Range": f"bytes */{file_size}"},
            text="Invalid Range header",
        )

    # Clamp range
    if start < 0:
        start = 0
    if end >= file_size:
        end = file_size - 1
    if start > end:
        return _cors_response(
            status=416,
            headers={"Content-Range": f"bytes */{file_size}"},
            text="Range not satisfiable",
        )

    content_length = end - start + 1

    if request.method == "HEAD":
        resp = _cors_response(
            status=206,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(content_length),
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
            },
        )
        return resp

    resp = _cors_response(
        status=206,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
        },
        body=_read_file_bytes(file_path, start, end),
    )
    return resp


def _read_file_bytes(file_path: str, start: int, end: int) -> bytes:
    """Read a byte range from a file (inclusive start and end)."""
    length = end - start + 1
    with open(file_path, "rb") as f:
        f.seek(start)
        return f.read(length)


async def _handle_hls(request: web.Request) -> web.Response:
    """Serve HLS playlist and segment files from a TranscodeStream's hls_dir.

    Route: /media/{media_id}/hls/{filename}
    """
    media_id = request.match_info["media_id"]
    filename = request.match_info["filename"]
    entry = _registered_media.get(media_id)
    if entry is None:
        return _cors_response(status=404, text="Media not found")

    hls_dir = entry.get("hls_dir")
    if hls_dir is None:
        return _cors_response(status=404, text="No HLS directory registered for this media")

    # Prevent directory traversal
    safe_name = Path(filename).name
    target = Path(hls_dir) / safe_name
    if not target.is_file():
        return _cors_response(status=404, text=f"HLS file not found: {safe_name}")

    # Determine content type
    if safe_name.endswith(".m3u8"):
        content_type = "application/vnd.apple.mpegurl"
    elif safe_name.endswith(".ts"):
        content_type = "video/mp2t"
    else:
        content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

    data = target.read_bytes()
    return _cors_response(
        status=200,
        body=data,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        },
    )


async def _handle_subs(request: web.Request) -> web.Response:
    """Serve VTT subtitle files.

    Route: /media/{media_id}/subs/{filename}
    """
    media_id = request.match_info["media_id"]
    filename = request.match_info["filename"]
    entry = _registered_media.get(media_id)
    if entry is None:
        return _cors_response(status=404, text="Media not found")

    subtitle_paths = entry.get("subtitle_paths")
    if not subtitle_paths:
        return _cors_response(status=404, text="No subtitles registered for this media")

    # Find the matching subtitle file by name
    safe_name = Path(filename).name
    target = None
    for sub_path in subtitle_paths:
        if sub_path.name == safe_name:
            target = sub_path
            break

    if target is None or not target.is_file():
        return _cors_response(status=404, text=f"Subtitle file not found: {safe_name}")

    data = target.read_bytes()
    return _cors_response(
        status=200,
        body=data,
        headers={
            "Content-Type": "text/vtt; charset=utf-8",
            "Content-Length": str(len(data)),
        },
    )


# ── Server lifecycle ────────────────────────────────────────────────

def _create_app() -> web.Application:
    """Build the aiohttp application with all routes."""
    app = web.Application()
    app.router.add_route("OPTIONS", "/{path_info:.*}", _handle_options)
    app.router.add_get("/media/{media_id}/file", _handle_media_file)
    app.router.add_head("/media/{media_id}/file", _handle_media_file)
    app.router.add_get("/media/{media_id}/hls/{filename}", _handle_hls)
    app.router.add_get("/media/{media_id}/subs/{filename}", _handle_subs)
    return app


async def start_server(port: int = 8792) -> bool:
    """Start the cast media server on the given port.

    Returns True if the server started successfully, False otherwise.
    The server binds to 0.0.0.0 so it is reachable from any device on the LAN.
    """
    global _server_app, _server_runner, _server_site, _server_port

    if is_running():
        logger.warning("[CastServer] Server is already running")
        return True

    _server_port = port

    try:
        _server_app = _create_app()
        _server_runner = web.AppRunner(_server_app)
        await _server_runner.setup()
        _server_site = web.TCPSite(_server_runner, "0.0.0.0", port)
        await _server_site.start()

        ip = get_local_ip() or "0.0.0.0"
        logger.info(f"[CastServer] Started on http://{ip}:{port}")
        return True

    except Exception as e:
        logger.error(f"[CastServer] Failed to start on port {port}: {e}")
        # Clean up partial state
        if _server_runner is not None:
            try:
                await _server_runner.cleanup()
            except Exception:
                pass
        _server_app = None
        _server_runner = None
        _server_site = None
        return False


async def stop_server() -> None:
    """Stop the cast media server and clean up resources."""
    global _server_app, _server_runner, _server_site

    if not is_running():
        return

    logger.info("[CastServer] Stopping server...")

    try:
        if _server_site is not None:
            await _server_site.stop()
        if _server_runner is not None:
            await _server_runner.cleanup()
    except Exception as e:
        logger.warning(f"[CastServer] Error during shutdown: {e}")
    finally:
        _server_app = None
        _server_runner = None
        _server_site = None

    # Clear all registered media
    _registered_media.clear()
    logger.info("[CastServer] Server stopped")


def is_running() -> bool:
    """Return True if the cast media server is currently running."""
    return _server_site is not None


# ── atexit cleanup ──────────────────────────────────────────────────

def _cleanup_on_exit() -> None:
    """Best-effort cleanup when the process exits."""
    if not is_running():
        return

    logger.info("[CastServer] Cleaning up on exit...")

    # Try to run the async cleanup in whatever loop is available
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule cleanup but can't await it in atexit
            loop.create_task(stop_server())
        else:
            loop.run_until_complete(stop_server())
    except RuntimeError:
        # No event loop available — do synchronous best-effort cleanup
        global _server_app, _server_runner, _server_site
        _server_app = None
        _server_runner = None
        _server_site = None
        _registered_media.clear()


atexit.register(_cleanup_on_exit)
