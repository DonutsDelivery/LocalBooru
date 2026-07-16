"""
Cast media registry and compatibility checking.

Media files are registered here and served through the main FastAPI server
at /api/cast-media/ endpoints (see api/routers/cast.py).
"""
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .network import get_local_ip

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────

_registered_media: Dict[str, dict] = {}


# ── Media registration ──────────────────────────────────────────────

def register_media(
    media_id: str,
    file_path: str,
    hls_dir: Optional[Path] = None,
    subtitle_paths: Optional[list] = None,
) -> None:
    """Register a media item so the cast routes can serve it.

    Args:
        media_id: Unique identifier for the media item.
        file_path: Absolute path to the original media file.
        hls_dir: Optional path to directory containing HLS segments/playlist.
        subtitle_paths: Optional list of Path objects pointing to VTT subtitle files.
    """
    filename = Path(file_path).name
    _registered_media[media_id] = {
        "file_path": file_path,
        "filename": filename,
        "hls_dir": Path(hls_dir) if hls_dir else None,
        "subtitle_paths": [Path(p) for p in subtitle_paths] if subtitle_paths else None,
    }
    logger.info(f"[CastMedia] Registered media {media_id}: {file_path} (filename={filename})")


def unregister_media(media_id: str) -> None:
    """Remove a media item from the registry."""
    if media_id in _registered_media:
        del _registered_media[media_id]
        logger.info(f"[CastMedia] Unregistered media {media_id}")


# ── Chromecast compatibility check ──────────────────────────────────

def _parse_fps(value: str) -> float:
    """Parse ffprobe frame-rate strings like '30000/1001'."""
    if "/" in value:
        num, den = value.split("/", 1)
        den_float = float(den)
        return float(num) / den_float if den_float > 0 else 0.0
    return float(value)


def is_chromecast_compatible(path: str) -> bool:
    """Check whether a video file can be direct-played by Chromecast.

    This intentionally uses a conservative common-denominator profile:
    MP4/MOV, H.264 8-bit 4:2:0, level <= 4.1, AAC/MP3 audio, and
    1080p30 or 720p60 limits. Anything else is sent through HLS transcoding.
    Uses ffprobe to inspect the file.

    Returns True if the file is compatible, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries",
                "format=format_name:stream=codec_name,profile,level,pix_fmt,width,height,avg_frame_rate",
                "-select_streams", "v:0",
                "-of", "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False

        data = json.loads(result.stdout)
        format_name = data.get("format", {}).get("format_name", "").lower()
        # format_name can be comma-separated list like "mov,mp4,m4a,3gp,3g2,mj2"
        compatible_formats = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
        format_parts = {f.strip() for f in format_name.split(",")}
        if not format_parts & compatible_formats:
            return False

        streams = data.get("streams") or []
        if not streams:
            return False

        video = streams[0]
        if str(video.get("codec_name", "")).lower() != "h264":
            return False

        profile = str(video.get("profile", "")).lower()
        if "10" in profile or "4:2:2" in profile or "4:4:4" in profile:
            return False

        pix_fmt = str(video.get("pix_fmt", "")).lower()
        if pix_fmt not in {"yuv420p", "yuvj420p", "nv12"}:
            return False

        level = int(video.get("level") or 0)
        if level > 41:
            return False

        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        if width <= 0 or height <= 0 or width > 1920 or height > 1080:
            return False

        try:
            fps = _parse_fps(str(video.get("avg_frame_rate") or "30/1"))
        except (ValueError, ZeroDivisionError):
            fps = 30.0
        if height > 720 and fps > 30.5:
            return False
        if height <= 720 and fps > 60.5:
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
        if audio_codec and audio_codec not in {"aac", "mp3"}:
            return False

        return True

    except Exception as e:
        logger.warning(f"[CastMedia] ffprobe check failed for {path}: {e}")
        return False


# ── URL construction ────────────────────────────────────────────────

def get_media_url(media_id: str, media_type: str = "file") -> Optional[str]:
    """Construct a LAN-accessible URL for a registered media item.

    Uses the main FastAPI server port so no extra firewall rules are needed.

    Args:
        media_id: The registered media identifier.
        media_type: One of "file", "hls", "subs".

    Returns:
        Full URL string like ``http://192.168.1.100:8790/api/cast-media/{id}/file/media.mp4``,
        or None if the IP cannot be determined.
    """
    from ..routers.settings.models import get_network_settings

    ip = get_local_ip()
    if ip is None:
        return None

    port = get_network_settings()["local_port"]
    base = f"http://{ip}:{port}/api/cast-media/{media_id}"

    if media_type == "file":
        # Include a clean extension in the URL so Chromecast can identify the format.
        entry = _registered_media.get(media_id)
        if entry:
            ext = Path(entry["file_path"]).suffix or ".mp4"
            clean_name = f"media{ext}"
        else:
            clean_name = "media.mp4"
        return f"{base}/file/{clean_name}"
    elif media_type == "hls":
        return f"{base}/hls/playlist.m3u8"
    elif media_type == "subs":
        return f"{base}/subs/"
    else:
        return f"{base}/{media_type}"
