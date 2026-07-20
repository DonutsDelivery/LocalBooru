"""
Web video downloader service using yt-dlp.

Downloads videos from web URLs for SVP processing.
Supports YouTube, Vimeo, Twitch VODs, and direct video URLs.
"""
import asyncio
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Cache directory for downloaded videos
CACHE_DIR = Path(tempfile.gettempdir()) / "localbooru_web_videos"
CACHE_DIR.mkdir(exist_ok=True)

# Active downloads registry
_active_downloads: Dict[str, "WebVideoDownload"] = {}

# DRM-protected sites that cannot be downloaded
DRM_SITES = {
    # Streaming services
    "netflix.com",
    "disneyplus.com",
    "hulu.com",
    "max.com",  # HBO Max rebranded
    "hbomax.com",
    "primevideo.com",
    "amazon.com/gp/video",
    "peacocktv.com",
    "paramountplus.com",
    "tv.apple.com",
    "appletv.com",
    # Regional streaming
    "stan.com.au",
    "binge.com.au",
    "crave.ca",
    "britbox.com",
    "now.com",
    "sky.com",
    # Sports
    "espn.com/watch",
    "dazn.com",
}

# Known supported sites (not exhaustive, yt-dlp supports many more)
SUPPORTED_SITES = {
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "twitch.tv",  # VODs only
    "dailymotion.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
}


@dataclass
class DownloadResult:
    """Result of a video download operation."""
    success: bool
    download_id: str
    file_path: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


@dataclass
class WebVideoDownload:
    """Tracks an active or completed download."""
    download_id: str
    url: str
    quality: str
    file_path: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    progress: float = 0.0
    status: str = "pending"  # pending, downloading, processing, complete, error
    error: Optional[str] = None
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)


def _url_to_cache_key(url: str, quality: str) -> str:
    """Generate a cache key from URL and quality."""
    # Normalize URL for caching
    normalized = url.lower().strip()
    # Remove tracking parameters
    for param in ["?si=", "&si=", "?utm", "&utm", "?ref", "&ref"]:
        if param in normalized:
            normalized = normalized.split(param)[0]

    key = f"{normalized}:{quality}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_cached_video(url: str, quality: str) -> Optional[Path]:
    """Check if video is already cached."""
    cache_key = _url_to_cache_key(url, quality)
    for ext in [".mp4", ".webm", ".mkv"]:
        cached = CACHE_DIR / f"{cache_key}{ext}"
        if cached.exists():
            logger.info(f"Cache hit for {url[:50]}...")
            return cached
    return None


def is_drm_site(url: str) -> bool:
    """Check if URL is from a known DRM-protected site."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check against known DRM sites
        for drm_site in DRM_SITES:
            if drm_site in domain or domain.endswith(drm_site.split("/")[0]):
                return True

        # Check for DRM-protected subdomains/paths
        full_url = f"{domain}{parsed.path}".lower()
        for drm_site in DRM_SITES:
            if drm_site in full_url:
                return True

    except Exception:
        pass

    return False


def is_live_stream(url: str) -> Optional[bool]:
    """
    Check if URL is a live stream (returns None if can't determine).

    Note: This is a heuristic check. Some URLs may need actual yt-dlp
    extraction to determine if they're live.
    """
    url_lower = url.lower()

    # YouTube live indicators
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        if "/live" in url_lower or "live=1" in url_lower:
            return True

    # Twitch live (not VOD)
    if "twitch.tv" in url_lower:
        # VODs have /videos/ in the path
        if "/videos/" not in url_lower:
            return True

    return None  # Can't determine from URL alone


def get_download(download_id: str) -> Optional[WebVideoDownload]:
    """Get a download by ID."""
    return _active_downloads.get(download_id)


def get_download_progress(download_id: str) -> Optional[float]:
    """Get download progress (0.0 to 1.0)."""
    download = _active_downloads.get(download_id)
    if download:
        return download.progress
    return None


async def download_video(url: str, quality: str = "best") -> DownloadResult:
    """
    Download a video from a web URL using yt-dlp.

    Args:
        url: Video URL (YouTube, Vimeo, Twitch VOD, direct video, etc.)
        quality: Quality preference - "best", "1080p", "720p", "480p"

    Returns:
        DownloadResult with file path on success or error message on failure.
    """
    # Generate download ID
    download_id = _url_to_cache_key(url, quality)

    # Check if already downloading
    if download_id in _active_downloads:
        existing = _active_downloads[download_id]
        if existing.status == "complete" and existing.file_path:
            return DownloadResult(
                success=True,
                download_id=download_id,
                file_path=existing.file_path,
                title=existing.title,
                duration=existing.duration,
            )
        elif existing.status == "error":
            return DownloadResult(
                success=False,
                download_id=download_id,
                error=existing.error or "Download failed",
            )
        elif existing.status in ("pending", "downloading", "processing"):
            # Return the in-progress download ID so frontend can poll
            return DownloadResult(
                success=True,
                download_id=download_id,
            )

    # Check DRM
    if is_drm_site(url):
        return DownloadResult(
            success=False,
            download_id=download_id,
            error="This site uses DRM protection and cannot be downloaded",
        )

    # Check for live streams (v2 feature)
    is_live = is_live_stream(url)
    if is_live:
        return DownloadResult(
            success=False,
            download_id=download_id,
            error="Live streams are not supported yet (coming in v2)",
        )

    # Check cache
    cached = _get_cached_video(url, quality)
    if cached:
        _active_downloads[download_id] = WebVideoDownload(
            download_id=download_id,
            url=url,
            quality=quality,
            file_path=str(cached),
            progress=1.0,
            status="complete",
        )
        return DownloadResult(
            success=True,
            download_id=download_id,
            file_path=str(cached),
        )

    # Check if yt-dlp is available
    if not shutil.which("yt-dlp"):
        return DownloadResult(
            success=False,
            download_id=download_id,
            error="yt-dlp is not installed. Install with: pip install yt-dlp",
        )

    # Create download tracker
    download = WebVideoDownload(
        download_id=download_id,
        url=url,
        quality=quality,
        status="pending",
    )
    _active_downloads[download_id] = download

    # Start download in background
    asyncio.create_task(_do_download(download))

    return DownloadResult(
        success=True,
        download_id=download_id,
    )


async def _do_download(download: WebVideoDownload):
    """Execute the download using yt-dlp."""
    try:
        download.status = "downloading"

        # Build quality format string
        if download.quality == "best":
            format_str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        elif download.quality == "1080p":
            format_str = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best"
        elif download.quality == "720p":
            format_str = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best"
        elif download.quality == "480p":
            format_str = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best"
        else:
            format_str = "best"

        # Output path
        output_path = CACHE_DIR / f"{download.download_id}.%(ext)s"

        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "--no-playlist",  # Don't download playlists
            "-f", format_str,
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-mtime",  # Don't set file modification time
            "--no-part",   # Don't use .part files
            "--progress",
            "--newline",   # Progress on separate lines for parsing
            download.url,
        ]

        logger.info(f"[Download {download.download_id}] Starting: {' '.join(cmd[:6])}...")

        # Run yt-dlp
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        download._proc = proc

        # Parse progress from output
        async def read_output():
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                line = line.decode().strip()

                # Parse progress percentage
                if "[download]" in line and "%" in line:
                    try:
                        # Extract percentage from lines like "[download]  45.3% of 123.45MiB"
                        match = re.search(r"(\d+\.?\d*)%", line)
                        if match:
                            download.progress = float(match.group(1)) / 100.0
                    except (ValueError, AttributeError):
                        pass

                # Extract title
                if "[info]" in line and "Downloading" in line:
                    # [info] Downloading 1 format(s): title
                    pass

        await read_output()
        await proc.wait()

        if proc.returncode != 0:
            stderr = await proc.stderr.read()
            error_msg = stderr.decode()[:500] if stderr else "Download failed"
            download.status = "error"
            download.error = error_msg
            logger.error(f"[Download {download.download_id}] Failed: {error_msg}")
            return

        # Find the downloaded file
        download.status = "processing"
        for ext in [".mp4", ".webm", ".mkv"]:
            output_file = CACHE_DIR / f"{download.download_id}{ext}"
            if output_file.exists():
                download.file_path = str(output_file)
                download.progress = 1.0
                download.status = "complete"
                logger.info(f"[Download {download.download_id}] Complete: {output_file}")
                return

        # File not found
        download.status = "error"
        download.error = "Download completed but file not found"
        logger.error(f"[Download {download.download_id}] File not found after download")

    except asyncio.CancelledError:
        download.status = "error"
        download.error = "Download cancelled"
        if download._proc:
            download._proc.terminate()
    except Exception as e:
        download.status = "error"
        download.error = str(e)
        logger.error(f"[Download {download.download_id}] Error: {e}")


def cancel_download(download_id: str) -> bool:
    """Cancel an active download."""
    download = _active_downloads.get(download_id)
    if download and download._proc:
        download._proc.terminate()
        download.status = "error"
        download.error = "Cancelled"
        return True
    return False


def cleanup_cache(max_age_hours: int = 24):
    """Remove cached videos older than max_age_hours."""
    import time
    now = time.time()
    max_age_seconds = max_age_hours * 3600

    for file in CACHE_DIR.iterdir():
        if file.is_file():
            age = now - file.stat().st_mtime
            if age > max_age_seconds:
                try:
                    file.unlink()
                    logger.info(f"Cleaned up old cache file: {file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file.name}: {e}")
