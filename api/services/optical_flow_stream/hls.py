"""
HLS segment generation and management for optical flow streaming.

Provides utilities for checking HLS playlist and segment status.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def count_segments(temp_dir: Optional[Path]) -> int:
    """
    Count of HLS segments ready.

    Args:
        temp_dir: HLS output directory

    Returns:
        Number of .ts segment files in directory
    """
    if not temp_dir:
        return 0
    return len(list(temp_dir.glob("segment_*.ts")))


def is_playlist_ready(playlist_path: Optional[Path]) -> bool:
    """
    Check if HLS playlist exists and has content.

    Args:
        playlist_path: Path to m3u8 playlist file

    Returns:
        True if playlist exists and contains segments
    """
    if not playlist_path or not playlist_path.exists():
        return False
    try:
        content = playlist_path.read_text()
        return "segment_" in content and "#EXTINF" in content
    except:
        return False


def get_file_path(temp_dir: Optional[Path], filename: str) -> Optional[Path]:
    """
    Get path to an HLS file (playlist or segment).

    Args:
        temp_dir: HLS output directory
        filename: Name of file to find

    Returns:
        Path to file if it exists, None otherwise
    """
    if not temp_dir:
        return None
    file_path = temp_dir / filename
    if file_path.exists():
        return file_path
    return None
