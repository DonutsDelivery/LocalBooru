"""
SVP stream lifecycle management.

Handles stream registry, cleanup, and orphaned process killing.
"""
import atexit
import csv
import io
import logging
import os
import signal
import subprocess
import sys
import threading
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hls import SVPStream

logger = logging.getLogger(__name__)

# Active SVP streams registry (protected by _streams_lock)
_active_svp_streams: Dict[str, 'SVPStream'] = {}
_streams_lock = threading.Lock()


def get_active_svp_stream(stream_id: str) -> Optional['SVPStream']:
    """Get an active SVP stream by ID."""
    with _streams_lock:
        return _active_svp_streams.get(stream_id)


def register_stream(stream: 'SVPStream') -> None:
    """Register a stream in the active streams registry."""
    with _streams_lock:
        _active_svp_streams[stream.stream_id] = stream


def unregister_stream(stream_id: str) -> None:
    """Unregister a stream from the active streams registry."""
    with _streams_lock:
        _active_svp_streams.pop(stream_id, None)


def stop_all_svp_streams() -> None:
    """Stop all active SVP streams."""
    with _streams_lock:
        streams = list(_active_svp_streams.values())
    for stream in streams:
        stream.stop()
    # Also kill any orphaned processes
    kill_orphaned_svp_processes()


def kill_orphaned_svp_processes() -> None:
    """Kill any orphaned SVP-related processes that escaped normal cleanup."""
    try:
        if sys.platform == "win32":
            _kill_orphaned_windows()
        else:
            _kill_orphaned_unix()
    except Exception as e:
        logger.debug(f"Error cleaning up orphaned processes: {e}")


def _kill_orphaned_unix() -> None:
    """Kill orphaned SVP processes on Linux/macOS.

    Only kills processes whose parent is PID 1 (truly orphaned/reparented to init).
    This avoids killing SVP processes belonging to another running backend instance.
    """
    my_pid = str(os.getpid())
    for pattern in ['svp_process\\.(py|vpy)', 'ffmpeg.*svp_stream']:
        result = subprocess.run(
            ['pgrep', '-f', pattern],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if not pid or pid == my_pid:
                    continue
                try:
                    # Check if process is truly orphaned (parent is init/PID 1)
                    # or is a child of our own process
                    ppid_result = subprocess.run(
                        ['ps', '-o', 'ppid=', '-p', pid],
                        capture_output=True, text=True, timeout=2
                    )
                    ppid = ppid_result.stdout.strip()
                    if ppid in ('1', my_pid):
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"Killed orphaned process: {pid} (ppid={ppid})")
                except (ProcessLookupError, ValueError, subprocess.TimeoutExpired):
                    pass


def _kill_orphaned_windows() -> None:
    """Kill orphaned SVP processes on Windows using tasklist + taskkill."""
    for image_name in ["python.exe", "python3.exe", "ffmpeg.exe"]:
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'IMAGENAME eq {image_name}', '/FO', 'CSV', '/V'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                continue
            reader = csv.reader(io.StringIO(result.stdout))
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                pid = row[1].strip().strip('"')
                # Check window title / command line for svp_process or svp_stream
                row_text = " ".join(row).lower()
                if "svp_process" in row_text or "svp_stream" in row_text:
                    subprocess.run(
                        ['taskkill', '/F', '/PID', pid],
                        capture_output=True, timeout=5
                    )
                    logger.info(f"Killed orphaned process: {pid}")
        except Exception:
            pass


def _cleanup_svp_on_exit() -> None:
    """Clean up all SVP streams on process exit."""
    logger.info("[SVP] Cleaning up on exit...")
    stop_all_svp_streams()
    kill_orphaned_svp_processes()


# Register cleanup on exit
atexit.register(_cleanup_svp_on_exit)
