"""
Video preview frame extraction service - generates preview frames for video files
"""
import asyncio
import subprocess
import shutil
import platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ..config import get_settings

settings = get_settings()

# Thread pool for video operations - limited to avoid overwhelming the system
# Even with fast-seeking, each ffmpeg process uses CPU/GPU resources
_executor = ThreadPoolExecutor(max_workers=4)

# Cache for ionice/nice availability
_ionice_available = None
_nice_available = None


def _check_ionice_available() -> bool:
    """Check if ionice is available (Linux only)"""
    global _ionice_available
    if _ionice_available is not None:
        return _ionice_available
    _ionice_available = platform.system() == 'Linux' and shutil.which('ionice') is not None
    return _ionice_available


def _check_nice_available() -> bool:
    """Check if nice is available"""
    global _nice_available
    if _nice_available is not None:
        return _nice_available
    _nice_available = shutil.which('nice') is not None
    return _nice_available


def get_low_priority_prefix() -> list[str]:
    """Get command prefix to run at low CPU and I/O priority.

    Uses ionice -c 3 (idle I/O class) and nice -n 19 (lowest CPU priority)
    to ensure background tasks don't interfere with interactive use.

    Based on Jellyfin and other media servers' approach to background processing.
    See: https://github.com/jellyfin/jellyfin/issues/12740
    """
    prefix = []
    if _check_ionice_available():
        prefix.extend(['ionice', '-c', '3'])  # Idle I/O class
    if _check_nice_available():
        prefix.extend(['nice', '-n', '19'])  # Lowest CPU priority
    return prefix

# Semaphore to limit concurrent video preview generation tasks
# This prevents bulk imports from spawning hundreds of ffmpeg processes
_preview_semaphore = None  # Created lazily to avoid event loop issues


def _get_preview_semaphore():
    """Get or create the preview semaphore (lazy init for event loop compatibility)"""
    global _preview_semaphore
    if _preview_semaphore is None:
        _preview_semaphore = asyncio.Semaphore(2)  # Max 2 videos generating previews at once
    return _preview_semaphore

# Cache ffmpeg availability
_ffmpeg_available = None
_nvdec_available = None


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and available"""
    global _ffmpeg_available
    if _ffmpeg_available is not None:
        return _ffmpeg_available

    _ffmpeg_available = shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None
    return _ffmpeg_available


def check_nvdec_available() -> bool:
    """Check if ffmpeg supports NVDEC hardware decoding"""
    global _nvdec_available
    if _nvdec_available is not None:
        return _nvdec_available

    if not check_ffmpeg_available():
        _nvdec_available = False
        return False

    try:
        # Test if ffmpeg can use cuda hwaccel
        result = subprocess.run(
            ['ffmpeg', '-hwaccels'],
            capture_output=True,
            text=True,
            timeout=5
        )
        _nvdec_available = 'cuda' in result.stdout.lower()
        if _nvdec_available:
            print("[VideoPreview] NVDEC/CUDA hardware acceleration available")
        return _nvdec_available
    except Exception:
        _nvdec_available = False
        return False


def get_hwaccel_args() -> list[str]:
    """Get ffmpeg hardware acceleration arguments if available.

    Note: We only use -hwaccel cuda without -hwaccel_output_format cuda
    because the scale filter is CPU-only. Using cuda output format would
    require scale_cuda/scale_npp filters or explicit hwdownload.
    """
    if check_nvdec_available():
        return ['-hwaccel', 'cuda']
    return []


def get_video_duration(video_path: str) -> float | None:
    """Get video duration in seconds using ffprobe"""
    if not check_ffmpeg_available():
        return None

    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"[VideoPreview] Error getting duration: {e}")

    return None


async def get_video_duration_async(video_path: str) -> float | None:
    """Async wrapper for getting video duration"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, get_video_duration, video_path)


def extract_preview_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 8,
    frame_width: int = 400
) -> list[str]:
    """
    Extract evenly-spaced preview frames from a video using batched fast-seeking.

    Uses a SINGLE ffmpeg command with multiple -ss/-i pairs to extract all frames
    at once. This is 3-4x faster than spawning separate processes per frame.
    Uses JPEG for faster encoding, then converts to WebP.

    Args:
        video_path: Path to the video file
        output_dir: Directory to store the frame images
        num_frames: Number of frames to extract (default 8)
        frame_width: Width of the output frames (height auto-scaled)

    Returns:
        List of paths to the extracted frame images
    """
    if not check_ffmpeg_available():
        return []

    # Get video duration
    duration = get_video_duration(video_path)
    if duration is None or duration < 1:
        return []

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate timestamps for evenly-spaced frames
    # Skip first and last 5% to avoid black frames
    start_offset = duration * 0.05
    end_offset = duration * 0.95
    usable_duration = end_offset - start_offset

    if usable_duration <= 0:
        # Very short video, just grab one frame from the middle
        timestamps = [duration / 2]
        num_frames = 1
    else:
        # Calculate evenly spaced timestamps
        timestamps = [
            start_offset + (usable_duration * i / (num_frames - 1))
            for i in range(num_frames)
        ]

    hwaccel_args = get_hwaccel_args()

    # Build a SINGLE ffmpeg command with multiple inputs (batched seeking)
    # This is 3-4x faster than spawning separate ffmpeg processes
    # Format: ffmpeg -ss T0 -i video -ss T1 -i video ... -map 0:v -frames:v 1 out0.jpg -map 1:v -frames:v 1 out1.jpg ...

    # Use low priority to avoid interfering with video playback
    cmd = get_low_priority_prefix() + ['ffmpeg', '-y']

    # Add all inputs with their seek positions
    # Use -skip_frame nokey to only decode keyframes for faster extraction
    for ts in timestamps:
        cmd.extend(['-skip_frame', 'nokey'])
        cmd.extend(hwaccel_args)
        cmd.extend(['-ss', str(ts), '-i', video_path])

    # Add all outputs with stream mapping
    # WebP output - slightly slower to encode but smaller files
    for i in range(len(timestamps)):
        output_file = output_path / f"frame_{i}.webp"
        cmd.extend([
            '-map', f'{i}:v:0',
            '-frames:v', '1',
            '-vf', f'scale={frame_width}:-1',
            '-c:v', 'libwebp',
            '-quality', '80',
            str(output_file)
        ])

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            stderr = result.stderr.decode()[:300] if result.stderr else "unknown error"
            print(f"[VideoPreview] ffmpeg error: {stderr}")
            return []
    except subprocess.TimeoutExpired:
        print(f"[VideoPreview] Timeout extracting frames from {video_path}")
        return []
    except Exception as e:
        print(f"[VideoPreview] Error: {e}")
        return []

    # Collect extracted frames
    extracted = []
    for i in range(len(timestamps)):
        webp_file = output_path / f"frame_{i}.webp"
        if webp_file.exists():
            extracted.append(str(webp_file))

    return extracted


async def extract_preview_frames_async(
    video_path: str,
    output_dir: str,
    num_frames: int = 8,
    frame_width: int = 400
) -> list[str]:
    """Async wrapper for frame extraction"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        extract_preview_frames,
        video_path,
        output_dir,
        num_frames,
        frame_width
    )


def get_preview_dir(file_hash: str) -> Path:
    """Get the preview frames directory for a given file hash"""
    from ..config import get_settings
    settings = get_settings()
    previews_dir = Path(settings.data_dir) / 'previews'
    return previews_dir / file_hash[:16]


def get_preview_frames(file_hash: str) -> list[str]:
    """Get list of existing preview frame paths for a file hash"""
    preview_dir = get_preview_dir(file_hash)
    if not preview_dir.exists():
        return []

    frames = []
    for i in range(8):  # Max 8 frames
        frame_path = preview_dir / f"frame_{i}.webp"
        if frame_path.exists():
            frames.append(str(frame_path))
        else:
            break  # Assume frames are sequential

    return frames


def delete_preview_frames(file_hash: str) -> bool:
    """Delete preview frames for a file hash"""
    preview_dir = get_preview_dir(file_hash)
    if preview_dir.exists():
        try:
            shutil.rmtree(preview_dir)
            return True
        except Exception as e:
            print(f"[VideoPreview] Error deleting preview frames: {e}")
    return False


async def generate_video_previews(video_path: str, file_hash: str, num_frames: int = 8) -> list[str]:
    """
    Generate preview frames for a video file

    Args:
        video_path: Path to the video file
        file_hash: The file's hash (used for directory naming)
        num_frames: Number of frames to extract

    Returns:
        List of paths to extracted frames, empty if generation failed
    """
    if not check_ffmpeg_available():
        print("[VideoPreview] ffmpeg not available, skipping preview generation")
        return []

    preview_dir = get_preview_dir(file_hash)

    # Check if previews already exist
    existing = get_preview_frames(file_hash)
    if existing:
        return existing

    # Use semaphore to limit concurrent preview generation
    # This prevents bulk imports from overwhelming the system with ffmpeg processes
    async with _get_preview_semaphore():
        return await extract_preview_frames_async(
            video_path,
            str(preview_dir),
            num_frames=num_frames
        )


def shutdown():
    """Shutdown the video preview service and cleanup thread pool"""
    print("[VideoPreview] Shutting down...")
    _executor.shutdown(wait=True, cancel_futures=True)
    print("[VideoPreview] Shutdown complete")
