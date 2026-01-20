"""
Video preview frame extraction service - generates preview frames for video files
"""
import asyncio
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ..config import get_settings

settings = get_settings()

# Thread pool for video operations (more workers since ffmpeg is I/O bound)
_executor = ThreadPoolExecutor(max_workers=4)

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


def _extract_single_frame(args: tuple) -> str | None:
    """Extract a single frame at a specific timestamp (for parallel execution)."""
    video_path, output_file, timestamp, frame_width, hwaccel_args = args

    cmd = ['ffmpeg', '-y'] + hwaccel_args + [
        '-ss', str(timestamp),  # Fast seek BEFORE -i
        '-i', video_path,
        '-vframes', '1',
        '-vf', f'scale={frame_width}:-1',
        '-c:v', 'libwebp',
        '-quality', '85',
        str(output_file)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode == 0 and output_file.exists():
            return str(output_file)
    except Exception as e:
        print(f"[VideoPreview] Error extracting frame at {timestamp}s: {e}")

    return None


def extract_preview_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 8,
    frame_width: int = 400
) -> list[str]:
    """
    Extract evenly-spaced preview frames from a video using parallel fast-seeking.

    Uses -ss before -i for fast keyframe seeking, running extractions in parallel
    for maximum speed. This is much faster than the select filter approach which
    requires decoding the entire video.

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

    # Build args for parallel extraction
    extraction_args = [
        (video_path, output_path / f"frame_{i}.webp", ts, frame_width, hwaccel_args)
        for i, ts in enumerate(timestamps)
    ]

    # Extract frames in parallel using thread pool
    # Each extraction uses fast-seeking (-ss before -i) so they're independent
    with ThreadPoolExecutor(max_workers=min(num_frames, 4)) as pool:
        results = list(pool.map(_extract_single_frame, extraction_args))

    # Filter out failed extractions and return in order
    return [r for r in results if r is not None]


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
