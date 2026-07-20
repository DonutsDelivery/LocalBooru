"""
FFmpeg encoding pipeline for optical flow streaming.

Handles FFmpeg command construction and process management for HLS output.
Supports both NVENC hardware encoding and libx264 software encoding.
"""
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Check for NVENC availability (cached at module load)
_NVENC_AVAILABLE: Optional[bool] = None


def check_nvenc_available() -> bool:
    """Check if NVENC hardware encoding is available."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    try:
        # Test if ffmpeg can use nvenc
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        _NVENC_AVAILABLE = 'h264_nvenc' in result.stdout
        if _NVENC_AVAILABLE:
            logger.info("NVENC hardware encoder detected")
        else:
            logger.info("NVENC not available, using libx264")
    except Exception as e:
        logger.debug(f"NVENC check failed: {e}")
        _NVENC_AVAILABLE = False

    return _NVENC_AVAILABLE


def build_ffmpeg_command(
    width: int,
    height: int,
    target_fps: int,
    output_dir: Path,
    playlist_path: Path,
    use_nvenc: bool = True,
    target_bitrate: Optional[str] = None,
    target_resolution: Optional[tuple] = None,
    stream_id: str = ""
) -> List[str]:
    """
    Build FFmpeg command with optimal encoder selection.

    Args:
        width: Input frame width
        height: Input frame height
        target_fps: Target output frame rate
        output_dir: Directory for HLS segment output
        playlist_path: Path to HLS playlist file
        use_nvenc: Whether to use NVENC hardware encoding
        target_bitrate: Target bitrate (e.g., "4M", "1536K")
        target_resolution: Target resolution (width, height) or None
        stream_id: Stream ID for logging

    Returns:
        List of command arguments for subprocess
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(target_fps),
        '-i', '-',  # Read from stdin
    ]

    # Add scaling filter if needed
    if target_resolution:
        target_width, target_height = target_resolution
        cmd.extend([
            '-vf', f'scale={target_width}:{target_height}:flags=lanczos'
        ])

    # Select encoder
    if use_nvenc:
        # NVENC hardware encoding
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',  # Fastest NVENC preset
            '-tune', 'll',   # Low latency
        ])

        # Use CBR mode for predictable bitrate, or VBR for original quality
        if target_bitrate:
            # CBR mode for consistent bitrate
            cmd.extend([
                '-rc', 'cbr',
                '-b:v', target_bitrate,
                '-maxrate', target_bitrate,
                '-bufsize', f'{int(target_bitrate.rstrip("MK")) * 2}M' if 'M' in target_bitrate else f'{int(target_bitrate.rstrip("MK")) * 2}K',
            ])
        else:
            # VBR mode (original quality)
            cmd.extend([
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '0',     # VBR mode
            ])

        logger.info(f"[Stream {stream_id}] Using NVENC hardware encoder")
    else:
        # Software encoding
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
        ])

        if target_bitrate:
            # Use specified bitrate
            cmd.extend([
                '-b:v', target_bitrate,
                '-bufsize', f'{int(target_bitrate.rstrip("MK")) * 2}M' if 'M' in target_bitrate else f'{int(target_bitrate.rstrip("MK")) * 2}K',
            ])
        else:
            # Use CRF for original quality (default)
            cmd.extend([
                '-crf', '23',
            ])

    # Common output options
    cmd.extend([
        '-g', str(target_fps * 2),  # GOP size = 2 seconds
        '-f', 'hls',
        '-hls_time', '2',
        '-hls_list_size', '10',
        '-hls_flags', 'delete_segments+append_list',
        '-hls_segment_filename', str(output_dir / 'segment_%03d.ts'),
        str(playlist_path)
    ])

    return cmd


def start_ffmpeg_process(
    ffmpeg_cmd: List[str]
) -> subprocess.Popen:
    """
    Start FFmpeg process for HLS encoding.

    Args:
        ffmpeg_cmd: Command arguments from build_ffmpeg_command

    Returns:
        Popen process with stdin pipe for frame input
    """
    return subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )


def finish_ffmpeg(ffmpeg_proc: Optional[subprocess.Popen], stream_id: str = ""):
    """
    Gracefully close FFmpeg process.

    Args:
        ffmpeg_proc: FFmpeg subprocess to close
        stream_id: Stream ID for logging
    """
    if ffmpeg_proc:
        try:
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
        except Exception as e:
            logger.warning(f"[Stream {stream_id}] FFmpeg cleanup error: {e}")
