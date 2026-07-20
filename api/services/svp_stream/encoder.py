"""
FFmpeg encoding pipeline for SVP streams.

Handles video info extraction, hardware encoder detection, and encoding configuration.
"""
import logging
import subprocess

logger = logging.getLogger(__name__)


def check_nvenc_available() -> bool:
    """Check if NVENC hardware encoding is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


def check_nvof_available() -> bool:
    """
    Check if NVIDIA Optical Flow (NVOF) hardware is available.

    NVOF requires:
    - NVIDIA GPU with Turing architecture or newer (RTX 20xx, 30xx, 40xx)
    - NVIDIA driver with NVOF support
    """
    try:
        # Check GPU compute capability (Turing = 7.5+, Ada = 8.9)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip()
            # Parse compute capability (e.g., "8.9" or "7.5")
            try:
                major, minor = map(int, compute_cap.split('.'))
                # Turing (7.5) and newer support NVOF
                return major > 7 or (major == 7 and minor >= 5)
            except ValueError:
                pass
    except Exception:
        pass
    return False


def get_video_info(video_path: str) -> dict:
    """Get video dimensions, FPS, frame count, and duration using ffprobe.

    Returns:
        dict with keys: width, height, src_fps, src_fps_num, src_fps_den,
                       num_frames, duration, success
    """
    result = {
        "width": 0,
        "height": 0,
        "src_fps": 0.0,
        "src_fps_num": 0,
        "src_fps_den": 1,
        "num_frames": 0,
        "duration": 0.0,
        "success": False,
    }

    try:
        proc_result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            video_path
        ], capture_output=True, text=True, timeout=10)

        if proc_result.returncode != 0:
            logger.error(f"ffprobe failed: {proc_result.stderr}")
            return result

        lines = proc_result.stdout.strip().split('\n')
        if len(lines) >= 1:
            parts = lines[0].split(',')
            if len(parts) >= 3:
                result["width"] = int(parts[0])
                result["height"] = int(parts[1])
                # Parse frame rate (e.g., "30000/1001" or "30/1")
                fps_parts = parts[2].split('/')
                if len(fps_parts) == 2:
                    result["src_fps_num"] = int(fps_parts[0])
                    result["src_fps_den"] = int(fps_parts[1])
                    result["src_fps"] = result["src_fps_num"] / result["src_fps_den"]
                else:
                    result["src_fps"] = float(fps_parts[0])
                    result["src_fps_num"] = int(result["src_fps"] * 1000)
                    result["src_fps_den"] = 1000

            # Get frame count if available
            if len(parts) >= 4 and parts[3]:
                try:
                    result["num_frames"] = int(parts[3])
                except ValueError:
                    pass

        # Get duration from format line
        if len(lines) >= 2 and lines[1]:
            try:
                result["duration"] = float(lines[1])
            except ValueError:
                pass

        # Estimate frame count from duration if not available
        if result["num_frames"] == 0 and result["duration"] > 0 and result["src_fps"] > 0:
            result["num_frames"] = int(result["duration"] * result["src_fps"])

        result["success"] = True
        return result

    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return result


# Aliases for backward compatibility
_check_nvenc_available = check_nvenc_available
_check_nvof_available = check_nvof_available
