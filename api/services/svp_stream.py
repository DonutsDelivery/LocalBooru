"""
SVP (SmoothVideo Project) streaming service.

Uses VapourSynth + SVPflow plugins to interpolate video and stream via HLS.
This is a separate pipeline from the built-in optical flow interpolation.

Architecture:
    Video File → VapourSynth (SVPflow) → vspipe → FFmpeg → HLS segments → Browser

SVPflow provides high-quality motion-compensated frame interpolation,
similar to what SVP Manager does for video players.

Requirements:
    - VapourSynth with Python bindings
    - SVPflow plugins (libsvpflow1.so, libsvpflow2.so)
    - ffms2 or lsmas VapourSynth plugin for video loading
    - FFmpeg for HLS encoding
"""
import atexit
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List

from .svp_platform import (
    get_svp_plugin_path, get_svp_plugin_full_paths,
    get_source_filter_paths, get_system_python, get_clean_env,
)

logger = logging.getLogger(__name__)

# SVP plugin path (resolved per-platform, env override supported)
SVP_PLUGIN_PATH = get_svp_plugin_path() or "/opt/svp/plugins"

# Active SVP streams registry
_active_svp_streams: Dict[str, 'SVPStream'] = {}


def _cleanup_svp_on_exit():
    """Clean up all SVP streams on process exit."""
    logger.info("[SVP] Cleaning up on exit...")
    stop_all_svp_streams()
    kill_orphaned_svp_processes()


# Register cleanup on exit
atexit.register(_cleanup_svp_on_exit)

# Check for vspipe availability
_VSPIPE_AVAILABLE: Optional[bool] = None


def _get_clean_env() -> dict:
    """
    Get a clean environment for running vspipe/VapourSynth commands.

    Delegates to the platform module for cross-platform support.
    """
    return get_clean_env()


def _check_vspipe_available() -> bool:
    """Check if vspipe (VapourSynth pipe) is available."""
    global _VSPIPE_AVAILABLE
    if _VSPIPE_AVAILABLE is not None:
        return _VSPIPE_AVAILABLE

    try:
        result = subprocess.run(
            ['vspipe', '--version'],
            capture_output=True,
            text=True,
            timeout=5,
            env=_get_clean_env()
        )
        _VSPIPE_AVAILABLE = result.returncode == 0
        if _VSPIPE_AVAILABLE:
            logger.info(f"vspipe available: {result.stdout.strip()}")
        else:
            logger.debug(f"vspipe check failed: {result.stderr}")
    except Exception as e:
        logger.debug(f"vspipe check failed: {e}")
        _VSPIPE_AVAILABLE = False

    return _VSPIPE_AVAILABLE


def _check_nvenc_available() -> bool:
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


def _check_nvof_available() -> bool:
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


def get_svp_status() -> dict:
    """
    Get SVP availability status.

    Uses vspipe subprocess to check capabilities since VapourSynth Python
    module may be installed for a different Python version than the venv.
    """
    vspipe_available = _check_vspipe_available()

    # Check VapourSynth and plugins via vspipe subprocess
    vs_available = False
    vs_version = None
    svp_available = False
    nvof_func_available = False
    ffms2_available = False
    lsmas_available = False
    bestsource_available = False

    if vspipe_available:
        # Resolve platform-specific paths for injection into subprocess script
        _flow1_path, _flow2_path = get_svp_plugin_full_paths()
        _src_filters = get_source_filter_paths()
        _sys_python = get_system_python()

        # Build source filter list literal for the test script
        _sf_literal = ", ".join(
            f'("{p}", "{k}", "{a}")' for p, k, a in _src_filters
        )

        # Create a test script to check all capabilities
        test_script = f'''
import vapoursynth as vs
import json
import sys

results = {{
    "vs_version": vs.__version__,
    "svp_available": False,
    "nvof_func_available": False,
    "ffms2_available": False,
    "lsmas_available": False,
    "bestsource_available": False,
}}

core = vs.core

# Check SVP plugins
try:
    core.std.LoadPlugin("{_flow1_path}")
    core.std.LoadPlugin("{_flow2_path}")
    results["svp_available"] = True
    # Check if SmoothFps_NVOF function exists
    if hasattr(core.svp2, 'SmoothFps_NVOF'):
        results["nvof_func_available"] = True
except:
    pass

# Check source filters
plugin_paths = [{_sf_literal}]

for path, key, attr in plugin_paths:
    if hasattr(core, attr):
        results[key] = True
    else:
        try:
            core.std.LoadPlugin(path)
            results[key] = True
        except:
            pass

print(json.dumps(results))
'''
        try:
            # Run the test script via system Python with clean environment
            result = subprocess.run(
                [_sys_python, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=10,
                env=_get_clean_env()
            )
            if result.returncode == 0:
                import json as json_mod
                data = json_mod.loads(result.stdout.strip())
                vs_available = True
                vs_version = data.get("vs_version")
                svp_available = data.get("svp_available", False)
                nvof_func_available = data.get("nvof_func_available", False)
                ffms2_available = data.get("ffms2_available", False)
                lsmas_available = data.get("lsmas_available", False)
                bestsource_available = data.get("bestsource_available", False)
        except Exception as e:
            logger.debug(f"VapourSynth check failed: {e}")

    source_filter_available = ffms2_available or lsmas_available or bestsource_available

    # Check NVOF hardware availability (Turing GPU or newer)
    nvof_hw_available = _check_nvof_available()
    # NVOF is fully ready if both the plugin function and hardware support it
    nvof_ready = nvof_func_available and nvof_hw_available

    # Determine ready status and any missing requirements
    # Note: We use FFmpeg decode, so source_filter isn't strictly required
    # but it's nice to have for the native vspipe path
    ready = vs_available and svp_available and vspipe_available

    missing = []
    if not vs_available:
        missing.append("VapourSynth")
    if not svp_available:
        missing.append("SVPflow plugins")
    if not vspipe_available:
        missing.append("vspipe")

    return {
        "vapoursynth_available": vs_available,
        "vapoursynth_version": vs_version,
        "svp_plugins_available": svp_available,
        "svp_plugin_path": SVP_PLUGIN_PATH,
        "vspipe_available": vspipe_available,
        "ffms2_available": ffms2_available,
        "lsmas_available": lsmas_available,
        "bestsource_available": bestsource_available,
        "source_filter_available": source_filter_available,
        "nvenc_available": _check_nvenc_available(),
        "nvof_func_available": nvof_func_available,  # SmoothFps_NVOF function exists
        "nvof_hw_available": nvof_hw_available,      # GPU supports NVOF (Turing+)
        "nvof_ready": nvof_ready,                    # Both plugin and hardware ready
        "ready": ready,
        "missing": missing if missing else None,
    }


def get_active_svp_stream(stream_id: str) -> Optional['SVPStream']:
    """Get an active SVP stream by ID."""
    return _active_svp_streams.get(stream_id)


def stop_all_svp_streams():
    """Stop all active SVP streams."""
    for stream in list(_active_svp_streams.values()):
        stream.stop()
    # Also kill any orphaned processes
    kill_orphaned_svp_processes()


def kill_orphaned_svp_processes():
    """Kill any orphaned SVP-related processes that escaped normal cleanup."""
    import signal
    try:
        if sys.platform == "win32":
            _kill_orphaned_windows()
        else:
            _kill_orphaned_unix()
    except Exception as e:
        logger.debug(f"Error cleaning up orphaned processes: {e}")


def _kill_orphaned_unix():
    """Kill orphaned SVP processes on Linux/macOS using pgrep + SIGKILL."""
    import signal
    for pattern in ['svp_process\\.(py|vpy)', 'ffmpeg.*svp_stream']:
        result = subprocess.run(
            ['pgrep', '-f', pattern],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"Killed orphaned process: {pid}")
                    except (ProcessLookupError, ValueError):
                        pass


def _kill_orphaned_windows():
    """Kill orphaned SVP processes on Windows using tasklist + taskkill."""
    import csv
    import io
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


# =============================================================================
# SVP Quality Presets
# =============================================================================
# Based on SVPflow documentation: https://www.svp-team.com/wiki/Manual:SVPflow
#
# Key parameters:
# - pel: motion estimation accuracy (1=pixel, 2=half-pixel, 4=quarter-pixel)
# - block.w/h: block size (8=detailed/slow, 16=balanced, 32=fast/coarse)
# - overlap: block overlap (0=none, 1=1/8, 2=1/4, 3=1/2)
# - algo: rendering algorithm (13=uniform, 23=adaptive - best quality)
# - mask.area: artifact masking (0-100, higher = less smoothing but fewer artifacts)
# =============================================================================

SVP_PRESETS = {
    # Fastest preset - good for real-time playback on weaker GPUs
    "fast": {
        "name": "Fast",
        "description": "Fastest processing, lower quality. Good for real-time on weaker hardware.",
        "super": "{gpu:1,pel:1,scale:{up:0,down:4}}",
        "analyse": "{gpu:1,block:{w:32,h:32,overlap:0},main:{search:{coarse:{type:2,distance:-6,bad:{sad:2000,range:24}},type:2,distance:6}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:13,mask:{area:100},scene:{}}",
        "nvof_blk": 32,  # Larger blocks for faster NVOF processing
    },
    # Balanced preset - good tradeoff between speed and quality
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of speed and quality. Recommended for most videos.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
        "nvof_blk": 16,  # Balanced block size for NVOF
    },
    # Quality preset - higher quality, slower processing
    "quality": {
        "name": "Quality",
        "description": "Higher quality motion estimation. Slower but smoother results.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
        "nvof_blk": 8,   # Smaller blocks for higher quality NVOF
    },
    # Maximum quality preset - best results, significantly slower
    "max": {
        "name": "Maximum",
        "description": "Maximum quality settings. Best for pre-rendering, not real-time.",
        "super": "{gpu:1,pel:4,scale:{up:2,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:3},main:{search:{coarse:{type:4,distance:-12,bad:{sad:1000,range:24}},type:4,distance:12}},refine:[{thsad:200},{thsad:100},{thsad:50}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:50,cover:80},scene:{}}",
        "nvof_blk": 4,   # Smallest blocks for maximum quality NVOF
    },
    # Animation preset - optimized for anime/cartoons
    "animation": {
        "name": "Animation",
        "description": "Optimized for anime and cartoons with flat colors and sharp edges.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:1500,range:24}},type:2,distance:10},penalty:{lambda:3.0}},refine:[{thsad:150}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:150},scene:{mode:0}}",
        "nvof_blk": 16,  # Balanced for animation
    },
    # Film preset - optimized for live action with natural motion
    "film": {
        "name": "Film",
        "description": "Optimized for live action movies with natural motion blur.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:80,cover:80},scene:{blend:true}}",
        "nvof_blk": 16,  # Balanced for film
    },
}

# Default SVP settings
DEFAULT_SVP_SETTINGS = {
    "enabled": False,
    "preset": "balanced",
    "target_fps": 60,
    "use_gpu": True,
    # Key settings (override preset values)
    "use_nvof": True,           # Use NVIDIA Optical Flow
    "shader": 23,               # SVP shader/algo (13=uniform, 23=adaptive)
    "artifact_masking": 100,    # Artifact masking area (0=off, 50-200)
    "frame_interpolation": 2,   # Frame interpolation mode (1=uniform, 2=adaptive)
    # Advanced settings (full override when set)
    "custom_super": None,
    "custom_analyse": None,
    "custom_smooth": None,
}

# Available algorithms for UI dropdown
SVP_ALGORITHMS = {
    1: "Block-based (fastest)",
    2: "Block-based with masking",
    11: "Pixel-based (smoother)",
    13: "Pixel-based uniform (recommended)",
    21: "Pixel-based with masking",
    23: "Pixel-based adaptive (best quality)",
}

# Block sizes for UI dropdown
SVP_BLOCK_SIZES = {
    8: "8x8 (highest quality, slowest)",
    16: "16x16 (balanced)",
    32: "32x32 (fastest, lower quality)",
}

# Motion accuracy (pel) for UI dropdown
SVP_PEL_OPTIONS = {
    1: "Pixel (fastest)",
    2: "Half-pixel (recommended)",
    4: "Quarter-pixel (highest quality)",
}

# Mask area settings for UI
SVP_MASK_AREA = {
    0: "Off (maximum smoothness)",
    50: "Low (smoother)",
    100: "Medium (balanced)",
    150: "High (fewer artifacts)",
    200: "Maximum (least smoothing)",
}


def _build_svp_params(
    target_fps: int,
    preset: str = "balanced",
    shader: int = 23,
    artifact_masking: int = 100,
    frame_interpolation: int = 2,
    custom_super: Optional[str] = None,
    custom_analyse: Optional[str] = None,
    custom_smooth: Optional[str] = None,
) -> tuple:
    """Build SVP parameters from preset and overrides.

    Note: NVOF (NVIDIA Optical Flow) is handled at the pipeline level by using
    SmoothFps_NVOF instead of Super/Analyse/SmoothFps. These params are only
    used for the regular (non-NVOF) pipeline.
    """
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    # Use custom params if provided, otherwise use preset
    if custom_super:
        super_params = custom_super
    else:
        super_params = preset_config["super"]

    if custom_analyse:
        analyse_params = custom_analyse
    else:
        analyse_params = preset_config["analyse"]

    if custom_smooth:
        smooth_params = custom_smooth
    else:
        # Build smooth params with rate, algorithm, and masking
        algo = shader  # shader setting maps to SVP algorithm
        smooth_params = f"{{rate:{{num:{target_fps},den:1,abs:true}},gpuid:0,algo:{algo},mask:{{area:{artifact_masking}}},scene:{{}}}}"

    return super_params, analyse_params, smooth_params


def _generate_vspipe_stdin_script(
    target_fps: int,
    preset: str = "balanced",
    use_nvof: bool = True,
    shader: int = 23,
    artifact_masking: int = 100,
    frame_interpolation: int = 2,
    custom_super: Optional[str] = None,
    custom_analyse: Optional[str] = None,
    custom_smooth: Optional[str] = None,
) -> str:
    """
    Generate a VapourSynth script that reads Y4M from stdin and outputs SVP-processed Y4M.

    This script is meant to be run with vspipe, reading from FFmpeg's Y4M output:
        ffmpeg -i video.mp4 -f yuv4mpegpipe - | vspipe -c y4m script.vpy - | ffmpeg -f yuv4mpegpipe -i - ...

    NO PYTHON IN THE FRAME PATH - rawsource reads directly from stdin.
    """
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    # Get SVP parameters
    super_params, analyse_params, smooth_params = _build_svp_params(
        target_fps, preset, shader, artifact_masking, frame_interpolation,
        custom_super, custom_analyse, custom_smooth
    )

    # Get NVOF block size for this preset
    nvof_blk = preset_config.get("nvof_blk", 16)

    # Resolve platform-specific SVP plugin paths
    _flow1_path, _flow2_path = get_svp_plugin_full_paths()

    # Rawsource plugin path (user-installed)
    rawsource_path = os.path.expanduser("~/.local/lib/vapoursynth/libvsrawsource.so")

    if use_nvof:
        svp_processing = f'''
# NVOF Pipeline - uses NVIDIA Optical Flow hardware
# Pick optimal vec_src ratio based on resolution
NVOF_MIN_WIDTH = 160
NVOF_MIN_HEIGHT = 128

for ratio in [8, 6, 4, 2, 1]:
    test_w = clip.width // ratio
    test_h = clip.height // ratio
    test_w = (test_w // 2) * 2
    test_h = (test_h // 2) * 2
    if test_w >= NVOF_MIN_WIDTH and test_h >= NVOF_MIN_HEIGHT:
        if {nvof_blk} >= 16 and ratio <= 4:
            break
        elif {nvof_blk} >= 8 and ratio <= 2:
            break
        elif ratio <= 1:
            break

new_w = clip.width // ratio
new_h = clip.height // ratio
new_w = (new_w // 2) * 2
new_h = (new_h // 2) * 2

if new_w < NVOF_MIN_WIDTH or new_h < NVOF_MIN_HEIGHT:
    new_w = (clip.width // 2) * 2
    new_h = (clip.height // 2) * 2

nvof_src = clip.resize.Bicubic(new_w, new_h)
smooth = core.svp2.SmoothFps_NVOF(clip, '{smooth_params}', vec_src=nvof_src, src=clip, fps=src_fps)
'''
    else:
        svp_processing = f'''
# Regular SVP Pipeline - CPU motion estimation, GPU rendering
super_clip = core.svp1.Super(clip, '{super_params}')
vectors = core.svp1.Analyse(super_clip["clip"], super_clip["data"], clip, '{analyse_params}')
smooth = core.svp2.SmoothFps(clip, super_clip["clip"], super_clip["data"],
    vectors["clip"], vectors["data"], '{smooth_params}', src=clip, fps=src_fps)
'''

    script = f'''import vapoursynth as vs
core = vs.core

# Load plugins
core.std.LoadPlugin("{rawsource_path}")
core.std.LoadPlugin("{_flow1_path}")
core.std.LoadPlugin("{_flow2_path}")

# Read Y4M from stdin - NO PYTHON FRAME HANDLING
clip = core.raws.Source("-")

# Get source fps for SVP
src_fps = float(clip.fps)

# Convert to YUV420P8 if needed
if clip.format.id != vs.YUV420P8:
    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_in_s="709", matrix_s="709")

{svp_processing}

smooth.set_output()
'''
    return script


def _generate_ffmpeg_svp_script(
    video_path: str,
    target_fps: int,
    width: int,
    height: int,
    src_fps_num: int,
    src_fps_den: int,
    num_frames: int,
    preset: str = "balanced",
    use_nvof: bool = True,
    shader: int = 23,
    artifact_masking: int = 100,
    frame_interpolation: int = 2,
    custom_super: Optional[str] = None,
    custom_analyse: Optional[str] = None,
    custom_smooth: Optional[str] = None,
    start_position: float = 0.0,
) -> str:
    """
    Generate a Python script that uses FFmpeg to decode video and feeds frames
    to VapourSynth for SVP processing. This bypasses bestsource indexing entirely.

    Supports two pipelines:
    - NVOF (use_nvof=True): Uses SmoothFps_NVOF with hardware optical flow (2x faster)
    - Regular (use_nvof=False): Uses Super/Analyse/SmoothFps pipeline

    Output is Y4M to stdout for FFmpeg to encode to HLS.
    """
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    # Get SVP parameters (for regular pipeline fallback)
    super_params, analyse_params, smooth_params = _build_svp_params(
        target_fps, preset, shader, artifact_masking, frame_interpolation,
        custom_super, custom_analyse, custom_smooth
    )

    # Get NVOF block size for this preset
    nvof_blk = preset_config.get("nvof_blk", 16)

    # Resolve platform-specific SVP plugin paths for injection
    _flow1_path, _flow2_path = get_svp_plugin_full_paths()

    escaped_path = video_path.replace("\\", "\\\\").replace("'", "\\'")
    src_fps = src_fps_num / src_fps_den

    # Calculate frames to skip and remaining frames
    start_frame = int(start_position * src_fps)
    remaining_frames = max(1, num_frames - start_frame)

    # Build FFmpeg seek arguments using hybrid seeking for accuracy:
    # - Input seek (-ss before -i) for fast approximate positioning
    # - Output seek (-ss after -i) for frame-accurate final positioning
    if start_position > 2:
        # Seek to 2 seconds before target with input seeking (fast, keyframe-based)
        # Then use output seeking for the remaining 2 seconds (accurate)
        input_seek = start_position - 2
        output_seek = 2.0
        ffmpeg_input_seek = f"'-ss', '{input_seek:.3f}', "
        ffmpeg_output_seek = f"'-ss', '{output_seek:.3f}', "
    elif start_position > 0:
        # For short seeks, just use output seeking (accurate but acceptable speed)
        ffmpeg_input_seek = ""
        ffmpeg_output_seek = f"'-ss', '{start_position:.3f}', "
    else:
        ffmpeg_input_seek = ""
        ffmpeg_output_seek = ""

    # Build the SVP processing section based on NVOF setting
    if use_nvof:
        svp_processing = f'''
# ================================================================
# NVOF Pipeline (NVIDIA Optical Flow Hardware Accelerator)
# ================================================================
# Uses dedicated optical flow hardware on RTX 20xx+ GPUs
# NVOF vec_src MUST be exactly 1/1, 1/2, 1/4, 1/6, or 1/8 of source size

# NVOF minimum vec_src requirements: 40 blocks × 4 pixels = 160 width, 32 blocks × 4 pixels = 128 height
NVOF_MIN_WIDTH = 160
NVOF_MIN_HEIGHT = 128

# Pick ratio based on block size emulation setting
# nvof_blk=16 -> 1/4, nvof_blk=8 -> 1/2, etc.
nvof_blk = {nvof_blk}

# Try ratios from largest (fastest) to smallest until we meet NVOF minimums
for ratio in [8, 6, 4, 2, 1]:
    test_w = WIDTH // ratio
    test_h = HEIGHT // ratio
    # Ensure even dimensions
    test_w = (test_w // 2) * 2
    test_h = (test_h // 2) * 2
    if test_w >= NVOF_MIN_WIDTH and test_h >= NVOF_MIN_HEIGHT:
        # Also respect the nvof_blk setting - don't use larger ratio than requested
        if nvof_blk >= 16 and ratio <= 4:
            break
        elif nvof_blk >= 8 and ratio <= 2:
            break
        elif ratio <= 1:
            break

new_w = WIDTH // ratio
new_h = HEIGHT // ratio

# Ensure even dimensions for video processing
new_w = (new_w // 2) * 2
new_h = (new_h // 2) * 2

# Final safety check - if still below minimum, use source resolution
if new_w < NVOF_MIN_WIDTH or new_h < NVOF_MIN_HEIGHT:
    new_w = (WIDTH // 2) * 2
    new_h = (HEIGHT // 2) * 2
    ratio = 1

print(f"[NVOF] Video: {{WIDTH}}x{{HEIGHT}}, vec_src: {{new_w}}x{{new_h}} (1/{{ratio}} ratio)", file=sys.stderr)

# Prepare NVOF vector source (exact fraction of source)
nvof_src = clip.resize.Bicubic(new_w, new_h)

smooth_params = '{smooth_params}'

# Use NVIDIA Optical Flow for motion estimation and interpolation
smooth = core.svp2.SmoothFps_NVOF(clip, smooth_params, vec_src=nvof_src, src=clip, fps=SRC_FPS)
'''
    else:
        svp_processing = f'''
# ================================================================
# Regular SVP Pipeline (GPU-accelerated frame rendering)
# ================================================================
# Uses CPU for motion estimation, GPU for frame rendering

super_params = '{super_params}'
analyse_params = '{analyse_params}'
smooth_params = '{smooth_params}'

# SVP processing pipeline
super_clip = core.svp1.Super(clip, super_params)
vectors = core.svp1.Analyse(super_clip["clip"], super_clip["data"], clip, analyse_params)
smooth = core.svp2.SmoothFps(clip, super_clip["clip"], super_clip["data"],
    vectors["clip"], vectors["data"], smooth_params, src=clip, fps=SRC_FPS)
'''

    script = f'''#!/usr/bin/env python3
"""SVP processing script - uses FFmpeg decode to bypass bestsource indexing."""
import vapoursynth as vs
import subprocess
import numpy as np
import sys
import signal
import atexit
import threading
from collections import deque

# Video parameters (pre-computed)
VIDEO_PATH = '{escaped_path}'
WIDTH = {width}
HEIGHT = {height}
FPS_NUM = {src_fps_num}
FPS_DEN = {src_fps_den}
NUM_FRAMES = {remaining_frames}
SRC_FPS = {src_fps}
TARGET_FPS = {target_fps}
START_POSITION = {start_position}

# Frame size for YUV420P
Y_SIZE = WIDTH * HEIGHT
UV_SIZE = (WIDTH // 2) * (HEIGHT // 2)
FRAME_SIZE = Y_SIZE + 2 * UV_SIZE

# Buffer size: read ahead up to 30 frames to keep GPU fed
BUFFER_SIZE = 30

# Initialize VapourSynth
core = vs.core
core.std.LoadPlugin("{_flow1_path}")
core.std.LoadPlugin("{_flow2_path}")

# Start FFmpeg with hardware decoding (NVDEC/VAAPI auto-fallback to software)
# Uses hybrid seeking: input seek for fast positioning, output seek for accuracy
ffmpeg_proc = subprocess.Popen([
    'ffmpeg',
    '-hwaccel', 'auto',        # Auto-select best hardware decoder (NVDEC/VAAPI/etc)
    '-threads', '0',           # Use all CPU threads for software decode fallback
    {ffmpeg_input_seek}'-i', VIDEO_PATH,
    {ffmpeg_output_seek}'-f', 'rawvideo', '-pix_fmt', 'yuv420p',
    '-'
], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=FRAME_SIZE * 4)

# Cleanup handler to ensure FFmpeg is terminated
def cleanup_ffmpeg():
    try:
        ffmpeg_proc.terminate()
        ffmpeg_proc.wait(timeout=2)
    except:
        try:
            ffmpeg_proc.kill()
        except:
            pass

# Register cleanup for normal exit and signals
atexit.register(cleanup_ffmpeg)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

# Debug/performance metrics - writes to log file for real-time monitoring
import time as _time

# Log file path - fixed location for easy monitoring
_LOG_PATH = '/tmp/svp_perf.log'

class PerfMetrics:
    def __init__(self):
        self.start_time = _time.time()
        self.decode_times = []  # FFmpeg read times
        self.process_times = []  # VS frame get times
        self.output_times = []  # Y4M write times
        self.buffer_levels = []
        self.frame_count = 0
        self.last_report = _time.time()
        self.stalls = 0  # times we had to wait for buffer
        # Clear log file on start
        with open(_LOG_PATH, 'w') as f:
            f.write(f"[SVP PERF] Started at {{_time.strftime('%H:%M:%S')}}\\n")
            f.write(f"[SVP PERF] Video: {{WIDTH}}x{{HEIGHT}} @ {{SRC_FPS:.2f}}fps -> {{TARGET_FPS}}fps\\n")
            f.write(f"[SVP PERF] Frame size: {{FRAME_SIZE / 1024 / 1024:.1f}} MB\\n")

    def log(self, msg):
        """Write to log file (unbuffered)."""
        with open(_LOG_PATH, 'a') as f:
            f.write(msg + '\\n')

    def report(self, buffer_fill, force=False):
        """Print performance report every 2 seconds."""
        now = _time.time()
        if not force and now - self.last_report < 2.0:
            return
        self.last_report = now

        elapsed = now - self.start_time
        overall_fps = self.frame_count / elapsed if elapsed > 0 else 0

        avg_decode = sum(self.decode_times[-100:]) / len(self.decode_times[-100:]) * 1000 if self.decode_times else 0
        avg_process = sum(self.process_times[-100:]) / len(self.process_times[-100:]) * 1000 if self.process_times else 0
        avg_output = sum(self.output_times[-100:]) / len(self.output_times[-100:]) * 1000 if self.output_times else 0

        decode_fps = 1000 / avg_decode if avg_decode > 0 else float('inf')
        process_fps = 1000 / avg_process if avg_process > 0 else float('inf')

        self.log(f"[PERF] frame={{self.frame_count}} buf={{buffer_fill}}/{{BUFFER_SIZE}} "
                 f"fps={{overall_fps:.1f}} stalls={{self.stalls}}")
        self.log(f"       decode={{avg_decode:.1f}}ms ({{decode_fps:.0f}}fps) "
                 f"process={{avg_process:.1f}}ms ({{process_fps:.0f}}fps) "
                 f"output={{avg_output:.1f}}ms")

metrics = PerfMetrics()

# Threaded frame reader for parallel decoding
class FrameReader:
    def __init__(self, proc, frame_size, buffer_size):
        self.proc = proc
        self.frame_size = frame_size
        self.frames = {{}}  # frame_num -> data
        self.next_read = 0
        self.eof = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.buffer_size = buffer_size
        self.last_requested = -1

        # Start reader thread
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        """Background thread that reads frames ahead."""
        while not self.eof:
            with self.lock:
                # Wait if buffer is full (too far ahead of consumer)
                while (self.next_read - self.last_requested > self.buffer_size
                       and not self.eof and self.last_requested >= 0):
                    self.condition.wait(timeout=0.1)

            # Read next frame outside lock
            t0 = _time.time()
            data = self.proc.stdout.read(self.frame_size)
            decode_time = _time.time() - t0

            if len(data) < self.frame_size:
                with self.lock:
                    self.eof = True
                    self.condition.notify_all()
                break

            with self.lock:
                self.frames[self.next_read] = data
                metrics.decode_times.append(decode_time)
                self.next_read += 1
                self.condition.notify_all()

    def get_frame(self, n):
        """Get frame n, blocking until available."""
        with self.lock:
            self.last_requested = max(self.last_requested, n)
            self.condition.notify_all()  # Wake reader if it was waiting

            # Wait for frame to be available
            waited = False
            while n not in self.frames and not self.eof:
                waited = True
                self.condition.wait(timeout=0.5)

            if waited:
                metrics.stalls += 1

            data = self.frames.get(n)
            buffer_fill = len(self.frames)

            # Clean old frames to save memory
            for old_n in list(self.frames.keys()):
                if old_n < n - 5:
                    del self.frames[old_n]

            return data, buffer_fill

# Create threaded frame reader
frame_reader = FrameReader(ffmpeg_proc, FRAME_SIZE, BUFFER_SIZE)
_last_buffer_fill = 0

def read_frame_data(n):
    """Read frame n from threaded buffer."""
    global _last_buffer_fill
    data, buffer_fill = frame_reader.get_frame(n)
    _last_buffer_fill = buffer_fill
    return data

def modify_frame(n, f):
    """Replace blank frame with FFmpeg decoded frame."""
    frame_data = read_frame_data(n)
    if frame_data is None:
        return f.copy()

    fout = f.copy()

    # Copy Y plane
    y_arr = np.frombuffer(frame_data[:Y_SIZE], dtype=np.uint8).reshape(HEIGHT, WIDTH)
    np.copyto(np.asarray(fout[0]), y_arr)

    # Copy U plane
    u_arr = np.frombuffer(frame_data[Y_SIZE:Y_SIZE+UV_SIZE], dtype=np.uint8).reshape(HEIGHT//2, WIDTH//2)
    np.copyto(np.asarray(fout[1]), u_arr)

    # Copy V plane
    v_arr = np.frombuffer(frame_data[Y_SIZE+UV_SIZE:], dtype=np.uint8).reshape(HEIGHT//2, WIDTH//2)
    np.copyto(np.asarray(fout[2]), v_arr)

    return fout

# Create clip from FFmpeg decoded frames
blank = core.std.BlankClip(width=WIDTH, height=HEIGHT, format=vs.YUV420P8,
                           length=NUM_FRAMES, fpsnum=FPS_NUM, fpsden=FPS_DEN)
clip = core.std.ModifyFrame(blank, blank, modify_frame)
{svp_processing}
# Output Y4M to stdout
def write_y4m_header():
    """Write Y4M header."""
    header = f"YUV4MPEG2 W{{smooth.width}} H{{smooth.height}} F{{TARGET_FPS}}:1 Ip A0:0 C420\\n"
    sys.stdout.buffer.write(header.encode())
    sys.stdout.buffer.flush()

def write_y4m_frame(frame):
    """Write a single Y4M frame."""
    sys.stdout.buffer.write(b"FRAME\\n")
    for plane_idx in range(3):
        plane = frame[plane_idx]
        arr = np.asarray(plane)
        sys.stdout.buffer.write(arr.tobytes())
    sys.stdout.buffer.flush()

# Main output loop
write_y4m_header()
for i in range(len(smooth)):
    try:
        # Time VapourSynth frame processing
        t0 = _time.time()
        frame = smooth.get_frame(i)
        process_time = _time.time() - t0
        metrics.process_times.append(process_time)

        # Time Y4M output
        t0 = _time.time()
        write_y4m_frame(frame)
        output_time = _time.time() - t0
        metrics.output_times.append(output_time)

        metrics.frame_count += 1
        metrics.report(_last_buffer_fill)

    except Exception as e:
        print(f"Frame {{i}} error: {{e}}", file=sys.stderr)
        break

# Final report
metrics.report(_last_buffer_fill, force=True)
ffmpeg_proc.terminate()
'''
    return script


def _generate_svp_script(
    video_path: str,
    target_fps: int,
    preset: str = "balanced",
    use_nvof: bool = True,
    shader: int = 23,
    artifact_masking: int = 100,
    frame_interpolation: int = 2,
    custom_super: Optional[str] = None,
    custom_analyse: Optional[str] = None,
    custom_smooth: Optional[str] = None,
) -> str:
    """
    Generate a VapourSynth script for SVP interpolation.

    Args:
        video_path: Path to input video
        target_fps: Target output frame rate
        preset: Quality preset (fast, balanced, quality, max, animation, film)
        use_nvof: Use NVIDIA Optical Flow for motion estimation
        shader: SVP shader/algorithm (13=uniform, 23=adaptive)
        artifact_masking: Artifact masking area (0=off, 50-200)
        frame_interpolation: Frame interpolation mode (1=uniform, 2=adaptive)
        custom_super: Custom super params (full override)
        custom_analyse: Custom analyse params (full override)
        custom_smooth: Custom smooth params (full override)

    Returns:
        VapourSynth script as string
    """
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    # Use custom params if provided, otherwise build from preset + overrides
    if custom_super:
        super_params = custom_super
    else:
        super_params = preset_config["super"]

    if custom_analyse:
        analyse_params = custom_analyse
    else:
        # Start with preset and override nvof setting
        analyse_params = preset_config["analyse"]
        # Update nvof setting
        if use_nvof and "nvof:" not in analyse_params:
            analyse_params = analyse_params.replace("{gpu:1,", "{gpu:1,nvof:1,")
        elif not use_nvof:
            analyse_params = analyse_params.replace(",nvof:1", "").replace("nvof:1,", "")

    if custom_smooth:
        smooth_params = custom_smooth
    else:
        # Build smooth params with user settings
        # Shader algo combines with frame_interpolation: algo = shader_base + frame_mode
        # shader 13/23 are pixel-based, frame_interpolation 1=uniform 2=adaptive
        # Actually algo is just the shader value directly
        algo = shader
        smooth_params = f"{{gpuid:0,algo:{algo},mask:{{area:{artifact_masking}}},scene:{{}}}}"

    # Inject target FPS into smooth params
    if "rate:" not in smooth_params:
        smooth_params = smooth_params.replace(
            "{",
            f"{{rate:{{num:{target_fps},den:1,abs:true}},",
            1
        )

    # Resolve platform-specific paths for injection
    _fb_flow1, _fb_flow2 = get_svp_plugin_full_paths()
    _fb_src_filters = get_source_filter_paths()
    # Build source plugin list literal: [('path', 'ns'), ...]
    _fb_sf_literal = ", ".join(
        f"('{p}', '{a}')" for p, _, a in _fb_src_filters
    )

    # Escape path for Python string
    escaped_path = video_path.replace("\\", "\\\\").replace('"', '\\"')

    script = f'''
import vapoursynth as vs
core = vs.core

# Load SVP plugins
core.std.LoadPlugin("{_fb_flow1}")
core.std.LoadPlugin("{_fb_flow2}")

# Try to load source filter plugins from system paths
source_plugins = [{_fb_sf_literal}]
for plugin_path, _ in source_plugins:
    try:
        core.std.LoadPlugin(plugin_path)
    except:
        pass

# Load video source - try available source filters in order of preference
clip = None

# Try bestsource (most reliable for seeking)
if clip is None and hasattr(core, 'bs'):
    try:
        clip = core.bs.VideoSource("{escaped_path}")
    except Exception as e:
        print(f"bestsource failed: {{e}}")

# Try ffms2
if clip is None and hasattr(core, 'ffms2'):
    try:
        clip = core.ffms2.Source("{escaped_path}")
    except Exception as e:
        print(f"ffms2 failed: {{e}}")

# Try lsmas
if clip is None and hasattr(core, 'lsmas'):
    try:
        clip = core.lsmas.LWLibavSource("{escaped_path}")
    except Exception as e:
        print(f"lsmas failed: {{e}}")

if clip is None:
    raise RuntimeError("No video source filter available. Install vapoursynth-plugin-bestsource.")

# Get source FPS
src_fps = clip.fps.numerator / clip.fps.denominator
target_fps = {target_fps}

# Convert to YUV420P8 for SVP processing (required format)
# Specify both input and output matrix to handle videos with unspecified colorspace
clip_yuv = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_in_s="709", matrix_s="709")

# SVP parameters (preset: {preset})
super_params = '{super_params}'
analyse_params = '{analyse_params}'
smooth_params = '{smooth_params}'

# SVP processing pipeline
# 1. Create super clip (hierarchical representation for motion estimation)
super_clip = core.svp1.Super(clip_yuv, super_params)

# 2. Analyze motion vectors
vectors = core.svp1.Analyse(
    super_clip["clip"],
    super_clip["data"],
    clip_yuv,
    analyse_params
)

# 3. Interpolate frames
smooth = core.svp2.SmoothFps(
    clip_yuv,
    super_clip["clip"],
    super_clip["data"],
    vectors["clip"],
    vectors["data"],
    smooth_params,
    src=clip_yuv,
    fps=src_fps
)

# Keep as YUV420P8 for Y4M output (Y4M only supports YUV/Gray formats)
# FFmpeg will handle any necessary color space conversion
smooth.set_output()
'''
    return script


class SVPStream:
    """
    Manages an SVP-interpolated video stream.

    Pipeline: Video → VapourSynth/SVPflow → vspipe → FFmpeg → HLS segments

    This uses vspipe to run the VapourSynth script and pipe raw frames
    directly to FFmpeg for HLS encoding. Much more efficient than
    frame-by-frame Python processing.
    """

    def __init__(
        self,
        video_path: str,
        target_fps: int = 60,
        preset: str = "balanced",
        use_nvenc: Optional[bool] = None,
        use_nvof: bool = True,
        shader: int = 23,
        artifact_masking: int = 100,
        frame_interpolation: int = 2,
        custom_super: Optional[str] = None,
        custom_analyse: Optional[str] = None,
        custom_smooth: Optional[str] = None,
        start_position: float = 0.0,  # Seek position in seconds
        target_bitrate: Optional[str] = None,  # Target bitrate (e.g., "4M", "1536K")
        target_resolution: Optional[tuple] = None,  # Target resolution (width, height)
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.preset = preset if preset in SVP_PRESETS else "balanced"
        self.use_nvenc = use_nvenc if use_nvenc is not None else _check_nvenc_available()
        self.stream_id = str(uuid.uuid4())[:8]
        self.start_position = start_position  # Where to start processing from

        # Quality settings
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution

        # Key SVP settings
        self.use_nvof = use_nvof
        self.shader = shader
        self.artifact_masking = artifact_masking
        self.frame_interpolation = frame_interpolation

        # Custom SVP parameters (full override when set)
        self.custom_super = custom_super
        self.custom_analyse = custom_analyse
        self.custom_smooth = custom_smooth

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._decode_proc: Optional[subprocess.Popen] = None
        self._vspipe_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._temp_dir: Optional[Path] = None
        self._script_path: Optional[Path] = None
        self._error: Optional[str] = None
        self._start_time: Optional[float] = None

        # Video info (populated on start)
        self._width: int = 0
        self._height: int = 0
        self._src_fps: float = 0
        self._src_fps_num: int = 0
        self._src_fps_den: int = 1
        self._num_frames: int = 0
        self._duration: float = 0  # Source video duration in seconds

        # Register stream
        _active_svp_streams[self.stream_id] = self

    @property
    def hls_dir(self) -> Optional[Path]:
        """Get the HLS output directory."""
        return self._temp_dir

    @property
    def playlist_path(self) -> Optional[Path]:
        """Get path to the HLS playlist file."""
        if self._temp_dir:
            return self._temp_dir / "stream.m3u8"
        return None

    @property
    def is_running(self) -> bool:
        """Check if the stream is active."""
        return self._running

    @property
    def error(self) -> Optional[str]:
        """Get any error that occurred."""
        return self._error

    @property
    def segments_ready(self) -> int:
        """Count of HLS segments ready."""
        if not self._temp_dir:
            return 0
        return len(list(self._temp_dir.glob("segment_*.ts")))

    @property
    def playlist_ready(self) -> bool:
        """Check if HLS playlist exists and has content."""
        if not self.playlist_path or not self.playlist_path.exists():
            return False
        try:
            content = self.playlist_path.read_text()
            return "segment_" in content and "#EXTINF" in content
        except:
            return False

    def _get_video_info(self) -> bool:
        """Get video dimensions, FPS, frame count, and duration using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                self.video_path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return False

            lines = result.stdout.strip().split('\n')
            if len(lines) >= 1:
                parts = lines[0].split(',')
                if len(parts) >= 3:
                    self._width = int(parts[0])
                    self._height = int(parts[1])
                    # Parse frame rate (e.g., "30000/1001" or "30/1")
                    fps_parts = parts[2].split('/')
                    if len(fps_parts) == 2:
                        self._src_fps_num = int(fps_parts[0])
                        self._src_fps_den = int(fps_parts[1])
                        self._src_fps = self._src_fps_num / self._src_fps_den
                    else:
                        self._src_fps = float(fps_parts[0])
                        self._src_fps_num = int(self._src_fps * 1000)
                        self._src_fps_den = 1000

                # Get frame count if available
                if len(parts) >= 4 and parts[3]:
                    try:
                        self._num_frames = int(parts[3])
                    except ValueError:
                        pass

            # Get duration from format line
            if len(lines) >= 2 and lines[1]:
                try:
                    self._duration = float(lines[1])
                except ValueError:
                    pass

            # Estimate frame count from duration if not available
            if self._num_frames == 0 and self._duration > 0 and self._src_fps > 0:
                self._num_frames = int(self._duration * self._src_fps)

            logger.info(f"[SVP {self.stream_id}] Video: {self._width}x{self._height} @ {self._src_fps:.2f}fps, {self._num_frames} frames, {self._duration:.1f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")

        return False

    async def start(self) -> bool:
        """Start the SVP interpolated stream."""
        if self._running:
            return True

        # Get video info
        if not self._get_video_info():
            self._error = "Failed to get video info"
            return False

        # Check if interpolation is actually needed
        # If source fps is within 5% of target fps, SVP is pointless
        fps_ratio = self.target_fps / self._src_fps if self._src_fps > 0 else 2.0
        if 0.95 <= fps_ratio <= 1.05:
            self._error = f"Source fps ({self._src_fps:.2f}) already at target ({self.target_fps}fps), interpolation not needed"
            logger.warning(f"[SVP {self.stream_id}] {self._error}")
            return False

        # Create temp directory for HLS output
        self._temp_dir = Path(tempfile.mkdtemp(prefix='svp_stream_'))
        logger.info(f"[SVP {self.stream_id}] HLS output: {self._temp_dir}")

        # Generate vspipe stdin script (reads Y4M from stdin, no Python frame handling)
        script = _generate_vspipe_stdin_script(
            self.target_fps,
            self.preset,
            use_nvof=self.use_nvof,
            shader=self.shader,
            artifact_masking=self.artifact_masking,
            frame_interpolation=self.frame_interpolation,
            custom_super=self.custom_super,
            custom_analyse=self.custom_analyse,
            custom_smooth=self.custom_smooth,
        )
        self._script_path = self._temp_dir / "svp_stdin.vpy"
        self._script_path.write_text(script)
        logger.debug(f"[SVP {self.stream_id}] Script written to {self._script_path}")

        # Start the pipeline
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._run_pipeline())

        return True

    async def _run_pipeline(self):
        """Run the FFmpeg → vspipe → FFmpeg pipeline.

        Three-stage pipeline with NO PYTHON in the frame path:
        1. FFmpeg decodes video to Y4M (hardware accelerated)
        2. vspipe reads Y4M from stdin, processes with SVP, outputs Y4M
        3. FFmpeg encodes Y4M to HLS

        This is the same architecture that native SVP uses for real-time playback.
        """
        try:
            # Stage 1: FFmpeg decode to Y4M
            decode_cmd = [
                'ffmpeg',
                '-hwaccel', 'auto',  # Use NVDEC/VAAPI if available
                '-threads', '0',
            ]

            # Add seek if needed
            if self.start_position > 0:
                decode_cmd.extend(['-ss', str(self.start_position)])

            decode_cmd.extend([
                '-i', self.video_path,
            ])

            # Downscale BEFORE SVP if quality preset specifies resolution
            # This makes SVP process smaller frames = much faster
            if self.target_resolution:
                width, height = self.target_resolution
                decode_cmd.extend(['-vf', f'scale={width}:{height}:flags=lanczos'])

            decode_cmd.extend([
                '-f', 'yuv4mpegpipe',
                '-pix_fmt', 'yuv420p',
                '-'
            ])

            # Stage 2: vspipe with SVP processing
            vspipe_cmd = [
                'vspipe',
                '-c', 'y4m',
                str(self._script_path),
                '-'
            ]

            # Stage 3: FFmpeg encode to HLS
            encode_cmd = [
                'ffmpeg',
                '-y',
                '-probesize', '32',
                '-analyzeduration', '0',
                '-fflags', '+nobuffer+flush_packets',
                '-f', 'yuv4mpegpipe',
                '-i', '-',  # Y4M from vspipe
            ]

            # Add audio input with seek if needed
            if self.start_position > 0:
                encode_cmd.extend(['-ss', str(self.start_position)])
            encode_cmd.extend([
                '-probesize', '5000000',
                '-i', self.video_path,  # Original video for audio
                '-map', '0:v',
                '-map', '1:a?',
            ])

            # Pad to even dimensions (required for H.264)
            # Note: Scaling already done in decode stage before SVP
            encode_cmd.extend(['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])

            # Video encoder selection
            if self.use_nvenc:
                encode_cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p1',
                    '-tune', 'll',
                ])
                if self.target_bitrate:
                    encode_cmd.extend([
                        '-rc', 'cbr',
                        '-b:v', self.target_bitrate,
                        '-maxrate', self.target_bitrate,
                        '-bufsize', f'{int(self.target_bitrate.rstrip("MK")) * 2}M' if 'M' in self.target_bitrate else f'{int(self.target_bitrate.rstrip("MK")) * 2}K',
                    ])
                else:
                    encode_cmd.extend(['-rc', 'vbr', '-cq', '23', '-b:v', '0'])
            else:
                encode_cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency'])
                if self.target_bitrate:
                    encode_cmd.extend(['-b:v', self.target_bitrate])
                else:
                    encode_cmd.extend(['-crf', '23'])

            # Audio encoder
            encode_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

            # HLS output with MPEG-TS segments
            encode_cmd.extend([
                '-g', str(self.target_fps * 2),  # Keyframe every 2 seconds
                '-keyint_min', str(self.target_fps),
                '-f', 'hls',
                '-hls_time', '4',
                '-hls_list_size', '0',
                '-hls_flags', 'append_list+split_by_time',
                '-hls_segment_filename', str(self._temp_dir / 'segment_%03d.ts'),
                str(self.playlist_path)
            ])

            logger.info(f"[SVP {self.stream_id}] Starting pipeline: FFmpeg decode → vspipe/SVP → FFmpeg encode")
            logger.debug(f"[SVP {self.stream_id}] Decode: {' '.join(decode_cmd)}")
            logger.debug(f"[SVP {self.stream_id}] vspipe: {' '.join(vspipe_cmd)}")
            logger.debug(f"[SVP {self.stream_id}] Encode: {' '.join(encode_cmd)}")

            # Start the three-stage pipeline
            clean_env = _get_clean_env()

            # Stage 1: FFmpeg decode
            self._decode_proc = subprocess.Popen(
                decode_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )

            # Stage 2: vspipe (reads from decode, writes to encode)
            self._vspipe_proc = subprocess.Popen(
                vspipe_cmd,
                stdin=self._decode_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )
            self._decode_proc.stdout.close()  # Allow SIGPIPE

            # Stage 3: FFmpeg encode
            self._ffmpeg_proc = subprocess.Popen(
                encode_cmd,
                stdin=self._vspipe_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )
            self._vspipe_proc.stdout.close()  # Allow SIGPIPE

            # Monitor the three processes
            while self._running:
                # Check decode process
                if self._decode_proc and self._decode_proc.poll() is not None:
                    decode_stderr = self._decode_proc.stderr.read().decode() if self._decode_proc.stderr else ""
                    if self._decode_proc.returncode != 0:
                        self._error = f"Decode FFmpeg exited with code {self._decode_proc.returncode}: {decode_stderr[-500:]}"
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                        break

                # Check vspipe process
                if self._vspipe_proc and self._vspipe_proc.poll() is not None:
                    vspipe_stderr = self._vspipe_proc.stderr.read().decode() if self._vspipe_proc.stderr else ""
                    if self._vspipe_proc.returncode != 0:
                        self._error = f"vspipe exited with code {self._vspipe_proc.returncode}: {vspipe_stderr[-1000:]}"
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                        print(f"[SVP {self.stream_id}] vspipe stderr:\n{vspipe_stderr}")
                        break

                # Check encode FFmpeg process
                if self._ffmpeg_proc and self._ffmpeg_proc.poll() is not None:
                    ffmpeg_stderr = self._ffmpeg_proc.stderr.read().decode() if self._ffmpeg_proc.stderr else ""
                    if self._ffmpeg_proc.returncode != 0:
                        # Collect all stderr for debugging
                        vspipe_stderr = ""
                        decode_stderr = ""
                        if self._vspipe_proc and self._vspipe_proc.stderr:
                            try:
                                vspipe_stderr = self._vspipe_proc.stderr.read().decode()
                            except:
                                pass
                        if self._decode_proc and self._decode_proc.stderr:
                            try:
                                decode_stderr = self._decode_proc.stderr.read().decode()
                            except:
                                pass
                        combined_error = f"Encode FFmpeg exit {self._ffmpeg_proc.returncode}"
                        if decode_stderr:
                            combined_error += f"\nDecode stderr: {decode_stderr[-300:]}"
                        if vspipe_stderr:
                            combined_error += f"\nvspipe stderr: {vspipe_stderr[-300:]}"
                        combined_error += f"\nEncode stderr: {ffmpeg_stderr[-300:]}"
                        self._error = combined_error
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                    else:
                        logger.info(f"[SVP {self.stream_id}] Pipeline finished normally")
                    break

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info(f"[SVP {self.stream_id}] Pipeline cancelled")
        except Exception as e:
            self._error = str(e)
            logger.error(f"[SVP {self.stream_id}] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            self._cleanup_processes()

    def _cleanup_processes(self):
        """Clean up subprocess resources."""
        # Clean up decode process
        if self._decode_proc:
            try:
                self._decode_proc.terminate()
                self._decode_proc.wait(timeout=2)
            except:
                try:
                    self._decode_proc.kill()
                except:
                    pass
            self._decode_proc = None

        # Clean up vspipe process
        if self._vspipe_proc:
            try:
                self._vspipe_proc.terminate()
                self._vspipe_proc.wait(timeout=2)
            except:
                try:
                    self._vspipe_proc.kill()
                except:
                    pass
            self._vspipe_proc = None

        # Clean up encode FFmpeg process
        if self._ffmpeg_proc:
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=2)
            except:
                try:
                    self._ffmpeg_proc.kill()
                except:
                    pass
            self._ffmpeg_proc = None

    def stop(self):
        """Stop the stream."""
        logger.info(f"[SVP {self.stream_id}] Stopping stream")
        self._running = False

        if self._task:
            self._task.cancel()
            self._task = None

        self._cleanup_processes()

        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
                logger.debug(f"[SVP {self.stream_id}] Cleaned up {self._temp_dir}")
            except Exception as e:
                logger.warning(f"[SVP {self.stream_id}] Failed to clean up: {e}")
            self._temp_dir = None

        # Unregister stream
        if self.stream_id in _active_svp_streams:
            del _active_svp_streams[self.stream_id]

    def get_stats(self) -> dict:
        """Get stream statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "stream_id": self.stream_id,
            "video_path": self.video_path,
            "target_fps": self.target_fps,
            "preset": self.preset,
            "resolution": f"{self._width}x{self._height}",
            "src_fps": self._src_fps,
            "duration": self._duration,  # Source video duration
            "running": self._running,
            "elapsed_seconds": elapsed,
            "segments_ready": self.segments_ready,
            "playlist_ready": self.playlist_ready,
            "error": self._error,
            "use_nvenc": self.use_nvenc,
        }
