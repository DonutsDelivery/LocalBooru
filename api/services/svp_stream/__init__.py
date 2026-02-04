"""
SVP (SmoothVideo Project) streaming service.

Uses VapourSynth + SVPflow plugins to interpolate video and stream via HLS.
This is a separate pipeline from the built-in optical flow interpolation.

Architecture:
    Video File -> VapourSynth (SVPflow) -> vspipe -> FFmpeg -> HLS segments -> Browser

SVPflow provides high-quality motion-compensated frame interpolation,
similar to what SVP Manager does for video players.

Requirements:
    - VapourSynth with Python bindings
    - SVPflow plugins (libsvpflow1.so, libsvpflow2.so)
    - ffms2 or lsmas VapourSynth plugin for video loading
    - FFmpeg for HLS encoding
"""
import logging
import subprocess
from typing import Optional

from ..svp_platform import (
    get_svp_plugin_path, get_svp_plugin_full_paths,
    get_source_filter_paths, get_system_python, get_clean_env,
)

# Import all components from submodules
from .config import (
    SVP_PRESETS,
    DEFAULT_SVP_SETTINGS,
    SVP_ALGORITHMS,
    SVP_BLOCK_SIZES,
    SVP_PEL_OPTIONS,
    SVP_MASK_AREA,
    build_svp_params,
    _build_svp_params,  # backward compat alias
)

from .encoder import (
    check_nvenc_available,
    check_nvof_available,
    get_video_info,
    _check_nvenc_available,  # backward compat alias
    _check_nvof_available,   # backward compat alias
)

from .manager import (
    _active_svp_streams,
    get_active_svp_stream,
    register_stream,
    unregister_stream,
    stop_all_svp_streams,
    kill_orphaned_svp_processes,
)

from .svp_integration import (
    generate_vspipe_stdin_script,
    generate_ffmpeg_svp_script,
    generate_svp_script,
    _generate_vspipe_stdin_script,  # backward compat alias
    _generate_ffmpeg_svp_script,    # backward compat alias
    _generate_svp_script,           # backward compat alias
)

from .hls import SVPStream

logger = logging.getLogger(__name__)

# SVP plugin path (resolved per-platform, env override supported)
SVP_PLUGIN_PATH = get_svp_plugin_path() or "/opt/svp/plugins"

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
    nvof_hw_available = check_nvof_available()
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
        "nvenc_available": check_nvenc_available(),
        "nvof_func_available": nvof_func_available,  # SmoothFps_NVOF function exists
        "nvof_hw_available": nvof_hw_available,      # GPU supports NVOF (Turing+)
        "nvof_ready": nvof_ready,                    # Both plugin and hardware ready
        "ready": ready,
        "missing": missing if missing else None,
    }


# Export all public API
__all__ = [
    # Config
    "SVP_PRESETS",
    "DEFAULT_SVP_SETTINGS",
    "SVP_ALGORITHMS",
    "SVP_BLOCK_SIZES",
    "SVP_PEL_OPTIONS",
    "SVP_MASK_AREA",
    "build_svp_params",
    "_build_svp_params",
    # Encoder
    "check_nvenc_available",
    "check_nvof_available",
    "get_video_info",
    "_check_nvenc_available",
    "_check_nvof_available",
    # Manager
    "_active_svp_streams",
    "get_active_svp_stream",
    "register_stream",
    "unregister_stream",
    "stop_all_svp_streams",
    "kill_orphaned_svp_processes",
    # SVP integration
    "generate_vspipe_stdin_script",
    "generate_ffmpeg_svp_script",
    "generate_svp_script",
    "_generate_vspipe_stdin_script",
    "_generate_ffmpeg_svp_script",
    "_generate_svp_script",
    # HLS
    "SVPStream",
    # Status
    "get_svp_status",
    "SVP_PLUGIN_PATH",
    "_get_clean_env",
    "_check_vspipe_available",
]
