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
import asyncio
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# SVP plugin path
SVP_PLUGIN_PATH = "/opt/svp/plugins"

# Active SVP streams registry
_active_svp_streams: Dict[str, 'SVPStream'] = {}

# Check for vspipe availability
_VSPIPE_AVAILABLE: Optional[bool] = None


def _get_clean_env() -> dict:
    """
    Get a clean environment for running vspipe/VapourSynth commands.

    VapourSynth's vspipe can fail when run from a Python venv due to
    environment variable conflicts. This returns a clean environment
    with only essential variables.
    """
    home = os.environ.get('HOME', '/tmp')
    return {
        'PATH': '/usr/bin:/bin:/usr/local/bin',
        'HOME': home,
        'USER': os.environ.get('USER', 'user'),
        'LANG': os.environ.get('LANG', 'en_US.UTF-8'),
        # Required for GPU access
        'DISPLAY': os.environ.get('DISPLAY', ':0'),
        'XDG_RUNTIME_DIR': os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}'),
    }


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
    ffms2_available = False
    lsmas_available = False
    bestsource_available = False

    if vspipe_available:
        # Create a test script to check all capabilities
        test_script = f'''
import vapoursynth as vs
import json
import sys

results = {{
    "vs_version": vs.__version__,
    "svp_available": False,
    "ffms2_available": False,
    "lsmas_available": False,
    "bestsource_available": False,
}}

core = vs.core

# Check SVP plugins
try:
    core.std.LoadPlugin("{SVP_PLUGIN_PATH}/libsvpflow1.so")
    core.std.LoadPlugin("{SVP_PLUGIN_PATH}/libsvpflow2.so")
    results["svp_available"] = True
except:
    pass

# Check source filters
plugin_paths = [
    ("/usr/lib/vapoursynth/libbestsource.so", "bestsource_available", "bs"),
    ("/usr/lib/vapoursynth/libffms2.so", "ffms2_available", "ffms2"),
    ("/usr/lib/vapoursynth/liblsmas.so", "lsmas_available", "lsmas"),
]

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
                ['/usr/bin/python3', '-c', test_script],
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
                ffms2_available = data.get("ffms2_available", False)
                lsmas_available = data.get("lsmas_available", False)
                bestsource_available = data.get("bestsource_available", False)
        except Exception as e:
            logger.debug(f"VapourSynth check failed: {e}")

    source_filter_available = ffms2_available or lsmas_available or bestsource_available

    # Determine ready status and any missing requirements
    ready = vs_available and svp_available and vspipe_available and source_filter_available

    missing = []
    if not vs_available:
        missing.append("VapourSynth")
    if not svp_available:
        missing.append("SVPflow plugins")
    if not vspipe_available:
        missing.append("vspipe")
    if not source_filter_available:
        missing.append("source filter (install vapoursynth-plugin-bestsource)")

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
    },
    # Balanced preset - good tradeoff between speed and quality
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of speed and quality. Recommended for most videos.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
    },
    # Quality preset - higher quality, slower processing
    "quality": {
        "name": "Quality",
        "description": "Higher quality motion estimation. Slower but smoother results.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
    },
    # Maximum quality preset - best results, significantly slower
    "max": {
        "name": "Maximum",
        "description": "Maximum quality settings. Best for pre-rendering, not real-time.",
        "super": "{gpu:1,pel:4,scale:{up:2,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:3},main:{search:{coarse:{type:4,distance:-12,bad:{sad:1000,range:24}},type:4,distance:12}},refine:[{thsad:200},{thsad:100},{thsad:50}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:50,cover:80},scene:{}}",
    },
    # Animation preset - optimized for anime/cartoons
    "animation": {
        "name": "Animation",
        "description": "Optimized for anime and cartoons with flat colors and sharp edges.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:1500,range:24}},type:2,distance:10},penalty:{lambda:3.0}},refine:[{thsad:150}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:150},scene:{mode:0}}",
    },
    # Film preset - optimized for live action with natural motion
    "film": {
        "name": "Film",
        "description": "Optimized for live action movies with natural motion blur.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:80,cover:80},scene:{blend:true}}",
    },
}

# Default SVP settings
DEFAULT_SVP_SETTINGS = {
    "enabled": False,
    "preset": "balanced",
    "target_fps": 60,
    "use_gpu": True,
    # Advanced settings (override preset when set)
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


def _generate_svp_script(
    video_path: str,
    target_fps: int,
    preset: str = "balanced",
    custom_super: Optional[str] = None,
    custom_analyse: Optional[str] = None,
    custom_smooth: Optional[str] = None,
    output_format: str = "RGB24"
) -> str:
    """
    Generate a VapourSynth script for SVP interpolation.

    Args:
        video_path: Path to input video
        target_fps: Target output frame rate
        preset: Quality preset (fast, balanced, quality, max, animation, film)
        custom_super: Custom super params (overrides preset)
        custom_analyse: Custom analyse params (overrides preset)
        custom_smooth: Custom smooth params (overrides preset)
        output_format: Output pixel format (RGB24 for FFmpeg rawvideo)

    Returns:
        VapourSynth script as string
    """
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    # Use custom params if provided, otherwise use preset
    super_params = custom_super if custom_super else preset_config["super"]
    analyse_params = custom_analyse if custom_analyse else preset_config["analyse"]

    # For smooth params, we need to inject the target FPS into the rate settings
    # Get base smooth params from custom or preset
    base_smooth = custom_smooth if custom_smooth else preset_config["smooth"]

    # Build smooth params with target FPS
    # We inject the rate:{num:N,den:1,abs:true} into the smooth params
    # The preset smooth params have the base settings, we need to add rate
    if "rate:" in base_smooth:
        # Custom already has rate, use as-is
        smooth_params = base_smooth
    else:
        # Inject rate into smooth params (after first {)
        smooth_params = base_smooth.replace(
            "{",
            f"{{rate:{{num:{target_fps},den:1,abs:true}},",
            1
        )

    # Escape path for Python string
    escaped_path = video_path.replace("\\", "\\\\").replace('"', '\\"')

    script = f'''
import vapoursynth as vs
core = vs.core

# Load SVP plugins
core.std.LoadPlugin("{SVP_PLUGIN_PATH}/libsvpflow1.so")
core.std.LoadPlugin("{SVP_PLUGIN_PATH}/libsvpflow2.so")

# Try to load source filter plugins from system paths
source_plugins = [
    ('/usr/lib/vapoursynth/libbestsource.so', 'bs'),
    ('/usr/lib/vapoursynth/libffms2.so', 'ffms2'),
    ('/usr/lib/vapoursynth/liblsmas.so', 'lsmas'),
]
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
clip_yuv = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

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
        custom_super: Optional[str] = None,
        custom_analyse: Optional[str] = None,
        custom_smooth: Optional[str] = None,
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.preset = preset if preset in SVP_PRESETS else "balanced"
        self.use_nvenc = use_nvenc if use_nvenc is not None else _check_nvenc_available()
        self.stream_id = str(uuid.uuid4())[:8]

        # Custom SVP parameters (override preset when set)
        self.custom_super = custom_super
        self.custom_analyse = custom_analyse
        self.custom_smooth = custom_smooth

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
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
        """Get video dimensions and FPS using ffprobe."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'csv=p=0',
                self.video_path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return False

            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                self._width = int(parts[0])
                self._height = int(parts[1])
                # Parse frame rate (e.g., "30000/1001" or "30/1")
                fps_parts = parts[2].split('/')
                if len(fps_parts) == 2:
                    self._src_fps = int(fps_parts[0]) / int(fps_parts[1])
                else:
                    self._src_fps = float(fps_parts[0])

                logger.info(f"[SVP {self.stream_id}] Video: {self._width}x{self._height} @ {self._src_fps:.2f}fps")
                return True

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")

        return False

    def _build_ffmpeg_command(self) -> List[str]:
        """Build FFmpeg command for HLS encoding from raw RGB24 input."""
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',  # vspipe outputs RGB24
            '-s', f'{self._width}x{self._height}',
            '-r', str(self.target_fps),
            '-i', '-',  # Read from stdin (vspipe output)
        ]

        # Select encoder
        if self.use_nvenc:
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',  # Fastest NVENC preset
                '-tune', 'll',   # Low latency
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '0',
            ])
            logger.info(f"[SVP {self.stream_id}] Using NVENC hardware encoder")
        else:
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-crf', '23',
            ])

        # HLS output options
        cmd.extend([
            '-g', str(self.target_fps * 2),  # GOP = 2 seconds
            '-f', 'hls',
            '-hls_time', '2',
            '-hls_list_size', '10',
            '-hls_flags', 'delete_segments+append_list',
            '-hls_segment_filename', str(self._temp_dir / 'segment_%03d.ts'),
            str(self.playlist_path)
        ])

        return cmd

    async def start(self) -> bool:
        """Start the SVP interpolated stream."""
        if self._running:
            return True

        # Check prerequisites
        status = get_svp_status()
        if not status["ready"]:
            self._error = "SVP not ready: " + str({k: v for k, v in status.items() if not v})
            logger.error(f"[SVP {self.stream_id}] {self._error}")
            return False

        # Get video info
        if not self._get_video_info():
            self._error = "Failed to get video info"
            return False

        # Create temp directory for HLS output
        self._temp_dir = Path(tempfile.mkdtemp(prefix='svp_stream_'))
        logger.info(f"[SVP {self.stream_id}] HLS output: {self._temp_dir}")

        # Generate and write VapourSynth script
        script = _generate_svp_script(
            self.video_path,
            self.target_fps,
            self.preset,
            custom_super=self.custom_super,
            custom_analyse=self.custom_analyse,
            custom_smooth=self.custom_smooth,
        )
        self._script_path = self._temp_dir / "interpolate.vpy"
        self._script_path.write_text(script)
        logger.debug(f"[SVP {self.stream_id}] Script written to {self._script_path}")

        # Start the pipeline
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._run_pipeline())

        return True

    async def _run_pipeline(self):
        """Run the vspipe → FFmpeg pipeline."""
        try:
            # Build commands
            vspipe_cmd = [
                'vspipe',
                '--container', 'y4m',  # Output as Y4M (includes header info)
                str(self._script_path),
                '-'  # Output to stdout
            ]

            # For Y4M input, FFmpeg auto-detects format
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-i', '-',  # Read Y4M from stdin
            ]

            # Encoder selection
            if self.use_nvenc:
                ffmpeg_cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p1',
                    '-tune', 'll',
                    '-rc', 'vbr',
                    '-cq', '23',
                    '-b:v', '0',
                ])
            else:
                ffmpeg_cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',
                    '-crf', '23',
                ])

            # HLS output
            ffmpeg_cmd.extend([
                '-g', str(self.target_fps * 2),
                '-f', 'hls',
                '-hls_time', '2',
                '-hls_list_size', '10',
                '-hls_flags', 'delete_segments+append_list',
                '-hls_segment_filename', str(self._temp_dir / 'segment_%03d.ts'),
                str(self.playlist_path)
            ])

            logger.info(f"[SVP {self.stream_id}] Starting pipeline: vspipe → FFmpeg")
            logger.debug(f"[SVP {self.stream_id}] vspipe: {' '.join(vspipe_cmd)}")
            logger.debug(f"[SVP {self.stream_id}] ffmpeg: {' '.join(ffmpeg_cmd)}")

            # Start vspipe with clean environment (avoids venv conflicts)
            clean_env = _get_clean_env()
            self._vspipe_proc = subprocess.Popen(
                vspipe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # Large buffer for video frames
                env=clean_env
            )

            # Start FFmpeg with vspipe output as input
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self._vspipe_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )

            # Allow vspipe to receive SIGPIPE if FFmpeg exits
            self._vspipe_proc.stdout.close()

            # Monitor the processes
            while self._running:
                # Check if FFmpeg is still running
                if self._ffmpeg_proc.poll() is not None:
                    if self._ffmpeg_proc.returncode != 0:
                        stderr = self._ffmpeg_proc.stderr.read().decode() if self._ffmpeg_proc.stderr else ""
                        self._error = f"FFmpeg exited with code {self._ffmpeg_proc.returncode}: {stderr[-500:]}"
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                    else:
                        logger.info(f"[SVP {self.stream_id}] FFmpeg finished normally")
                    break

                # Check if vspipe failed
                if self._vspipe_proc.poll() is not None and self._vspipe_proc.returncode != 0:
                    stderr = self._vspipe_proc.stderr.read().decode() if self._vspipe_proc.stderr else ""
                    self._error = f"vspipe exited with code {self._vspipe_proc.returncode}: {stderr[-500:]}"
                    logger.error(f"[SVP {self.stream_id}] {self._error}")
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
        if self._vspipe_proc:
            try:
                self._vspipe_proc.terminate()
                self._vspipe_proc.wait(timeout=5)
            except:
                self._vspipe_proc.kill()
            self._vspipe_proc = None

        if self._ffmpeg_proc:
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=5)
            except:
                self._ffmpeg_proc.kill()
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
            "running": self._running,
            "elapsed_seconds": elapsed,
            "segments_ready": self.segments_ready,
            "playlist_ready": self.playlist_ready,
            "error": self._error,
            "use_nvenc": self.use_nvenc,
        }
