"""
SVP/VapourSynth script generation for frame interpolation.

Generates VapourSynth scripts that use SVPflow plugins for motion-compensated
frame interpolation. Supports both regular SVP pipeline and NVOF (NVIDIA Optical Flow).
"""
import os
from typing import Optional

from ..svp_platform import get_svp_plugin_full_paths, get_source_filter_paths
from .config import SVP_PRESETS, build_svp_params


def generate_vspipe_stdin_script(
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
    super_params, analyse_params, smooth_params = build_svp_params(
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


def generate_ffmpeg_svp_script(
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
    super_params, analyse_params, smooth_params = build_svp_params(
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


def generate_svp_script(
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


# Aliases for backward compatibility
_generate_vspipe_stdin_script = generate_vspipe_stdin_script
_generate_ffmpeg_svp_script = generate_ffmpeg_svp_script
_generate_svp_script = generate_svp_script
