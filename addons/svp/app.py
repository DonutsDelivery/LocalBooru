"""
SVP (SmoothVideo Project) Sidecar — Standalone FastAPI app.

Uses VapourSynth + SVPflow for high-quality frame interpolation.
Pipeline: FFmpeg decode → vspipe/SVP → FFmpeg HLS encode.

Endpoints:
  GET  /health                          → check VapourSynth/SVP availability
  POST /svp/play                        → start SVP interpolated HLS stream
  POST /svp/stop                        → stop all streams
  GET  /svp/stream/{stream_id}/{file}   → serve HLS files
"""

import asyncio
import logging
import os
import platform
import shutil
import signal
import select
import subprocess
import sys
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("svp")

# ─── Platform helpers ─────────────────────────────────────────────────────────

def _is_linux():
    return sys.platform.startswith("linux")

def _is_windows():
    return sys.platform == "win32"

def _is_macos():
    return sys.platform == "darwin"


def get_svp_plugin_path() -> Optional[str]:
    env = os.environ.get("LOCALBOORU_SVP_PLUGIN_PATH")
    if env and os.path.isdir(env):
        return env
    candidates = []
    if _is_windows():
        pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        candidates = [os.path.join(pf86, "SVP 4", "plugins64"),
                      os.path.join(pf, "SVP 4", "plugins64")]
    elif _is_macos():
        candidates = ["/Applications/SVP 4 Mac.app/Contents/Resources/plugins"]
    else:
        candidates = ["/opt/svp/plugins", "/usr/lib/svp/plugins"]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def get_svp_plugin_names() -> Tuple[str, str]:
    if _is_windows():
        return ("svpflow1_vs.dll", "svpflow2_vs.dll")
    elif _is_macos():
        return ("libsvpflow1_vs64.dylib", "libsvpflow2_vs64.dylib")
    return ("libsvpflow1.so", "libsvpflow2.so")


def get_svp_plugin_full_paths() -> Tuple[str, str]:
    base = get_svp_plugin_path() or "/opt/svp/plugins"
    f1, f2 = get_svp_plugin_names()
    return (Path(base, f1).as_posix(), Path(base, f2).as_posix())


def get_clean_env() -> dict:
    """Minimal environment for vspipe subprocess (avoids venv conflicts)."""
    if _is_linux():
        home = os.environ.get("HOME", "/tmp")
        path_parts = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        if os.environ.get("PATH"):
            path_parts.extend(os.environ["PATH"].split(":"))
        seen = set()
        unique = [p for p in path_parts if p and p not in seen and not seen.add(p)]
        env = {
            "PATH": ":".join(unique),
            "HOME": home,
            "USER": os.environ.get("USER", "user"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "DISPLAY": os.environ.get("DISPLAY", ":0"),
        }
        for v in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PKG_CONFIG_PATH"):
            if v in os.environ:
                env[v] = os.environ[v]
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            env["XDG_RUNTIME_DIR"] = xdg
        else:
            try:
                env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
            except AttributeError:
                pass
    elif _is_windows():
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\WINDOWS"),
            "TEMP": os.environ.get("TEMP", os.environ.get("TMP", "")),
            "APPDATA": os.environ.get("APPDATA", ""),
        }
    else:  # macOS
        home = os.environ.get("HOME", "/tmp")
        paths = ["/usr/bin", "/bin", "/usr/local/bin"]
        if platform.machine() == "arm64":
            paths.append("/opt/homebrew/bin")
        env = {"PATH": ":".join(paths), "HOME": home, "LANG": "en_US.UTF-8"}
    for key in ("LOCALBOORU_VS_PYTHON", "LOCALBOORU_SVP_PLUGIN_PATH", "LOCALBOORU_VS_PLUGIN_PATH"):
        val = os.environ.get(key)
        if val:
            env[key] = val
    return env


# ─── SVP availability detection ──────────────────────────────────────────────

def check_vspipe() -> bool:
    try:
        r = subprocess.run(["vspipe", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def check_vapoursynth() -> bool:
    try:
        import vapoursynth
        return True
    except ImportError:
        return False


def check_svp_plugins() -> bool:
    path = get_svp_plugin_path()
    if not path:
        return False
    f1, f2 = get_svp_plugin_names()
    return os.path.isfile(os.path.join(path, f1)) and os.path.isfile(os.path.join(path, f2))


def check_nvenc() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=5)
        return "h264_nvenc" in r.stdout if r.returncode == 0 else False
    except Exception:
        return False


# ─── SVP presets ──────────────────────────────────────────────────────────────

SVP_PRESETS = {
    "fast": {
        "super": "{gpu:1,pel:1,scale:{up:0,down:4}}",
        "analyse": "{gpu:1,block:{w:32,h:32,overlap:0},main:{search:{coarse:{type:2,distance:-6,bad:{sad:2000,range:24}},type:2,distance:6}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:13,mask:{area:100},scene:{}}",
    },
    "balanced": {
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
    },
    "quality": {
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
    },
    "max": {
        "super": "{gpu:1,pel:4,scale:{up:2,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:3},main:{search:{coarse:{type:4,distance:-12,bad:{sad:1000,range:24}},type:4,distance:12}},refine:[{thsad:200},{thsad:100},{thsad:50}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:50,cover:80},scene:{}}",
    },
    "animation": {
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:32,h:32,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:1500,range:24}},type:2,distance:8}},refine:[{thsad:150}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:150},scene:{}}",
    },
    "film": {
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:80},scene:{}}",
    },
}


# ─── VapourSynth script generation ───────────────────────────────────────────

def generate_vspipe_stdin_script(
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    num_frames: int,
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
    """Generate a VapourSynth script that reads raw yuv420p frames from stdin and applies SVP."""
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    super_params = custom_super or preset_config["super"]
    analyse_params = custom_analyse or preset_config["analyse"]

    if not custom_analyse:
        if use_nvof and "nvof:" not in analyse_params:
            analyse_params = analyse_params.replace("{gpu:1,", "{gpu:1,nvof:1,")
        elif not use_nvof:
            analyse_params = analyse_params.replace(",nvof:1", "").replace("nvof:1,", "")

    if custom_smooth:
        smooth_params = custom_smooth
    else:
        smooth_params = f"{{gpuid:0,algo:{shader},mask:{{area:{artifact_masking}}},scene:{{}}}}"

    if "rate:" not in smooth_params:
        smooth_params = smooth_params.replace(
            "{",
            f"{{rate:{{num:{target_fps},den:1,abs:true}},",
            1,
        )

    if not use_nvof:
        super_params = super_params.replace("gpu:1", "gpu:0")
        analyse_params = analyse_params.replace("gpu:1", "gpu:0")
        analyse_params = analyse_params.replace(",nvof:1", "").replace("nvof:1,", "")
        if not custom_smooth:
            smooth_params = smooth_params.replace("gpuid:0,", "")

    flow1, flow2 = get_svp_plugin_full_paths()

    frame_size = width * height * 3 // 2
    chroma_width = width // 2
    chroma_height = height // 2

    return f'''import ctypes
import sys
import vapoursynth as vs
core = vs.core

WIDTH = {width}
HEIGHT = {height}
FPS_NUM = {fps_num}
FPS_DEN = {fps_den}
NUM_FRAMES = {num_frames}
FRAME_SIZE = {frame_size}
CHROMA_WIDTH = {chroma_width}
CHROMA_HEIGHT = {chroma_height}
stdin = sys.stdin.buffer
next_frame = 0
last_frame = None

# Load SVP plugins
core.std.LoadPlugin("{flow1}")
core.std.LoadPlugin("{flow2}")

# vspipe has no built-in stdin video source. Create a clip with the source
# geometry, then fill each requested frame from FFmpeg's rawvideo pipe.
clip = core.std.BlankClip(
    width=WIDTH,
    height=HEIGHT,
    format=vs.YUV420P8,
    length=max(NUM_FRAMES, 1),
    fpsnum=FPS_NUM,
    fpsden=FPS_DEN,
)

def read_exact(size):
    data = bytearray()
    while len(data) < size:
        chunk = stdin.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)

def write_plane(frame, plane, src, width, height):
    stride = frame.get_stride(plane)
    ptr = frame.get_write_ptr(plane)
    pos = 0
    for y in range(height):
        ctypes.memmove(ptr.value + y * stride, src[pos:pos + width], width)
        pos += width

def source_frame(n, f):
    global next_frame, last_frame
    if n < next_frame and last_frame is not None:
        return last_frame
    while next_frame < n:
        skipped = read_exact(FRAME_SIZE)
        if len(skipped) < FRAME_SIZE:
            break
        next_frame += 1

    raw = read_exact(FRAME_SIZE)
    if len(raw) < FRAME_SIZE:
        if last_frame is not None:
            return last_frame
        raw = raw + bytes(FRAME_SIZE - len(raw))

    out = f.copy()
    y_size = WIDTH * HEIGHT
    uv_size = CHROMA_WIDTH * CHROMA_HEIGHT
    write_plane(out, 0, raw[:y_size], WIDTH, HEIGHT)
    write_plane(out, 1, raw[y_size:y_size + uv_size], CHROMA_WIDTH, CHROMA_HEIGHT)
    write_plane(out, 2, raw[y_size + uv_size:y_size + uv_size * 2], CHROMA_WIDTH, CHROMA_HEIGHT)
    next_frame = n + 1
    last_frame = out
    return out

clip = core.std.ModifyFrame(clip, clip, source_frame)

# SVP parameters (preset: {preset})
super_params = '{super_params}'
analyse_params = '{analyse_params}'
smooth_params = '{smooth_params}'

# SVP processing pipeline
super_clip = core.svp1.Super(clip, super_params)
vectors = core.svp1.Analyse(super_clip["clip"], super_clip["data"], clip, analyse_params)

src_fps = clip.fps.numerator / clip.fps.denominator
smooth = core.svp2.SmoothFps(
    clip, super_clip["clip"], super_clip["data"],
    vectors["clip"], vectors["data"],
    smooth_params, src=clip, fps=src_fps
)

smooth.set_output()
'''


# ─── Video info via ffprobe ───────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    """Get video metadata via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration,r_frame_rate,avg_frame_rate,nb_frames",
             "-show_entries", "format=duration",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return {"success": False}

        import json
        data = json.loads(r.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))

        # Parse FPS
        afr = stream.get("avg_frame_rate", "30/1")
        if "/" in afr:
            num, den = afr.split("/")
            fps_num, fps_den = int(num), int(den)
            fps = fps_num / fps_den if fps_den > 0 else 30
        else:
            fps = float(afr) if afr else 30
            fps_num, fps_den = int(fps * 1000), 1000

        # Duration
        dur = float(stream.get("duration", 0) or fmt.get("duration", 0) or 0)

        # Frame count
        nf = int(stream.get("nb_frames", 0) or 0)
        if nf == 0 and dur > 0 and fps > 0:
            nf = int(dur * fps)

        # Check for audio stream
        has_audio = False
        audio_check = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if audio_check.returncode == 0:
            has_audio = bool(audio_check.stdout.strip())

        return {
            "success": True,
            "width": w, "height": h,
            "src_fps": fps, "src_fps_num": fps_num, "src_fps_den": fps_den,
            "duration": dur, "num_frames": nf, "has_audio": has_audio,
        }
    except Exception as e:
        logger.error(f"ffprobe error: {e}")
        return {"success": False}


def detect_audio_gain(video_path: str) -> Optional[float]:
    """Detect gain needed to normalize the loudest peak to -2 dB.
    Uses ffmpeg volumedetect on first 15 seconds. Returns dB gain or None."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn",
             "-af", "volumedetect", "-t", "15",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None

        m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", r.stderr)
        if not m:
            return None

        max_volume = float(m.group(1))
        gain_db = -2.0 - max_volume
        return max(-12.0, min(24.0, gain_db))
    except Exception as e:
        logger.warning(f"Audio gain detection failed: {e}")
        return None


# ─── Stream manager ──────────────────────────────────────────────────────────

_active_streams: Dict[str, "SVPStream"] = {}


class SVPStream:
    """Manages an SVP-interpolated stream pipeline."""

    def __init__(
        self,
        video_path: str,
        target_fps: int = 60,
        preset: str = "balanced",
        use_nvof: bool = True,
        shader: int = 23,
        artifact_masking: int = 100,
        frame_interpolation: int = 2,
        custom_super: Optional[str] = None,
        custom_analyse: Optional[str] = None,
        custom_smooth: Optional[str] = None,
        start_position: float = 0.0,
        target_bitrate: Optional[str] = None,
        target_resolution: Optional[tuple] = None,
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.preset = preset if preset in SVP_PRESETS else "balanced"
        self.use_nvenc = check_nvenc()
        self.stream_id = uuid.uuid4().hex[:8]
        self.start_position = start_position
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution

        self.use_nvof = use_nvof
        self.shader = shader
        self.artifact_masking = artifact_masking
        self.frame_interpolation = frame_interpolation
        self.custom_super = custom_super
        self.custom_analyse = custom_analyse
        self.custom_smooth = custom_smooth

        self._running = False
        self._task = None
        self._decode_proc = None
        self._vspipe_proc = None
        self._ffmpeg_proc = None
        self._decode_stderr = b""
        self._vspipe_stderr = b""
        self._ffmpeg_stderr = b""
        self._temp_dir = None
        self._error = None
        self._audio_gain_db = None

        self._width = 0
        self._height = 0
        self._src_fps = 0
        self._duration = 0

        _active_streams[self.stream_id] = self

    @property
    def hls_dir(self):
        return self._temp_dir

    @property
    def playlist_path(self):
        return self._temp_dir / "stream.m3u8" if self._temp_dir else None

    @property
    def error(self):
        return self._error

    @property
    def playlist_ready(self):
        if not self.playlist_path or not self.playlist_path.exists():
            return False
        try:
            c = self.playlist_path.read_text()
            return "segment_" in c and "#EXTINF" in c
        except Exception:
            return False

    async def start(self) -> bool:
        info = get_video_info(self.video_path)
        if not info["success"]:
            self._error = "Failed to get video info"
            return False

        self._width = info["width"]
        self._height = info["height"]
        self._src_fps = info["src_fps"]
        self._duration = info["duration"]

        fps_ratio = self.target_fps / self._src_fps if self._src_fps > 0 else 2.0
        if 0.95 <= fps_ratio <= 1.05:
            self._error = f"Source fps ({self._src_fps:.2f}) already near target ({self.target_fps}fps)"
            return False

        self._temp_dir = Path(tempfile.mkdtemp(prefix="svp_stream_"))

        stream_width, stream_height = self.target_resolution or (self._width, self._height)

        script = generate_vspipe_stdin_script(
            stream_width, stream_height, info["src_fps_num"], info["src_fps_den"],
            info["num_frames"],
            self.target_fps, self.preset,
            use_nvof=self.use_nvof, shader=self.shader,
            artifact_masking=self.artifact_masking,
            frame_interpolation=self.frame_interpolation,
            custom_super=self.custom_super,
            custom_analyse=self.custom_analyse,
            custom_smooth=self.custom_smooth,
        )
        script_path = self._temp_dir / "svp_stdin.vpy"
        script_path.write_text(script)

        # Detect audio gain for normalization to -2 dB peak
        self._audio_gain_db = None
        if info.get("has_audio", True):
            self._audio_gain_db = detect_audio_gain(self.video_path)

        self._running = True
        self._task = asyncio.create_task(self._run_pipeline(script_path))
        return True

    async def _run_pipeline(self, script_path: Path):
        try:
            # Stage 1: FFmpeg decode to raw yuv420p frames
            decode_cmd = ["ffmpeg", "-hwaccel", "auto", "-threads", "0"]
            if self.start_position > 0:
                decode_cmd.extend(["-ss", str(self.start_position)])
            decode_cmd.extend(["-i", self.video_path])
            if self.target_resolution:
                w, h = self.target_resolution
                decode_cmd.extend(["-vf", f"scale={w}:{h}:flags=lanczos"])
            decode_cmd.extend(["-an", "-sn", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])

            # Stage 2: vspipe with SVP
            vspipe_cmd = ["vspipe", "--requests", "1", "-c", "y4m", str(script_path), "-"]

            # Stage 3: FFmpeg encode to HLS
            encode_cmd = [
                "ffmpeg", "-y",
                "-probesize", "1000000", "-analyzeduration", "1000000",
                "-fflags", "+nobuffer+flush_packets",
                "-f", "yuv4mpegpipe", "-i", "-",
            ]
            if self.start_position > 0:
                encode_cmd.extend(["-ss", str(self.start_position)])
            encode_cmd.extend(["-probesize", "5000000", "-i", self.video_path,
                               "-map", "0:v", "-map", "1:a?"])
            encode_cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])

            if self.use_nvenc:
                encode_cmd.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll"])
                if self.target_bitrate:
                    encode_cmd.extend(["-rc", "cbr", "-b:v", self.target_bitrate])
                else:
                    encode_cmd.extend(["-rc", "vbr", "-cq", "23", "-b:v", "0"])
            else:
                encode_cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency"])
                if self.target_bitrate:
                    encode_cmd.extend(["-b:v", self.target_bitrate])
                else:
                    encode_cmd.extend(["-crf", "23"])

            encode_cmd.extend(["-c:a", "aac", "-b:a", "192k"])

            # Normalize peak to -2 dB (uniform gain, no dynamics processing)
            if self._audio_gain_db is not None and abs(self._audio_gain_db) > 0.5:
                encode_cmd.extend(["-af", f"volume={self._audio_gain_db}dB"])
                logger.info(
                    f"[SVP {self.stream_id}] Audio normalization: {self._audio_gain_db:=+.1f} dB (peak → -2 dB)"
                )

            gop = self.target_fps * 2
            encode_cmd.extend([
                "-g", str(gop), "-keyint_min", str(self.target_fps),
                "-f", "hls", "-hls_time", "4", "-hls_list_size", "0",
                "-hls_flags", "append_list+split_by_time",
                "-hls_segment_filename", str(self._temp_dir / "segment_%03d.ts"),
                str(self.playlist_path),
            ])

            logger.info(f"[SVP {self.stream_id}] Starting pipeline")
            env = get_clean_env()

            self._decode_proc = subprocess.Popen(
                decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            self._vspipe_proc = subprocess.Popen(
                vspipe_cmd, stdin=self._decode_proc.stdout,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            self._decode_proc.stdout.close()
            self._ffmpeg_proc = subprocess.Popen(
                encode_cmd, stdin=self._vspipe_proc.stdout,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            self._vspipe_proc.stdout.close()
            # stdin was passed as a file descriptor (not PIPE), so it's None
            if self._vspipe_proc.stdin:
                self._vspipe_proc.stdin.close()

            while self._running:
                self._drain_stderr()
                if self._vspipe_proc.poll() is not None and self._vspipe_proc.returncode != 0:
                    self._drain_stderr(final=True)
                    stderr = self._decode_tail(self._vspipe_stderr, 2000)
                    decode_stderr = self._decode_tail(self._decode_stderr, 1000)
                    self._error = self._format_vspipe_error(stderr)
                    if decode_stderr:
                        self._error += f"\nDecode stderr: {decode_stderr}"
                    logger.error(f"[SVP {self.stream_id}] {self._error}")
                    break
                if self._ffmpeg_proc.poll() is not None:
                    self._drain_stderr(final=True)
                    if self._ffmpeg_proc.returncode != 0:
                        stderr = self._decode_tail(self._ffmpeg_stderr, 1000)
                        try:
                            self._vspipe_proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                        if self._vspipe_proc.poll() is not None and self._vspipe_proc.returncode != 0:
                            self._drain_stderr(final=True)
                            vspipe_stderr = self._decode_tail(self._vspipe_stderr, 2000)
                            decode_stderr = self._decode_tail(self._decode_stderr, 1000)
                            self._error = self._format_vspipe_error(vspipe_stderr)
                            if decode_stderr:
                                self._error += f"\nDecode stderr: {decode_stderr}"
                            logger.error(f"[SVP {self.stream_id}] {self._error}")
                            break
                        self._error = f"FFmpeg encode error: {stderr}"
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                    else:
                        logger.info(f"[SVP {self.stream_id}] Pipeline finished")
                    break
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info(f"[SVP {self.stream_id}] Cancelled")
        except Exception as e:
            self._error = str(e)
            logger.error(f"[SVP {self.stream_id}] Error: {e}", exc_info=True)
        finally:
            self._running = False
            self._cleanup()

    def _drain_stderr(self, final: bool = False):
        procs = (
            ("decode", self._decode_proc, "_decode_stderr"),
            ("vspipe", self._vspipe_proc, "_vspipe_stderr"),
            ("ffmpeg", self._ffmpeg_proc, "_ffmpeg_stderr"),
        )
        for _name, proc, attr in procs:
            pipe = proc.stderr if proc and proc.stderr else None
            if not pipe:
                continue
            try:
                while final or select.select([pipe], [], [], 0)[0]:
                    chunk = pipe.read(4096 if final else 1024)
                    if not chunk:
                        break
                    setattr(self, attr, (getattr(self, attr) + chunk)[-12000:])
                    if not final and len(chunk) < 1024:
                        break
            except Exception:
                pass

    @staticmethod
    def _decode_tail(data: bytes, limit: int) -> str:
        return data.decode(errors="replace")[-limit:].strip()

    def _format_vspipe_error(self, stderr: str) -> str:
        if not stderr:
            code = self._vspipe_proc.returncode if self._vspipe_proc else "unknown"
            return f"vspipe exited with code {code} but produced no stderr"
        if "unable to init GPU-based renderer" in stderr:
            return (
                "SVP NVIDIA Optical Flow/GPU renderer failed to initialize. "
                f"vspipe error: {stderr}"
            )
        return f"vspipe error: {stderr}"

    def _cleanup(self):
        for proc in (self._ffmpeg_proc, self._vspipe_proc, self._decode_proc):
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._decode_proc = self._vspipe_proc = self._ffmpeg_proc = None

    async def stop_async(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=3)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._task = None
        self._cleanup()
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        _active_streams.pop(self.stream_id, None)

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self._cleanup()
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        _active_streams.pop(self.stream_id, None)

    async def wait_for_ready(self, timeout: float = 45) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.playlist_ready:
                return True
            if self._error:
                return False
            await asyncio.sleep(0.3)
        return False


async def stop_all_streams():
    await asyncio.gather(
        *(s.stop_async() for s in list(_active_streams.values())),
        return_exceptions=True,
    )


def is_gpu_renderer_error(error: Optional[str]) -> bool:
    return bool(error and "unable to init GPU-based renderer" in error)


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="SVP Sidecar")


@app.get("/health")
async def health():
    vs_ok = check_vapoursynth()
    vspipe_ok = check_vspipe()
    svp_ok = check_svp_plugins()
    nvenc_ok = check_nvenc()
    ready = vs_ok and vspipe_ok and svp_ok

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "vapoursynth_available": vs_ok,
        "vspipe_available": vspipe_ok,
        "svp_plugins_available": svp_ok,
        "nvenc_available": nvenc_ok,
    }


class PlayRequest(BaseModel):
    file_path: str
    target_fps: int = 60
    preset: str = "balanced"
    start_position: float = 0.0
    use_nvof: bool = True
    shader: int = 23
    artifact_masking: int = 100
    frame_interpolation: int = 2
    custom_super: Optional[str] = None
    custom_analyse: Optional[str] = None
    custom_smooth: Optional[str] = None
    target_bitrate: Optional[str] = None
    target_resolution: Optional[str] = None


@app.post("/svp/play")
async def play(req: PlayRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    if not check_vspipe():
        raise HTTPException(status_code=503, detail="vspipe not available")
    if not check_svp_plugins():
        raise HTTPException(status_code=503, detail="SVP plugins not found")

    await stop_all_streams()

    res = None
    if req.target_resolution:
        presets = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080),
                   "1440p": (2560, 1440), "4k": (3840, 2160)}
        res = presets.get(req.target_resolution)

    def make_stream(use_nvof: bool) -> SVPStream:
        return SVPStream(
            video_path=req.file_path,
            target_fps=req.target_fps,
            preset=req.preset,
            use_nvof=use_nvof,
            shader=req.shader,
            artifact_masking=req.artifact_masking,
            frame_interpolation=req.frame_interpolation,
            custom_super=req.custom_super,
            custom_analyse=req.custom_analyse,
            custom_smooth=req.custom_smooth,
            start_position=req.start_position,
            target_bitrate=req.target_bitrate,
            target_resolution=res,
        )

    stream = make_stream(req.use_nvof)

    started = await stream.start()
    if not started:
        error = stream.error or "Failed to start SVP stream"
        stream.stop()
        if "already near target" in error:
            return {
                "success": True,
                "skipped": True,
                "reason": error,
                "duration": stream._duration,
                "source_resolution": {"width": stream._width, "height": stream._height},
                "message": "SVP stream not needed",
            }
        raise HTTPException(status_code=500, detail=error)

    ready = await stream.wait_for_ready(timeout=45)
    if not ready and stream.error:
        error = stream.error
        stream.stop()
        if req.use_nvof:
            logger.warning(
                "[SVP] vspipe failed with NVOF enabled; retrying with CPU SVP"
            )
            await asyncio.sleep(0)  # yield to let GPU task finalizer complete
            stream = make_stream(False)
            started = await stream.start()
            if not started:
                fallback_error = stream.error or "Failed to start SVP CPU fallback stream"
                stream.stop()
                raise HTTPException(
                    status_code=500,
                    detail=f"{error}\nCPU fallback also failed: {fallback_error}",
                )
            ready = await stream.wait_for_ready(timeout=45)
            if ready:
                return {
                    "success": True,
                    "stream_id": stream.stream_id,
                    "stream_url": f"/svp/stream/{stream.stream_id}/stream.m3u8",
                    "duration": stream._duration,
                    "source_resolution": {"width": stream._width, "height": stream._height},
                    "message": "SVP stream started with CPU fallback (NVOF unavailable or failed)",
                    "nvof_fallback": True,
                    "fallback_reason": error.split("\n")[0],
                }
            fallback_error = stream.error or "SVP CPU fallback timed out"
            stream.stop()
            raise HTTPException(
                status_code=500,
                detail=f"{error}\nCPU fallback also failed: {fallback_error}",
            )
        raise HTTPException(status_code=500, detail=error)

    return {
        "success": True,
        "stream_id": stream.stream_id,
        "stream_url": f"/svp/stream/{stream.stream_id}/stream.m3u8",
        "duration": stream._duration,
        "source_resolution": {"width": stream._width, "height": stream._height},
        "message": "SVP stream started",
    }


@app.post("/svp/stop")
async def stop():
    await stop_all_streams()
    return {"success": True, "message": "All SVP streams stopped"}


@app.get("/svp/stream/{stream_id}/{filename}")
async def stream_file(stream_id: str, filename: str):
    stream = _active_streams.get(stream_id)
    if not stream or not stream.hls_dir:
        raise HTTPException(status_code=404, detail="Stream not found")

    file_path = stream.hls_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if filename.endswith(".m3u8"):
        media_type = "application/vnd.apple.mpegurl"
    elif filename.endswith(".ts"):
        media_type = "video/mp2t"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        str(file_path),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )
