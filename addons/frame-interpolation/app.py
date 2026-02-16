"""
Frame Interpolation Sidecar — Standalone FastAPI app.

Provides HLS streaming with optical-flow frame interpolation.
Backends (in priority order): RIFE-NCNN, OpenCV CUDA, OpenCV CPU, simple blend.

Endpoints:
  GET  /health                                  → health check
  POST /optical-flow/play                       → start interpolated HLS stream
  POST /optical-flow/stop                       → stop all streams
  GET  /optical-flow/stream/{stream_id}/{file}  → serve HLS files
"""

import asyncio
import glob
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("frame-interpolation")

# ─── Backend detection ────────────────────────────────────────────────────────

HAS_CV2 = False
HAS_CV2_CUDA = False
HAS_RIFE_NCNN = False

try:
    import cv2
    HAS_CV2 = True
    try:
        _ = cv2.cuda.getCudaEnabledDeviceCount()
        HAS_CV2_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        pass
except ImportError:
    pass

try:
    from rife_ncnn_vulkan_python import Rife
    HAS_RIFE_NCNN = True
except ImportError:
    pass

_NVENC_AVAILABLE = None

def check_nvenc_available() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        _NVENC_AVAILABLE = "h264_nvenc" in result.stdout if result.returncode == 0 else False
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def get_backend_status():
    return {
        "cv2_available": HAS_CV2,
        "cv2_cuda_available": HAS_CV2_CUDA,
        "rife_ncnn_available": HAS_RIFE_NCNN,
        "cuda_available": HAS_CV2_CUDA,
        "nvenc_available": check_nvenc_available(),
        "any_backend_available": HAS_CV2 or HAS_RIFE_NCNN,
        "gpu_backend": (
            "rife_ncnn" if HAS_RIFE_NCNN
            else "opencv_cuda" if HAS_CV2_CUDA
            else "opencv_cpu" if HAS_CV2
            else None
        ),
    }


# ─── Quality presets ──────────────────────────────────────────────────────────

QUALITY_PRESETS = {
    "fast": {"levels": 3, "winsize": 13, "iterations": 3, "pyr_scale": 0.5,
             "poly_n": 5, "poly_sigma": 1.1, "flow_scale": 0.5},
    "balanced": {"levels": 5, "winsize": 15, "iterations": 5, "pyr_scale": 0.5,
                 "poly_n": 7, "poly_sigma": 1.5, "flow_scale": 0.75},
    "quality": {"levels": 7, "winsize": 21, "iterations": 7, "pyr_scale": 0.5,
                "poly_n": 7, "poly_sigma": 1.5, "flow_scale": 1.0},
}


# ─── Frame Interpolator ──────────────────────────────────────────────────────

class FrameInterpolator:
    """Unified interpolation interface — picks best available backend."""

    def __init__(self, quality: str = "fast", use_gpu: bool = True):
        self.quality = quality
        self.params = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["fast"])
        self._rife_ncnn = None
        self._cuda_farneback = None
        self._initialized = False

        self.use_rife_ncnn = HAS_RIFE_NCNN and use_gpu
        self.use_cv2_cuda = HAS_CV2_CUDA and use_gpu and not self.use_rife_ncnn

    def initialize(self):
        if self._initialized:
            return
        try:
            if self.use_rife_ncnn:
                self._rife_ncnn = Rife(gpuid=0, model="rife-v4")
                self._initialized = True
                logger.info("RIFE-NCNN interpolator initialized")
                return

            if self.use_cv2_cuda:
                p = self.params
                self._cuda_farneback = cv2.cuda.FarnebackOpticalFlow.create(
                    numLevels=p["levels"], pyrScale=p["pyr_scale"],
                    fastPyramids=True, winSize=p["winsize"],
                    numIters=p["iterations"], polyN=p["poly_n"],
                    polySigma=p["poly_sigma"], flags=0,
                )
                self._initialized = True
                logger.info("OpenCV CUDA Farneback initialized")
        except Exception as e:
            logger.error(f"Failed to initialize interpolation backend: {e}")

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        if self.use_rife_ncnn and self._rife_ncnn is not None:
            return self._rife_ncnn.process(frame1, frame2, timestep=t)

        if self.use_cv2_cuda and self._cuda_farneback is not None:
            return self._interpolate_cv2_cuda(frame1, frame2, t)

        if HAS_CV2:
            return self._interpolate_farneback(frame1, frame2, t)

        return self._blend(frame1, frame2, t)

    def _interpolate_cv2_cuda(self, frame1, frame2, t):
        try:
            h, w = frame1.shape[:2]
            gpu1, gpu2 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            gpu1.upload(frame1)
            gpu2.upload(frame2)
            gray1 = cv2.cuda.cvtColor(gpu1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cuda.cvtColor(gpu2, cv2.COLOR_BGR2GRAY)
            gpu_flow = self._cuda_farneback.calc(gray1, gray2, None)

            gpu_fx = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            gpu_fy = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            cv2.cuda.split(gpu_flow, [gpu_fx, gpu_fy])
            fx, fy = gpu_fx.download(), gpu_fy.download()

            x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                               np.arange(h, dtype=np.float32))
            map_x = (x - fx * t).astype(np.float32)
            map_y = (y - fy * t).astype(np.float32)
            gx, gy = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            gx.upload(map_x)
            gy.upload(map_y)
            warped = cv2.cuda.remap(gpu1, gx, gy, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
            result = cv2.cuda.addWeighted(warped, 1 - t, gpu2, t, 0)
            return result.download()
        except Exception:
            return self._blend(frame1, frame2, t)

    def _interpolate_farneback(self, frame1, frame2, t):
        try:
            h, w = frame1.shape[:2]
            p = self.params
            fs = p["flow_scale"]
            if fs < 1.0:
                sh, sw = int(h * fs), int(w * fs)
                s1 = cv2.resize(frame1, (sw, sh), interpolation=cv2.INTER_AREA)
                s2 = cv2.resize(frame2, (sw, sh), interpolation=cv2.INTER_AREA)
                g1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
            else:
                g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, pyr_scale=p["pyr_scale"], levels=p["levels"],
                winsize=p["winsize"], iterations=p["iterations"],
                poly_n=p["poly_n"], poly_sigma=p["poly_sigma"], flags=0,
            )
            if fs < 1.0:
                flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / fs

            x, y = np.meshgrid(np.arange(w), np.arange(h))
            ft = flow * t
            mx = (x - ft[:, :, 0]).astype(np.float32)
            my = (y - ft[:, :, 1]).astype(np.float32)
            warped = cv2.remap(frame1, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return cv2.addWeighted(warped, 1 - t, frame2, t, 0)
        except Exception:
            return self._blend(frame1, frame2, t)

    @staticmethod
    def _blend(frame1, frame2, t):
        return ((1 - t) * frame1.astype(np.float32) + t * frame2.astype(np.float32)).astype(np.uint8)

    def cleanup(self):
        self._rife_ncnn = None
        self._cuda_farneback = None
        self._initialized = False


# ─── FFmpeg encoder ───────────────────────────────────────────────────────────

def build_ffmpeg_command(
    width: int, height: int, target_fps: float,
    output_dir: Path, playlist_path: Path,
    use_nvenc: bool = False, target_bitrate: Optional[str] = None,
    target_resolution: Optional[tuple] = None,
) -> List[str]:
    """Build FFmpeg command for HLS encoding from raw BGR24 stdin."""
    gop = int(target_fps * 2)

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(target_fps), "-i", "pipe:0",
    ]

    vf = []
    if target_resolution:
        tw, th = target_resolution
        vf.append(f"scale={tw}:{th}:flags=lanczos")
    vf.append("format=yuv420p")
    if vf:
        cmd.extend(["-vf", ",".join(vf)])

    if use_nvenc:
        cmd.extend(["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                     "-g", str(gop), "-keyint_min", str(gop)])
        if target_bitrate:
            cmd.extend(["-b:v", target_bitrate])
        else:
            cmd.extend(["-cq", "23"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                     "-g", str(gop), "-keyint_min", str(gop)])
        if target_bitrate:
            cmd.extend(["-b:v", target_bitrate])
        else:
            cmd.extend(["-crf", "23"])

    cmd.extend([
        "-an",
        "-f", "hls", "-hls_time", "2",
        "-hls_list_size", "10",
        "-hls_flags", "delete_segments+append_list",
        "-hls_segment_filename", str(output_dir / "segment_%d.ts"),
        str(playlist_path),
    ])
    return cmd


# ─── Stream manager ──────────────────────────────────────────────────────────

_active_streams: Dict[str, "InterpolatedStream"] = {}
_executor = ThreadPoolExecutor(max_workers=4)


class InterpolatedStream:
    def __init__(
        self,
        video_path: str,
        target_fps: int = 60,
        use_gpu: bool = True,
        quality: str = "fast",
        use_nvenc: bool = False,
        target_bitrate: Optional[str] = None,
        target_resolution: Optional[tuple] = None,
        start_position: float = 0.0,
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.use_gpu = use_gpu
        self.quality = quality
        self.use_nvenc = use_nvenc or check_nvenc_available()
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution
        self.start_position = start_position

        self.stream_id = uuid.uuid4().hex[:8]
        self._running = False
        self._task = None
        self._temp_dir = None
        self.hls_dir = None
        self._error = None
        self._process = None
        self._interpolator = None

        _active_streams[self.stream_id] = self

    @property
    def error(self):
        return self._error

    @property
    def playlist_ready(self):
        if self.hls_dir is None:
            return False
        p = self.hls_dir / "playlist.m3u8"
        if not p.exists():
            return False
        content = p.read_text()
        return "segment_" in content and "#EXTINF" in content

    async def start(self):
        if self._running:
            return False
        self._running = True

        self._temp_dir = Path(tempfile.mkdtemp(prefix="optflow_"))
        self.hls_dir = self._temp_dir / "hls"
        self.hls_dir.mkdir(exist_ok=True)

        self._task = asyncio.create_task(self._process_video())
        return True

    async def _process_video(self):
        loop = asyncio.get_event_loop()
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self._error = "Failed to open video"
                return

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps_multiplier = self.target_fps / src_fps
            frames_between = max(0, int(round(fps_multiplier)) - 1)

            if self.start_position > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.start_position * 1000)

            self._interpolator = FrameInterpolator(quality=self.quality, use_gpu=self.use_gpu)
            await loop.run_in_executor(_executor, self._interpolator.initialize)

            ffmpeg_cmd = build_ffmpeg_command(
                width, height, self.target_fps,
                self.hls_dir, self.hls_dir / "playlist.m3u8",
                use_nvenc=self.use_nvenc,
                target_bitrate=self.target_bitrate,
                target_resolution=self.target_resolution,
            )
            logger.info(f"[Stream {self.stream_id}] FFmpeg: {' '.join(ffmpeg_cmd)}")

            self._process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            prev_frame = None
            frame_count = 0
            while self._running:
                ret, frame = await loop.run_in_executor(_executor, cap.read)
                if not ret:
                    break

                if prev_frame is not None and frames_between > 0:
                    for i in range(1, frames_between + 1):
                        t = i / (frames_between + 1)
                        interp = await loop.run_in_executor(
                            _executor, self._interpolator.interpolate, prev_frame, frame, t
                        )
                        try:
                            self._process.stdin.write(interp.tobytes())
                            await self._process.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            self._running = False
                            break

                if not self._running:
                    break

                try:
                    self._process.stdin.write(frame.tobytes())
                    await self._process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    break

                prev_frame = frame
                frame_count += 1

            cap.release()

            if self._process and self._process.stdin:
                self._process.stdin.close()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self._process.kill()

            logger.info(f"[Stream {self.stream_id}] Processed {frame_count} source frames")

        except asyncio.CancelledError:
            logger.info(f"[Stream {self.stream_id}] Cancelled")
        except Exception as e:
            logger.error(f"[Stream {self.stream_id}] Error: {e}", exc_info=True)
            self._error = str(e)
        finally:
            self._running = False

    def stop(self):
        self._running = False
        if self._process:
            try:
                pid = self._process.pid
                if pid:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.3)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except Exception:
                pass
            self._process = None

        if self._task:
            self._task.cancel()
            self._task = None

        if self._interpolator:
            self._interpolator.cleanup()
            self._interpolator = None

        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        _active_streams.pop(self.stream_id, None)

    async def wait_for_ready(self, timeout: float = 30) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.playlist_ready:
                return True
            if self._error:
                return False
            await asyncio.sleep(0.2)
        return False


def stop_all_streams():
    for stream in list(_active_streams.values()):
        stream.stop()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Frame Interpolation Sidecar")


@app.get("/health")
async def health():
    status = get_backend_status()
    return {"status": "ok", "backend": status}


class PlayRequest(BaseModel):
    file_path: str
    quality_preset: str = "fast"
    start_position: float = 0.0
    target_fps: int = 60
    target_resolution: Optional[str] = None
    target_bitrate: Optional[str] = None
    use_gpu: bool = True


@app.post("/optical-flow/play")
async def play(req: PlayRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    stop_all_streams()

    # Parse resolution
    res = None
    if req.target_resolution:
        presets = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
        res = presets.get(req.target_resolution)

    quality = req.quality_preset if req.quality_preset in QUALITY_PRESETS else "fast"

    stream = InterpolatedStream(
        video_path=req.file_path,
        target_fps=req.target_fps,
        use_gpu=req.use_gpu,
        quality=quality,
        target_bitrate=req.target_bitrate,
        target_resolution=res,
        start_position=req.start_position,
    )

    started = await stream.start()
    if not started:
        raise HTTPException(status_code=500, detail=stream.error or "Failed to start stream")

    ready = await stream.wait_for_ready(timeout=30)
    if not ready and stream.error:
        raise HTTPException(status_code=500, detail=stream.error)

    # Probe source resolution
    src_width, src_height = 1920, 1080
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", req.file_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                src_width, src_height = int(parts[0]), int(parts[1])
    except Exception:
        pass

    return {
        "success": True,
        "stream_id": stream.stream_id,
        "stream_url": f"/optical-flow/stream/{stream.stream_id}/playlist.m3u8",
        "source_resolution": {"width": src_width, "height": src_height},
        "message": "Optical flow stream started",
    }


@app.post("/optical-flow/stop")
async def stop():
    stop_all_streams()
    return {"success": True, "message": "All streams stopped"}


@app.get("/optical-flow/stream/{stream_id}/{filename}")
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
