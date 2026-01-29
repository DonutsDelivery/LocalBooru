"""
Optical flow HLS streaming service.

Manages the video → interpolation → FFmpeg → HLS pipeline for streaming
interpolated video to clients via hls.js.

Optimized for RIFE-based interpolation with:
- Async frame loading with pinned memory for faster CPU→GPU transfer
- Batched interpolation for multiple t values in single forward pass
- NVENC hardware encoding when available
- Performance monitoring and logging

GPU Native mode (quality="gpu_native"):
- GPU video decode via PyNvVideoCodec (when available)
- NVIDIA Optical Flow Accelerator for motion estimation
- PyTorch GPU warping
- Full pipeline stays on GPU until final encode
- Target: <16.7ms per frame at 1440p (60fps real-time)
"""
import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    import numpy
    import torch

logger = logging.getLogger(__name__)

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=4)

# Active streams registry
_active_streams: Dict[str, 'InterpolatedStream'] = {}

# Performance monitoring interval (log every N frames)
_PERF_LOG_INTERVAL = 100

# Check for NVENC availability (cached at module load)
_NVENC_AVAILABLE: Optional[bool] = None


def _check_nvenc_available() -> bool:
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


# Check for GPU video pipeline availability (cached at module load)
_GPU_PIPELINE_AVAILABLE: Optional[bool] = None
_GPU_NATIVE_AVAILABLE: Optional[bool] = None


def _check_gpu_pipeline_available() -> bool:
    """Check if GPU video pipeline (PyNvVideoCodec) is available."""
    global _GPU_PIPELINE_AVAILABLE
    if _GPU_PIPELINE_AVAILABLE is not None:
        return _GPU_PIPELINE_AVAILABLE

    try:
        from .gpu_video_pipeline import get_gpu_pipeline_status
        status = get_gpu_pipeline_status()
        _GPU_PIPELINE_AVAILABLE = status.get('pynvvideocodec_available', False)
        if _GPU_PIPELINE_AVAILABLE:
            logger.info("GPU video pipeline (PyNvVideoCodec) available")
        else:
            logger.debug("GPU video pipeline not available, using cv2.VideoCapture")
    except ImportError:
        _GPU_PIPELINE_AVAILABLE = False
        logger.debug("gpu_video_pipeline module not available")
    except Exception as e:
        _GPU_PIPELINE_AVAILABLE = False
        logger.debug(f"GPU pipeline check failed: {e}")

    return _GPU_PIPELINE_AVAILABLE


def _check_gpu_native_available() -> bool:
    """Check if GPU native interpolation (NVOF + GPU warp) is available."""
    global _GPU_NATIVE_AVAILABLE
    if _GPU_NATIVE_AVAILABLE is not None:
        return _GPU_NATIVE_AVAILABLE

    try:
        from .optical_flow import get_backend_status
        status = get_backend_status()
        _GPU_NATIVE_AVAILABLE = status.get('gpu_native_available', False)
        if _GPU_NATIVE_AVAILABLE:
            logger.info("GPU native interpolation (NVOF + GPU warp) available")
    except ImportError:
        _GPU_NATIVE_AVAILABLE = False
    except Exception as e:
        _GPU_NATIVE_AVAILABLE = False
        logger.debug(f"GPU native check failed: {e}")

    return _GPU_NATIVE_AVAILABLE


def get_active_stream(stream_id: str) -> Optional['InterpolatedStream']:
    """Get an active stream by ID."""
    return _active_streams.get(stream_id)


def stop_all_streams():
    """Stop all active streams."""
    for stream in list(_active_streams.values()):
        stream.stop()


class FrameLoader:
    """
    Async frame loader with pinned memory for faster CPU→GPU transfer.

    Pre-loads the next frame while the current frame is being processed,
    hiding I/O latency behind computation.
    """

    def __init__(self, cap, width: int, height: int, use_pinned_memory: bool = True):
        self.cap = cap
        self.width = width
        self.height = height
        self.use_pinned_memory = use_pinned_memory
        self._next_frame = None
        self._next_ret = None
        self._preload_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Try to set up pinned memory if torch is available
        self._pinned_buffer = None
        if use_pinned_memory:
            try:
                import torch
                if torch.cuda.is_available():
                    # Create pinned memory buffer for frame data
                    self._pinned_buffer = torch.empty(
                        (height, width, 3),
                        dtype=torch.uint8,
                        pin_memory=True
                    )
                    logger.debug("Pinned memory buffer allocated for frame loading")
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Could not allocate pinned memory: {e}")

    def _read_frame_sync(self) -> Tuple[bool, Optional['numpy.ndarray']]:
        """Synchronously read a frame (runs in thread pool)."""
        import cv2

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Resize if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        # Copy to pinned memory if available
        if self._pinned_buffer is not None:
            import numpy as np
            self._pinned_buffer.numpy()[:] = frame
            return True, self._pinned_buffer.numpy().copy()

        return True, frame

    async def preload_next(self):
        """Start preloading the next frame asynchronously."""
        async with self._lock:
            if self._preload_task is None or self._preload_task.done():
                loop = asyncio.get_event_loop()
                # run_in_executor returns a Future, not a coroutine
                self._preload_task = loop.run_in_executor(_executor, self._read_frame_sync)

    async def get_frame(self) -> Tuple[bool, Optional['numpy.ndarray']]:
        """
        Get the next frame, using preloaded data if available.
        Automatically starts preloading the following frame.
        """
        async with self._lock:
            # If we have a preload task, wait for it
            if self._preload_task is not None:
                try:
                    ret, frame = await self._preload_task
                    self._preload_task = None

                    # Start preloading the next frame
                    loop = asyncio.get_event_loop()
                    # run_in_executor returns a Future, not a coroutine
                    self._preload_task = loop.run_in_executor(_executor, self._read_frame_sync)

                    return ret, frame
                except Exception as e:
                    logger.warning(f"Preload failed: {e}")
                    self._preload_task = None

            # Fallback: read synchronously
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(_executor, self._read_frame_sync)

    def cleanup(self):
        """Clean up resources."""
        self._pinned_buffer = None
        if self._preload_task and not self._preload_task.done():
            self._preload_task.cancel()


class GPUFrameLoader:
    """
    GPU-native frame loader using PyNvVideoCodec.

    Decodes video frames directly to GPU memory, eliminating CPU→GPU transfer
    bottleneck. Provides both numpy arrays (for FFmpeg) and GPU tensors
    (for gpu_native interpolation).

    Performance benefit: ~14ms saved per frame at 1440p (GPU decode vs CPU decode + upload)
    """

    def __init__(self, video_path: str, width: int, height: int):
        self.video_path = video_path
        self.width = width
        self.height = height
        self._decoder = None
        self._frame_index = 0
        self._total_frames = 0
        self._lock = asyncio.Lock()
        self._initialized = False

        # Try to initialize GPU decoder
        try:
            from .gpu_video_pipeline import GPUVideoDecoder, get_gpu_pipeline_status
            status = get_gpu_pipeline_status()

            if status.get('pynvvideocodec_available', False):
                self._decoder = GPUVideoDecoder(video_path)
                self._total_frames = self._decoder.total_frames
                self._initialized = True
                logger.info(f"GPU frame loader initialized for {video_path}")
            else:
                logger.debug("PyNvVideoCodec not available for GPUFrameLoader")
        except ImportError:
            logger.debug("gpu_video_pipeline not available")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU frame loader: {e}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _get_frame_sync(self) -> Tuple[bool, Optional['numpy.ndarray'], Optional['torch.Tensor']]:
        """
        Synchronously get next frame.

        Returns:
            Tuple of (success, numpy_frame, gpu_tensor)
            - numpy_frame for FFmpeg output
            - gpu_tensor for GPU-native interpolation
        """
        if not self._initialized or self._decoder is None:
            return False, None, None

        if self._frame_index >= self._total_frames:
            return False, None, None

        try:
            # Get GPU frame
            gpu_frame = self._decoder.get_frame_gpu(self._frame_index)
            if gpu_frame is None:
                return False, None, None

            # Get numpy version for FFmpeg
            numpy_frame = self._decoder.get_frame(self._frame_index)

            self._frame_index += 1
            return True, numpy_frame, gpu_frame

        except Exception as e:
            logger.warning(f"GPU frame read failed at {self._frame_index}: {e}")
            return False, None, None

    async def get_frame(self) -> Tuple[bool, Optional['numpy.ndarray'], Optional['torch.Tensor']]:
        """
        Get the next frame (both numpy and GPU tensor versions).

        Returns:
            Tuple of (success, numpy_frame, gpu_tensor)
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(_executor, self._get_frame_sync)

    async def get_frame_gpu_only(self) -> Tuple[bool, Optional['torch.Tensor']]:
        """
        Get only the GPU tensor (faster when numpy not needed).

        For full GPU pipeline where FFmpeg reads from GPU encoder.
        """
        if not self._initialized or self._decoder is None:
            return False, None

        async with self._lock:
            if self._frame_index >= self._total_frames:
                return False, None

            try:
                loop = asyncio.get_event_loop()
                gpu_frame = await loop.run_in_executor(
                    _executor,
                    self._decoder.get_frame_gpu,
                    self._frame_index
                )
                if gpu_frame is not None:
                    self._frame_index += 1
                    return True, gpu_frame
                return False, None
            except Exception as e:
                logger.warning(f"GPU-only frame read failed: {e}")
                return False, None

    def seek(self, frame_index: int):
        """Seek to a specific frame."""
        if 0 <= frame_index < self._total_frames:
            self._frame_index = frame_index

    def cleanup(self):
        """Clean up resources."""
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None
        self._initialized = False


class PerformanceMonitor:
    """Tracks and logs interpolation performance metrics."""

    def __init__(self, stream_id: str, target_fps: float):
        self.stream_id = stream_id
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.033

        self._frame_times: List[float] = []
        self._interpolation_times: List[float] = []
        self._frames_processed = 0
        self._behind_realtime_count = 0
        self._last_gpu_log_time = 0
        self._start_time = time.time()

    def record_frame(self, interpolation_time: float, total_frame_time: float):
        """Record timing for a processed frame."""
        self._frame_times.append(total_frame_time)
        self._interpolation_times.append(interpolation_time)
        self._frames_processed += 1

        # Check if we're falling behind real-time
        if total_frame_time > self.target_frame_time:
            self._behind_realtime_count += 1

        # Periodic logging
        if self._frames_processed % _PERF_LOG_INTERVAL == 0:
            self._log_performance()

    def _log_performance(self):
        """Log performance statistics."""
        if not self._frame_times:
            return

        avg_frame_time = sum(self._frame_times[-_PERF_LOG_INTERVAL:]) / min(len(self._frame_times), _PERF_LOG_INTERVAL)
        avg_interp_time = sum(self._interpolation_times[-_PERF_LOG_INTERVAL:]) / min(len(self._interpolation_times), _PERF_LOG_INTERVAL)
        actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        behind_pct = (self._behind_realtime_count / self._frames_processed * 100) if self._frames_processed > 0 else 0

        logger.info(
            f"[Stream {self.stream_id}] Perf: {self._frames_processed} frames, "
            f"avg {avg_interp_time*1000:.1f}ms/interp, "
            f"{actual_fps:.1f} fps (target {self.target_fps}), "
            f"{behind_pct:.1f}% behind realtime"
        )

        # Log GPU memory periodically
        current_time = time.time()
        if current_time - self._last_gpu_log_time > 30:  # Every 30 seconds
            self._log_gpu_memory()
            self._last_gpu_log_time = current_time

    def _log_gpu_memory(self):
        """Log GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(
                    f"[Stream {self.stream_id}] GPU memory: "
                    f"{allocated:.0f}MB allocated, {reserved:.0f}MB reserved"
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU memory stats: {e}")

    def get_summary(self) -> dict:
        """Get performance summary."""
        elapsed = time.time() - self._start_time
        return {
            "frames_processed": self._frames_processed,
            "elapsed_seconds": elapsed,
            "avg_fps": self._frames_processed / elapsed if elapsed > 0 else 0,
            "behind_realtime_pct": (self._behind_realtime_count / self._frames_processed * 100) if self._frames_processed > 0 else 0
        }


class InterpolatedStream:
    """
    Manages an interpolated video stream.

    Standard Pipeline: Video File → cv2.VideoCapture → FrameInterpolator → FFmpeg stdin → HLS segments

    GPU Native Pipeline (quality="gpu_native"):
    Video File → GPU Decode → NVOF → GPU Warp → FFmpeg/NVENC → HLS segments
    (Frames stay on GPU through interpolation for maximum performance)

    Optimizations:
    - Async frame preloading with pinned memory
    - Batched interpolation for multiple t values
    - NVENC hardware encoding when available
    - GPU video decode when available (PyNvVideoCodec)
    - GPU native interpolation (NVOF + GPU warp) for real-time performance
    - Performance monitoring
    """

    def __init__(
        self,
        video_path: str,
        target_fps: int = 60,
        use_gpu: bool = True,
        quality: str = "fast",
        use_nvenc: Optional[bool] = None,  # None = auto-detect
        enable_perf_logging: bool = True,
        use_gpu_native: Optional[bool] = None,  # None = auto (use if quality="gpu_native")
        target_bitrate: Optional[str] = None,  # Target bitrate (e.g., "4M", "1536K")
        target_resolution: Optional[tuple] = None,  # Target resolution (width, height)
        start_position: float = 0.0  # Seek position in seconds
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.use_gpu = use_gpu
        self.quality = quality
        self.use_nvenc = use_nvenc if use_nvenc is not None else _check_nvenc_available()
        self.enable_perf_logging = enable_perf_logging
        self.stream_id = str(uuid.uuid4())[:8]

        # Quality settings
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution
        self.start_position = start_position

        # Determine if we should use GPU native mode
        if use_gpu_native is not None:
            self.use_gpu_native = use_gpu_native
        elif quality == "gpu_native":
            # Auto-detect based on availability
            self.use_gpu_native = _check_gpu_native_available() and use_gpu
        else:
            self.use_gpu_native = False

        # Try GPU video decode for gpu_native mode
        self.use_gpu_decode = self.use_gpu_native and _check_gpu_pipeline_available()

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._temp_dir: Optional[Path] = None
        self._error: Optional[str] = None
        self._perf_monitor: Optional[PerformanceMonitor] = None
        self._gpu_frame_loader: Optional[GPUFrameLoader] = None

        # Log pipeline selection
        if self.use_gpu_native:
            logger.info(f"[Stream {self.stream_id}] Using GPU Native pipeline "
                       f"(decode={self.use_gpu_decode}, nvenc={self.use_nvenc})")
        else:
            logger.debug(f"[Stream {self.stream_id}] Using standard pipeline "
                        f"(quality={quality}, nvenc={self.use_nvenc})")

        # Register stream
        _active_streams[self.stream_id] = self

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

    def _build_ffmpeg_command(self, width: int, height: int) -> List[str]:
        """Build FFmpeg command with optimal encoder selection."""
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(self.target_fps),
            '-i', '-',  # Read from stdin
        ]

        # Add scaling filter if needed
        if self.target_resolution:
            target_width, target_height = self.target_resolution
            cmd.extend([
                '-vf', f'scale={target_width}:{target_height}:flags=lanczos'
            ])

        # Select encoder
        if self.use_nvenc:
            # NVENC hardware encoding
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',  # Fastest NVENC preset
                '-tune', 'll',   # Low latency
            ])

            # Use CBR mode for predictable bitrate, or VBR for original quality
            if self.target_bitrate:
                # CBR mode for consistent bitrate
                cmd.extend([
                    '-rc', 'cbr',
                    '-b:v', self.target_bitrate,
                    '-maxrate', self.target_bitrate,
                    '-bufsize', f'{int(self.target_bitrate.rstrip("MK")) * 2}M' if 'M' in self.target_bitrate else f'{int(self.target_bitrate.rstrip("MK")) * 2}K',
                ])
            else:
                # VBR mode (original quality)
                cmd.extend([
                    '-rc', 'vbr',
                    '-cq', '23',
                    '-b:v', '0',     # VBR mode
                ])

            logger.info(f"[Stream {self.stream_id}] Using NVENC hardware encoder")
        else:
            # Software encoding
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
            ])

            if self.target_bitrate:
                # Use specified bitrate
                cmd.extend([
                    '-b:v', self.target_bitrate,
                    '-bufsize', f'{int(self.target_bitrate.rstrip("MK")) * 2}M' if 'M' in self.target_bitrate else f'{int(self.target_bitrate.rstrip("MK")) * 2}K',
                ])
            else:
                # Use CRF for original quality (default)
                cmd.extend([
                    '-crf', '23',
                ])

        # Common output options
        cmd.extend([
            '-g', str(self.target_fps * 2),  # GOP size = 2 seconds
            '-f', 'hls',
            '-hls_time', '2',
            '-hls_list_size', '10',
            '-hls_flags', 'delete_segments+append_list',
            '-hls_segment_filename', str(self._temp_dir / 'segment_%03d.ts'),
            str(self.playlist_path)
        ])

        return cmd

    async def start(self) -> bool:
        """Start the interpolated stream."""
        if self._running:
            return True

        try:
            # Import here to avoid circular imports
            import cv2
            from .optical_flow import FrameInterpolator, HAS_CV2

            if not HAS_CV2:
                self._error = "OpenCV not available"
                return False

            # Create temp directory for HLS output
            self._temp_dir = Path(tempfile.mkdtemp(prefix="localbooru_hls_"))
            logger.info(f"[Stream {self.stream_id}] HLS output: {self._temp_dir}")

            # Open video file - try GPU decode first if enabled
            cap = None
            source_fps = 0
            width = 0
            height = 0
            total_frames = 0

            if self.use_gpu_decode:
                try:
                    # Try GPU video decoder
                    self._gpu_frame_loader = GPUFrameLoader(self.video_path, 0, 0)
                    if self._gpu_frame_loader.is_initialized:
                        # Get video properties from GPU decoder
                        # Note: We still need cv2 for metadata
                        cap = cv2.VideoCapture(self.video_path)
                        source_fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        cap = None  # We'll use GPU loader instead
                        logger.info(f"[Stream {self.stream_id}] Using GPU video decode")
                    else:
                        self._gpu_frame_loader.cleanup()
                        self._gpu_frame_loader = None
                        self.use_gpu_decode = False
                except Exception as e:
                    logger.warning(f"[Stream {self.stream_id}] GPU decode init failed: {e}")
                    if self._gpu_frame_loader:
                        self._gpu_frame_loader.cleanup()
                        self._gpu_frame_loader = None
                    self.use_gpu_decode = False

            # Fall back to cv2.VideoCapture if GPU decode unavailable
            if cap is None and self._gpu_frame_loader is None:
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    self._error = f"Failed to open video: {self.video_path}"
                    return False

                # Get video properties
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(
                f"[Stream {self.stream_id}] Source: {width}x{height} @ {source_fps:.2f} fps, "
                f"{total_frames} frames"
            )

            # Calculate interpolation ratio
            fps_multiplier = self.target_fps / source_fps if source_fps > 0 else 2
            frames_between = max(0, int(fps_multiplier) - 1)

            logger.info(
                f"[Stream {self.stream_id}] Target: {self.target_fps} fps "
                f"({frames_between} interpolated frames between each source frame)"
            )

            # Initialize interpolator
            interpolator = FrameInterpolator(use_gpu=self.use_gpu, quality=self.quality)

            # Initialize performance monitor
            if self.enable_perf_logging:
                self._perf_monitor = PerformanceMonitor(self.stream_id, self.target_fps)

            # Start FFmpeg HLS muxer
            ffmpeg_cmd = self._build_ffmpeg_command(width, height)

            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            self._running = True

            # Start processing task - choose pipeline based on mode
            if self.use_gpu_native and self._gpu_frame_loader is not None:
                # Full GPU pipeline: GPU decode → GPU interpolation → FFmpeg
                self._task = asyncio.create_task(
                    self._process_video_gpu_native(
                        interpolator, width, height, frames_between
                    )
                )
            elif self._gpu_frame_loader is not None:
                # GPU decode with standard interpolation
                self._task = asyncio.create_task(
                    self._process_video_gpu_decode(
                        interpolator, width, height, frames_between
                    )
                )
            else:
                # Standard pipeline: cv2.VideoCapture → FrameInterpolator → FFmpeg
                self._task = asyncio.create_task(
                    self._process_video(cap, interpolator, width, height, frames_between)
                )

            return True

        except Exception as e:
            logger.error(f"[Stream {self.stream_id}] Failed to start: {e}")
            self._error = str(e)
            self.stop()
            return False

    async def _interpolate_batched(
        self,
        interpolator,
        frame1: 'numpy.ndarray',
        frame2: 'numpy.ndarray',
        frames_between: int
    ) -> List['numpy.ndarray']:
        """
        Batch multiple interpolation positions for efficiency.

        When frames_between > 1, this generates all t values and attempts
        to process them in a single batch through the interpolator.
        """
        if frames_between <= 0:
            return []

        # Check if interpolator supports batch interpolation (RIFE)
        if hasattr(interpolator, 'interpolate_batch'):
            # Generate all t values
            t_values = [i / (frames_between + 1) for i in range(1, frames_between + 1)]

            # Run batched interpolation in thread pool
            loop = asyncio.get_event_loop()
            try:
                interp_frames = await loop.run_in_executor(
                    _executor,
                    interpolator.interpolate_batch,
                    frame1,
                    frame2,
                    t_values
                )
                return interp_frames
            except Exception as e:
                logger.debug(f"Batch interpolation failed, falling back to sequential: {e}")

        # Fallback: sequential interpolation
        interp_frames = []
        loop = asyncio.get_event_loop()

        for i in range(1, frames_between + 1):
            t = i / (frames_between + 1)
            interp_frame = await loop.run_in_executor(
                _executor,
                interpolator.interpolate,
                frame1,
                frame2,
                t
            )
            interp_frames.append(interp_frame)

        return interp_frames

    async def _process_video(
        self,
        cap,
        interpolator,
        width: int,
        height: int,
        frames_between: int
    ):
        """Process video frames and write to FFmpeg with optimizations."""
        import cv2

        # Create async frame loader with pinned memory
        frame_loader = FrameLoader(cap, width, height, use_pinned_memory=self.use_gpu)

        # Start preloading first frame
        await frame_loader.preload_next()

        try:
            prev_frame = None
            frame_count = 0

            while self._running:
                frame_start_time = time.time()

                # Get next frame (uses preloaded data)
                ret, frame = await frame_loader.get_frame()
                if not ret:
                    logger.info(f"[Stream {self.stream_id}] End of video")
                    break

                interp_time = 0

                # If we have a previous frame, generate interpolated frames
                if prev_frame is not None and frames_between > 0:
                    interp_start = time.time()

                    # Use batched interpolation for efficiency
                    interp_frames = await self._interpolate_batched(
                        interpolator, prev_frame, frame, frames_between
                    )

                    interp_time = time.time() - interp_start

                    # Write interpolated frames to FFmpeg
                    for interp_frame in interp_frames:
                        if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                            try:
                                self._ffmpeg_proc.stdin.write(interp_frame.tobytes())
                            except BrokenPipeError:
                                logger.warning(f"[Stream {self.stream_id}] FFmpeg pipe broken")
                                self._running = False
                                break

                # Write original frame
                if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                    try:
                        self._ffmpeg_proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.warning(f"[Stream {self.stream_id}] FFmpeg pipe broken")
                        self._running = False
                        break

                prev_frame = frame.copy()
                frame_count += 1

                # Record performance metrics
                frame_total_time = time.time() - frame_start_time
                if self._perf_monitor:
                    self._perf_monitor.record_frame(interp_time, frame_total_time)

                # Yield to event loop periodically
                if frame_count % 10 == 0:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info(f"[Stream {self.stream_id}] Processing cancelled")
        except Exception as e:
            logger.error(f"[Stream {self.stream_id}] Processing error: {e}")
            self._error = str(e)
        finally:
            # Log final performance summary
            if self._perf_monitor:
                summary = self._perf_monitor.get_summary()
                logger.info(
                    f"[Stream {self.stream_id}] Final stats: "
                    f"{summary['frames_processed']} frames in {summary['elapsed_seconds']:.1f}s "
                    f"({summary['avg_fps']:.1f} avg fps)"
                )

            frame_loader.cleanup()
            cap.release()
            interpolator.cleanup()
            self._finish_ffmpeg()

    async def _process_video_gpu_decode(
        self,
        interpolator,
        width: int,
        height: int,
        frames_between: int
    ):
        """
        Process video using GPU decode with standard interpolation.

        Uses GPUFrameLoader for decode, but standard interpolation pipeline.
        This is faster than cv2.VideoCapture but doesn't use GPU-to-GPU interpolation.
        """
        if self._gpu_frame_loader is None:
            logger.error(f"[Stream {self.stream_id}] GPU frame loader not initialized")
            return

        try:
            prev_frame = None
            frame_count = 0

            while self._running:
                frame_start_time = time.time()

                # Get frame from GPU loader (returns both numpy and GPU tensor)
                ret, frame, _ = await self._gpu_frame_loader.get_frame()
                if not ret or frame is None:
                    logger.info(f"[Stream {self.stream_id}] End of video (GPU decode)")
                    break

                interp_time = 0

                # If we have a previous frame, generate interpolated frames
                if prev_frame is not None and frames_between > 0:
                    interp_start = time.time()

                    # Use batched interpolation
                    interp_frames = await self._interpolate_batched(
                        interpolator, prev_frame, frame, frames_between
                    )

                    interp_time = time.time() - interp_start

                    # Write interpolated frames to FFmpeg
                    for interp_frame in interp_frames:
                        if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                            try:
                                self._ffmpeg_proc.stdin.write(interp_frame.tobytes())
                            except BrokenPipeError:
                                logger.warning(f"[Stream {self.stream_id}] FFmpeg pipe broken")
                                self._running = False
                                break

                # Write original frame
                if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                    try:
                        self._ffmpeg_proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.warning(f"[Stream {self.stream_id}] FFmpeg pipe broken")
                        self._running = False
                        break

                prev_frame = frame.copy()
                frame_count += 1

                # Record performance metrics
                frame_total_time = time.time() - frame_start_time
                if self._perf_monitor:
                    self._perf_monitor.record_frame(interp_time, frame_total_time)

                # Yield to event loop periodically
                if frame_count % 10 == 0:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info(f"[Stream {self.stream_id}] Processing cancelled (GPU decode)")
        except Exception as e:
            logger.error(f"[Stream {self.stream_id}] Processing error (GPU decode): {e}")
            self._error = str(e)
        finally:
            if self._perf_monitor:
                summary = self._perf_monitor.get_summary()
                logger.info(
                    f"[Stream {self.stream_id}] Final stats (GPU decode): "
                    f"{summary['frames_processed']} frames in {summary['elapsed_seconds']:.1f}s "
                    f"({summary['avg_fps']:.1f} avg fps)"
                )

            if self._gpu_frame_loader:
                self._gpu_frame_loader.cleanup()
                self._gpu_frame_loader = None
            interpolator.cleanup()
            self._finish_ffmpeg()

    async def _process_video_gpu_native(
        self,
        interpolator,
        width: int,
        height: int,
        frames_between: int
    ):
        """
        Full GPU pipeline: GPU decode → GPU interpolation (NVOF + warp) → FFmpeg.

        This is the fastest pipeline for real-time interpolation.
        Frames stay on GPU through interpolation, only downloaded for FFmpeg encoding.

        Expected performance at 1440p: ~12ms per interpolated frame (83 fps)
        """
        if self._gpu_frame_loader is None:
            logger.error(f"[Stream {self.stream_id}] GPU frame loader not initialized")
            return

        # Check if interpolator has GPU-native support
        gpu_native_interp = getattr(interpolator, '_gpu_native', None)
        use_gpu_native_interp = (gpu_native_interp is not None and
                                  hasattr(gpu_native_interp, 'interpolate_gpu'))

        if use_gpu_native_interp:
            logger.info(f"[Stream {self.stream_id}] Using full GPU-to-GPU interpolation")
        else:
            logger.info(f"[Stream {self.stream_id}] GPU decode with standard interpolation")

        try:
            prev_frame_np = None
            prev_frame_gpu = None
            frame_count = 0

            while self._running:
                frame_start_time = time.time()

                # Get frame from GPU loader (both numpy and GPU tensor)
                ret, frame_np, frame_gpu = await self._gpu_frame_loader.get_frame()
                if not ret or frame_np is None:
                    logger.info(f"[Stream {self.stream_id}] End of video (GPU native)")
                    break

                interp_time = 0

                # If we have a previous frame, generate interpolated frames
                if prev_frame_np is not None and frames_between > 0:
                    interp_start = time.time()

                    interp_frames = []

                    # Try GPU-to-GPU interpolation if available
                    if use_gpu_native_interp and prev_frame_gpu is not None and frame_gpu is not None:
                        loop = asyncio.get_event_loop()

                        for i in range(1, frames_between + 1):
                            t = i / (frames_between + 1)
                            try:
                                # GPU-to-GPU interpolation
                                interp_gpu = await loop.run_in_executor(
                                    _executor,
                                    gpu_native_interp.interpolate_gpu,
                                    prev_frame_gpu,
                                    frame_gpu,
                                    t
                                )
                                # Download for FFmpeg (necessary until we have GPU encoder)
                                interp_frames.append(interp_gpu.cpu().numpy())
                            except Exception as e:
                                logger.debug(f"GPU-to-GPU interpolation failed: {e}")
                                # Fall back to standard interpolation
                                interp_frame = await loop.run_in_executor(
                                    _executor,
                                    interpolator.interpolate,
                                    prev_frame_np,
                                    frame_np,
                                    t
                                )
                                interp_frames.append(interp_frame)
                    else:
                        # Standard interpolation path
                        interp_frames = await self._interpolate_batched(
                            interpolator, prev_frame_np, frame_np, frames_between
                        )

                    interp_time = time.time() - interp_start

                    # Write interpolated frames to FFmpeg
                    for interp_frame in interp_frames:
                        if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                            try:
                                self._ffmpeg_proc.stdin.write(interp_frame.tobytes())
                            except BrokenPipeError:
                                logger.warning(f"[Stream {self.stream_id}] FFmpeg pipe broken")
                                self._running = False
                                break

                # Write original frame
                if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                    try:
                        self._ffmpeg_proc.stdin.write(frame_np.tobytes())
                    except BrokenPipeError:
                        logger.warning(f"[Stream {self.stream_id}] FFmpeg pipe broken")
                        self._running = False
                        break

                prev_frame_np = frame_np.copy()
                prev_frame_gpu = frame_gpu
                frame_count += 1

                # Record performance metrics
                frame_total_time = time.time() - frame_start_time
                if self._perf_monitor:
                    self._perf_monitor.record_frame(interp_time, frame_total_time)

                # Yield to event loop periodically
                if frame_count % 10 == 0:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info(f"[Stream {self.stream_id}] Processing cancelled (GPU native)")
        except Exception as e:
            logger.error(f"[Stream {self.stream_id}] Processing error (GPU native): {e}")
            self._error = str(e)
        finally:
            if self._perf_monitor:
                summary = self._perf_monitor.get_summary()
                mode = "GPU native" if use_gpu_native_interp else "GPU decode"
                logger.info(
                    f"[Stream {self.stream_id}] Final stats ({mode}): "
                    f"{summary['frames_processed']} frames in {summary['elapsed_seconds']:.1f}s "
                    f"({summary['avg_fps']:.1f} avg fps)"
                )

            if self._gpu_frame_loader:
                self._gpu_frame_loader.cleanup()
                self._gpu_frame_loader = None
            interpolator.cleanup()
            self._finish_ffmpeg()

    def _finish_ffmpeg(self):
        """Gracefully close FFmpeg."""
        if self._ffmpeg_proc:
            try:
                if self._ffmpeg_proc.stdin:
                    self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.kill()
            except Exception as e:
                logger.warning(f"[Stream {self.stream_id}] FFmpeg cleanup error: {e}")
            finally:
                self._ffmpeg_proc = None

    def stop(self):
        """Stop the stream and clean up."""
        logger.info(f"[Stream {self.stream_id}] Stopping")
        self._running = False

        # Cancel processing task
        if self._task and not self._task.done():
            self._task.cancel()

        # Stop FFmpeg
        self._finish_ffmpeg()

        # Clean up GPU frame loader
        if self._gpu_frame_loader:
            self._gpu_frame_loader.cleanup()
            self._gpu_frame_loader = None

        # Clean up temp directory (after a delay to allow final segment reads)
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"[Stream {self.stream_id}] Cleanup error: {e}")
            self._temp_dir = None

        # Unregister stream
        if self.stream_id in _active_streams:
            del _active_streams[self.stream_id]

    def get_file_path(self, filename: str) -> Optional[Path]:
        """Get path to an HLS file (playlist or segment)."""
        if not self._temp_dir:
            return None
        file_path = self._temp_dir / filename
        if file_path.exists():
            return file_path
        return None

    def get_performance_stats(self) -> Optional[dict]:
        """Get current performance statistics."""
        if self._perf_monitor:
            return self._perf_monitor.get_summary()
        return None

    async def wait_for_ready(self, min_segments: int = 2, timeout: float = 30.0) -> bool:
        """Wait for enough HLS segments to be buffered before playback."""
        import time
        start = time.time()

        while self._running and (time.time() - start) < timeout:
            if self.playlist_ready and self.segments_ready >= min_segments:
                logger.info(
                    f"[Stream {self.stream_id}] Ready with {self.segments_ready} segments"
                )
                return True

            # Check for errors
            if self._error:
                logger.warning(f"[Stream {self.stream_id}] Error during buffering: {self._error}")
                return False

            await asyncio.sleep(0.5)

        logger.warning(
            f"[Stream {self.stream_id}] Timeout waiting for segments "
            f"(got {self.segments_ready}, need {min_segments})"
        )
        return False


async def create_interpolated_stream(
    video_path: str,
    target_fps: int = 60,
    use_gpu: bool = True,
    quality: str = "fast",
    wait_for_buffer: bool = True,
    min_segments: int = 2,
    use_nvenc: Optional[bool] = None,
    enable_perf_logging: bool = True,
    use_gpu_native: Optional[bool] = None,
    target_bitrate: Optional[str] = None,
    target_resolution: Optional[tuple] = None,
    start_position: float = 0.0
) -> Optional[InterpolatedStream]:
    """
    Create and start an interpolated stream.

    Args:
        video_path: Path to source video file
        target_fps: Target frame rate for output
        use_gpu: Use GPU acceleration if available
        quality: Interpolation quality preset ("fast", "balanced", "quality", "gpu_native")
        wait_for_buffer: Wait for initial segments before returning
        min_segments: Minimum segments to buffer before ready
        use_nvenc: Use NVENC hardware encoding (None = auto-detect)
        enable_perf_logging: Enable performance monitoring and logging
        use_gpu_native: Force GPU native pipeline (None = auto, based on quality preset)
        target_bitrate: Target bitrate (e.g., "4M", "1536K")
        target_resolution: Target resolution (width, height)
        start_position: Seek position in seconds

    Returns:
        InterpolatedStream instance or None on failure

    Quality Presets:
        - "fast": RIFE-NCNN or OpenCV CUDA (good balance of speed/quality)
        - "balanced": RIFE with medium settings
        - "quality": RIFE with high quality settings
        - "gpu_native": Full GPU pipeline with NVOF + GPU warp (fastest real-time)
    """
    # Stop any existing streams first
    stop_all_streams()

    stream = InterpolatedStream(
        video_path=video_path,
        target_fps=target_fps,
        use_gpu=use_gpu,
        quality=quality,
        use_nvenc=use_nvenc,
        enable_perf_logging=enable_perf_logging,
        use_gpu_native=use_gpu_native,
        target_bitrate=target_bitrate,
        target_resolution=target_resolution,
        start_position=start_position
    )

    if not await stream.start():
        return None

    # Wait for buffer to fill before returning
    if wait_for_buffer:
        if not await stream.wait_for_ready(min_segments=min_segments):
            stream.stop()
            return None

    return stream


def shutdown():
    """Shutdown the optical flow stream service and cleanup resources"""
    print("[OpticalFlowStream] Shutting down...")
    stop_all_streams()
    _executor.shutdown(wait=True, cancel_futures=True)
    print("[OpticalFlowStream] Shutdown complete")
