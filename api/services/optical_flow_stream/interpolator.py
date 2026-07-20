"""
Frame interpolation integration for optical flow streaming.

Provides frame loaders with optimized memory handling and batched interpolation
support for efficient GPU-accelerated frame interpolation.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy
    import torch

logger = logging.getLogger(__name__)

# Thread pool for blocking operations (shared with other modules)
_executor = ThreadPoolExecutor(max_workers=4)


def get_executor() -> ThreadPoolExecutor:
    """Get the shared thread pool executor."""
    return _executor


def shutdown_executor():
    """Shutdown the shared thread pool executor."""
    _executor.shutdown(wait=True, cancel_futures=True)


class FrameLoader:
    """
    Async frame loader with pinned memory for faster CPU->GPU transfer.

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

    Decodes video frames directly to GPU memory, eliminating CPU->GPU transfer
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
            from ..gpu_video_pipeline import GPUVideoDecoder, get_gpu_pipeline_status
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


async def interpolate_batched(
    interpolator,
    frame1: 'numpy.ndarray',
    frame2: 'numpy.ndarray',
    frames_between: int
) -> List['numpy.ndarray']:
    """
    Batch multiple interpolation positions for efficiency.

    When frames_between > 1, this generates all t values and attempts
    to process them in a single batch through the interpolator.

    Args:
        interpolator: FrameInterpolator instance
        frame1: First source frame
        frame2: Second source frame
        frames_between: Number of interpolated frames to generate

    Returns:
        List of interpolated frames
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
