"""
Main FrameInterpolator class - unified interface for all interpolation backends.
"""
import logging
from typing import Optional, List
import numpy as np

from .gpu_utils import (
    HAS_TORCH, CUDA_AVAILABLE, HAS_CV2, HAS_CV2_CUDA, HAS_RIFE_NCNN,
    RIFE_AVAILABLE, RIFE_MODEL_DIR, check_rife_availability,
)
from .base import QUALITY_PRESETS
from .frame_buffer import FastGPUBlendInterpolator
from .rife_ncnn import RifeNCNNInterpolator
from .rife import RIFEInterpolator
from .opencv_flow import GPUNativeInterpolator

logger = logging.getLogger(__name__)

# Conditional imports
if HAS_TORCH:
    import torch

if HAS_CV2:
    import cv2


class FrameInterpolator:
    """
    Frame interpolation using optical flow.

    Backend priority (depends on quality setting):
    - "gpu_native": Full GPU pipeline with NVOF - 60+fps at 1440p, motion-compensated
    - "realtime": FastGPUBlend - 60fps at 1440p, simple blend
    - "fast"/"balanced"/"quality": RIFE-NCNN > RIFE PyTorch > OpenCV CUDA > CPU

    For true real-time 60fps at 1440p with motion compensation, use quality="gpu_native".
    For simple blending, use quality="realtime".
    For high-quality pre-processing, use quality="fast" or higher.
    """

    def __init__(self, use_gpu: bool = True, quality: str = "fast", data_dir: Optional[str] = None):
        self.quality = quality if quality in QUALITY_PRESETS else "fast"
        self.params = QUALITY_PRESETS[self.quality]

        # Get data_dir from settings if not provided
        if data_dir is None:
            try:
                from api.config import get_settings
                settings = get_settings()
                data_dir = settings.data_dir
            except Exception:
                pass

        self.data_dir = data_dir

        # Check if gpu_native mode requested - full GPU pipeline with NVOF
        self.use_gpu_native = (self.params.get("backend") == "gpu_native" and
                               use_gpu and HAS_TORCH and CUDA_AVAILABLE)

        # Check if realtime mode requested - use fast GPU blend
        self.use_realtime_blend = (not self.use_gpu_native and
                                   self.params.get("backend") == "gpu_blend" and
                                   use_gpu and HAS_TORCH and CUDA_AVAILABLE)

        # Check RIFE PyTorch availability (will auto-download if needed)
        if not self.use_gpu_native and not self.use_realtime_blend:
            check_rife_availability(data_dir)

        # Backend selection priority: gpu_native > realtime_blend > RIFE-NCNN > RIFE PyTorch > OpenCV CUDA > OpenCV CPU
        self.use_rife_ncnn = (not self.use_gpu_native and not self.use_realtime_blend and use_gpu and HAS_RIFE_NCNN)
        self.use_rife_torch = (not self.use_gpu_native and not self.use_realtime_blend and use_gpu and RIFE_AVAILABLE and not self.use_rife_ncnn)
        self.use_cv2_cuda = (not self.use_gpu_native and not self.use_realtime_blend and use_gpu and HAS_CV2_CUDA and
                            not self.use_rife_ncnn and not self.use_rife_torch)
        self.use_torch_cuda = (not self.use_gpu_native and not self.use_realtime_blend and use_gpu and HAS_TORCH and CUDA_AVAILABLE and
                              not self.use_rife_ncnn and not self.use_rife_torch and not self.use_cv2_cuda)
        self.use_gpu = self.use_gpu_native or self.use_realtime_blend or self.use_rife_ncnn or self.use_rife_torch or self.use_cv2_cuda or self.use_torch_cuda

        # Interpolator instances
        self._gpu_native: Optional[GPUNativeInterpolator] = None
        self._fast_blend: Optional[FastGPUBlendInterpolator] = None
        self._rife_ncnn: Optional[RifeNCNNInterpolator] = None
        self._rife_torch: Optional[RIFEInterpolator] = None

        # OpenCV CUDA resources
        self._cuda_farneback = None

        self.device = (
            torch.device("cuda" if (self.use_gpu_native or self.use_realtime_blend or self.use_rife_torch or self.use_torch_cuda) else "cpu")
            if HAS_TORCH
            else None
        )
        self._initialized = False
        self._active_backend = None

        if self.use_gpu_native:
            logger.info(f"FrameInterpolator using GPU Native (NVOF + GPU warp) (quality={self.quality}) - REAL-TIME 60fps+ with motion compensation")
            self._active_backend = "gpu_native"
        elif self.use_realtime_blend:
            logger.info(f"FrameInterpolator using FastGPUBlend (quality={self.quality}) - REAL-TIME 60fps capable")
            self._active_backend = "gpu_blend"
        elif self.use_rife_ncnn:
            logger.info(f"FrameInterpolator using RIFE-NCNN Vulkan (quality={self.quality})")
            self._active_backend = "rife_ncnn"
        elif self.use_rife_torch:
            logger.info(f"FrameInterpolator using RIFE PyTorch (quality={self.quality})")
            self._active_backend = "rife_torch"
        elif self.use_cv2_cuda:
            logger.info(f"FrameInterpolator using OpenCV CUDA Farneback (quality={self.quality})")
            self._active_backend = "opencv_cuda"
        elif self.use_torch_cuda:
            logger.info(f"FrameInterpolator using PyTorch CUDA (quality={self.quality})")
            self._active_backend = "torch_cuda"
        elif HAS_CV2:
            logger.info(f"FrameInterpolator using OpenCV CPU Farneback (quality={self.quality})")
            self._active_backend = "farneback_cpu"
        else:
            logger.warning("No interpolation backend available")
            self._active_backend = "blend"

    def initialize(self):
        """Initialize GPU resources."""
        if self._initialized:
            return

        try:
            # Try GPU Native first (full GPU pipeline with NVOF)
            if self.use_gpu_native:
                preset = self.params.get("preset", "fast")
                use_ipc_worker = self.params.get("use_ipc_worker", True)  # Default to worker for Py3.14 opencv-cuda
                self._gpu_native = GPUNativeInterpolator(preset=preset, use_ipc_worker=use_ipc_worker)
                if self._gpu_native.initialize():
                    self._initialized = True
                    self._active_backend = "gpu_native"
                    logger.info("GPU Native interpolator initialized")
                    return
                else:
                    logger.warning("GPU Native initialization failed, falling back")
                    self._gpu_native = None
                    self.use_gpu_native = False
                    self.use_rife_ncnn = HAS_RIFE_NCNN  # Try RIFE-NCNN

            # Try FastGPUBlend (realtime mode)
            if self.use_realtime_blend and not self._initialized:
                self._fast_blend = FastGPUBlendInterpolator()
                if self._fast_blend.initialize():
                    self._initialized = True
                    self._active_backend = "gpu_blend"
                    logger.info("FastGPUBlend interpolator initialized (realtime)")
                    return
                else:
                    logger.warning("FastGPUBlend initialization failed, falling back")
                    self._fast_blend = None
                    self.use_realtime_blend = False
                    self.use_rife_ncnn = HAS_RIFE_NCNN

            # Try RIFE-NCNN (fastest neural network backend)
            if self.use_rife_ncnn and not self._initialized:
                # Use rife-v4 with single thread (multiple threads compete for GPU)
                self._rife_ncnn = RifeNCNNInterpolator(gpu_id=0, model="rife-v4")
                if self._rife_ncnn.initialize():
                    self._initialized = True
                    self._active_backend = "rife_ncnn"
                    logger.info("RIFE-NCNN interpolator initialized")
                    return
                else:
                    logger.warning("RIFE-NCNN initialization failed, falling back")
                    self._rife_ncnn = None
                    self.use_rife_ncnn = False
                    self.use_rife_torch = RIFE_AVAILABLE  # Try PyTorch RIFE

            # Try RIFE PyTorch
            if self.use_rife_torch and not self._initialized:
                self._rife_torch = RIFEInterpolator(
                    model_dir=RIFE_MODEL_DIR,
                    device='cuda' if CUDA_AVAILABLE else 'cpu'
                )
                if self._rife_torch.initialize(warm_up=True):
                    self._initialized = True
                    self._active_backend = "rife_torch"
                    logger.info("RIFE PyTorch interpolator initialized")
                    return
                else:
                    logger.warning("RIFE PyTorch initialization failed, falling back")
                    self._rife_torch = None
                    self.use_rife_torch = False
                    self.use_cv2_cuda = HAS_CV2_CUDA

            # Try OpenCV CUDA Farneback
            if self.use_cv2_cuda and not self._initialized:
                p = self.params
                self._cuda_farneback = cv2.cuda.FarnebackOpticalFlow.create(
                    numLevels=p["levels"],
                    pyrScale=p["pyr_scale"],
                    fastPyramids=True,
                    winSize=p["winsize"],
                    numIters=p["iterations"],
                    polyN=p["poly_n"],
                    polySigma=p["poly_sigma"],
                    flags=0
                )
                self._initialized = True
                self._active_backend = "opencv_cuda"
                logger.info("OpenCV CUDA Farneback initialized")

        except Exception as e:
            logger.error(f"Failed to initialize optical flow: {e}")
            self._rife_ncnn = None
            self._rife_torch = None
            self._cuda_farneback = None

    def interpolate(
        self, frame1: np.ndarray, frame2: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Generate intermediate frame at position t.

        Args:
            frame1: First frame (H, W, 3) uint8 BGR
            frame2: Second frame (H, W, 3) uint8 BGR
            t: Position between frames (0.0 = frame1, 1.0 = frame2)

        Returns:
            Interpolated frame (H, W, 3) uint8 BGR
        """
        # GPU Native path - full GPU pipeline with NVOF (fastest with motion compensation)
        if self.use_gpu_native and self._gpu_native is not None:
            return self._gpu_native.interpolate(frame1, frame2, t)

        # Fast GPU blend path - realtime simple blending
        if self.use_realtime_blend and self._fast_blend is not None:
            return self._fast_blend.interpolate(frame1, frame2, t)

        # RIFE-NCNN path - fastest Vulkan GPU interpolation
        if self.use_rife_ncnn and self._rife_ncnn is not None:
            return self._rife_ncnn.interpolate(frame1, frame2, t)

        # RIFE PyTorch path - neural network interpolation
        if self.use_rife_torch and self._rife_torch is not None:
            return self._rife_torch.interpolate(frame1, frame2, t)

        # OpenCV CUDA path - fast GPU optical flow
        if self.use_cv2_cuda:
            return self._interpolate_cv2_cuda(frame1, frame2, t)

        # CPU fallback - OpenCV Farneback
        if HAS_CV2:
            return self._interpolate_farneback(frame1, frame2, t)

        # Fallback - simple blend
        return self._blend_frames(frame1, frame2, t)

    def _interpolate_cv2_cuda(
        self, frame1: np.ndarray, frame2: np.ndarray, t: float
    ) -> np.ndarray:
        """GPU interpolation using OpenCV CUDA Farneback - all on GPU."""
        if not self._initialized:
            self.initialize()

        if not self._initialized or self._cuda_farneback is None:
            return self._blend_frames(frame1, frame2, t)

        try:
            h, w = frame1.shape[:2]

            # Upload frames to GPU
            gpu_frame1 = cv2.cuda_GpuMat()
            gpu_frame2 = cv2.cuda_GpuMat()
            gpu_frame1.upload(frame1)
            gpu_frame2.upload(frame2)

            # Convert to grayscale on GPU
            gpu_gray1 = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)
            gpu_gray2 = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)

            # Compute optical flow on GPU
            gpu_flow = self._cuda_farneback.calc(gpu_gray1, gpu_gray2, None)

            # Split flow into x and y components on GPU
            gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
            cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

            # Create coordinate maps on CPU (small overhead, but necessary)
            x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

            # Download flow and compute maps (flow is small relative to frame)
            flow_x = gpu_flow_x.download()
            flow_y = gpu_flow_y.download()

            map_x = (x - flow_x * t).astype(np.float32)
            map_y = (y - flow_y * t).astype(np.float32)

            # Upload maps and remap on GPU
            gpu_map_x = cv2.cuda_GpuMat()
            gpu_map_y = cv2.cuda_GpuMat()
            gpu_map_x.upload(map_x)
            gpu_map_y.upload(map_y)

            gpu_warped1 = cv2.cuda.remap(gpu_frame1, gpu_map_x, gpu_map_y,
                                         cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)

            # Blend on GPU
            gpu_result = cv2.cuda.addWeighted(gpu_warped1, 1 - t, gpu_frame2, t, 0)

            # Download final result
            return gpu_result.download()

        except Exception as e:
            logger.warning(f"OpenCV CUDA interpolation failed: {e}")
            return self._blend_frames(frame1, frame2, t)

    def _interpolate_farneback(
        self, frame1: np.ndarray, frame2: np.ndarray, t: float
    ) -> np.ndarray:
        """CPU interpolation using OpenCV Farneback optical flow."""
        if not HAS_CV2:
            return self._blend_frames(frame1, frame2, t)

        try:
            h, w = frame1.shape[:2]
            p = self.params
            flow_scale = p["flow_scale"]

            # Downscale for flow computation if needed
            if flow_scale < 1.0:
                small_h, small_w = int(h * flow_scale), int(w * flow_scale)
                small1 = cv2.resize(frame1, (small_w, small_h), interpolation=cv2.INTER_AREA)
                small2 = cv2.resize(frame2, (small_w, small_h), interpolation=cv2.INTER_AREA)
                gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Compute forward flow only (skip backward for speed)
            flow_fwd = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=p["pyr_scale"],
                levels=p["levels"],
                winsize=p["winsize"],
                iterations=p["iterations"],
                poly_n=p["poly_n"],
                poly_sigma=p["poly_sigma"],
                flags=0
            )

            # Upscale flow if we downscaled
            if flow_scale < 1.0:
                flow_fwd = cv2.resize(flow_fwd, (w, h), interpolation=cv2.INTER_LINEAR)
                flow_fwd = flow_fwd / flow_scale  # Scale flow vectors

            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Forward warp with simple interpolation (skip backward flow for speed)
            flow_t = flow_fwd * t
            map_x = (x - flow_t[:, :, 0]).astype(np.float32)
            map_y = (y - flow_t[:, :, 1]).astype(np.float32)

            # Warp frame1 toward frame2
            warped1 = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

            # Blend with original frame2 position
            result = cv2.addWeighted(warped1, 1 - t, frame2, t, 0)

            return result

        except Exception as e:
            logger.warning(f"Farneback interpolation failed: {e}")
            return self._blend_frames(frame1, frame2, t)

    def interpolate_sequence(
        self, frame1: np.ndarray, frame2: np.ndarray, num_frames: int
    ) -> List[np.ndarray]:
        """Generate multiple intermediate frames."""
        frames = []
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            frames.append(self.interpolate(frame1, frame2, t))
        return frames

    def _blend_frames(
        self, frame1: np.ndarray, frame2: np.ndarray, t: float
    ) -> np.ndarray:
        """Simple alpha blend fallback."""
        return (
            (1 - t) * frame1.astype(np.float32) + t * frame2.astype(np.float32)
        ).astype(np.uint8)

    def cleanup(self):
        """Release resources."""
        if self._gpu_native:
            self._gpu_native.cleanup()
            self._gpu_native = None
        if self._fast_blend:
            self._fast_blend.cleanup()
            self._fast_blend = None
        if self._rife_ncnn:
            self._rife_ncnn.cleanup()
            self._rife_ncnn = None
        if self._rife_torch:
            self._rife_torch.cleanup()
            self._rife_torch = None
        self._cuda_farneback = None
        self._initialized = False
        if HAS_TORCH and CUDA_AVAILABLE:
            torch.cuda.empty_cache()
