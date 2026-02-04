"""
OpenCV-based optical flow implementations.

Contains:
- GPUNativeInterpolator: Full GPU pipeline with NVIDIA Optical Flow
"""
import logging
from typing import Optional
import numpy as np

from .gpu_utils import HAS_TORCH, CUDA_AVAILABLE, HAS_CV2, HAS_CV2_CUDA

logger = logging.getLogger(__name__)

# Conditional imports
if HAS_TORCH:
    import torch
    import torch.nn.functional as F

if HAS_CV2:
    import cv2


class GPUNativeInterpolator:
    """
    Full GPU pipeline frame interpolator using NVIDIA hardware acceleration.

    This interpolator keeps all operations on the GPU:
    - NVIDIA Optical Flow Accelerator for motion estimation (~10.8ms @ 1440p)
    - PyTorch grid_sample for frame warping (~0.17ms @ 1440p)
    - GPU blending (~0.3ms @ 1440p)

    Total expected: ~12ms per frame at 1440p (83 fps - REAL-TIME capable)

    Falls back to software paths if hardware unavailable:
    - OpenCV CUDA Farneback if NVOF unavailable
    - PyTorch GPU warp (always available with CUDA)
    """

    def __init__(self, width: int = 0, height: int = 0,
                 preset: str = "fast", gpu_id: int = 0,
                 use_ipc_worker: bool = False):
        """
        Initialize GPU native interpolator.

        Args:
            width: Frame width (0 = auto-detect from first frame)
            height: Frame height (0 = auto-detect from first frame)
            preset: NVOF performance preset ('slow', 'medium', 'fast')
            gpu_id: CUDA device ID
            use_ipc_worker: Use IPC worker for NVOF (default False - direct is faster, no CPU)
        """
        if not HAS_TORCH or not CUDA_AVAILABLE:
            raise RuntimeError("PyTorch with CUDA required for GPUNativeInterpolator")

        self.width = width
        self.height = height
        self.preset = preset
        self.gpu_id = gpu_id
        self.use_ipc_worker = use_ipc_worker

        self.device = torch.device(f'cuda:{gpu_id}')
        self._initialized = False

        # NVIDIA Optical Flow (direct or via worker)
        self._nvof_direct = None
        self._nvof_worker = None

        # Fallback: OpenCV CUDA Farneback
        self._cuda_farneback = None

        # Pre-allocated tensors for warping (PyTorch fallback path)
        self._grid_cache = None
        self._grid_shape = None

        # Backend tracking
        self._flow_backend = None
        self._warp_backend = "pytorch_cuda"

        # Frame caching for sequential interpolation (saves ~1.5ms per frame)
        self._cached_gpu_frame: Optional[cv2.cuda_GpuMat] = None
        self._cached_gpu_gray: Optional[cv2.cuda_GpuMat] = None
        self._cached_frame_id: Optional[int] = None  # id(frame) for identity check

    def initialize(self, width: int = 0, height: int = 0) -> bool:
        """Initialize GPU resources."""
        if self._initialized:
            return True

        # Update dimensions
        if width > 0:
            self.width = width
        if height > 0:
            self.height = height

        if self.width == 0 or self.height == 0:
            logger.warning("GPUNativeInterpolator: dimensions not set, will auto-detect")
            return True  # Will initialize on first frame

        try:
            # Try direct NVIDIA Optical Flow first (best performance - no CPU)
            # Only available if cv2.cuda is present in current Python
            if HAS_CV2_CUDA:
                self._init_nvof_direct()

            # Fall back to IPC worker if direct not available
            # This uses system Python with opencv-cuda via subprocess
            if self._flow_backend is None:
                self._init_nvof_worker()

            # Fall back to OpenCV CUDA Farneback if NVOF failed
            if self._flow_backend is None and HAS_CV2_CUDA:
                self._init_cuda_farneback()

            # Pre-allocate warp grid (for PyTorch fallback path)
            self._create_warp_grid()

            self._initialized = True
            logger.info(f"GPUNativeInterpolator initialized: {self.width}x{self.height}, "
                       f"flow={self._flow_backend}, warp={self._warp_backend}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GPUNativeInterpolator: {e}")
            return False

    def _init_nvof_direct(self):
        """Initialize direct NVIDIA Optical Flow."""
        try:
            from ..nvidia_of_worker import DirectNvidiaOpticalFlow, check_nvidia_of_available

            status = check_nvidia_of_available()
            if status['available']:
                self._nvof_direct = DirectNvidiaOpticalFlow(
                    self.width, self.height,
                    preset=self.preset,
                    gpu_id=self.gpu_id
                )
                if self._nvof_direct.initialize():
                    self._flow_backend = "nvidia_optflow"
                    logger.info("Using NVIDIA Optical Flow Accelerator")
                else:
                    self._nvof_direct = None
        except ImportError:
            logger.debug("nvidia_of_worker not available")
        except Exception as e:
            logger.warning(f"NVOF direct init failed: {e}")

    def _init_nvof_worker(self):
        """Initialize NVIDIA Optical Flow via IPC worker."""
        try:
            from ..nvidia_of_worker import NvidiaOpticalFlowWorker

            self._nvof_worker = NvidiaOpticalFlowWorker(
                self.width, self.height,
                preset=self.preset
            )
            # Worker initialization is async, defer to first use
            self._flow_backend = "nvidia_optflow_worker"
            logger.info("Using NVIDIA Optical Flow via IPC worker")
        except ImportError:
            logger.debug("nvidia_of_worker not available for IPC")
        except Exception as e:
            logger.warning(f"NVOF worker init failed: {e}")

    def _init_cuda_farneback(self):
        """Initialize OpenCV CUDA Farneback as fallback."""
        try:
            # Farneback parameters for fast flow
            self._cuda_farneback = cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=3,
                pyrScale=0.5,
                fastPyramids=True,
                winSize=15,
                numIters=3,
                polyN=5,
                polySigma=1.2,
                flags=0
            )
            self._flow_backend = "opencv_cuda_farneback"
            logger.info("Using OpenCV CUDA Farneback for optical flow")
        except Exception as e:
            logger.warning(f"OpenCV CUDA Farneback init failed: {e}")

    def _create_warp_grid(self):
        """Pre-allocate coordinate grid for PyTorch fallback warping."""
        if self.width == 0 or self.height == 0:
            return

        h, w = self.height, self.width

        # PyTorch path: normalized grid [-1, 1] for grid_sample
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        self._grid_cache = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        self._grid_shape = (h, w)

        # Note: cv2.cuda path uses WARP_RELATIVE_MAP which doesn't need
        # pre-computed coordinate grids - flow is used as relative offsets directly

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        Generate intermediate frame at position t.

        SVP-style full GPU pipeline: upload once, process on GPU, download once.

        Args:
            frame1: First frame (H, W, 3) uint8 BGR
            frame2: Second frame (H, W, 3) uint8 BGR
            t: Position between frames (0.0 = frame1, 1.0 = frame2)

        Returns:
            Interpolated frame (H, W, 3) uint8 BGR
        """
        h, w = frame1.shape[:2]

        # Auto-initialize with frame dimensions
        if not self._initialized or self._grid_shape != (h, w):
            self.width = w
            self.height = h
            self._initialized = False
            if not self.initialize(w, h):
                return self._blend_frames_cpu(frame1, frame2, t)

        try:
            # Full GPU pipeline using cv2.cuda (SVP-style)
            if self._nvof_direct is not None and HAS_CV2_CUDA:
                return self._interpolate_full_gpu(frame1, frame2, t)

            # Fallback: PyTorch-based pipeline
            f1_gpu = torch.from_numpy(frame1).to(self.device, non_blocking=True)
            f2_gpu = torch.from_numpy(frame2).to(self.device, non_blocking=True)

            flow_gpu = self._compute_flow_gpu(frame1, frame2)
            if flow_gpu is None:
                return self._blend_frames_gpu(f1_gpu, f2_gpu, t)

            result_gpu = self._warp_and_blend_gpu(f1_gpu, f2_gpu, flow_gpu, t)
            return result_gpu.cpu().numpy()

        except Exception as e:
            logger.warning(f"GPU interpolation failed: {e}")
            return self._blend_frames_cpu(frame1, frame2, t)

    def _interpolate_full_gpu(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        SVP-style full GPU interpolation using cv2.cuda with WARP_RELATIVE_MAP.

        Pipeline: Upload -> Grayscale -> NVOF -> Scale -> Split -> Remap -> Blend -> Download

        OPTIMIZATION: Uses WARP_RELATIVE_MAP flag which interprets flow as relative
        offsets, eliminating the need for coordinate grid computation entirely.

        OPTIMIZATION: Frame caching - when called sequentially (frame1 = previous frame2),
        reuses the cached GPU frame to avoid redundant upload (~1.5ms savings).

        Performance @ 1440p with FAST preset:
        - With frame caching: ~15.3ms (65 fps)
        - Without caching: ~17.0ms (59 fps)
        - NVOF hardware: ~9.3ms (bottleneck)
        """
        # Check if frame1 is the cached frame2 from the previous call
        frame1_id = id(frame1)
        cache_hit = (self._cached_frame_id == frame1_id and
                     self._cached_gpu_frame is not None and
                     self._cached_gpu_gray is not None)

        if cache_hit:
            # Reuse cached GPU data for frame1
            gpu_frame1 = self._cached_gpu_frame
            gpu_gray1 = self._cached_gpu_gray
        else:
            # Cache miss - upload frame1
            gpu_frame1 = cv2.cuda_GpuMat()
            gpu_frame1.upload(frame1)
            gpu_gray1 = cv2.cuda.cvtColor(gpu_frame1, cv2.COLOR_BGR2GRAY)

        # Always upload frame2
        gpu_frame2 = cv2.cuda_GpuMat()
        gpu_frame2.upload(frame2)
        gpu_gray2 = cv2.cuda.cvtColor(gpu_frame2, cv2.COLOR_BGR2GRAY)

        # Cache frame2 for the next call
        self._cached_gpu_frame = gpu_frame2
        self._cached_gpu_gray = gpu_gray2
        self._cached_frame_id = id(frame2)

        # Compute optical flow on GPU (NVOF hardware accelerated)
        # Returns int16 with 1/32 pixel precision as 2-channel GpuMat
        gpu_flow_raw, _ = self._nvof_direct._nvof.calc(gpu_gray1, gpu_gray2, None)

        # Scale flow on GPU: int16 -> float32, apply -t/32 scale factor
        # With WARP_RELATIVE_MAP: dst(x,y) = src(x + map_x, y + map_y)
        # For backward warp: we want src(x - flow_x*t, y - flow_y*t)
        # So use scale = -t/32 (negative for backward, /32 for NVOF precision)
        scale = -t / 32.0
        gpu_flow_scaled = gpu_flow_raw.convertTo(cv2.CV_32FC2, alpha=scale)

        # Split flow into x,y channels on GPU
        flow_channels = cv2.cuda.split(gpu_flow_scaled)
        gpu_flow_x = flow_channels[0]
        gpu_flow_y = flow_channels[1]

        # Warp using WARP_RELATIVE_MAP (flow as relative offsets - no grid needed!)
        gpu_warped1 = cv2.cuda.remap(
            gpu_frame1, gpu_flow_x, gpu_flow_y,
            cv2.INTER_LINEAR | cv2.WARP_RELATIVE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )

        # Blend on GPU
        gpu_result = cv2.cuda.addWeighted(gpu_warped1, 1 - t, gpu_frame2, t, 0)

        # Download result (only CPU transfer: output)
        return gpu_result.download()

    def interpolate_gpu(self, frame1_gpu: torch.Tensor, frame2_gpu: torch.Tensor,
                        t: float) -> torch.Tensor:
        """
        GPU-to-GPU interpolation (no CPU transfers).

        Use this when frames are already on GPU for maximum performance.

        Args:
            frame1_gpu: First frame as CUDA tensor (H, W, 3) uint8
            frame2_gpu: Second frame as CUDA tensor (H, W, 3) uint8
            t: Interpolation position

        Returns:
            Interpolated frame as CUDA tensor (H, W, 3) uint8
        """
        h, w = frame1_gpu.shape[:2]

        if not self._initialized or self._grid_shape != (h, w):
            self.width = w
            self.height = h
            self._initialized = False
            if not self.initialize(w, h):
                return torch.lerp(frame1_gpu.float(), frame2_gpu.float(), t).byte()

        try:
            # Convert to numpy for flow computation (flow APIs need numpy)
            # This is a necessary CPU roundtrip for now
            frame1_np = frame1_gpu.cpu().numpy()
            frame2_np = frame2_gpu.cpu().numpy()

            flow_gpu = self._compute_flow_gpu(frame1_np, frame2_np)
            if flow_gpu is None:
                return torch.lerp(frame1_gpu.float(), frame2_gpu.float(), t).byte()

            return self._warp_and_blend_gpu(frame1_gpu, frame2_gpu, flow_gpu, t)

        except Exception as e:
            logger.warning(f"GPU interpolation failed: {e}")
            return torch.lerp(frame1_gpu.float(), frame2_gpu.float(), t).byte()

    def _compute_flow_gpu(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[torch.Tensor]:
        """
        Compute optical flow and return as GPU tensor.

        Tries backends in order: NVOF direct > NVOF worker > OpenCV CUDA
        """
        try:
            flow_np = None

            # Try NVIDIA Optical Flow (direct)
            if self._nvof_direct is not None:
                flow_np = self._nvof_direct.compute_flow(frame1, frame2)

            # Try NVIDIA Optical Flow (worker) - this is async
            elif self._nvof_worker is not None:
                import asyncio
                # Run async in sync context
                loop = asyncio.new_event_loop()
                try:
                    flow_np = loop.run_until_complete(
                        self._nvof_worker.compute_flow(frame1, frame2)
                    )
                finally:
                    loop.close()

            # Fallback to OpenCV CUDA Farneback
            elif self._cuda_farneback is not None:
                flow_np = self._compute_flow_farneback(frame1, frame2)

            if flow_np is not None:
                # Upload flow to GPU
                return torch.from_numpy(flow_np).to(self.device)

            return None

        except Exception as e:
            logger.warning(f"Flow computation failed: {e}")
            return None

    def _compute_flow_farneback(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """Compute flow using OpenCV CUDA Farneback."""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Upload to GPU
            gpu_gray1 = cv2.cuda_GpuMat()
            gpu_gray2 = cv2.cuda_GpuMat()
            gpu_gray1.upload(gray1)
            gpu_gray2.upload(gray2)

            # Compute flow
            gpu_flow = self._cuda_farneback.calc(gpu_gray1, gpu_gray2, None)

            # Download (small compared to frames)
            return gpu_flow.download()

        except Exception as e:
            logger.warning(f"Farneback flow failed: {e}")
            return None

    def _warp_and_blend_gpu(self, frame1: torch.Tensor, frame2: torch.Tensor,
                            flow: torch.Tensor, t: float) -> torch.Tensor:
        """
        Warp both frames and blend them on GPU.

        Uses bidirectional warping for better quality:
        - Warp frame1 forward by t*flow
        - Warp frame2 backward by (1-t)*flow
        - Blend based on t

        Performance: ~0.5ms at 1440p (very fast)
        """
        h, w = frame1.shape[:2]

        # Ensure frames are float for interpolation
        f1 = frame1.float()
        f2 = frame2.float()

        # Ensure flow has correct shape (H, W, 2)
        if flow.dim() == 3 and flow.shape[2] == 2:
            flow_xy = flow
        elif flow.dim() == 3 and flow.shape[0] == 2:
            flow_xy = flow.permute(1, 2, 0)
        else:
            # Flow might be smaller (grid-based), upsample
            if flow.dim() == 3:
                flow_xy = F.interpolate(
                    flow.permute(2, 0, 1).unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0).permute(1, 2, 0)
            else:
                logger.warning(f"Unexpected flow shape: {flow.shape}")
                return torch.lerp(f1, f2, t).byte()

        # Normalize flow to [-1, 1] range for grid_sample
        flow_norm = torch.zeros_like(flow_xy)
        flow_norm[..., 0] = flow_xy[..., 0] * 2.0 / (w - 1)
        flow_norm[..., 1] = flow_xy[..., 1] * 2.0 / (h - 1)

        # Get base grid
        if self._grid_cache is None or self._grid_shape != (h, w):
            self._create_warp_grid()
        grid = self._grid_cache

        # Forward warp grid (frame1 -> intermediate)
        grid_forward = grid + t * flow_norm.unsqueeze(0)

        # Backward warp grid (frame2 -> intermediate)
        grid_backward = grid - (1 - t) * flow_norm.unsqueeze(0)

        # Warp frames
        f1_4d = f1.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        f2_4d = f2.permute(2, 0, 1).unsqueeze(0)

        warped1 = F.grid_sample(
            f1_4d, grid_forward,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        warped2 = F.grid_sample(
            f2_4d, grid_backward,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        # Blend warped frames
        blended = torch.lerp(warped1, warped2, t)

        # Convert back to (H, W, 3) uint8
        result = blended.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()
        return result

    def _blend_frames_gpu(self, frame1: torch.Tensor, frame2: torch.Tensor,
                          t: float) -> torch.Tensor:
        """Simple GPU alpha blend."""
        return torch.lerp(frame1.float(), frame2.float(), t).byte()

    def _blend_frames_cpu(self, frame1: np.ndarray, frame2: np.ndarray,
                          t: float) -> np.ndarray:
        """Simple CPU alpha blend fallback."""
        return ((1 - t) * frame1.astype(np.float32) +
                t * frame2.astype(np.float32)).astype(np.uint8)

    def cleanup(self):
        """Release GPU resources."""
        if self._nvof_direct is not None:
            self._nvof_direct.cleanup()
            self._nvof_direct = None

        if self._nvof_worker is not None:
            self._nvof_worker.stop()
            self._nvof_worker = None

        self._cuda_farneback = None
        self._grid_cache = None
        self._initialized = False

        # Clear frame cache
        self._cached_gpu_frame = None
        self._cached_gpu_gray = None
        self._cached_frame_id = None

        if HAS_TORCH and CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def flow_backend(self) -> Optional[str]:
        return self._flow_backend

    def get_stats(self) -> dict:
        """Get interpolator statistics."""
        stats = {
            'initialized': self._initialized,
            'width': self.width,
            'height': self.height,
            'flow_backend': self._flow_backend,
            'warp_backend': self._warp_backend,
            'preset': self.preset,
        }

        if self._nvof_worker is not None:
            stats['worker_stats'] = self._nvof_worker.get_stats()

        return stats
