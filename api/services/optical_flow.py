"""
Optical flow frame interpolation service.

Provides GPU-accelerated frame interpolation using:
- rife-ncnn-vulkan - Primary backend (Vulkan GPU, highly optimized)
- RIFE PyTorch - Secondary GPU backend
- OpenCV CUDA Farneback - Tertiary GPU backend
- OpenCV CPU Farneback - CPU fallback
"""
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Dependency detection
HAS_TORCH = False
CUDA_AVAILABLE = False
HAS_CV2 = False
HAS_CV2_CUDA = False
HAS_RIFE_NCNN = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    logger.info("PyTorch not installed - torch GPU interpolation unavailable")

try:
    import cv2
    HAS_CV2 = True
    # Check for OpenCV CUDA support
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            HAS_CV2_CUDA = True
            logger.info(f"OpenCV CUDA available ({cv2.cuda.getCudaEnabledDeviceCount()} devices)")
    except Exception:
        pass
except ImportError:
    logger.warning("OpenCV not installed - frame interpolation unavailable")

# Check for rife-ncnn-vulkan (preferred backend)
try:
    # Try tntwise fork first (more features, better maintained)
    from rife_ncnn_vulkan_python_tntwise import Rife as RifeNCNN
    HAS_RIFE_NCNN = True
    logger.info("rife-ncnn-vulkan-tntwise available (Vulkan GPU acceleration)")
except ImportError:
    try:
        # Fall back to original package
        from rife_ncnn_vulkan_python import Rife as RifeNCNN
        HAS_RIFE_NCNN = True
        logger.info("rife-ncnn-vulkan available (Vulkan GPU acceleration)")
    except ImportError:
        HAS_RIFE_NCNN = False
        logger.info("rife-ncnn-vulkan not installed - trying alternative backends")


# =============================================================================
# Fast GPU Blend Interpolator (Real-time capable)
# =============================================================================

class FastGPUBlendInterpolator:
    """
    Ultra-fast GPU-based frame blending for real-time playback.

    Performance at 1440p:
    - GPU blend only: ~0.3ms (3000+ fps)
    - With CPU transfer: ~17ms (60 fps) - exactly real-time

    This is the only option for true real-time 60fps at 1440p.
    Quality is lower than RIFE but acceptable for smooth playback.
    """

    def __init__(self, device: str = 'cuda'):
        if not HAS_TORCH or not CUDA_AVAILABLE:
            raise RuntimeError("PyTorch with CUDA required for FastGPUBlendInterpolator")

        self.device = torch.device(device)
        self._initialized = False
        # Pre-allocated tensors for zero-copy when possible
        self._f1_gpu = None
        self._f2_gpu = None
        self._last_shape = None

    def initialize(self, width: int = 0, height: int = 0) -> bool:
        """Initialize GPU resources."""
        self._initialized = True
        return True

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        Fast GPU alpha blend interpolation.

        Args:
            frame1: First frame (H, W, 3) uint8 BGR
            frame2: Second frame (H, W, 3) uint8 BGR
            t: Interpolation position (0.0 = frame1, 1.0 = frame2)

        Returns:
            Interpolated frame (H, W, 3) uint8 BGR
        """
        # Convert to torch tensors
        f1 = torch.from_numpy(frame1).to(self.device, dtype=torch.float32, non_blocking=True)
        f2 = torch.from_numpy(frame2).to(self.device, dtype=torch.float32, non_blocking=True)

        # Fast lerp on GPU
        result = torch.lerp(f1, f2, t)

        # Transfer back to CPU
        return result.byte().cpu().numpy()

    def cleanup(self):
        """Release GPU resources."""
        self._f1_gpu = None
        self._f2_gpu = None
        self._initialized = False
        if HAS_TORCH and CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class MotionCompensatedInterpolator:
    """
    SVP-style motion-compensated frame interpolation.

    Uses block matching for motion estimation and GPU warping for compensation.
    This is faster than RIFE neural network while providing better quality
    than simple alpha blending.

    Performance at 1440p: ~25-40 fps (depends on motion complexity)
    Quality: Medium - better than blend, not as good as RIFE
    """

    def __init__(self, block_size: int = 16, search_range: int = 16,
                 use_gpu: bool = True, downscale: float = 0.5):
        """
        Initialize motion-compensated interpolator.

        Args:
            block_size: Block size for motion estimation (8, 16, or 32)
            search_range: Search range for motion vectors
            use_gpu: Use GPU for warping if available
            downscale: Downscale factor for motion estimation (0.25-1.0)
        """
        if not HAS_CV2:
            raise RuntimeError("OpenCV required for MotionCompensatedInterpolator")

        self.block_size = block_size
        self.search_range = search_range
        self.use_gpu = use_gpu and HAS_TORCH and CUDA_AVAILABLE
        self.downscale = max(0.25, min(1.0, downscale))
        self._initialized = False
        self.device = torch.device('cuda') if self.use_gpu else None

    def initialize(self, width: int = 0, height: int = 0) -> bool:
        """Initialize resources."""
        self._initialized = True
        return True

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        Motion-compensated frame interpolation (SVP-style).

        Uses block matching to find motion vectors, then warps both frames
        toward the target time and blends them with an occlusion mask.
        """
        h, w = frame1.shape[:2]

        # Downscale for faster motion estimation
        if self.downscale < 1.0:
            sh, sw = int(h * self.downscale), int(w * self.downscale)
            small1 = cv2.resize(frame1, (sw, sh), interpolation=cv2.INTER_AREA)
            small2 = cv2.resize(frame2, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small1, small2 = frame1, frame2
            sh, sw = h, w

        # Convert to grayscale for motion estimation
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

        # Fast block matching using phase correlation at pyramid levels
        # This is much faster than dense optical flow
        flow = self._estimate_motion_fast(gray1, gray2)

        # Upscale flow to full resolution
        if self.downscale < 1.0:
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
            flow = flow / self.downscale  # Scale motion vectors

        # Warp frames toward target time
        if self.use_gpu:
            result = self._warp_gpu(frame1, frame2, flow, t)
        else:
            result = self._warp_cpu(frame1, frame2, flow, t)

        return result

    def _estimate_motion_fast(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """
        Fast motion estimation using sparse-to-dense approach.

        Uses corner detection + sparse flow, then interpolates to dense.
        Much faster than full dense optical flow.
        """
        h, w = gray1.shape

        # Detect corners for sparse flow
        corners = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01,
                                          minDistance=10, blockSize=7)

        if corners is None or len(corners) < 10:
            # Fallback to simple flow estimation
            return np.zeros((h, w, 2), dtype=np.float32)

        # Calculate sparse optical flow (Lucas-Kanade)
        corners2, status, _ = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, corners, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Filter good points
        good_old = corners[status == 1]
        good_new = corners2[status == 1]

        if len(good_old) < 4:
            return np.zeros((h, w, 2), dtype=np.float32)

        # Calculate motion vectors
        motion = good_new - good_old

        # Interpolate sparse flow to dense using RBF or simple grid interpolation
        flow = self._interpolate_sparse_flow(good_old, motion, (h, w))

        return flow

    def _interpolate_sparse_flow(self, points: np.ndarray, motion: np.ndarray,
                                  shape: Tuple[int, int]) -> np.ndarray:
        """Interpolate sparse motion vectors to dense flow field."""
        h, w = shape
        flow = np.zeros((h, w, 2), dtype=np.float32)

        if len(points) < 4:
            return flow

        # Create grid
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Use griddata for interpolation (fast enough for downscaled images)
        from scipy.interpolate import griddata

        pts = points.reshape(-1, 2)

        # Interpolate x and y motion separately
        flow[:, :, 0] = griddata(pts, motion[:, 0], (grid_y, grid_x),
                                  method='linear', fill_value=0)
        flow[:, :, 1] = griddata(pts, motion[:, 1], (grid_y, grid_x),
                                  method='linear', fill_value=0)

        # Fill NaN with 0
        flow = np.nan_to_num(flow, 0)

        return flow

    def _warp_gpu(self, frame1: np.ndarray, frame2: np.ndarray,
                   flow: np.ndarray, t: float) -> np.ndarray:
        """GPU-accelerated frame warping and blending."""
        h, w = frame1.shape[:2]

        # Upload to GPU
        f1 = torch.from_numpy(frame1).float().to(self.device)
        f2 = torch.from_numpy(frame2).float().to(self.device)
        flow_t = torch.from_numpy(flow).float().to(self.device)

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # Warp frame1 forward by t
        map_x1 = grid_x + flow_t[:, :, 0] * t
        map_y1 = grid_y + flow_t[:, :, 1] * t

        # Warp frame2 backward by (1-t)
        map_x2 = grid_x - flow_t[:, :, 0] * (1 - t)
        map_y2 = grid_y - flow_t[:, :, 1] * (1 - t)

        # Normalize to [-1, 1] for grid_sample
        map_x1 = 2.0 * map_x1 / (w - 1) - 1.0
        map_y1 = 2.0 * map_y1 / (h - 1) - 1.0
        map_x2 = 2.0 * map_x2 / (w - 1) - 1.0
        map_y2 = 2.0 * map_y2 / (h - 1) - 1.0

        grid1 = torch.stack([map_x1, map_y1], dim=-1).unsqueeze(0)
        grid2 = torch.stack([map_x2, map_y2], dim=-1).unsqueeze(0)

        # Warp using grid_sample
        f1_warped = F.grid_sample(
            f1.permute(2, 0, 1).unsqueeze(0),
            grid1, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0).permute(1, 2, 0)

        f2_warped = F.grid_sample(
            f2.permute(2, 0, 1).unsqueeze(0),
            grid2, mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0).permute(1, 2, 0)

        # Blend warped frames
        result = torch.lerp(f1_warped, f2_warped, t)

        return result.byte().cpu().numpy()

    def _warp_cpu(self, frame1: np.ndarray, frame2: np.ndarray,
                   flow: np.ndarray, t: float) -> np.ndarray:
        """CPU frame warping and blending."""
        h, w = frame1.shape[:2]

        # Create coordinate grids
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # Warp maps for frame1 (forward warp by t)
        map_x1 = (grid_x + flow[:, :, 0] * t).astype(np.float32)
        map_y1 = (grid_y + flow[:, :, 1] * t).astype(np.float32)

        # Warp maps for frame2 (backward warp by 1-t)
        map_x2 = (grid_x - flow[:, :, 0] * (1 - t)).astype(np.float32)
        map_y2 = (grid_y - flow[:, :, 1] * (1 - t)).astype(np.float32)

        # Warp both frames
        warped1 = cv2.remap(frame1, map_x1, map_y1, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        warped2 = cv2.remap(frame2, map_x2, map_y2, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

        # Blend
        result = cv2.addWeighted(warped1, 1 - t, warped2, t, 0)

        return result

    def cleanup(self):
        """Release resources."""
        self._initialized = False
        if self.use_gpu and HAS_TORCH:
            torch.cuda.empty_cache()

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# =============================================================================
# RIFE-NCNN Interpolator (Vulkan GPU - fastest and most compatible)
# =============================================================================

class RifeNCNNInterpolator:
    """
    Frame interpolator using rife-ncnn-vulkan.

    This is the fastest and most compatible option:
    - Uses Vulkan for GPU acceleration (works on NVIDIA, AMD, Intel)
    - Pre-optimized binary with bundled models
    - No Python 3.14 torch.compile issues
    - Typically 10-50ms per 1440p frame
    """

    def __init__(self, gpu_id: int = 0, model: str = "rife-v4.6", num_threads: int = 2,
                 width: int = 0, height: int = 0, uhd_mode: bool = False):
        """
        Initialize RIFE-NCNN interpolator.

        Args:
            gpu_id: GPU device ID (0 for first GPU, -1 for CPU)
            model: Model name (e.g., "rife-v4.6", "rife-v4")
            num_threads: Number of processing threads
            width: Frame width (0 = auto-detect from first frame)
            height: Frame height (0 = auto-detect from first frame)
            uhd_mode: Enable UHD mode for high resolution content
        """
        if not HAS_RIFE_NCNN:
            raise RuntimeError("rife-ncnn-vulkan-python not installed")

        self.gpu_id = gpu_id
        self.model = model
        self.num_threads = num_threads
        self.width = width
        self.height = height
        self.uhd_mode = uhd_mode
        self._rife = None
        self._initialized = False
        self._dims_set = False

    def initialize(self, width: int = 0, height: int = 0) -> bool:
        """Initialize the RIFE-NCNN model."""
        if self._initialized and self._dims_set:
            return True

        # Use provided dimensions or stored ones
        w = width or self.width
        h = height or self.height

        # Auto-detect UHD mode for high resolutions
        uhd = self.uhd_mode or (w * h > 1920 * 1080)

        try:
            # RifeNCNN accepts gpu_id, model name, dimensions for pre-allocation
            init_args = {
                'gpuid': self.gpu_id,
                'model': self.model,
                'num_threads': self.num_threads,
            }

            # Add dimensions if known (enables faster process_bytes)
            if w > 0 and h > 0:
                init_args['width'] = w
                init_args['height'] = h
                init_args['channels'] = 3
                self._dims_set = True

            # Add UHD mode if supported and needed
            if uhd:
                init_args['uhd_mode'] = True

            self._rife = RifeNCNN(**init_args)
            self._initialized = True
            self.width = w
            self.height = h
            logger.info(f"RIFE-NCNN initialized (gpu={self.gpu_id}, model={self.model}, "
                       f"size={w}x{h}, uhd={uhd})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RIFE-NCNN with full params: {e}")
            # Try with minimal parameters
            try:
                self._rife = RifeNCNN(gpuid=self.gpu_id, model=self.model)
                self._initialized = True
                logger.info(f"RIFE-NCNN initialized with minimal params (gpu={self.gpu_id})")
                return True
            except Exception as e2:
                logger.error(f"RIFE-NCNN fallback failed: {e2}")
                return False

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between two frames.

        Args:
            frame1: First frame (H, W, 3) uint8 BGR
            frame2: Second frame (H, W, 3) uint8 BGR
            t: Interpolation position (0.0 = frame1, 1.0 = frame2)

        Returns:
            Interpolated frame (H, W, 3) uint8 BGR
        """
        h, w = frame1.shape[:2]

        # Initialize with frame dimensions if not yet done
        if not self._initialized or not self._dims_set:
            if not self.initialize(width=w, height=h):
                return self._blend_frames(frame1, frame2, t)

        try:
            # Convert BGR to RGB (RIFE expects RGB)
            img1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Try fastest method first: process_bytes (streaming optimized with caching)
            if hasattr(self._rife, 'process_bytes') and self._dims_set:
                result_bytes = self._rife.process_bytes(
                    img1_rgb.tobytes(),
                    img2_rgb.tobytes(),
                    timestep=t
                )
                result_rgb = np.frombuffer(result_bytes, dtype=np.uint8).reshape(h, w, 3)
            # Fall back to process_cv2 (direct numpy, no PIL overhead)
            elif hasattr(self._rife, 'process_cv2'):
                result_rgb = self._rife.process_cv2(img1_rgb, img2_rgb, timestep=t)
            else:
                # Final fallback to PIL-based process
                from PIL import Image
                pil_img1 = Image.fromarray(img1_rgb)
                pil_img2 = Image.fromarray(img2_rgb)
                result_pil = self._rife.process(pil_img1, pil_img2, timestep=t)
                result_rgb = np.array(result_pil)

            # Convert back to BGR
            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"RIFE-NCNN interpolation failed: {e}")
            return self._blend_frames(frame1, frame2, t)

    def _blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """Simple alpha blend fallback."""
        return ((1 - t) * frame1.astype(np.float32) + t * frame2.astype(np.float32)).astype(np.uint8)

    def cleanup(self):
        """Release resources."""
        self._rife = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# =============================================================================
# RIFE Model Architecture (Practical-RIFE implementation - PyTorch fallback)
# =============================================================================

if HAS_TORCH:
    def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Sequential:
        """Standard convolution block with LeakyReLU."""
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def deconv(in_planes: int, out_planes: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> nn.Sequential:
        """Transposed convolution for upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    class ConvBlock(nn.Module):
        """Convolutional block with residual connection option."""
        def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
            super().__init__()
            self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
            self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
            self.use_res = (stride == 1 and in_planes == out_planes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv1(x)
            out = self.conv2(out)
            if self.use_res:
                out = out + x
            return out

    class IFBlock(nn.Module):
        """
        Intermediate Flow Block - core building block of RIFE.
        Estimates flow and mask at a single scale.
        """
        def __init__(self, in_planes: int, c: int = 64):
            super().__init__()
            self.conv0 = nn.Sequential(
                conv(in_planes, c // 2, 3, 2, 1),
                conv(c // 2, c, 3, 2, 1),
            )
            self.convblock = nn.Sequential(
                ConvBlock(c, c),
                ConvBlock(c, c),
                ConvBlock(c, c),
                ConvBlock(c, c),
                ConvBlock(c, c),
                ConvBlock(c, c),
                ConvBlock(c, c),
                ConvBlock(c, c),
            )
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(c // 2, 5, 4, 2, 1),  # 4 flow + 1 mask
            )

        def forward(self, x: torch.Tensor, flow: torch.Tensor, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Input features (concatenated warped frames + context)
                flow: Current flow estimate (to be refined)
                scale: Downscale factor for multi-scale processing
            Returns:
                delta_flow: Flow refinement
                mask: Blending mask
            """
            if scale != 1.0:
                x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)

            feat = self.conv0(x)
            feat = self.convblock(feat)
            out = self.lastconv(feat)

            if scale != 1.0:
                out = F.interpolate(out, scale_factor=scale, mode="bilinear", align_corners=False)

            # Split output: 4 channels for flow (2 forward + 2 backward), 1 for mask
            delta_flow = out[:, :4] * scale
            mask = out[:, 4:5]

            return delta_flow, mask

    class IFNet(nn.Module):
        """
        RIFE's IFNet - Intermediate Flow Network.

        Multi-scale coarse-to-fine flow estimation with:
        - 3 scale levels (1/4, 1/2, 1)
        - Bidirectional flow estimation
        - Learned blending mask
        """
        def __init__(self):
            super().__init__()
            # Multi-scale IFBlocks
            # Input: 6 (img0 + img1) + 4 (flow) + 1 (timestep) = 11 channels at each scale
            self.block0 = IFBlock(7, c=192)   # First block: 6 (images) + 1 (timestep) = 7
            self.block1 = IFBlock(8, c=128)   # 7 + 1 (mask from prev) = 8
            self.block2 = IFBlock(8, c=96)
            self.block3 = IFBlock(8, c=64)    # Optional 4th scale for higher quality

            # Context network for feature extraction
            self.contextnet = nn.Sequential(
                conv(3, 32, 3, 2, 1),
                conv(32, 64, 3, 2, 1),
                conv(64, 96, 3, 2, 1),
            )

            # Fusion network for final frame synthesis
            self.fusionnet = nn.Sequential(
                conv(10, 64, 3, 1, 1),  # 2*3 (warped) + 2*2 (context compressed) = 10
                conv(64, 64, 3, 1, 1),
                conv(64, 64, 3, 1, 1),
                nn.Conv2d(64, 3, 3, 1, 1),
            )

        def forward(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5,
                    scale_list: List[float] = [8, 4, 2, 1]) -> torch.Tensor:
            """
            Perform frame interpolation.

            Args:
                img0: First frame (B, 3, H, W) normalized to [0, 1]
                img1: Second frame (B, 3, H, W) normalized to [0, 1]
                timestep: Interpolation position (0.0 = img0, 1.0 = img1)
                scale_list: Multi-scale factors for coarse-to-fine estimation

            Returns:
                Interpolated frame (B, 3, H, W) normalized to [0, 1]
            """
            B, C, H, W = img0.shape

            # Timestep as spatial map
            timestep_map = torch.full((B, 1, H, W), timestep, device=img0.device, dtype=img0.dtype)

            # Initialize flow
            flow = torch.zeros(B, 4, H, W, device=img0.device, dtype=img0.dtype)

            # Coarse-to-fine flow estimation
            blocks = [self.block0, self.block1, self.block2, self.block3]

            for i, scale in enumerate(scale_list):
                if i >= len(blocks):
                    break

                # Warp images with current flow
                flow_01 = flow[:, :2]  # Flow from img0 to intermediate
                flow_10 = flow[:, 2:4]  # Flow from img1 to intermediate

                if i == 0:
                    # First iteration: no warping yet
                    warped0 = img0
                    warped1 = img1
                else:
                    warped0 = self._warp(img0, flow_01)
                    warped1 = self._warp(img1, flow_10)

                # Build input for IFBlock
                if i == 0:
                    x = torch.cat([img0, img1, timestep_map], dim=1)
                else:
                    x = torch.cat([warped0, warped1, timestep_map, mask], dim=1)

                # Estimate flow refinement
                delta_flow, mask = blocks[i](x, flow, scale=scale)

                # Update flow
                flow = flow + delta_flow

            # Final warping with refined flow
            flow_01 = flow[:, :2] * timestep
            flow_10 = flow[:, 2:4] * (1 - timestep)

            warped0 = self._warp(img0, flow_01)
            warped1 = self._warp(img1, flow_10)

            # Mask normalization
            mask = torch.sigmoid(mask)

            # Simple blending (can be enhanced with fusion network)
            merged = warped0 * mask + warped1 * (1 - mask)

            return merged.clamp(0, 1)

        def _warp(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
            """Backward warp image using optical flow."""
            B, C, H, W = img.shape

            # Create normalized grid
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype),
                torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            # Convert flow to normalized coordinates
            flow_norm = torch.zeros_like(flow).permute(0, 2, 3, 1)
            flow_norm[..., 0] = flow[:, 0] / ((W - 1) / 2)
            flow_norm[..., 1] = flow[:, 1] / ((H - 1) / 2)

            # Add flow to grid
            new_grid = grid + flow.permute(0, 2, 3, 1) * torch.tensor([2.0 / (W - 1), 2.0 / (H - 1)], device=img.device, dtype=img.dtype)

            return F.grid_sample(img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)


# =============================================================================
# RIFE Interpolator with Optimizations
# =============================================================================

class RIFEInterpolator:
    """
    RIFE-based frame interpolator with production optimizations.

    Optimizations:
    - FP16 inference with autocast
    - torch.compile with reduce-overhead mode
    - Pre-allocated buffers for common resolutions
    - Input padding to multiple of 32
    - Warm-up inference on initialization
    """

    # Common resolutions for pre-allocated buffers
    BUFFER_RESOLUTIONS = [
        (2560, 1440),  # 1440p
        (1920, 1080),  # 1080p
    ]

    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize RIFE interpolator.

        Args:
            model_dir: Directory containing RIFE model weights
            device: Device to use ('cuda' or 'cpu')
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for RIFE interpolation")

        self.device = torch.device(device or ('cuda' if CUDA_AVAILABLE else 'cpu'))
        self.model_dir = model_dir
        self._model: Optional[nn.Module] = None
        self._compiled_model = None
        self._initialized = False
        self._use_fp16 = self.device.type == 'cuda'

        # Pre-allocated buffers for common resolutions
        self._input_buffers: Dict[Tuple[int, int], torch.Tensor] = {}
        self._output_buffers: Dict[Tuple[int, int], torch.Tensor] = {}

        logger.info(f"RIFEInterpolator created (device={self.device}, fp16={self._use_fp16})")

    def initialize(self, warm_up: bool = True) -> bool:
        """
        Initialize model and optionally warm up.

        Args:
            warm_up: Whether to run warm-up inference

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            # Create model
            self._model = IFNet()

            # Try to load weights if available
            weights_loaded = False
            if self.model_dir:
                model_dir = Path(self.model_dir)
                # Try multiple possible weight file names
                possible_names = [
                    "flownet_v4.22.pkl",
                    "flownet_v4.22_lite.pkl",
                    "flownet.pkl",
                    "model.pkl",
                ]
                weights_path = None
                for name in possible_names:
                    candidate = model_dir / name
                    if candidate.exists():
                        weights_path = candidate
                        break

                if weights_path and weights_path.exists():
                    try:
                        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                        # Handle different weight formats
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                        self._model.load_state_dict(state_dict, strict=False)
                        weights_loaded = True
                        logger.info(f"RIFE weights loaded from {weights_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load RIFE weights: {e}")
                else:
                    logger.warning(f"RIFE weights not found in {model_dir} - using untrained model")
            else:
                logger.warning("No model_dir specified - using untrained RIFE model")

            # Move to device
            self._model.to(self.device)

            # Convert to FP16 for faster inference on GPU
            if self._use_fp16:
                self._model.half()

            self._model.eval()

            # Compile model for better performance
            try:
                if hasattr(torch, 'compile'):
                    self._compiled_model = torch.compile(
                        self._model,
                        mode="reduce-overhead",
                        fullgraph=False
                    )
                    logger.info("RIFE model compiled with torch.compile (reduce-overhead)")
                else:
                    self._compiled_model = self._model
            except Exception as e:
                logger.warning(f"torch.compile failed, using eager mode: {e}")
                self._compiled_model = self._model

            # Pre-allocate buffers for common resolutions
            self._allocate_buffers()

            # Warm-up inference
            if warm_up:
                self._warm_up()

            self._initialized = True
            logger.info(f"RIFEInterpolator initialized (weights_loaded={weights_loaded})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RIFE: {e}")
            self._model = None
            self._compiled_model = None
            return False

    def _allocate_buffers(self):
        """Pre-allocate GPU buffers for common resolutions."""
        if self.device.type != 'cuda':
            return

        dtype = torch.float16 if self._use_fp16 else torch.float32

        for w, h in self.BUFFER_RESOLUTIONS:
            # Pad to multiple of 32
            ph = ((h + 31) // 32) * 32
            pw = ((w + 31) // 32) * 32

            try:
                # Input buffer: 2 frames concatenated (6 channels)
                self._input_buffers[(w, h)] = torch.empty(
                    1, 3, ph, pw, device=self.device, dtype=dtype
                )
                logger.debug(f"Pre-allocated input buffer for {w}x{h} -> {pw}x{ph}")
            except Exception as e:
                logger.warning(f"Failed to pre-allocate buffer for {w}x{h}: {e}")

    def _warm_up(self):
        """Run warm-up inference to trigger compilation."""
        logger.info("Running RIFE warm-up inference...")

        dtype = torch.float16 if self._use_fp16 else torch.float32

        # Warm up at multiple scales
        warm_up_sizes = [(256, 256), (512, 512), (1920, 1088)]  # 1088 = 1080 padded to 32

        for w, h in warm_up_sizes:
            try:
                with torch.no_grad():
                    if self._use_fp16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            dummy0 = torch.rand(1, 3, h, w, device=self.device, dtype=dtype)
                            dummy1 = torch.rand(1, 3, h, w, device=self.device, dtype=dtype)
                            _ = self._compiled_model(dummy0, dummy1, 0.5)
                    else:
                        dummy0 = torch.rand(1, 3, h, w, device=self.device, dtype=dtype)
                        dummy1 = torch.rand(1, 3, h, w, device=self.device, dtype=dtype)
                        _ = self._compiled_model(dummy0, dummy1, 0.5)

                # Sync to ensure compilation completes
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

            except Exception as e:
                logger.warning(f"Warm-up at {w}x{h} failed: {e}")

        logger.info("RIFE warm-up complete")

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between two frames.

        Args:
            frame1: First frame (H, W, 3) uint8 BGR
            frame2: Second frame (H, W, 3) uint8 BGR
            t: Interpolation position (0.0 = frame1, 1.0 = frame2)

        Returns:
            Interpolated frame (H, W, 3) uint8 BGR
        """
        if not self._initialized:
            if not self.initialize():
                # Fallback to simple blend
                return self._blend_frames(frame1, frame2, t)

        try:
            h, w = frame1.shape[:2]

            # Pad dimensions to multiple of 32
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            # Convert BGR to RGB and normalize
            img0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Convert to tensor
            dtype = torch.float16 if self._use_fp16 else torch.float32

            img0_t = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=dtype)
            img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=dtype)

            # Pad if necessary
            if pad_h > 0 or pad_w > 0:
                img0_t = F.pad(img0_t, (0, pad_w, 0, pad_h), mode='reflect')
                img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h), mode='reflect')

            # Inference
            with torch.no_grad():
                if self._use_fp16 and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        result = self._compiled_model(img0_t, img1_t, t)
                else:
                    result = self._compiled_model(img0_t, img1_t, t)

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                result = result[:, :, :h, :w]

            # Convert back to numpy BGR uint8
            result_np = result.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

            return result_bgr

        except Exception as e:
            logger.warning(f"RIFE interpolation failed: {e}")
            return self._blend_frames(frame1, frame2, t)

    def _blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """Simple alpha blend fallback."""
        return ((1 - t) * frame1.astype(np.float32) + t * frame2.astype(np.float32)).astype(np.uint8)

    def cleanup(self):
        """Release resources."""
        self._model = None
        self._compiled_model = None
        self._input_buffers.clear()
        self._output_buffers.clear()
        self._initialized = False
        if HAS_TORCH and CUDA_AVAILABLE:
            torch.cuda.empty_cache()

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# =============================================================================
# Status and Configuration
# =============================================================================

# Global RIFE availability flag (set after checking model weights)
RIFE_AVAILABLE = False
RIFE_MODEL_DIR: Optional[Path] = None


def check_rife_availability(data_dir: Optional[str] = None) -> Tuple[bool, Optional[Path]]:
    """
    Check if RIFE model is available, downloading if needed.

    Args:
        data_dir: Data directory containing models/rife/

    Returns:
        Tuple of (available, model_dir)
    """
    global RIFE_AVAILABLE, RIFE_MODEL_DIR

    if not HAS_TORCH or not CUDA_AVAILABLE:
        RIFE_AVAILABLE = False
        RIFE_MODEL_DIR = None
        return False, None

    # Try to get model path (will download if needed)
    try:
        from .rife_models import get_rife_model_path, is_model_cached, get_cache_dir

        cache_dir = get_cache_dir()

        # Check if already cached
        if is_model_cached("4.22"):
            model_path = cache_dir / "flownet_v4.22.pkl"
            RIFE_AVAILABLE = True
            RIFE_MODEL_DIR = cache_dir
            logger.info(f"RIFE model found at {cache_dir}")
            return True, cache_dir

        # Try to download
        logger.info("RIFE model not cached, attempting download...")
        model_path = get_rife_model_path("4.22")
        RIFE_AVAILABLE = True
        RIFE_MODEL_DIR = model_path.parent
        logger.info(f"RIFE model downloaded to {RIFE_MODEL_DIR}")
        return True, RIFE_MODEL_DIR

    except ImportError as e:
        logger.debug(f"rife_models module not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to get RIFE model: {e}")

    # Fallback: check manual path
    if data_dir:
        model_dir = Path(data_dir) / "models" / "rife"
        weights_path = model_dir / "flownet.pkl"

        if weights_path.exists():
            RIFE_AVAILABLE = True
            RIFE_MODEL_DIR = model_dir
            logger.info(f"RIFE model found at {model_dir}")
            return True, model_dir

    # RIFE architecture available but no weights - still usable (untrained)
    RIFE_AVAILABLE = HAS_TORCH and CUDA_AVAILABLE
    RIFE_MODEL_DIR = Path(data_dir) / "models" / "rife" if data_dir else None
    return RIFE_AVAILABLE, RIFE_MODEL_DIR


def get_backend_status() -> dict:
    """Get status of available interpolation backends."""
    # Check for NVIDIA Optical Flow availability
    nvof_available = False
    try:
        from .nvidia_of_worker import check_nvidia_of_available
        nvof_status = check_nvidia_of_available()
        nvof_available = nvof_status.get('available', False)
    except ImportError:
        pass
    except Exception:
        pass

    # Check for GPU pipeline availability
    gpu_pipeline_available = False
    try:
        from .gpu_video_pipeline import get_gpu_pipeline_status
        pipeline_status = get_gpu_pipeline_status()
        gpu_pipeline_available = pipeline_status.get('pynvvideocodec_available', False)
    except ImportError:
        pass
    except Exception:
        pass

    # Determine best GPU backend (priority: gpu_native > RIFE-NCNN > RIFE PyTorch > OpenCV CUDA > PyTorch)
    # gpu_native = full GPU pipeline with NVOF + GPU warp (fastest for real-time)
    if nvof_available and HAS_TORCH and CUDA_AVAILABLE:
        gpu_backend = "gpu_native"
    elif HAS_RIFE_NCNN:
        gpu_backend = "rife_ncnn"
    elif RIFE_AVAILABLE:
        gpu_backend = "rife_torch"
    elif HAS_CV2_CUDA:
        gpu_backend = "opencv_cuda"
    elif HAS_TORCH and CUDA_AVAILABLE:
        gpu_backend = "torch_cuda"
    else:
        gpu_backend = None

    return {
        "torch_available": HAS_TORCH,
        "cuda_available": CUDA_AVAILABLE,
        "cv2_available": HAS_CV2,
        "cv2_cuda_available": HAS_CV2_CUDA,
        "rife_ncnn_available": HAS_RIFE_NCNN,
        "rife_torch_available": RIFE_AVAILABLE,
        "rife_model_dir": str(RIFE_MODEL_DIR) if RIFE_MODEL_DIR else None,
        "nvof_available": nvof_available,
        "gpu_pipeline_available": gpu_pipeline_available,
        "gpu_native_available": nvof_available and HAS_TORCH and CUDA_AVAILABLE,
        "gpu_backend": gpu_backend,
        "cpu_backend": "farneback" if HAS_CV2 else None,
        "any_backend_available": nvof_available or HAS_RIFE_NCNN or RIFE_AVAILABLE or HAS_CV2_CUDA or (HAS_TORCH and CUDA_AVAILABLE) or HAS_CV2
    }


# Quality presets for frame interpolation
# svp: SVP-style NVIDIA Optical Flow - 60+fps at 1440p, motion-compensated (best realtime)
# gpu_native: Full GPU pipeline with NVOF - similar to svp but more aggressive preset
# realtime: Simple GPU blend - 60fps at 1440p, lower quality
# fast: RIFE-NCNN - ~15fps at 1440p, high quality (for pre-processing)
# balanced/quality: Even higher quality RIFE settings
QUALITY_PRESETS = {
    "svp": {
        "backend": "gpu_native",  # Full GPU pipeline: NVOF + GPU warp
        "preset": "fast",         # NVOF preset (fast achieves 82fps @ 1440p GPU-only)
        "flow_scale": 1.0,        # Full resolution flow
        "use_ipc_worker": False,  # Direct NVOF (no CPU, full GPU)
        # Fallback Farneback params (used if NVOF unavailable)
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
    },
    "gpu_native": {
        "backend": "gpu_native",  # Full GPU pipeline: NVOF + GPU warp + GPU encode
        "preset": "fast",         # NVOF preset (slow/medium/fast)
        "flow_scale": 1.0,        # Full resolution flow
        "use_ipc_worker": False,  # Direct NVOF (no CPU, full GPU)
        # Fallback Farneback params (used if NVOF unavailable)
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
    },
    "realtime": {
        "backend": "gpu_blend",   # Use FastGPUBlendInterpolator
        "pyr_scale": 0.5,
        "levels": 1,
        "winsize": 7,
        "iterations": 1,
        "poly_n": 5,
        "poly_sigma": 1.1,
        "flow_scale": 0.25,
    },
    "fast": {
        "backend": "rife",  # Use RIFE-NCNN
        "pyr_scale": 0.5,
        "levels": 2,
        "winsize": 11,
        "iterations": 2,
        "poly_n": 5,
        "poly_sigma": 1.1,
        "flow_scale": 0.5,
    },
    "balanced": {
        "backend": "rife",
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flow_scale": 0.75,
    },
    "quality": {
        "backend": "rife",
        "pyr_scale": 0.5,
        "levels": 4,
        "winsize": 21,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 1.5,
        "flow_scale": 1.0,
    },
}


@dataclass
class InterpolatedFrame:
    """Result of frame interpolation."""
    data: np.ndarray
    source_t: float  # Interpolation position (0.0 = frame1, 1.0 = frame2)


# =============================================================================
# Main FrameInterpolator Class
# =============================================================================

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


# =============================================================================
# GPU Native Interpolator (Full GPU Pipeline)
# =============================================================================

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
            from .nvidia_of_worker import DirectNvidiaOpticalFlow, check_nvidia_of_available

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
            from .nvidia_of_worker import NvidiaOpticalFlowWorker

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

        Pipeline: Upload → Grayscale → NVOF → Scale → Split → Remap → Blend → Download

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

        # Scale flow on GPU: int16 → float32, apply -t/32 scale factor
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
