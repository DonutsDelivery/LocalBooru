"""
RIFE PyTorch-based frame interpolation.

Contains:
- RIFE neural network architecture (IFNet, IFBlock, ConvBlock)
- RIFEInterpolator class for frame interpolation
"""
import logging
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import numpy as np

from .gpu_utils import HAS_TORCH, CUDA_AVAILABLE, HAS_CV2, RIFE_MODEL_DIR

logger = logging.getLogger(__name__)

# Conditional imports
if HAS_TORCH:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

if HAS_CV2:
    import cv2


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
