"""
RIFE-NCNN Vulkan interpolator.

Uses rife-ncnn-vulkan for GPU-accelerated frame interpolation.
This is the fastest and most compatible neural network option.
"""
import logging
import numpy as np

from .gpu_utils import HAS_RIFE_NCNN, HAS_CV2

logger = logging.getLogger(__name__)

# Import RIFE-NCNN if available
if HAS_RIFE_NCNN:
    try:
        from rife_ncnn_vulkan_python_tntwise import Rife as RifeNCNN
    except ImportError:
        from rife_ncnn_vulkan_python import Rife as RifeNCNN

if HAS_CV2:
    import cv2


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
