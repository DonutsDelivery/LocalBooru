"""
Optical flow frame interpolation service.

Provides GPU-accelerated frame interpolation using:
- SVP (SmoothVideo Project) - Commercial VapourSynth plugin (best quality, requires license)
- rife-ncnn-vulkan - Primary backend (Vulkan GPU, highly optimized)
- RIFE PyTorch - Secondary GPU backend
- OpenCV CUDA Farneback - Tertiary GPU backend
- OpenCV CPU Farneback - CPU fallback
"""

# Re-export all public API from submodules
from .gpu_utils import (
    HAS_TORCH,
    CUDA_AVAILABLE,
    HAS_CV2,
    HAS_CV2_CUDA,
    HAS_RIFE_NCNN,
    HAS_VAPOURSYNTH,
    HAS_SVP,
    SVP_PLUGIN_PATH,
    RIFE_AVAILABLE,
    RIFE_MODEL_DIR,
    check_rife_availability,
    get_backend_status,
)

from .base import (
    InterpolatedFrame,
    QUALITY_PRESETS,
)

from .frame_buffer import (
    FastGPUBlendInterpolator,
    MotionCompensatedInterpolator,
)

from .rife import (
    RIFEInterpolator,
)

from .rife_ncnn import (
    RifeNCNNInterpolator,
)

from .svp import (
    SVPInterpolator,
)

from .opencv_flow import (
    GPUNativeInterpolator,
)

from .interpolator import (
    FrameInterpolator,
)

__all__ = [
    # GPU detection flags
    "HAS_TORCH",
    "CUDA_AVAILABLE",
    "HAS_CV2",
    "HAS_CV2_CUDA",
    "HAS_RIFE_NCNN",
    "HAS_VAPOURSYNTH",
    "HAS_SVP",
    "SVP_PLUGIN_PATH",
    "RIFE_AVAILABLE",
    "RIFE_MODEL_DIR",
    # Functions
    "check_rife_availability",
    "get_backend_status",
    # Data classes
    "InterpolatedFrame",
    # Presets
    "QUALITY_PRESETS",
    # Interpolator classes
    "FastGPUBlendInterpolator",
    "MotionCompensatedInterpolator",
    "RIFEInterpolator",
    "RifeNCNNInterpolator",
    "SVPInterpolator",
    "GPUNativeInterpolator",
    "FrameInterpolator",
]
