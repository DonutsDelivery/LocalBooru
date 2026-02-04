"""
GPU detection and utility functions for optical flow.

This module handles:
- PyTorch and CUDA availability detection
- OpenCV and OpenCV CUDA detection
- RIFE-NCNN Vulkan detection
- VapourSynth and SVP plugin detection
- RIFE model path management
- Backend status reporting
"""
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Dependency Detection
# =============================================================================

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

# Check for VapourSynth + SVP (commercial SVP plugin)
HAS_VAPOURSYNTH = False
HAS_SVP = False

from ..svp_platform import (
    get_svp_plugin_path, get_svp_plugin_full_paths,
)

SVP_PLUGIN_PATH = get_svp_plugin_path() or "/opt/svp/plugins"

try:
    import vapoursynth as vs
    HAS_VAPOURSYNTH = True
    logger.info(f"VapourSynth available (version {vs.__version__})")

    # Try to load SVP plugins
    try:
        _test_core = vs.core
        _flow1_path, _flow2_path = get_svp_plugin_full_paths(SVP_PLUGIN_PATH)
        _test_core.std.LoadPlugin(_flow1_path)
        _test_core.std.LoadPlugin(_flow2_path)
        HAS_SVP = True
        logger.info("SVP plugins available (svpflow1, svpflow2)")
        del _test_core
    except Exception as e:
        logger.info(f"SVP plugins not available: {e}")
except ImportError:
    logger.info("VapourSynth not installed - SVP interpolation unavailable")


# =============================================================================
# RIFE Model Management
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
        from ..rife_models import get_rife_model_path, is_model_cached, get_cache_dir

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


# =============================================================================
# Backend Status
# =============================================================================

def get_backend_status() -> dict:
    """Get status of available interpolation backends."""
    # Check for NVIDIA Optical Flow availability
    nvof_available = False
    try:
        from ..nvidia_of_worker import check_nvidia_of_available
        nvof_status = check_nvidia_of_available()
        nvof_available = nvof_status.get('available', False)
    except ImportError:
        pass
    except Exception:
        pass

    # Check for GPU pipeline availability
    gpu_pipeline_available = False
    try:
        from ..gpu_video_pipeline import get_gpu_pipeline_status
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
        "any_backend_available": nvof_available or HAS_RIFE_NCNN or RIFE_AVAILABLE or HAS_CV2_CUDA or (HAS_TORCH and CUDA_AVAILABLE) or HAS_CV2,
        # SVP (VapourSynth + SVPflow) - commercial plugin
        "vapoursynth_available": HAS_VAPOURSYNTH,
        "svp_available": HAS_SVP,
        "svp_plugin_path": SVP_PLUGIN_PATH if HAS_SVP else None,
    }
