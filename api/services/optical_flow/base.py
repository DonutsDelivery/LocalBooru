"""
Base classes and shared types for optical flow interpolation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class InterpolatedFrame:
    """Result of frame interpolation."""
    data: np.ndarray
    source_t: float  # Interpolation position (0.0 = frame1, 1.0 = frame2)


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
