"""
SVP configuration, presets, and parameter handling.

Contains quality presets, default settings, and parameter building functions.
"""
from typing import Optional

# =============================================================================
# SVP Quality Presets
# =============================================================================
# Based on SVPflow documentation: https://www.svp-team.com/wiki/Manual:SVPflow
#
# Key parameters:
# - pel: motion estimation accuracy (1=pixel, 2=half-pixel, 4=quarter-pixel)
# - block.w/h: block size (8=detailed/slow, 16=balanced, 32=fast/coarse)
# - overlap: block overlap (0=none, 1=1/8, 2=1/4, 3=1/2)
# - algo: rendering algorithm (13=uniform, 23=adaptive - best quality)
# - mask.area: artifact masking (0-100, higher = less smoothing but fewer artifacts)
# =============================================================================

SVP_PRESETS = {
    # Fastest preset - good for real-time playback on weaker GPUs
    "fast": {
        "name": "Fast",
        "description": "Fastest processing, lower quality. Good for real-time on weaker hardware.",
        "super": "{gpu:1,pel:1,scale:{up:0,down:4}}",
        "analyse": "{gpu:1,block:{w:32,h:32,overlap:0},main:{search:{coarse:{type:2,distance:-6,bad:{sad:2000,range:24}},type:2,distance:6}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:13,mask:{area:100},scene:{}}",
        "nvof_blk": 32,  # Larger blocks for faster NVOF processing
    },
    # Balanced preset - good tradeoff between speed and quality
    "balanced": {
        "name": "Balanced",
        "description": "Good balance of speed and quality. Recommended for most videos.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
        "nvof_blk": 16,  # Balanced block size for NVOF
    },
    # Quality preset - higher quality, slower processing
    "quality": {
        "name": "Quality",
        "description": "Higher quality motion estimation. Slower but smoother results.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:100},scene:{}}",
        "nvof_blk": 8,   # Smaller blocks for higher quality NVOF
    },
    # Maximum quality preset - best results, significantly slower
    "max": {
        "name": "Maximum",
        "description": "Maximum quality settings. Best for pre-rendering, not real-time.",
        "super": "{gpu:1,pel:4,scale:{up:2,down:2}}",
        "analyse": "{gpu:1,block:{w:8,h:8,overlap:3},main:{search:{coarse:{type:4,distance:-12,bad:{sad:1000,range:24}},type:4,distance:12}},refine:[{thsad:200},{thsad:100},{thsad:50}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:50,cover:80},scene:{}}",
        "nvof_blk": 4,   # Smallest blocks for maximum quality NVOF
    },
    # Animation preset - optimized for anime/cartoons
    "animation": {
        "name": "Animation",
        "description": "Optimized for anime and cartoons with flat colors and sharp edges.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:1500,range:24}},type:2,distance:10},penalty:{lambda:3.0}},refine:[{thsad:150}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:150},scene:{mode:0}}",
        "nvof_blk": 16,  # Balanced for animation
    },
    # Film preset - optimized for live action with natural motion
    "film": {
        "name": "Film",
        "description": "Optimized for live action movies with natural motion blur.",
        "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
        "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
        "smooth": "{gpuid:0,algo:23,mask:{area:80,cover:80},scene:{blend:true}}",
        "nvof_blk": 16,  # Balanced for film
    },
}

# Default SVP settings
DEFAULT_SVP_SETTINGS = {
    "enabled": False,
    "preset": "balanced",
    "target_fps": 60,
    "use_gpu": True,
    # Key settings (override preset values)
    "use_nvof": True,           # Use NVIDIA Optical Flow
    "shader": 23,               # SVP shader/algo (13=uniform, 23=adaptive)
    "artifact_masking": 100,    # Artifact masking area (0=off, 50-200)
    "frame_interpolation": 2,   # Frame interpolation mode (1=uniform, 2=adaptive)
    # Advanced settings (full override when set)
    "custom_super": None,
    "custom_analyse": None,
    "custom_smooth": None,
}

# Available algorithms for UI dropdown
SVP_ALGORITHMS = {
    1: "Block-based (fastest)",
    2: "Block-based with masking",
    11: "Pixel-based (smoother)",
    13: "Pixel-based uniform (recommended)",
    21: "Pixel-based with masking",
    23: "Pixel-based adaptive (best quality)",
}

# Block sizes for UI dropdown
SVP_BLOCK_SIZES = {
    8: "8x8 (highest quality, slowest)",
    16: "16x16 (balanced)",
    32: "32x32 (fastest, lower quality)",
}

# Motion accuracy (pel) for UI dropdown
SVP_PEL_OPTIONS = {
    1: "Pixel (fastest)",
    2: "Half-pixel (recommended)",
    4: "Quarter-pixel (highest quality)",
}

# Mask area settings for UI
SVP_MASK_AREA = {
    0: "Off (maximum smoothness)",
    50: "Low (smoother)",
    100: "Medium (balanced)",
    150: "High (fewer artifacts)",
    200: "Maximum (least smoothing)",
}


def build_svp_params(
    target_fps: int,
    preset: str = "balanced",
    shader: int = 23,
    artifact_masking: int = 100,
    frame_interpolation: int = 2,
    custom_super: Optional[str] = None,
    custom_analyse: Optional[str] = None,
    custom_smooth: Optional[str] = None,
) -> tuple:
    """Build SVP parameters from preset and overrides.

    Note: NVOF (NVIDIA Optical Flow) is handled at the pipeline level by using
    SmoothFps_NVOF instead of Super/Analyse/SmoothFps. These params are only
    used for the regular (non-NVOF) pipeline.
    """
    preset_config = SVP_PRESETS.get(preset, SVP_PRESETS["balanced"])

    # Use custom params if provided, otherwise use preset
    if custom_super:
        super_params = custom_super
    else:
        super_params = preset_config["super"]

    if custom_analyse:
        analyse_params = custom_analyse
    else:
        analyse_params = preset_config["analyse"]

    if custom_smooth:
        smooth_params = custom_smooth
    else:
        # Build smooth params with rate, algorithm, and masking
        algo = shader  # shader setting maps to SVP algorithm
        smooth_params = f"{{rate:{{num:{target_fps},den:1,abs:true}},gpuid:0,algo:{algo},mask:{{area:{artifact_masking}}},scene:{{}}}}"

    return super_params, analyse_params, smooth_params


# Alias for backward compatibility
_build_svp_params = build_svp_params
