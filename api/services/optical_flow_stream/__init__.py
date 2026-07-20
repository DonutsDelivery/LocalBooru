"""
Optical flow HLS streaming service.

Manages the video -> interpolation -> FFmpeg -> HLS pipeline for streaming
interpolated video to clients via hls.js.

Optimized for RIFE-based interpolation with:
- Async frame loading with pinned memory for faster CPU->GPU transfer
- Batched interpolation for multiple t values in single forward pass
- NVENC hardware encoding when available
- Performance monitoring and logging

GPU Native mode (quality="gpu_native"):
- GPU video decode via PyNvVideoCodec (when available)
- NVIDIA Optical Flow Accelerator for motion estimation
- PyTorch GPU warping
- Full pipeline stays on GPU until final encode
- Target: <16.7ms per frame at 1440p (60fps real-time)
"""

# Re-export public API from manager module
from .manager import (
    # Main classes
    InterpolatedStream,
    PerformanceMonitor,
    # Convenience functions
    create_interpolated_stream,
    get_active_stream,
    stop_all_streams,
    shutdown,
    # Internal registry (for task_queue compatibility)
    _active_streams,
)

# Re-export encoder utilities
from .encoder import (
    check_nvenc_available,
    build_ffmpeg_command,
)

# Re-export frame loaders
from .interpolator import (
    FrameLoader,
    GPUFrameLoader,
    interpolate_batched,
)

# Re-export HLS utilities
from .hls import (
    count_segments,
    is_playlist_ready,
    get_file_path,
)

# Backwards compatibility alias
_check_nvenc_available = check_nvenc_available

__all__ = [
    # Main classes
    'InterpolatedStream',
    'PerformanceMonitor',
    # Frame loaders
    'FrameLoader',
    'GPUFrameLoader',
    # Convenience functions
    'create_interpolated_stream',
    'get_active_stream',
    'stop_all_streams',
    'shutdown',
    # Encoder utilities
    'check_nvenc_available',
    'build_ffmpeg_command',
    # Interpolation utilities
    'interpolate_batched',
    # HLS utilities
    'count_segments',
    'is_playlist_ready',
    'get_file_path',
    # Internal (for backwards compat)
    '_active_streams',
    '_check_nvenc_available',
]
