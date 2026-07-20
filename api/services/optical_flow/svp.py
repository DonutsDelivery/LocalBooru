"""
SVP (SmoothVideo Project) interpolator using VapourSynth.

Uses commercial SVP plugins for high-quality motion-compensated interpolation.
"""
import logging
import numpy as np

from .gpu_utils import HAS_VAPOURSYNTH, HAS_SVP, HAS_CV2, SVP_PLUGIN_PATH
from ..svp_platform import get_svp_plugin_full_paths

logger = logging.getLogger(__name__)

if HAS_CV2:
    import cv2


class SVPInterpolator:
    """
    Frame interpolator using SVP (SmoothVideo Project) via VapourSynth.

    SVP is a commercial frame interpolation plugin that uses advanced
    motion estimation and compensation. It requires:
    - VapourSynth installed
    - SVP license and plugins (libsvpflow1.so, libsvpflow2.so)

    SVP provides excellent quality interpolation with configurable presets
    for different performance/quality tradeoffs.

    Performance: Depends on preset and resolution
    - Fast preset: ~20-30ms per frame at 1080p
    - Quality preset: ~40-60ms per frame at 1080p
    """

    # SVP super and analyse presets (JSON format)
    PRESETS = {
        "fast": {
            "super": "{gpu:1,pel:1,scale:{up:0,down:4}}",
            "analyse": "{gpu:1,block:{w:32,h:32,overlap:0},main:{search:{coarse:{type:2,distance:-6,bad:{sad:2000,range:24}},type:2,distance:6}},refine:[{thsad:200}]}",
            "smooth": "{gpuid:0,rate:{num:2,den:1,abs:false},algo:23,mask:{area:100},scene:{}}",
        },
        "balanced": {
            "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
            "analyse": "{gpu:1,block:{w:16,h:16,overlap:2},main:{search:{coarse:{type:2,distance:-8,bad:{sad:2000,range:24}},type:2,distance:8}},refine:[{thsad:200}]}",
            "smooth": "{gpuid:0,rate:{num:2,den:1,abs:false},algo:23,mask:{area:100},scene:{}}",
        },
        "quality": {
            "super": "{gpu:1,pel:2,scale:{up:0,down:2}}",
            "analyse": "{gpu:1,block:{w:8,h:8,overlap:2},main:{search:{coarse:{type:2,distance:-10,bad:{sad:2000,range:24}},type:2,distance:10}},refine:[{thsad:200},{thsad:100}]}",
            "smooth": "{gpuid:0,rate:{num:2,den:1,abs:false},algo:23,mask:{area:100},scene:{}}",
        },
    }

    def __init__(self, preset: str = "balanced", target_fps: int = 60,
                 plugin_path: str = SVP_PLUGIN_PATH):
        """
        Initialize SVP interpolator.

        Args:
            preset: Quality preset ('fast', 'balanced', 'quality')
            target_fps: Target output frame rate
            plugin_path: Path to SVP plugin directory
        """
        if not HAS_VAPOURSYNTH:
            raise RuntimeError("VapourSynth not installed - SVP interpolation unavailable")
        if not HAS_SVP:
            raise RuntimeError("SVP plugins not available - check installation")

        self.preset = preset if preset in self.PRESETS else "balanced"
        self.target_fps = target_fps
        self.plugin_path = plugin_path

        self._core = None
        self._initialized = False
        self._width = 0
        self._height = 0

    def initialize(self, width: int = 0, height: int = 0) -> bool:
        """Initialize VapourSynth core and load SVP plugins."""
        if self._initialized:
            return True

        try:
            import vapoursynth as vs
            self._core = vs.core

            # Set cache size (in MB)
            self._core.max_cache_size = 1024

            # Load SVP plugins
            _flow1, _flow2 = get_svp_plugin_full_paths(self.plugin_path)
            self._core.std.LoadPlugin(_flow1)
            self._core.std.LoadPlugin(_flow2)

            self._width = width
            self._height = height
            self._initialized = True

            logger.info(f"SVP interpolator initialized (preset={self.preset}, target_fps={self.target_fps})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SVP interpolator: {e}")
            return False

    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between two frames using SVP.

        Args:
            frame1: First frame (H, W, 3) uint8 BGR
            frame2: Second frame (H, W, 3) uint8 BGR
            t: Interpolation position (0.0 = frame1, 1.0 = frame2)

        Returns:
            Interpolated frame (H, W, 3) uint8 BGR
        """
        if not self._initialized:
            if not self.initialize(frame1.shape[1], frame1.shape[0]):
                return self._blend_frames(frame1, frame2, t)

        try:
            import vapoursynth as vs

            h, w = frame1.shape[:2]
            preset_config = self.PRESETS[self.preset]

            # Create a 2-frame clip from input frames
            # Convert BGR to RGB for VapourSynth
            rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Stack frames to create numpy array for BlankClip
            frames_data = np.stack([rgb1, rgb2], axis=0)

            # Create VapourSynth clip from numpy array
            # BlankClip + ModifyFrame approach
            clip = self._core.std.BlankClip(
                width=w, height=h,
                format=vs.RGB24,
                length=2,
                fpsnum=30, fpsden=1  # Dummy FPS, will be overridden
            )

            # Replace blank frames with our actual frames
            def _frame_func(n, f):
                fout = f.copy()
                frame_data = frames_data[n]
                for plane in range(3):
                    np.copyto(
                        np.asarray(fout[plane]),
                        frame_data[:, :, plane]
                    )
                return fout

            clip = self._core.std.ModifyFrame(clip, clip, _frame_func)

            # Convert to YUV for SVP processing
            clip_yuv = self._core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

            # SVP processing pipeline
            # 1. Create super clip
            super_params = preset_config["super"]
            super_clip = self._core.svp1.Super(clip_yuv, super_params)

            # 2. Analyze motion
            analyse_params = preset_config["analyse"]
            vectors = self._core.svp1.Analyse(
                super_clip["clip"],
                super_clip["data"],
                clip_yuv,
                analyse_params
            )

            # 3. Smooth/interpolate
            # Modify smooth params to output single intermediate frame
            # We need rate:{num:N,den:1,abs:true} where N gives us the frame at position t
            # For t=0.5, we want frame index 1 out of [0, 1, 2] (original 0, interp, original 1)
            smooth_params = preset_config["smooth"]

            smooth_clip = self._core.svp2.SmoothFps(
                clip_yuv,
                super_clip["clip"],
                super_clip["data"],
                vectors["clip"],
                vectors["data"],
                smooth_params
            )

            # Convert back to RGB
            smooth_rgb = self._core.resize.Bicubic(smooth_clip, format=vs.RGB24, matrix_s="709")

            # Extract the interpolated frame (frame 1 in a 2x multiplied clip)
            # With rate:{num:2,den:1}, output has frames at t=0, t=0.5, t=1.0
            # We want the middle frame (index 1)
            target_frame = int(t * (smooth_rgb.num_frames - 1))
            target_frame = max(0, min(target_frame, smooth_rgb.num_frames - 1))

            # Get frame synchronously
            frame = smooth_rgb.get_frame(target_frame)

            # Convert to numpy
            result = np.empty((h, w, 3), dtype=np.uint8)
            for plane in range(3):
                np.copyto(result[:, :, plane], np.asarray(frame[plane]))

            # Convert RGB back to BGR
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"SVP interpolation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._blend_frames(frame1, frame2, t)

    def interpolate_clip(self, video_path: str, target_fps: int = None) -> 'vs.VideoNode':
        """
        Create an interpolated VapourSynth clip from a video file.

        This is more efficient than frame-by-frame interpolation for
        processing entire videos.

        Args:
            video_path: Path to input video file
            target_fps: Target frame rate (None = use instance default)

        Returns:
            VapourSynth VideoNode with interpolated frames
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("SVP interpolator not initialized")

        import vapoursynth as vs

        target_fps = target_fps or self.target_fps
        preset_config = self.PRESETS[self.preset]

        # Load video using ffms2 or lsmas
        try:
            clip = self._core.ffms2.Source(video_path)
        except AttributeError:
            try:
                clip = self._core.lsmas.LWLibavSource(video_path)
            except AttributeError:
                raise RuntimeError("No video source filter available (need ffms2 or lsmas)")

        # Get source FPS
        src_fps = clip.fps.numerator / clip.fps.denominator

        # Convert to YUV if needed
        if clip.format.color_family != vs.YUV:
            clip = self._core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

        # SVP processing
        super_params = preset_config["super"]
        super_clip = self._core.svp1.Super(clip, super_params)

        analyse_params = preset_config["analyse"]
        vectors = self._core.svp1.Analyse(
            super_clip["clip"],
            super_clip["data"],
            clip,
            analyse_params
        )

        # Calculate rate multiplier
        rate_mult = target_fps / src_fps
        smooth_params = f"{{gpuid:0,rate:{{num:{int(rate_mult * 1000)},den:1000,abs:true}},algo:23,mask:{{area:100}},scene:{{}}}}"

        smooth_clip = self._core.svp2.SmoothFps(
            clip,
            super_clip["clip"],
            super_clip["data"],
            vectors["clip"],
            vectors["data"],
            smooth_params,
            fps=src_fps
        )

        return smooth_clip

    def _blend_frames(self, frame1: np.ndarray, frame2: np.ndarray, t: float) -> np.ndarray:
        """Simple alpha blend fallback."""
        return ((1 - t) * frame1.astype(np.float32) + t * frame2.astype(np.float32)).astype(np.uint8)

    def cleanup(self):
        """Release VapourSynth resources."""
        self._core = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
