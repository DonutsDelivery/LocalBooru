"""
HLS segment generation and serving for SVP streams.

Contains the SVPStream class that manages the full pipeline from video input
through SVP interpolation to HLS output.
"""
import asyncio
import logging
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from ..svp_platform import get_clean_env
from .config import SVP_PRESETS
from .encoder import check_nvenc_available, get_video_info
from .manager import register_stream, unregister_stream
from .svp_integration import generate_vspipe_stdin_script

logger = logging.getLogger(__name__)


class SVPStream:
    """
    Manages an SVP-interpolated video stream.

    Pipeline: Video -> VapourSynth/SVPflow -> vspipe -> FFmpeg -> HLS segments

    This uses vspipe to run the VapourSynth script and pipe raw frames
    directly to FFmpeg for HLS encoding. Much more efficient than
    frame-by-frame Python processing.
    """

    def __init__(
        self,
        video_path: str,
        target_fps: int = 60,
        preset: str = "balanced",
        use_nvenc: Optional[bool] = None,
        use_nvof: bool = True,
        shader: int = 23,
        artifact_masking: int = 100,
        frame_interpolation: int = 2,
        custom_super: Optional[str] = None,
        custom_analyse: Optional[str] = None,
        custom_smooth: Optional[str] = None,
        start_position: float = 0.0,  # Seek position in seconds
        target_bitrate: Optional[str] = None,  # Target bitrate (e.g., "4M", "1536K")
        target_resolution: Optional[tuple] = None,  # Target resolution (width, height)
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.preset = preset if preset in SVP_PRESETS else "balanced"
        self.use_nvenc = use_nvenc if use_nvenc is not None else check_nvenc_available()
        self.stream_id = str(uuid.uuid4())[:8]
        self.start_position = start_position  # Where to start processing from

        # Quality settings
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution

        # Key SVP settings
        self.use_nvof = use_nvof
        self.shader = shader
        self.artifact_masking = artifact_masking
        self.frame_interpolation = frame_interpolation

        # Custom SVP parameters (full override when set)
        self.custom_super = custom_super
        self.custom_analyse = custom_analyse
        self.custom_smooth = custom_smooth

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._decode_proc: Optional[subprocess.Popen] = None
        self._vspipe_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._temp_dir: Optional[Path] = None
        self._script_path: Optional[Path] = None
        self._error: Optional[str] = None
        self._start_time: Optional[float] = None

        # Video info (populated on start)
        self._width: int = 0
        self._height: int = 0
        self._src_fps: float = 0
        self._src_fps_num: int = 0
        self._src_fps_den: int = 1
        self._num_frames: int = 0
        self._duration: float = 0  # Source video duration in seconds

        # Register stream
        register_stream(self)

    @property
    def hls_dir(self) -> Optional[Path]:
        """Get the HLS output directory."""
        return self._temp_dir

    @property
    def playlist_path(self) -> Optional[Path]:
        """Get path to the HLS playlist file."""
        if self._temp_dir:
            return self._temp_dir / "stream.m3u8"
        return None

    @property
    def is_running(self) -> bool:
        """Check if the stream is active."""
        return self._running

    @property
    def error(self) -> Optional[str]:
        """Get any error that occurred."""
        return self._error

    @property
    def segments_ready(self) -> int:
        """Count of HLS segments ready."""
        if not self._temp_dir:
            return 0
        return len(list(self._temp_dir.glob("segment_*.ts")))

    @property
    def playlist_ready(self) -> bool:
        """Check if HLS playlist exists and has content."""
        if not self.playlist_path or not self.playlist_path.exists():
            return False
        try:
            content = self.playlist_path.read_text()
            return "segment_" in content and "#EXTINF" in content
        except:
            return False

    def _get_video_info(self) -> bool:
        """Get video dimensions, FPS, frame count, and duration using ffprobe."""
        info = get_video_info(self.video_path)
        if not info["success"]:
            return False

        self._width = info["width"]
        self._height = info["height"]
        self._src_fps = info["src_fps"]
        self._src_fps_num = info["src_fps_num"]
        self._src_fps_den = info["src_fps_den"]
        self._num_frames = info["num_frames"]
        self._duration = info["duration"]

        logger.info(f"[SVP {self.stream_id}] Video: {self._width}x{self._height} @ {self._src_fps:.2f}fps, {self._num_frames} frames, {self._duration:.1f}s")
        return True

    async def start(self) -> bool:
        """Start the SVP interpolated stream."""
        if self._running:
            return True

        # Get video info
        if not self._get_video_info():
            self._error = "Failed to get video info"
            return False

        # Check if interpolation is actually needed
        # If source fps is within 5% of target fps, SVP is pointless
        fps_ratio = self.target_fps / self._src_fps if self._src_fps > 0 else 2.0
        if 0.95 <= fps_ratio <= 1.05:
            self._error = f"Source fps ({self._src_fps:.2f}) already at target ({self.target_fps}fps), interpolation not needed"
            logger.warning(f"[SVP {self.stream_id}] {self._error}")
            return False

        # Create temp directory for HLS output
        self._temp_dir = Path(tempfile.mkdtemp(prefix='svp_stream_'))
        logger.info(f"[SVP {self.stream_id}] HLS output: {self._temp_dir}")

        # Generate vspipe stdin script (reads Y4M from stdin, no Python frame handling)
        script = generate_vspipe_stdin_script(
            self.target_fps,
            self.preset,
            use_nvof=self.use_nvof,
            shader=self.shader,
            artifact_masking=self.artifact_masking,
            frame_interpolation=self.frame_interpolation,
            custom_super=self.custom_super,
            custom_analyse=self.custom_analyse,
            custom_smooth=self.custom_smooth,
        )
        self._script_path = self._temp_dir / "svp_stdin.vpy"
        self._script_path.write_text(script)
        logger.debug(f"[SVP {self.stream_id}] Script written to {self._script_path}")

        # Start the pipeline
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._run_pipeline())

        return True

    async def _run_pipeline(self):
        """Run the FFmpeg -> vspipe -> FFmpeg pipeline.

        Three-stage pipeline with NO PYTHON in the frame path:
        1. FFmpeg decodes video to Y4M (hardware accelerated)
        2. vspipe reads Y4M from stdin, processes with SVP, outputs Y4M
        3. FFmpeg encodes Y4M to HLS

        This is the same architecture that native SVP uses for real-time playback.
        """
        try:
            # Stage 1: FFmpeg decode to Y4M
            decode_cmd = [
                'ffmpeg',
                '-hwaccel', 'auto',  # Use NVDEC/VAAPI if available
                '-threads', '0',
            ]

            # Add seek if needed
            if self.start_position > 0:
                decode_cmd.extend(['-ss', str(self.start_position)])

            decode_cmd.extend([
                '-i', self.video_path,
            ])

            # Downscale BEFORE SVP if quality preset specifies resolution
            # This makes SVP process smaller frames = much faster
            if self.target_resolution:
                width, height = self.target_resolution
                decode_cmd.extend(['-vf', f'scale={width}:{height}:flags=lanczos'])

            decode_cmd.extend([
                '-f', 'yuv4mpegpipe',
                '-pix_fmt', 'yuv420p',
                '-'
            ])

            # Stage 2: vspipe with SVP processing
            vspipe_cmd = [
                'vspipe',
                '-c', 'y4m',
                str(self._script_path),
                '-'
            ]

            # Stage 3: FFmpeg encode to HLS
            encode_cmd = [
                'ffmpeg',
                '-y',
                '-probesize', '32',
                '-analyzeduration', '0',
                '-fflags', '+nobuffer+flush_packets',
                '-f', 'yuv4mpegpipe',
                '-i', '-',  # Y4M from vspipe
            ]

            # Add audio input with seek if needed
            if self.start_position > 0:
                encode_cmd.extend(['-ss', str(self.start_position)])
            encode_cmd.extend([
                '-probesize', '5000000',
                '-i', self.video_path,  # Original video for audio
                '-map', '0:v',
                '-map', '1:a?',
            ])

            # Pad to even dimensions (required for H.264)
            # Note: Scaling already done in decode stage before SVP
            encode_cmd.extend(['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])

            # Video encoder selection
            if self.use_nvenc:
                encode_cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p1',
                    '-tune', 'll',
                ])
                if self.target_bitrate:
                    encode_cmd.extend([
                        '-rc', 'cbr',
                        '-b:v', self.target_bitrate,
                        '-maxrate', self.target_bitrate,
                        '-bufsize', f'{int(self.target_bitrate.rstrip("MK")) * 2}M' if 'M' in self.target_bitrate else f'{int(self.target_bitrate.rstrip("MK")) * 2}K',
                    ])
                else:
                    encode_cmd.extend(['-rc', 'vbr', '-cq', '23', '-b:v', '0'])
            else:
                encode_cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency'])
                if self.target_bitrate:
                    encode_cmd.extend(['-b:v', self.target_bitrate])
                else:
                    encode_cmd.extend(['-crf', '23'])

            # Audio encoder
            encode_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

            # HLS output with MPEG-TS segments
            encode_cmd.extend([
                '-g', str(self.target_fps * 2),  # Keyframe every 2 seconds
                '-keyint_min', str(self.target_fps),
                '-f', 'hls',
                '-hls_time', '4',
                '-hls_list_size', '20',
                '-hls_flags', 'delete_segments+append_list+split_by_time',
                '-hls_segment_filename', str(self._temp_dir / 'segment_%03d.ts'),
                str(self.playlist_path)
            ])

            logger.info(f"[SVP {self.stream_id}] Starting pipeline: FFmpeg decode -> vspipe/SVP -> FFmpeg encode")
            logger.debug(f"[SVP {self.stream_id}] Decode: {' '.join(decode_cmd)}")
            logger.debug(f"[SVP {self.stream_id}] vspipe: {' '.join(vspipe_cmd)}")
            logger.debug(f"[SVP {self.stream_id}] Encode: {' '.join(encode_cmd)}")

            # Start the three-stage pipeline
            clean_env = get_clean_env()

            # Stage 1: FFmpeg decode
            self._decode_proc = subprocess.Popen(
                decode_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )

            # Stage 2: vspipe (reads from decode, writes to encode)
            self._vspipe_proc = subprocess.Popen(
                vspipe_cmd,
                stdin=self._decode_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )
            self._decode_proc.stdout.close()  # Allow SIGPIPE

            # Stage 3: FFmpeg encode
            self._ffmpeg_proc = subprocess.Popen(
                encode_cmd,
                stdin=self._vspipe_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env
            )
            self._vspipe_proc.stdout.close()  # Allow SIGPIPE

            # Monitor the three processes
            while self._running:
                # Check decode process
                if self._decode_proc and self._decode_proc.poll() is not None:
                    decode_stderr = self._decode_proc.stderr.read().decode() if self._decode_proc.stderr else ""
                    if self._decode_proc.returncode != 0:
                        self._error = f"Decode FFmpeg exited with code {self._decode_proc.returncode}: {decode_stderr[-500:]}"
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                        break

                # Check vspipe process
                if self._vspipe_proc and self._vspipe_proc.poll() is not None:
                    vspipe_stderr = self._vspipe_proc.stderr.read().decode() if self._vspipe_proc.stderr else ""
                    if self._vspipe_proc.returncode != 0:
                        self._error = f"vspipe exited with code {self._vspipe_proc.returncode}: {vspipe_stderr[-1000:]}"
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                        print(f"[SVP {self.stream_id}] vspipe stderr:\n{vspipe_stderr}")
                        break

                # Check encode FFmpeg process
                if self._ffmpeg_proc and self._ffmpeg_proc.poll() is not None:
                    ffmpeg_stderr = self._ffmpeg_proc.stderr.read().decode() if self._ffmpeg_proc.stderr else ""
                    if self._ffmpeg_proc.returncode != 0:
                        # Collect all stderr for debugging
                        vspipe_stderr = ""
                        decode_stderr = ""
                        if self._vspipe_proc and self._vspipe_proc.stderr:
                            try:
                                vspipe_stderr = self._vspipe_proc.stderr.read().decode()
                            except:
                                pass
                        if self._decode_proc and self._decode_proc.stderr:
                            try:
                                decode_stderr = self._decode_proc.stderr.read().decode()
                            except:
                                pass
                        combined_error = f"Encode FFmpeg exit {self._ffmpeg_proc.returncode}"
                        if decode_stderr:
                            combined_error += f"\nDecode stderr: {decode_stderr[-300:]}"
                        if vspipe_stderr:
                            combined_error += f"\nvspipe stderr: {vspipe_stderr[-300:]}"
                        combined_error += f"\nEncode stderr: {ffmpeg_stderr[-300:]}"
                        self._error = combined_error
                        logger.error(f"[SVP {self.stream_id}] {self._error}")
                    else:
                        logger.info(f"[SVP {self.stream_id}] Pipeline finished normally")
                    break

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info(f"[SVP {self.stream_id}] Pipeline cancelled")
        except Exception as e:
            self._error = str(e)
            logger.error(f"[SVP {self.stream_id}] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
            self._cleanup_processes()

    def _cleanup_processes(self):
        """Clean up subprocess resources."""
        # Clean up decode process
        if self._decode_proc:
            try:
                self._decode_proc.terminate()
                self._decode_proc.wait(timeout=2)
            except:
                try:
                    self._decode_proc.kill()
                except:
                    pass
            self._decode_proc = None

        # Clean up vspipe process
        if self._vspipe_proc:
            try:
                self._vspipe_proc.terminate()
                self._vspipe_proc.wait(timeout=2)
            except:
                try:
                    self._vspipe_proc.kill()
                except:
                    pass
            self._vspipe_proc = None

        # Clean up encode FFmpeg process
        if self._ffmpeg_proc:
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=2)
            except:
                try:
                    self._ffmpeg_proc.kill()
                except:
                    pass
            self._ffmpeg_proc = None

    def stop(self):
        """Stop the stream."""
        logger.info(f"[SVP {self.stream_id}] Stopping stream")
        self._running = False

        if self._task:
            self._task.cancel()
            self._task = None

        self._cleanup_processes()

        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
                logger.debug(f"[SVP {self.stream_id}] Cleaned up {self._temp_dir}")
            except Exception as e:
                logger.warning(f"[SVP {self.stream_id}] Failed to clean up: {e}")
            self._temp_dir = None

        # Unregister stream
        unregister_stream(self.stream_id)

    def get_stats(self) -> dict:
        """Get stream statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "stream_id": self.stream_id,
            "video_path": self.video_path,
            "target_fps": self.target_fps,
            "preset": self.preset,
            "resolution": f"{self._width}x{self._height}",
            "src_fps": self._src_fps,
            "duration": self._duration,  # Source video duration
            "running": self._running,
            "elapsed_seconds": elapsed,
            "segments_ready": self.segments_ready,
            "playlist_ready": self.playlist_ready,
            "error": self._error,
            "use_nvenc": self.use_nvenc,
        }
