"""
Simple HLS transcoding service using FFmpeg only.

This provides quality/bitrate selection without requiring SVP or OpticalFlow.
Useful as a fallback when interpolation isn't available or needed.

Architecture:
    Video File → FFmpeg → HLS segments → Browser
"""

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Registry of active transcoding streams
_active_transcode_streams: Dict[str, 'TranscodeStream'] = {}

# Cache for hardware encoder availability (detected once at startup)
_hw_encoder_cache: Optional[Dict[str, bool]] = None


def _detect_hw_encoders() -> Dict[str, bool]:
    """Detect available hardware encoders (cached)."""
    global _hw_encoder_cache
    if _hw_encoder_cache is not None:
        return _hw_encoder_cache

    _hw_encoder_cache = {
        'nvenc': False,
        'vaapi': False,
        'qsv': False,
    }

    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout
            _hw_encoder_cache['nvenc'] = 'h264_nvenc' in output
            _hw_encoder_cache['vaapi'] = 'h264_vaapi' in output
            _hw_encoder_cache['qsv'] = 'h264_qsv' in output
            logger.info(f"[Transcode] Hardware encoders: NVENC={_hw_encoder_cache['nvenc']}, VAAPI={_hw_encoder_cache['vaapi']}, QSV={_hw_encoder_cache['qsv']}")
    except Exception as e:
        logger.warning(f"[Transcode] Could not detect hardware encoders: {e}")

    return _hw_encoder_cache


def _cleanup_on_exit():
    """Clean up all transcode streams on process exit."""
    logger.info("[Transcode] Cleaning up on exit...")
    stop_all_transcode_streams()


# Register cleanup on exit
atexit.register(_cleanup_on_exit)


def stop_all_transcode_streams():
    """Stop all active transcoding streams."""
    for stream in list(_active_transcode_streams.values()):
        stream.stop()


def get_active_transcode_stream(stream_id: str) -> Optional['TranscodeStream']:
    """Get an active transcoding stream by ID."""
    return _active_transcode_streams.get(stream_id)


class TranscodeStream:
    """HLS transcoding stream using FFmpeg only (no VapourSynth)."""

    def __init__(
        self,
        video_path: str,
        target_bitrate: Optional[str] = None,  # e.g., "4M", "1536K"
        target_resolution: Optional[tuple] = None,  # e.g., (1280, 720)
        start_position: float = 0.0,
        force_cfr: bool = False,  # Convert VFR to CFR
    ):
        """Initialize transcoding stream.

        Args:
            video_path: Path to source video
            target_bitrate: Target bitrate (e.g., "4M", "1536K") or None for original
            target_resolution: Target resolution (width, height) or None for original
            start_position: Start position in seconds
            force_cfr: If True, convert Variable Frame Rate to Constant Frame Rate
        """
        self.video_path = video_path
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution
        self.start_position = start_position
        self.force_cfr = force_cfr

        self.stream_id = str(uuid.uuid4())
        self._running = False
        self._task = None
        self._temp_dir: Optional[Path] = None
        self.hls_dir: Optional[Path] = None
        self.error: Optional[str] = None

        self._start_time = 0
        self._duration = 0
        self._width = 0
        self._height = 0
        self._avg_fps = 0  # Average frame rate (for VFR conversion)
        self._has_audio = True  # Assume video has audio by default
        self.segments_ready = 0
        self.playlist_ready = False
        self._process = None  # FFmpeg process reference

        # Register stream
        _active_transcode_streams[self.stream_id] = self

    async def start(self) -> bool:
        """Start the transcoding stream."""
        if self._running:
            self.error = "Stream already running"
            return False

        self._running = True
        self._start_time = time.time()

        try:
            # Create temp directory for HLS segments
            self._temp_dir = Path(tempfile.mkdtemp(prefix="transcode_"))
            self.hls_dir = self._temp_dir / "hls"
            self.hls_dir.mkdir(exist_ok=True)

            # Start encoding task
            self._task = asyncio.create_task(self._encode_loop())
            return True
        except Exception as e:
            logger.error(f"[Transcode {self.stream_id}] Failed to start: {e}")
            self.error = str(e)
            self._running = False
            return False

    async def _encode_loop(self):
        """Main encoding loop."""
        try:
            # Get video duration and dimensions
            await self._detect_video_info()

            # Build FFmpeg command
            ffmpeg_cmd = await self._build_ffmpeg_command()

            logger.info(f"[Transcode {self.stream_id}] Starting FFmpeg: {' '.join(ffmpeg_cmd)}")

            # Run FFmpeg with suppressed output
            self._process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            process = self._process

            # Wait for playlist file AND at least one segment to be created
            max_wait = 30
            for attempt in range(max_wait * 10):
                playlist_exists = (self.hls_dir / "playlist.m3u8").exists()
                segment_exists = (self.hls_dir / "segment_0.ts").exists()

                if playlist_exists and segment_exists:
                    # Check if segment has some content (not empty)
                    segment_size = (self.hls_dir / "segment_0.ts").stat().st_size
                    if segment_size > 1000:  # At least 1KB
                        logger.info(f"[Transcode {self.stream_id}] Playlist and first segment ready after {attempt * 0.1:.1f}s (segment: {segment_size} bytes)")
                        self.playlist_ready = True
                        break

                # Check if FFmpeg process failed
                if process.returncode is not None and process.returncode != 0:
                    error_output = (await process.stderr.read()).decode('utf-8', errors='replace')
                    error_lines = [l.strip() for l in error_output.split('\n') if l.strip()]
                    meaningful_error = '; '.join(error_lines[-3:]) if error_lines else "FFmpeg exited early"
                    self.error = f"FFmpeg error: {meaningful_error[:200]}"
                    logger.error(f"[Transcode {self.stream_id}] {self.error}")
                    return

                if attempt % 100 == 0:  # Log every 10 seconds
                    logger.debug(f"[Transcode {self.stream_id}] Waiting for playlist... ({attempt * 0.1:.1f}s)")
                await asyncio.sleep(0.1)

            if not self.playlist_ready:
                logger.warning(f"[Transcode {self.stream_id}] Playlist not ready after {max_wait}s")

            # Wait for process to complete
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='replace')
                logger.error(f"[Transcode {self.stream_id}] FFmpeg failed with code {process.returncode}: {error_output[-500:]}")
                # Extract meaningful error message from FFmpeg output
                error_lines = [l.strip() for l in error_output.split('\n') if l.strip()]
                # Get last few lines which usually contain the actual error
                meaningful_error = '; '.join(error_lines[-3:]) if error_lines else "Unknown FFmpeg error"
                self.error = f"FFmpeg error: {meaningful_error[:200]}"
            else:
                logger.info(f"[Transcode {self.stream_id}] FFmpeg completed successfully")

        except asyncio.CancelledError:
            logger.info(f"[Transcode {self.stream_id}] Encoding cancelled")
        except Exception as e:
            import traceback
            logger.error(f"[Transcode {self.stream_id}] Encoding error: {e}")
            logger.error(traceback.format_exc())
            self.error = str(e)
        finally:
            self._running = False

    async def _detect_video_info(self):
        """Detect source video duration, dimensions, frame rate, and audio presence."""
        try:
            result = await asyncio.create_subprocess_exec(
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration,avg_frame_rate',
                '-of', 'csv=p=0',
                self.video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                line = stdout.decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) >= 3:
                    self._width = int(parts[0])
                    self._height = int(parts[1])
                    # Stream-level duration may be "N/A" for containers like MKV
                    try:
                        self._duration = float(parts[2]) if parts[2] and parts[2] != 'N/A' else 0
                    except (ValueError, TypeError):
                        self._duration = 0
                    # Parse avg_frame_rate (e.g., "25000/1001" or "30/1")
                    if len(parts) >= 4 and parts[3]:
                        try:
                            fps_parts = parts[3].split('/')
                            if len(fps_parts) == 2:
                                self._avg_fps = float(fps_parts[0]) / float(fps_parts[1])
                            else:
                                self._avg_fps = float(parts[3])
                        except (ValueError, ZeroDivisionError):
                            self._avg_fps = 30  # Default fallback
                    logger.info(f"[Transcode {self.stream_id}] Detected: {self._width}x{self._height}, {self._duration}s, {self._avg_fps:.2f}fps")

            # If stream-level duration is missing (MKV, etc.), query format-level duration
            if self._duration <= 0:
                fmt_result = await asyncio.create_subprocess_exec(
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'csv=p=0',
                    self.video_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                fmt_stdout, _ = await fmt_result.communicate()
                if fmt_result.returncode == 0:
                    dur_str = fmt_stdout.decode('utf-8').strip()
                    try:
                        self._duration = float(dur_str) if dur_str and dur_str != 'N/A' else 0
                        logger.info(f"[Transcode {self.stream_id}] Format-level duration: {self._duration}s")
                    except (ValueError, TypeError):
                        pass

            # Check for audio stream
            audio_result = await asyncio.create_subprocess_exec(
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                self.video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            audio_stdout, _ = await audio_result.communicate()
            self._has_audio = bool(audio_stdout.decode('utf-8').strip())
            logger.info(f"[Transcode {self.stream_id}] Has audio: {self._has_audio}")
        except Exception as e:
            logger.warning(f"[Transcode {self.stream_id}] Could not detect video info: {e}")
            # Use defaults
            self._width = 1920
            self._height = 1080
            self._duration = 0
            self._avg_fps = 30
            self._has_audio = True

    async def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command for HLS transcoding."""
        cmd = ['ffmpeg', '-y']  # -y to overwrite output files

        # Clamp to video duration to avoid seeking past end
        effective_start = self.start_position
        if self._duration > 0 and effective_start >= self._duration:
            effective_start = max(0, self._duration - 1)
            logger.warning(f"[Transcode {self.stream_id}] Start position {self.start_position}s >= duration {self._duration}s, clamping to {effective_start}s")

        # Use hybrid seeking for frame-accurate positioning:
        # - Input seek (-ss before -i): fast, keyframe-based positioning
        # - Output seek (-ss after -i): frame-accurate final positioning
        input_seek = 0
        output_seek = 0
        if effective_start > 2:
            # Seek to 2 seconds before target with input seeking (fast)
            # Then use output seeking for remaining 2 seconds (accurate)
            input_seek = effective_start - 2
            output_seek = 2.0
        elif effective_start > 0:
            # For short seeks, just use output seeking (accurate)
            output_seek = effective_start

        if input_seek > 0:
            cmd.extend(['-ss', f'{input_seek:.3f}'])

        cmd.extend(['-i', self.video_path])

        if output_seek > 0:
            cmd.extend(['-ss', f'{output_seek:.3f}'])

        # Build video filter chain
        vf_filters = []

        # Add scaling if target resolution specified
        if self.target_resolution:
            width, height = self.target_resolution
            # Scale to target width, maintaining aspect ratio (-2 ensures height is divisible by 2)
            vf_filters.append(f'scale={width}:-2:flags=lanczos')

        # Always pad to multiple of 2
        vf_filters.append('pad=ceil(iw/2)*2:ceil(ih/2)*2')

        # Ensure yuv420p pixel format for encoder compatibility (handles 10-bit, yuv444, etc.)
        vf_filters.append('format=yuv420p')

        if vf_filters:
            cmd.extend(['-vf', ','.join(vf_filters)])

        # For VFR videos, convert to constant frame rate to prevent stuttering
        if self.force_cfr:
            # Use -vsync cfr to ensure constant frame rate output
            # Use the detected average frame rate as target
            if self._avg_fps > 0:
                cmd.extend(['-r', str(self._avg_fps)])
            cmd.extend(['-vsync', 'cfr'])
            logger.info(f"[Transcode {self.stream_id}] Converting VFR to CFR at {self._avg_fps:.2f} fps")

        # Video codec and bitrate - prefer hardware encoding for speed
        hw_encoders = _detect_hw_encoders()
        use_nvenc = hw_encoders['nvenc']

        # Force keyframes every 2 seconds for proper HLS segmentation
        # This is required for both hardware and software encoders
        cmd.extend(['-force_key_frames', 'expr:gte(t,n_forced*2)'])

        # Calculate GOP size based on frame rate (2 seconds worth of frames)
        gop_size = int(self._avg_fps * 2) if self._avg_fps > 0 else 60

        if use_nvenc:
            # NVIDIA NVENC - very fast hardware encoding
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-g', str(gop_size),
                '-keyint_min', str(gop_size),
            ])
            logger.info(f"[Transcode {self.stream_id}] Using NVENC hardware encoding (GOP={gop_size})")
        else:
            # Software fallback (VAAPI/QSV require more complex filter setup)
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-g', str(gop_size),
                '-keyint_min', str(gop_size),
            ])

        # Add bitrate if specified
        if self.target_bitrate:
            cmd.extend([
                '-b:v', self.target_bitrate,
                '-bufsize', f'{int(self.target_bitrate.rstrip("MK")) * 2}M' if 'M' in self.target_bitrate else f'{int(self.target_bitrate.rstrip("MK")) * 2}K',
            ])
        else:
            # Original quality - use CRF
            cmd.extend(['-crf', '23'])

        # Audio codec (only if video has audio)
        if self._has_audio:
            cmd.extend(['-c:a', 'aac', '-ar', '48000', '-ac', '2', '-b:a', '192k'])
        else:
            cmd.extend(['-an'])  # No audio

        # HLS format - use shorter segments for faster startup
        # Use a sliding window to limit disk usage: keep only recent segments,
        # delete old ones. Seeking restarts the stream anyway, so we don't need
        # all historical segments. 20 segments × 2s = 40s buffer (~100-200MB max).
        cmd.extend([
            '-f', 'hls',
            '-hls_time', '2',  # 2-second segments for faster startup
            '-hls_list_size', '20',  # Keep last 20 segments in playlist
            '-hls_flags', 'delete_segments+append_list',  # Delete old segments from disk
            '-hls_segment_filename', str(self.hls_dir / 'segment_%d.ts'),
            str(self.hls_dir / 'playlist.m3u8')
        ])

        return cmd

    def stop(self):
        """Stop the transcoding stream."""
        logger.info(f"[Transcode {self.stream_id}] Stopping")
        self._running = False

        # Kill FFmpeg process first
        if self._process:
            try:
                self._process.terminate()
                # Give it a moment to terminate gracefully
                import asyncio
                try:
                    asyncio.get_event_loop().run_until_complete(
                        asyncio.wait_for(self._process.wait(), timeout=2)
                    )
                except (asyncio.TimeoutError, RuntimeError):
                    # Force kill if it doesn't terminate
                    self._process.kill()
            except Exception as e:
                logger.warning(f"[Transcode {self.stream_id}] Error killing process: {e}")
            self._process = None

        if self._task:
            self._task.cancel()
            self._task = None

        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._temp_dir)
                logger.debug(f"[Transcode {self.stream_id}] Cleaned up {self._temp_dir}")
            except Exception as e:
                logger.warning(f"[Transcode {self.stream_id}] Failed to clean up: {e}")
            self._temp_dir = None

        # Unregister stream
        if self.stream_id in _active_transcode_streams:
            del _active_transcode_streams[self.stream_id]
