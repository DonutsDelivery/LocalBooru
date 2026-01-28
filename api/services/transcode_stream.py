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
    ):
        """Initialize transcoding stream.

        Args:
            video_path: Path to source video
            target_bitrate: Target bitrate (e.g., "4M", "1536K") or None for original
            target_resolution: Target resolution (width, height) or None for original
            start_position: Start position in seconds
        """
        self.video_path = video_path
        self.target_bitrate = target_bitrate
        self.target_resolution = target_resolution
        self.start_position = start_position

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

            # Wait for playlist file to be created
            max_wait = 30
            for attempt in range(max_wait * 10):
                if (self.hls_dir / "playlist.m3u8").exists():
                    logger.info(f"[Transcode {self.stream_id}] Playlist created after {attempt * 0.1:.1f}s")
                    self.playlist_ready = True
                    break
                if attempt % 100 == 0:  # Log every 10 seconds
                    logger.debug(f"[Transcode {self.stream_id}] Waiting for playlist... ({attempt * 0.1:.1f}s)")
                await asyncio.sleep(0.1)

            if not self.playlist_ready:
                logger.warning(f"[Transcode {self.stream_id}] Playlist not created after {max_wait}s")

            # Wait for process to complete
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_output = stderr.decode('utf-8', errors='replace')
                logger.error(f"[Transcode {self.stream_id}] FFmpeg failed with code {process.returncode}: {error_output[-500:]}")
                self.error = "Encoding failed"
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
        """Detect source video duration and dimensions."""
        try:
            result = await asyncio.create_subprocess_exec(
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration',
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
                    self._duration = float(parts[2])
                    logger.info(f"[Transcode {self.stream_id}] Detected: {self._width}x{self._height}, {self._duration}s")
        except Exception as e:
            logger.warning(f"[Transcode {self.stream_id}] Could not detect video info: {e}")
            # Use defaults
            self._width = 1920
            self._height = 1080
            self._duration = 0

    async def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command for HLS transcoding."""
        cmd = [
            'ffmpeg',
            '-i', self.video_path,
        ]

        # Add start position if specified
        if self.start_position > 0:
            cmd.extend(['-ss', str(self.start_position)])

        # Build video filter chain
        vf_filters = []

        # Add scaling if target resolution specified
        if self.target_resolution:
            width, height = self.target_resolution
            vf_filters.append(f'scale={width}:{height}:flags=lanczos')

        # Always pad to multiple of 2
        vf_filters.append('pad=ceil(iw/2)*2:ceil(ih/2)*2')

        if vf_filters:
            cmd.extend(['-vf', ','.join(vf_filters)])

        # Video codec and bitrate
        cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency'])

        # Add bitrate if specified
        if self.target_bitrate:
            cmd.extend([
                '-b:v', self.target_bitrate,
                '-bufsize', f'{int(self.target_bitrate.rstrip("MK")) * 2}M' if 'M' in self.target_bitrate else f'{int(self.target_bitrate.rstrip("MK")) * 2}K',
            ])
        else:
            # Original quality - use CRF
            cmd.extend(['-crf', '23'])

        # Audio codec
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

        # HLS format
        cmd.extend([
            '-f', 'hls',
            '-hls_time', '10',
            '-hls_list_size', '0',
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
