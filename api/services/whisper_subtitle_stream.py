"""
Real-time subtitle generation using faster-whisper.

Architecture:
    Video File -> FFmpeg (audio extraction) -> faster-whisper -> VTT cues -> SSE events

Follows transcode_stream.py patterns (global registry, UUID IDs, atexit cleanup, async subprocess).
"""

import asyncio
import atexit
import io
import logging
import shutil
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Registry of active subtitle streams
_active_subtitle_streams: Dict[str, 'WhisperSubtitleStream'] = {}

# Cached model (reused across streams)
_cached_model = None
_cached_model_key = None  # (size, device, compute_type)


def _cleanup_on_exit():
    """Clean up all subtitle streams on process exit."""
    logger.info("[Whisper] Cleaning up on exit...")
    stop_all_subtitle_streams()


atexit.register(_cleanup_on_exit)


def stop_all_subtitle_streams():
    """Stop all active subtitle streams."""
    for stream in list(_active_subtitle_streams.values()):
        stream.stop()


def get_active_subtitle_stream(stream_id: str) -> Optional['WhisperSubtitleStream']:
    """Get an active subtitle stream by ID."""
    return _active_subtitle_streams.get(stream_id)


def _load_model(model_size: str, device: str, compute_type: str):
    """Load or reuse a cached faster-whisper model.

    Returns the model instance, or raises ImportError/RuntimeError.
    """
    global _cached_model, _cached_model_key

    key = (model_size, device, compute_type)
    if _cached_model is not None and _cached_model_key == key:
        return _cached_model

    from faster_whisper import WhisperModel

    # Resolve "auto" device
    if device == "auto":
        try:
            import torch
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            actual_device = "cpu"
    else:
        actual_device = device

    # Resolve "auto" compute type
    if compute_type == "auto":
        actual_compute = "float16" if actual_device == "cuda" else "int8"
    else:
        actual_compute = compute_type

    logger.info(f"[Whisper] Loading model: {model_size} on {actual_device} ({actual_compute})")
    model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute)
    logger.info(f"[Whisper] Model loaded successfully")

    _cached_model = model
    _cached_model_key = key
    return model


def find_cached_vtt(video_path: str, language: str, task: str) -> Optional[Path]:
    """Check for a cached VTT file alongside the video."""
    video = Path(video_path)
    # Pattern: video_stem.lang.vtt or video_stem.lang.translated.vtt
    suffix = f".{language}.translated.vtt" if task == "translate" else f".{language}.vtt"
    vtt_path = video.with_suffix(suffix)
    if vtt_path.exists():
        return vtt_path
    return None


class WhisperSubtitleStream:
    """Real-time subtitle generation stream using faster-whisper."""

    def __init__(
        self,
        video_path: str,
        model_size: str = "medium",
        language: str = "ja",
        task: str = "translate",
        chunk_duration: int = 30,
        beam_size: int = 8,
        device: str = "auto",
        compute_type: str = "auto",
        vad_filter: bool = True,
        suppress_nst: bool = True,
        cache_subtitles: bool = True,
        start_position: float = 0.0,
    ):
        self.video_path = video_path
        self.model_size = model_size
        self.language = language
        self.task = task
        self.chunk_duration = chunk_duration
        self.beam_size = beam_size
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.suppress_nst = suppress_nst
        self.cache_subtitles = cache_subtitles
        self.start_position = start_position

        self.stream_id = str(uuid.uuid4())
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._temp_dir: Optional[Path] = None
        self.vtt_path: Optional[Path] = None
        self.cached_vtt_path: Optional[Path] = None  # Set if serving a pre-existing VTT
        self.error: Optional[str] = None
        self.completed = False
        self._duration = 0.0
        self._has_audio = True
        self._process = None  # FFmpeg process

        # Register stream
        _active_subtitle_streams[self.stream_id] = self

    async def start(self) -> bool:
        """Start subtitle generation. Returns True if started or cached VTT found."""
        if self._running:
            self.error = "Stream already running"
            return False

        # Check for cached VTT
        cached = find_cached_vtt(self.video_path, self.language, self.task)
        if cached:
            logger.info(f"[Whisper {self.stream_id}] Found cached VTT: {cached}")
            self.cached_vtt_path = cached
            self.vtt_path = cached
            self.completed = True
            return True

        self._running = True

        try:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="whisper_"))
            self.vtt_path = self._temp_dir / "subtitles.vtt"

            # Write VTT header
            with open(self.vtt_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")

            # Check for audio stream
            has_audio = await self._check_audio()
            if not has_audio:
                self.error = "No audio stream found in video"
                self._running = False
                return False

            # Detect duration
            await self._detect_duration()

            # Start transcription loop as async task
            self._task = asyncio.create_task(self._transcription_loop())
            return True

        except Exception as e:
            logger.error(f"[Whisper {self.stream_id}] Failed to start: {e}")
            self.error = str(e)
            self._running = False
            return False

    async def _check_audio(self) -> bool:
        """Check if the video has an audio stream."""
        try:
            result = await asyncio.create_subprocess_exec(
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                self.video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            return bool(stdout.decode('utf-8').strip())
        except Exception as e:
            logger.warning(f"[Whisper {self.stream_id}] Audio check failed: {e}")
            return True  # Assume yes, FFmpeg will fail later if no audio

    async def _detect_duration(self):
        """Detect video duration."""
        try:
            result = await asyncio.create_subprocess_exec(
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                self.video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            duration_str = stdout.decode('utf-8').strip()
            if duration_str:
                self._duration = float(duration_str)
                logger.info(f"[Whisper {self.stream_id}] Duration: {self._duration:.1f}s")
        except Exception as e:
            logger.warning(f"[Whisper {self.stream_id}] Duration detection failed: {e}")

    async def _transcription_loop(self):
        """Main transcription loop: extract audio chunks and transcribe."""
        from .events import subtitle_events, SubtitleEventType

        try:
            # Load model (blocking, run in executor)
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None, _load_model, self.model_size, self.device, self.compute_type
            )

            await subtitle_events.broadcast(SubtitleEventType.STARTED, {
                "stream_id": self.stream_id,
                "duration": self._duration,
            })

            chunk_idx = 0
            chunk_offset = self.start_position

            while self._running and (self._duration <= 0 or chunk_offset < self._duration):
                chunk_end = chunk_offset + self.chunk_duration
                if self._duration > 0:
                    chunk_end = min(chunk_end, self._duration)

                actual_duration = chunk_end - chunk_offset
                if actual_duration < 0.5:
                    break  # Remaining audio too short

                # Extract audio chunk as 16kHz mono PCM WAV
                audio_data = await self._extract_audio_chunk(chunk_offset, actual_duration)

                if audio_data is None:
                    # No more audio or error
                    if chunk_idx == 0:
                        self.error = "Failed to extract audio"
                        await subtitle_events.broadcast(SubtitleEventType.ERROR, {
                            "stream_id": self.stream_id,
                            "error": self.error,
                        })
                    break

                if not self._running:
                    break

                # Transcribe chunk (blocking, run in executor)
                segments = await loop.run_in_executor(
                    None, self._transcribe_chunk, model, audio_data
                )

                if not self._running:
                    break

                # Process segments: offset timestamps and write VTT cues
                for segment in segments:
                    start_time = chunk_offset + segment.start
                    end_time = chunk_offset + segment.end
                    text = segment.text.strip()

                    if not text:
                        continue

                    # Write VTT cue to file
                    cue_text = f"{_format_vtt_time(start_time)} --> {_format_vtt_time(end_time)}\n{text}\n\n"
                    with open(self.vtt_path, 'a', encoding='utf-8') as f:
                        f.write(cue_text)

                    # Broadcast cue via SSE
                    await subtitle_events.broadcast(SubtitleEventType.CUE, {
                        "stream_id": self.stream_id,
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                    })

                # Broadcast progress
                progress = (chunk_end / self._duration * 100) if self._duration > 0 else 0
                await subtitle_events.broadcast(SubtitleEventType.PROGRESS, {
                    "stream_id": self.stream_id,
                    "chunk": chunk_idx,
                    "progress": round(progress, 1),
                    "processed_time": chunk_end,
                    "total_time": self._duration,
                })

                chunk_offset = chunk_end
                chunk_idx += 1

            if self._running:
                self.completed = True

                # Cache VTT alongside video if enabled (only for full transcriptions)
                if self.cache_subtitles and self.start_position == 0 and self.vtt_path and self.vtt_path.exists():
                    self._cache_vtt()

                await subtitle_events.broadcast(SubtitleEventType.COMPLETED, {
                    "stream_id": self.stream_id,
                    "chunks_processed": chunk_idx,
                    "cached": self.cache_subtitles,
                })

        except asyncio.CancelledError:
            logger.info(f"[Whisper {self.stream_id}] Transcription cancelled")
        except ImportError as e:
            self.error = f"faster-whisper not installed: {e}"
            logger.error(f"[Whisper {self.stream_id}] {self.error}")
            await subtitle_events.broadcast(SubtitleEventType.ERROR, {
                "stream_id": self.stream_id,
                "error": self.error,
            })
        except Exception as e:
            import traceback
            self.error = str(e)
            logger.error(f"[Whisper {self.stream_id}] Transcription error: {e}")
            logger.error(traceback.format_exc())
            await subtitle_events.broadcast(SubtitleEventType.ERROR, {
                "stream_id": self.stream_id,
                "error": self.error,
            })
        finally:
            self._running = False

    async def _extract_audio_chunk(self, offset: float, duration: float) -> Optional[bytes]:
        """Extract a chunk of audio as 16kHz mono PCM WAV bytes via FFmpeg."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', f'{offset:.3f}',
                '-i', self.video_path,
                '-t', f'{duration:.3f}',
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                'pipe:1',
            ]

            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await self._process.communicate()
            self._process = None

            if not stdout or len(stdout) < 100:
                # Empty or too small - likely past end of audio
                return None

            return stdout

        except Exception as e:
            logger.error(f"[Whisper {self.stream_id}] Audio extraction failed: {e}")
            self._process = None
            return None

    def _transcribe_chunk(self, model, audio_data: bytes) -> list:
        """Transcribe an audio chunk using faster-whisper. Runs in thread executor."""
        # faster-whisper can accept a file path, BinaryIO, or numpy array
        # We'll wrap the WAV bytes in a BytesIO
        audio_io = io.BytesIO(audio_data)

        segments, info = model.transcribe(
            audio_io,
            language=self.language if self.language else None,
            task=self.task,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            # Suppress non-speech tokens like music notes, etc.
            suppress_tokens=[-1] if self.suppress_nst else None,
        )

        # segments is a generator; consume it into a list
        return list(segments)

    def _cache_vtt(self):
        """Copy VTT file alongside the video for future reuse."""
        try:
            video = Path(self.video_path)
            suffix = f".{self.language}.translated.vtt" if self.task == "translate" else f".{self.language}.vtt"
            cache_path = video.with_suffix(suffix)
            shutil.copy2(str(self.vtt_path), str(cache_path))
            logger.info(f"[Whisper {self.stream_id}] Cached VTT to {cache_path}")
        except Exception as e:
            logger.warning(f"[Whisper {self.stream_id}] Failed to cache VTT: {e}")

    def stop(self):
        """Stop the subtitle generation stream."""
        logger.info(f"[Whisper {self.stream_id}] Stopping")
        self._running = False

        # Kill FFmpeg process if running
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
            try:
                self._process.kill()
            except Exception:
                pass
            self._process = None

        # Cancel async task
        if self._task:
            self._task.cancel()
            self._task = None

        # Clean up temp directory (but not cached VTTs)
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"[Whisper {self.stream_id}] Failed to clean up: {e}")
            self._temp_dir = None

        # Unregister stream
        if self.stream_id in _active_subtitle_streams:
            del _active_subtitle_streams[self.stream_id]


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
