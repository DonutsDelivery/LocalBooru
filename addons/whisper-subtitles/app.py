"""
Whisper Subtitles Sidecar — Standalone FastAPI app.

Real-time subtitle generation using faster-whisper.
Architecture: Video → FFmpeg (audio extraction) → faster-whisper → VTT cues → SSE events

Endpoints:
  GET  /health                                   → health check + capabilities
  POST /whisper/generate                         → start subtitle generation
  POST /whisper/stop                             → stop active generation
  GET  /whisper/vtt/{stream_id}/subtitles.vtt    → serve growing VTT file
  GET  /whisper/events/{stream_id}               → SSE progress/cue stream
"""

import asyncio
import atexit
import io
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("whisper-subtitles")

# ─── Model cache ──────────────────────────────────────────────────────────────

_cached_model = None
_cached_model_key = None  # (size, device, compute_type)


def _load_model(model_size: str, device: str, compute_type: str):
    """Load or reuse a cached faster-whisper model."""
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

    logger.info(f"Loading model: {model_size} on {actual_device} ({actual_compute})")
    model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute)
    logger.info("Model loaded successfully")

    _cached_model = model
    _cached_model_key = key
    return model


# ─── Stream registry ──────────────────────────────────────────────────────────

_active_streams: Dict[str, "WhisperSubtitleStream"] = {}


def _cleanup_on_exit():
    for stream in list(_active_streams.values()):
        stream.stop()


atexit.register(_cleanup_on_exit)


# ─── VTT time formatting ─────────────────────────────────────────────────────

def _format_vtt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# ─── Cached VTT lookup ───────────────────────────────────────────────────────

def _find_cached_vtt(video_path: str, language: str, task: str) -> Optional[Path]:
    video = Path(video_path)
    suffix = f".{language}.translated.vtt" if task == "translate" else f".{language}.vtt"
    vtt_path = video.with_suffix(suffix)
    if vtt_path.exists():
        return vtt_path
    return None


# ─── Subtitle stream ─────────────────────────────────────────────────────────

class WhisperSubtitleStream:
    """Real-time subtitle generation stream."""

    def __init__(
        self,
        stream_id: str,
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
        self.stream_id = stream_id
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

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._temp_dir: Optional[Path] = None
        self.vtt_path: Optional[Path] = None
        self.cached_vtt_path: Optional[Path] = None
        self.error: Optional[str] = None
        self.completed = False
        self._duration = 0.0
        self._process = None

        # Per-stream event queue for SSE
        self._event_queues: list[asyncio.Queue] = []

        _active_streams[self.stream_id] = self

    def _broadcast_event(self, event_type: str, data: dict):
        """Push an event to all connected SSE clients for this stream."""
        import json
        event = {"type": event_type, **data}
        for q in self._event_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        """Create and return a new SSE event queue for this stream."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._event_queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove an SSE event queue."""
        try:
            self._event_queues.remove(q)
        except ValueError:
            pass

    async def start(self) -> bool:
        if self._running:
            return False

        # Check for cached VTT
        cached = _find_cached_vtt(self.video_path, self.language, self.task)
        if cached:
            logger.info(f"[{self.stream_id}] Found cached VTT: {cached}")
            self.cached_vtt_path = cached
            self.vtt_path = cached
            self.completed = True
            self._broadcast_event("completed", {"stream_id": self.stream_id, "cached": True})
            return True

        self._running = True
        self._temp_dir = Path(tempfile.mkdtemp(prefix="whisper_"))
        self.vtt_path = self._temp_dir / "subtitles.vtt"

        with open(self.vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

        # Check for audio
        has_audio = await self._check_audio()
        if not has_audio:
            self.error = "No audio stream found in video"
            self._running = False
            return False

        await self._detect_duration()
        self._task = asyncio.create_task(self._transcription_loop())
        return True

    async def _check_audio(self) -> bool:
        try:
            result = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                self.video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            return bool(stdout.decode("utf-8").strip())
        except Exception:
            return True

    async def _detect_duration(self):
        try:
            result = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                self.video_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            dur = stdout.decode("utf-8").strip()
            if dur:
                self._duration = float(dur)
                logger.info(f"[{self.stream_id}] Duration: {self._duration:.1f}s")
        except Exception:
            pass

    async def _transcription_loop(self):
        try:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None, _load_model, self.model_size, self.device, self.compute_type
            )

            self._broadcast_event("started", {
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
                    break

                audio_data = await self._extract_audio_chunk(chunk_offset, actual_duration)
                if audio_data is None:
                    if chunk_idx == 0:
                        self.error = "Failed to extract audio"
                        self._broadcast_event("error", {
                            "stream_id": self.stream_id,
                            "error": self.error,
                        })
                    break

                if not self._running:
                    break

                segments = await loop.run_in_executor(
                    None, self._transcribe_chunk, model, audio_data
                )

                if not self._running:
                    break

                for segment in segments:
                    start_time = chunk_offset + segment.start
                    end_time = chunk_offset + segment.end
                    text = segment.text.strip()
                    if not text:
                        continue

                    cue = f"{_format_vtt_time(start_time)} --> {_format_vtt_time(end_time)}\n{text}\n\n"
                    with open(self.vtt_path, "a", encoding="utf-8") as f:
                        f.write(cue)

                    self._broadcast_event("cue", {
                        "stream_id": self.stream_id,
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                    })

                progress = (chunk_end / self._duration * 100) if self._duration > 0 else 0
                self._broadcast_event("progress", {
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
                if self.cache_subtitles and self.start_position == 0 and self.vtt_path and self.vtt_path.exists():
                    self._cache_vtt()

                self._broadcast_event("completed", {
                    "stream_id": self.stream_id,
                    "chunks_processed": chunk_idx,
                    "cached": self.cache_subtitles,
                })

        except asyncio.CancelledError:
            logger.info(f"[{self.stream_id}] Cancelled")
        except ImportError as e:
            self.error = f"faster-whisper not installed: {e}"
            logger.error(f"[{self.stream_id}] {self.error}")
            self._broadcast_event("error", {"stream_id": self.stream_id, "error": self.error})
        except Exception as e:
            self.error = str(e)
            logger.error(f"[{self.stream_id}] Error: {e}", exc_info=True)
            self._broadcast_event("error", {"stream_id": self.stream_id, "error": self.error})
        finally:
            self._running = False

    async def _extract_audio_chunk(self, offset: float, duration: float) -> Optional[bytes]:
        try:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{offset:.3f}",
                "-i", self.video_path,
                "-t", f"{duration:.3f}",
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-f", "wav", "pipe:1",
            ]
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await self._process.communicate()
            self._process = None

            if not stdout or len(stdout) < 100:
                return None
            return stdout
        except Exception as e:
            logger.error(f"[{self.stream_id}] Audio extraction failed: {e}")
            self._process = None
            return None

    def _transcribe_chunk(self, model, audio_data: bytes) -> list:
        audio_io = io.BytesIO(audio_data)
        segments, info = model.transcribe(
            audio_io,
            language=self.language if self.language else None,
            task=self.task,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            suppress_tokens=[-1] if self.suppress_nst else None,
        )
        return list(segments)

    def _cache_vtt(self):
        try:
            video = Path(self.video_path)
            suffix = f".{self.language}.translated.vtt" if self.task == "translate" else f".{self.language}.vtt"
            cache_path = video.with_suffix(suffix)
            shutil.copy2(str(self.vtt_path), str(cache_path))
            logger.info(f"[{self.stream_id}] Cached VTT to {cache_path}")
        except Exception as e:
            logger.warning(f"[{self.stream_id}] Failed to cache VTT: {e}")

    def stop(self):
        self._running = False
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

        if self._task:
            self._task.cancel()
            self._task = None

        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None

        # Signal completion to any connected SSE clients
        self._broadcast_event("stopped", {"stream_id": self.stream_id})

        _active_streams.pop(self.stream_id, None)


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Whisper Subtitles Sidecar")


@app.get("/health")
async def health():
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    return {
        "status": "ok",
        "cuda_available": cuda_available,
        "model_loaded": _cached_model is not None,
    }


class GenerateRequest(BaseModel):
    file_path: str
    stream_id: Optional[str] = None
    image_id: Optional[int] = None
    language: Optional[str] = None
    task: Optional[str] = None
    start_position: Optional[float] = None
    config: Optional[dict] = None


@app.post("/whisper/generate")
async def generate(req: GenerateRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    stream_id = req.stream_id or str(uuid.uuid4())

    # Extract settings from config or use defaults
    cfg = req.config or {}
    language = req.language or cfg.get("language", "ja")
    task = req.task or cfg.get("task", "translate")
    model_size = cfg.get("model_size", "medium")
    beam_size = cfg.get("beam_size", 8)
    device = cfg.get("device", "auto")
    compute_type = cfg.get("compute_type", "auto")
    vad_filter = cfg.get("vad_filter", True)
    suppress_nst = cfg.get("suppress_nst", True)
    cache_subtitles = cfg.get("cache_subtitles", True)
    chunk_duration = cfg.get("chunk_duration", 30)
    start_pos = req.start_position or cfg.get("start_position", 0.0)

    stream = WhisperSubtitleStream(
        stream_id=stream_id,
        video_path=req.file_path,
        model_size=model_size,
        language=language,
        task=task,
        chunk_duration=chunk_duration,
        beam_size=beam_size,
        device=device,
        compute_type=compute_type,
        vad_filter=vad_filter,
        suppress_nst=suppress_nst,
        cache_subtitles=cache_subtitles,
        start_position=start_pos,
    )

    started = await stream.start()
    if not started:
        raise HTTPException(status_code=500, detail=stream.error or "Failed to start")

    return {
        "stream_id": stream_id,
        "cached": stream.cached_vtt_path is not None,
        "completed": stream.completed,
    }


@app.post("/whisper/stop")
async def stop_generation():
    for stream in list(_active_streams.values()):
        stream.stop()
    return {"success": True, "message": "All streams stopped"}


@app.get("/whisper/vtt/{stream_id}/subtitles.vtt")
async def serve_vtt(stream_id: str):
    stream = _active_streams.get(stream_id)
    if not stream or not stream.vtt_path or not stream.vtt_path.exists():
        raise HTTPException(status_code=404, detail="VTT not found")

    return FileResponse(
        str(stream.vtt_path),
        media_type="text/vtt",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/whisper/events/{stream_id}")
async def events(stream_id: str):
    stream = _active_streams.get(stream_id)
    if not stream:
        raise HTTPException(status_code=404, detail="Stream not found")

    queue = stream.subscribe()

    async def event_generator():
        import json
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("completed", "error", "stopped"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            stream.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
