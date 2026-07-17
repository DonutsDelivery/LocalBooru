import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

from fmp4_stream import find_fragment_decode_time, iter_fragmented_mp4, parse_video_track
from processing_session import GenerationConflict, ProcessingSession, SegmentRejected
from session_protocol import InitSegmentDescriptor, SegmentDescriptor, SessionState


ScriptBuilder = Callable[..., str]
EnvironmentProvider = Callable[[], dict]


class Fmp4SessionProcessor:
    def __init__(
        self,
        session: ProcessingSession,
        script_builder: ScriptBuilder,
        environment_provider: EnvironmentProvider,
        ffmpeg_executable: str = "ffmpeg",
        vspipe_executable: str = "vspipe",
    ):
        self.session = session
        self.script_builder = script_builder
        self.environment_provider = environment_provider
        self.ffmpeg_executable = ffmpeg_executable
        self.vspipe_executable = vspipe_executable
        self._stop_event = threading.Event()
        self._process_lock = threading.Lock()
        self._processes: list[subprocess.Popen] = []
        self._thread = threading.Thread(target=self._run, name=f"svp-fmp4-{session.session_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._terminate_processes()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        generation = self.session.generation
        try:
            self._produce(generation)
            if not self._stop_event.is_set() and self.session.generation == generation:
                self.session.end(generation)
        except (GenerationConflict, SegmentRejected):
            return
        except Exception as error:
            if not self._stop_event.is_set() and self.session.generation == generation:
                try:
                    self.session.fail(generation, "fmp4_pipeline_failed", str(error))
                except (GenerationConflict, SegmentRejected):
                    pass
        finally:
            self._terminate_processes()

    def _produce(self, generation: int) -> None:
        metadata = self.session.metadata
        start_position = self.session.start_position
        remaining_duration = metadata.source_duration - start_position
        if remaining_duration <= 0:
            return
        source_rate = Fraction(metadata.source_fps).limit_denominator(100_000)
        source_frames = max(1, math.ceil(remaining_duration * float(source_rate)))
        target_fps = max(1, round(metadata.output_fps))

        with tempfile.TemporaryDirectory(prefix="localbooru-fmp4-") as temp_dir:
            script_path = Path(temp_dir) / "processor.vpy"
            script_path.write_text(
                self.script_builder(
                    metadata.width,
                    metadata.height,
                    source_rate.numerator,
                    source_rate.denominator,
                    source_frames,
                    target_fps,
                )
            )
            stderr_files = [tempfile.TemporaryFile() for _ in range(3)]
            try:
                decode = subprocess.Popen(
                    self._decode_command(start_position),
                    stdout=subprocess.PIPE,
                    stderr=stderr_files[0],
                    env=self.environment_provider(),
                )
                vspipe = subprocess.Popen(
                    [self.vspipe_executable, "--requests", "1", "-c", "y4m", str(script_path), "-"],
                    stdin=decode.stdout,
                    stdout=subprocess.PIPE,
                    stderr=stderr_files[1],
                    env=self.environment_provider(),
                )
                if decode.stdout is not None:
                    decode.stdout.close()
                encoder = subprocess.Popen(
                    self._encode_command(start_position, remaining_duration, target_fps),
                    stdin=vspipe.stdout,
                    stdout=subprocess.PIPE,
                    stderr=stderr_files[2],
                    env=self.environment_provider(),
                )
                if vspipe.stdout is not None:
                    vspipe.stdout.close()
                with self._process_lock:
                    self._processes = [decode, vspipe, encoder]
                if encoder.stdout is None:
                    raise RuntimeError("FFmpeg encoder stdout is unavailable")
                with encoder.stdout:
                    self._publish_stream(encoder.stdout, generation)
                return_codes = [process.wait(timeout=10) for process in (decode, vspipe, encoder)]
                if any(code != 0 for code in return_codes):
                    details = self._stderr_summary(stderr_files)
                    raise RuntimeError(f"processing pipeline exited with {return_codes}: {details}")
            finally:
                for stderr_file in stderr_files:
                    stderr_file.close()

    def _publish_stream(self, stream, generation: int) -> None:
        track = None
        pending: Optional[tuple[bytes, float]] = None
        sequence = 0
        for part in iter_fragmented_mp4(stream):
            self._require_current(generation)
            if part.kind == "init":
                track = parse_video_track(part.data)
                self._wait_for_capacity(generation, len(part.data), 0.0)
                filename = f"{generation}-init-{uuid.uuid4().hex}.mp4"
                self._write_atomic(filename, part.data)
                self.session.publish_init(
                    InitSegmentDescriptor(
                        generation=generation,
                        byte_length=len(part.data),
                        filename=filename,
                    )
                )
                continue
            if track is None:
                raise RuntimeError("media fragment arrived before initialization")
            decode_time = find_fragment_decode_time(part.data, track.track_id)
            source_start = self.session.start_position + decode_time / track.timescale
            if pending is not None:
                previous_data, previous_start = pending
                sequence = self._publish_media(
                    generation,
                    sequence,
                    previous_start,
                    source_start - previous_start,
                    previous_data,
                )
            pending = (part.data, source_start)

        if pending is not None:
            data, source_start = pending
            duration = self.session.metadata.source_duration - source_start
            if duration > 0.001:
                self._publish_media(generation, sequence, source_start, duration, data)

    def _publish_media(
        self,
        generation: int,
        sequence: int,
        source_start: float,
        duration: float,
        data: bytes,
    ) -> int:
        if duration <= 0:
            raise RuntimeError("fragment duration is not positive")
        self._wait_for_capacity(generation, len(data), duration)
        filename = f"{generation}-{sequence}-{uuid.uuid4().hex}.m4s"
        self._write_atomic(filename, data)
        accepted = self.session.publish_segment(
            SegmentDescriptor(
                generation=generation,
                sequence=sequence,
                source_start=source_start,
                duration=duration,
                byte_length=len(data),
                filename=filename,
                independent=True,
                av_drift_ms=0.0,
            )
        )
        if not accepted:
            raise RuntimeError("fragment exceeded the configured buffer limit")
        return sequence + 1

    def _wait_for_capacity(self, generation: int, byte_length: int, duration: float) -> None:
        while not self._stop_event.is_set():
            self._require_current(generation)
            if self.session.state == SessionState.RUNNING and self.session.has_capacity(byte_length, duration):
                return
            time.sleep(0.02)
        raise GenerationConflict("processing generation was stopped")

    def _require_current(self, generation: int) -> None:
        if self._stop_event.is_set() or self.session.generation != generation:
            raise GenerationConflict("processing generation is no longer current")

    def _write_atomic(self, filename: str, data: bytes) -> None:
        temporary = self.session.output_dir / f".{filename}.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, self.session.output_dir / filename)

    def _decode_command(self, start_position: float) -> list[str]:
        command = [self.ffmpeg_executable, "-hide_banner", "-loglevel", "error", "-threads", "0"]
        if start_position > 0:
            command.extend(["-ss", str(start_position)])
        command.extend(
            [
                "-i",
                self.session.file_path,
                "-an",
                "-sn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ]
        )
        return command

    def _encode_command(self, start_position: float, duration: float, target_fps: int) -> list[str]:
        command = [
            self.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "yuv4mpegpipe",
            "-i",
            "-",
        ]
        if start_position > 0:
            command.extend(["-ss", str(start_position)])
        command.extend(
            [
                "-i",
                self.session.file_path,
                "-map",
                "0:v",
                "-map",
                "1:a?",
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-profile:v",
                "high",
                "-level:v",
                "5.2",
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(target_fps),
                "-keyint_min",
                str(target_fps),
                "-sc_threshold",
                "0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "frag_keyframe+empty_moov+default_base_moof",
                "-f",
                "mp4",
                "-",
            ]
        )
        return command

    def _terminate_processes(self) -> None:
        with self._process_lock:
            processes = self._processes
            self._processes = []
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    @staticmethod
    def _stderr_summary(files) -> str:
        messages = []
        for stderr_file in files:
            stderr_file.seek(0)
            text = stderr_file.read().decode("utf-8", errors="replace").strip()
            if text:
                messages.append(text[-2000:])
        return " | ".join(messages)
