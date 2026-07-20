import shutil
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Union

from session_protocol import (
    InitSegmentDescriptor,
    OpenSessionRequest,
    SegmentDescriptor,
    SessionEvent,
    SessionMetadata,
    SessionState,
)


class GenerationConflict(RuntimeError):
    pass


class SegmentRejected(RuntimeError):
    pass


class EventHistoryExpired(RuntimeError):
    pass


StoredDescriptor = Union[InitSegmentDescriptor, SegmentDescriptor]


class BoundedSegmentStore:
    def __init__(self, output_dir: Path, max_bytes: int, max_duration: float):
        self.output_dir = output_dir
        self.max_bytes = max_bytes
        self.max_duration = max_duration
        self._items: Deque[StoredDescriptor] = deque()
        self._bytes = 0
        self._duration = 0.0

    @property
    def buffered_bytes(self) -> int:
        return self._bytes

    @property
    def buffered_duration(self) -> float:
        return self._duration

    def owns_filename(self, filename: str) -> bool:
        return any(item.filename == filename for item in self._items)

    def can_accept(self, byte_length: int, duration: float) -> bool:
        return self._bytes + byte_length <= self.max_bytes and self._duration + duration <= self.max_duration

    def try_add(self, item: StoredDescriptor) -> bool:
        duration = item.duration if isinstance(item, SegmentDescriptor) else 0.0
        if not self.can_accept(item.byte_length, duration):
            self.unlink(item.filename)
            return False
        self._items.append(item)
        self._bytes += item.byte_length
        self._duration += duration
        return True

    def acknowledge_segment(self, generation: int, sequence: int) -> Optional[SegmentDescriptor]:
        for item in self._items:
            if isinstance(item, SegmentDescriptor) and item.generation == generation and item.sequence == sequence:
                self._remove(item)
                return item
        return None

    def acknowledge_init(self, generation: int) -> bool:
        for item in self._items:
            if isinstance(item, InitSegmentDescriptor) and item.generation == generation:
                self._remove(item)
                return True
        return False

    def clear(self) -> None:
        while self._items:
            self.unlink(self._items.popleft().filename)
        self._bytes = 0
        self._duration = 0.0

    def discard(self) -> None:
        self._items.clear()
        self._bytes = 0
        self._duration = 0.0

    def unlink(self, filename: str) -> None:
        try:
            (self.output_dir / filename).unlink()
        except FileNotFoundError:
            pass

    def _remove(self, item: StoredDescriptor) -> None:
        self._items.remove(item)
        self._bytes -= item.byte_length
        if isinstance(item, SegmentDescriptor):
            self._duration -= item.duration
        self.unlink(item.filename)


class ProcessingSession:
    def __init__(
        self,
        request: OpenSessionRequest,
        metadata: SessionMetadata,
        output_dir: Path,
        session_id: Optional[str] = None,
        event_history_limit: int = 512,
    ):
        if abs(metadata.initial_source_position - request.start_position) > 0.001:
            raise ValueError("metadata and request start positions differ")
        if event_history_limit <= 0:
            raise ValueError("event history limit must be positive")
        self.session_id = session_id or uuid.uuid4().hex
        self.file_path = request.file_path
        self.graph = request.graph
        self.metadata = metadata
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.generation = request.generation
        self.start_position = request.start_position
        self.state = SessionState.RUNNING
        self._segments = BoundedSegmentStore(
            output_dir,
            request.max_buffer_bytes,
            request.max_buffer_duration,
        )
        self._last_sequence = -1
        self._last_source_end = request.start_position
        self._last_acknowledged_end = request.start_position
        self._init_published = False
        self._events: Deque[SessionEvent] = deque(maxlen=event_history_limit)
        self._event_sequence = 0
        self._lock = threading.RLock()
        self._emit("metadata", metadata=metadata)

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return self._segments.buffered_bytes

    @property
    def buffered_duration(self) -> float:
        with self._lock:
            return self._segments.buffered_duration

    @property
    def latest_event_sequence(self) -> int:
        with self._lock:
            return self._event_sequence - 1

    def read_owned_file(self, generation: int, filename: str) -> bytes:
        with self._lock:
            self._require_generation(generation)
            if not self._segments.owns_filename(filename):
                raise SegmentRejected("segment is not retained by this session")
            return (self.output_dir / filename).read_bytes()

    def publish_init(self, descriptor: InitSegmentDescriptor) -> bool:
        with self._lock:
            try:
                self._require_generation(descriptor.generation)
                self._require_publishable()
                if self._init_published:
                    raise SegmentRejected("initialization segment is already published")
                self._validate_file(descriptor)
                if not self._segments.try_add(descriptor):
                    self._emit_buffer_state()
                    return False
                self._init_published = True
                self._emit("init_ready", init_segment=descriptor)
                self._emit_buffer_state()
                return True
            except (GenerationConflict, SegmentRejected):
                self._discard_rejected(descriptor.filename)
                raise

    def publish_segment(self, segment: SegmentDescriptor) -> bool:
        with self._lock:
            try:
                self._require_generation(segment.generation)
                self._require_publishable()
                if not self._init_published:
                    raise SegmentRejected("initialization segment is not published")
                if segment.sequence != self._last_sequence + 1:
                    raise SegmentRejected("segment sequence is not contiguous")
                if abs(segment.source_start - self._last_source_end) > 0.001:
                    raise SegmentRejected("segment timestamp is not contiguous")
                if abs(segment.av_drift_ms) > self.metadata.max_av_drift_ms:
                    raise SegmentRejected("segment exceeds A/V drift tolerance")
                if segment.source_start + segment.duration > self.metadata.source_duration + 0.001:
                    raise SegmentRejected("segment exceeds source duration")
                self._validate_file(segment)
                if not self._segments.try_add(segment):
                    self._emit_buffer_state()
                    return False
                self._last_sequence = segment.sequence
                self._last_source_end = segment.source_start + segment.duration
                self._emit("segment_ready", segment=segment)
                self._emit_buffer_state()
                return True
            except (GenerationConflict, SegmentRejected):
                self._discard_rejected(segment.filename)
                raise

    def acknowledge(self, generation: int, sequence: int) -> bool:
        with self._lock:
            self._require_generation(generation)
            acknowledged = self._segments.acknowledge_segment(generation, sequence)
            if acknowledged is not None:
                self._last_acknowledged_end = max(
                    self._last_acknowledged_end,
                    acknowledged.source_start + acknowledged.duration,
                )
                self._emit_buffer_state()
            return acknowledged is not None

    def acknowledge_init(self, generation: int) -> bool:
        with self._lock:
            self._require_generation(generation)
            removed = self._segments.acknowledge_init(generation)
            if removed:
                self._emit_buffer_state()
            return removed

    def production_cursor(self) -> Optional[tuple[int, int, float]]:
        with self._lock:
            if self.state != SessionState.RUNNING or not self._init_published:
                return None
            return self.generation, self._last_sequence + 1, self._last_source_end

    def has_capacity(self, byte_length: int, duration: float) -> bool:
        with self._lock:
            return self._segments.can_accept(byte_length, duration)

    def seek(self, expected_generation: int, generation: int, position: float) -> None:
        with self._lock:
            self._require_generation(expected_generation)
            self._require_active()
            if not self.metadata.seekable:
                raise SegmentRejected("session is not seekable")
            if generation <= self.generation:
                raise GenerationConflict("seek generation must increase")
            if position < 0 or position > self.metadata.source_duration:
                raise ValueError("seek position is outside the source duration")
            was_paused = self.state == SessionState.PAUSED
            self._clear_output()
            self.generation = generation
            self.start_position = position
            self._last_sequence = -1
            self._last_source_end = position
            self._last_acknowledged_end = position
            self._init_published = False
            self.state = SessionState.PAUSED if was_paused else SessionState.RUNNING
            self._emit_buffer_state()

    def pause(self, expected_generation: int) -> None:
        with self._lock:
            self._require_generation(expected_generation)
            if self.state == SessionState.PAUSED:
                return
            if self.state != SessionState.RUNNING:
                raise SegmentRejected(f"session is {self.state.value}")
            self.state = SessionState.PAUSED
            self._emit("paused")

    def resume(self, expected_generation: int) -> None:
        with self._lock:
            self._require_generation(expected_generation)
            if self.state == SessionState.RUNNING:
                return
            if self.state != SessionState.PAUSED:
                raise SegmentRejected(f"session is {self.state.value}")
            self.state = SessionState.RUNNING
            self._emit("resumed")

    def end(self, expected_generation: int) -> None:
        with self._lock:
            self._require_generation(expected_generation)
            if self.state == SessionState.ENDED:
                return
            self._require_active()
            self.state = SessionState.ENDED
            self._emit("ended")

    def fail(self, expected_generation: int, code: str, message: str) -> None:
        with self._lock:
            self._require_generation(expected_generation)
            if self.state == SessionState.FAILED:
                return
            self._require_active()
            self.state = SessionState.FAILED
            self._emit(
                "terminal_error",
                error_code=code,
                message=message,
                last_safe_position=self._last_acknowledged_end,
            )
            self._clear_output()

    def stop(self, expected_generation: int) -> None:
        with self._lock:
            self._require_generation(expected_generation)
            self._stop_locked()

    def stop_current(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self.state == SessionState.STOPPED:
            return
        self.state = SessionState.STOPPED
        self._segments.discard()
        self._emit("stopped")
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def events_after(self, event_sequence: int) -> tuple[SessionEvent, ...]:
        with self._lock:
            if self._events and event_sequence < self._events[0].event_sequence - 1:
                raise EventHistoryExpired("event cursor is older than retained history")
            return tuple(event for event in self._events if event.event_sequence > event_sequence)

    def _require_generation(self, generation: int) -> None:
        if generation != self.generation:
            raise GenerationConflict(
                f"generation {generation} does not own session generation {self.generation}"
            )

    def _require_active(self) -> None:
        if self.state not in {SessionState.RUNNING, SessionState.PAUSED}:
            raise SegmentRejected(f"session is {self.state.value}")

    def _require_publishable(self) -> None:
        if self.state != SessionState.RUNNING:
            raise SegmentRejected(f"session is {self.state.value}")

    def _validate_file(self, descriptor: StoredDescriptor) -> None:
        if self._segments.owns_filename(descriptor.filename):
            raise SegmentRejected("segment filename is already owned")
        path = self.output_dir / descriptor.filename
        if path.is_symlink() or not path.is_file():
            raise SegmentRejected("segment file is missing or invalid")
        if path.stat().st_size != descriptor.byte_length:
            raise SegmentRejected("segment byte length does not match file size")

    def _clear_output(self) -> None:
        self._segments.clear()
        if self.output_dir.exists():
            for path in self.output_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()

    def _discard_rejected(self, filename: str) -> None:
        if not self._segments.owns_filename(filename):
            self._segments.unlink(filename)

    def _emit_buffer_state(self) -> None:
        self._emit(
            "buffer_state",
            buffered_bytes=self._segments.buffered_bytes,
            buffered_duration=self._segments.buffered_duration,
        )

    def _emit(self, event_type: str, **values) -> None:
        event = SessionEvent(
            event_sequence=self._event_sequence,
            session_id=self.session_id,
            generation=self.generation,
            type=event_type,
            **values,
        )
        self._event_sequence += 1
        self._events.append(event)


class SyntheticSegmentProducer:
    def __init__(self, session: ProcessingSession, segment_duration: float = 1.0, byte_length: int = 1024):
        if segment_duration <= 0 or byte_length <= 0:
            raise ValueError("fake segment dimensions must be positive")
        self.session = session
        self.segment_duration = segment_duration
        self.byte_length = byte_length
        self._producer_id = uuid.uuid4().hex
        self._lock = threading.Lock()

    def produce_init(self) -> bool:
        with self._lock:
            generation = self.session.generation
            filename = f"{generation}-init-{self._producer_id}.mp4"
            payload = b"synthetic-init"
            if not self.session.has_capacity(len(payload), 0.0):
                return False
            (self.session.output_dir / filename).write_bytes(payload)
            return self.session.publish_init(
                InitSegmentDescriptor(
                    generation=generation,
                    byte_length=len(payload),
                    filename=filename,
                )
            )

    def produce_one(self) -> bool:
        with self._lock:
            cursor = self.session.production_cursor()
            if cursor is None:
                return False
            generation, sequence, start = cursor
            remaining = self.session.metadata.source_duration - start
            if remaining <= 0:
                self.session.end(generation)
                return False
            duration = min(self.segment_duration, remaining)
            if not self.session.has_capacity(self.byte_length, duration):
                return False
            filename = f"{generation}-{sequence}-{self._producer_id}.m4s"
            payload = bytes([sequence % 251]) * self.byte_length
            (self.session.output_dir / filename).write_bytes(payload)
            segment = SegmentDescriptor(
                generation=generation,
                sequence=sequence,
                source_start=start,
                duration=duration,
                byte_length=len(payload),
                filename=filename,
                independent=True,
                av_drift_ms=0.0,
            )
            return self.session.publish_segment(segment)


class ProcessingSessionRegistry:
    def __init__(self, root_dir: Path, event_history_limit: int = 512):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.event_history_limit = event_history_limit
        self._sessions: Dict[str, ProcessingSession] = {}
        self._lock = threading.RLock()

    def open(self, request: OpenSessionRequest, metadata: SessionMetadata) -> ProcessingSession:
        with self._lock:
            session_id = uuid.uuid4().hex
            session = ProcessingSession(
                request,
                metadata,
                self.root_dir / session_id,
                session_id=session_id,
                event_history_limit=self.event_history_limit,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Optional[ProcessingSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def stop(self, session_id: str, expected_generation: int) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        session.stop(expected_generation)
        with self._lock:
            self._sessions.pop(session_id, None)
        return True

    def stop_all(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.stop_current()
