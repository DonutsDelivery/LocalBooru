from pathlib import Path
import threading
from typing import Callable, Optional, Protocol

from fastapi import APIRouter, HTTPException, Query, Response

from processing_session import (
    EventHistoryExpired,
    GenerationConflict,
    ProcessingSession,
    ProcessingSessionRegistry,
    SegmentRejected,
)
from session_protocol import (
    EventBatchResponse,
    GenerationCommand,
    OpenSessionRequest,
    OpenSessionResponse,
    PROTOCOL_VERSION,
    SeekSessionRequest,
    SegmentAcknowledgement,
    SessionMetadata,
    SessionStatusResponse,
)


class ProcessorUnavailable(RuntimeError):
    pass


class SessionProcessor(Protocol):
    def stop(self) -> None:
        ...


MetadataProvider = Callable[[OpenSessionRequest], SessionMetadata]
SessionStarted = Callable[[ProcessingSession], Optional[SessionProcessor]]


class ProcessingSessionService:
    def __init__(
        self,
        root_dir: Path,
        metadata_provider: Optional[MetadataProvider] = None,
        session_started: Optional[SessionStarted] = None,
    ):
        self.registry = ProcessingSessionRegistry(root_dir)
        self.metadata_provider = metadata_provider
        self.session_started = session_started
        self._processors: dict[str, SessionProcessor] = {}
        self._lock = threading.RLock()
        self._stop_generation = 0

    def open(self, request: OpenSessionRequest) -> ProcessingSession:
        if self.metadata_provider is None:
            raise ProcessorUnavailable("bounded fMP4 processing is not available")
        with self._lock:
            stop_generation = self._stop_generation
        metadata = self.metadata_provider(request)
        with self._lock:
            if stop_generation != self._stop_generation:
                raise ProcessorUnavailable("processing session open was superseded")
            session = self.registry.open(request, metadata)
            try:
                if self.session_started is not None:
                    processor = self.session_started(session)
                    if processor is not None:
                        self._processors[session.session_id] = processor
            except Exception:
                self.registry.stop(session.session_id, session.generation)
                raise
            return session

    def get(self, session_id: str) -> ProcessingSession:
        session = self.registry.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def pause(self, session: ProcessingSession, generation: int) -> None:
        session.pause(generation)

    def resume(self, session: ProcessingSession, generation: int) -> None:
        session.resume(generation)

    def seek(self, session: ProcessingSession, request: SeekSessionRequest) -> None:
        with self._lock:
            session.seek(request.expected_generation, request.generation, request.position)
            processor = self._processors.pop(session.session_id, None)
            if processor is not None:
                processor.stop()
            try:
                if self.session_started is not None:
                    replacement = self.session_started(session)
                    if replacement is not None:
                        self._processors[session.session_id] = replacement
            except Exception as error:
                session.fail(request.generation, "processor_start_failed", str(error))
                raise

    def stop(self, session: ProcessingSession, expected_generation: int) -> None:
        with self._lock:
            self.registry.stop(session.session_id, expected_generation)
            processor = self._processors.pop(session.session_id, None)
            if processor is not None:
                processor.stop()

    def stop_all(self) -> None:
        with self._lock:
            self._stop_generation += 1
            processors = tuple(self._processors.values())
            self._processors.clear()
            self.registry.stop_all()
        for processor in processors:
            processor.stop()

    @staticmethod
    def status(session: ProcessingSession) -> SessionStatusResponse:
        return SessionStatusResponse(
            session_id=session.session_id,
            generation=session.generation,
            state=session.state,
            buffered_bytes=session.buffered_bytes,
            buffered_duration=session.buffered_duration,
            latest_event_sequence=session.latest_event_sequence,
        )


def create_session_router(service: ProcessingSessionService) -> APIRouter:
    router = APIRouter(prefix="/svp/sessions", tags=["svp-sessions"])

    def session_or_404(session_id: str) -> ProcessingSession:
        try:
            return service.get(session_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": "processing session not found"},
            ) from error

    def run(command):
        try:
            return command()
        except GenerationConflict as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "generation_conflict", "message": str(error)},
            ) from error
        except SegmentRejected as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "invalid_state", "message": str(error)},
            ) from error
        except EventHistoryExpired as error:
            raise HTTPException(
                status_code=410,
                detail={"code": "cursor_expired", "message": str(error)},
            ) from error

    @router.post("", response_model=OpenSessionResponse)
    def open_session(request: OpenSessionRequest):
        try:
            session = service.open(request)
        except ProcessorUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "processor_unavailable", "message": str(error)},
            ) from error
        return OpenSessionResponse(
            session_id=session.session_id,
            generation=session.generation,
            state=session.state,
        )

    @router.get("/{session_id}", response_model=SessionStatusResponse)
    def get_session(session_id: str):
        return service.status(session_or_404(session_id))

    @router.get("/{session_id}/events", response_model=EventBatchResponse)
    def get_events(
        session_id: str,
        after: int = Query(default=-1, ge=-1),
        limit: int = Query(default=128, ge=1, le=256),
    ):
        session = session_or_404(session_id)
        events = run(lambda: session.events_after(after))[:limit]
        next_cursor = events[-1].event_sequence if events else after
        return EventBatchResponse(events=list(events), next_cursor=next_cursor)

    @router.post("/{session_id}/pause", response_model=SessionStatusResponse)
    def pause_session(session_id: str, command: GenerationCommand):
        session = session_or_404(session_id)
        run(lambda: service.pause(session, command.generation))
        return service.status(session)

    @router.post("/{session_id}/resume", response_model=SessionStatusResponse)
    def resume_session(session_id: str, command: GenerationCommand):
        session = session_or_404(session_id)
        run(lambda: service.resume(session, command.generation))
        return service.status(session)

    @router.post("/{session_id}/seek", response_model=SessionStatusResponse)
    def seek_session(session_id: str, request: SeekSessionRequest):
        session = session_or_404(session_id)
        run(lambda: service.seek(session, request))
        return service.status(session)

    @router.get("/{session_id}/segments/{generation}/{filename}")
    def get_segment(session_id: str, generation: int, filename: str):
        session = session_or_404(session_id)
        content = run(lambda: session.read_owned_file(generation, filename))
        return Response(content=content, media_type="video/mp4")

    @router.delete("/{session_id}/segments/init", response_model=SessionStatusResponse)
    def acknowledge_init(session_id: str, command: GenerationCommand):
        session = session_or_404(session_id)
        run(lambda: session.acknowledge_init(command.generation))
        return service.status(session)

    @router.delete("/{session_id}/segments", response_model=SessionStatusResponse)
    def acknowledge_segment(session_id: str, command: SegmentAcknowledgement):
        session = session_or_404(session_id)
        run(lambda: session.acknowledge(command.generation, command.sequence))
        return service.status(session)

    @router.delete("/{session_id}", response_model=SessionStatusResponse)
    def stop_session(
        session_id: str,
        expected_generation: int = Query(ge=1),
        protocol_version: int = Query(default=PROTOCOL_VERSION),
    ):
        if protocol_version != PROTOCOL_VERSION:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_protocol_version", "message": "unsupported protocol version"},
            )
        session = session_or_404(session_id)
        run(lambda: service.stop(session, expected_generation))
        return service.status(session)

    return router
