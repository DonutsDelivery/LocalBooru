from enum import Enum
from pathlib import PurePath
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROTOCOL_VERSION = 1


class ProtocolModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class SessionState(str, Enum):
    OPENING = "opening"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"
    FAILED = "failed"
    STOPPED = "stopped"


class SessionGraph(ProtocolModel):
    kind: Literal["manager_snapshot"]
    revision: int = Field(ge=1)
    snapshot_path: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OpenSessionRequest(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION]
    generation: int = Field(ge=1)
    file_path: str = Field(min_length=1)
    start_position: float = Field(default=0.0, ge=0.0)
    graph: SessionGraph
    max_buffer_bytes: int = Field(default=128 * 1024 * 1024, gt=0)
    max_buffer_duration: float = Field(default=30.0, gt=0.0)


class OpenSessionResponse(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    session_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    state: SessionState


class GenerationCommand(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION]
    generation: int = Field(ge=1)


class SegmentAcknowledgement(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION]
    generation: int = Field(ge=1)
    sequence: int = Field(ge=0)


class SessionStatusResponse(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    session_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    state: SessionState
    buffered_bytes: int = Field(ge=0)
    buffered_duration: float = Field(ge=0.0)
    latest_event_sequence: int = Field(ge=0)


class EventBatchResponse(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    events: list["SessionEvent"]
    next_cursor: int = Field(ge=-1)


class SeekSessionRequest(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION]
    expected_generation: int = Field(ge=1)
    generation: int = Field(ge=1)
    position: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_generation(self):
        if self.generation <= self.expected_generation:
            raise ValueError("seek generation must increase")
        return self


class SessionMetadata(ProtocolModel):
    source_duration: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_fps: float = Field(gt=0.0)
    output_fps: float = Field(gt=0.0)
    mime_type: str = Field(min_length=1)
    initial_source_position: float = Field(ge=0.0)
    max_av_drift_ms: float = Field(default=50.0, ge=0.0)
    seekable: bool = True

    @model_validator(mode="after")
    def validate_initial_position(self):
        if self.initial_source_position > self.source_duration:
            raise ValueError("initial position exceeds source duration")
        return self


def _validate_filename(value: str) -> str:
    path = PurePath(value)
    if path.name != value or value in {".", ".."}:
        raise ValueError("segment filename must be a basename")
    return value


class InitSegmentDescriptor(ProtocolModel):
    generation: int = Field(ge=1)
    byte_length: int = Field(gt=0)
    filename: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _validate_filename(value)


class SegmentDescriptor(ProtocolModel):
    generation: int = Field(ge=1)
    sequence: int = Field(ge=0)
    source_start: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    byte_length: int = Field(gt=0)
    filename: str = Field(min_length=1)
    independent: bool
    av_drift_ms: float = 0.0

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _validate_filename(value)


class SessionEvent(ProtocolModel):
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    event_sequence: int = Field(ge=0)
    session_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    type: Literal[
        "metadata",
        "init_ready",
        "segment_ready",
        "buffer_state",
        "paused",
        "resumed",
        "ended",
        "terminal_error",
        "stopped",
    ]
    metadata: Optional[SessionMetadata] = None
    init_segment: Optional[InitSegmentDescriptor] = None
    segment: Optional[SegmentDescriptor] = None
    buffered_bytes: Optional[int] = Field(default=None, ge=0)
    buffered_duration: Optional[float] = Field(default=None, ge=0.0)
    error_code: Optional[str] = None
    message: Optional[str] = None
    last_safe_position: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_payload(self):
        required = {
            "metadata": ("metadata",),
            "init_ready": ("init_segment",),
            "segment_ready": ("segment",),
            "buffer_state": ("buffered_bytes", "buffered_duration"),
            "terminal_error": ("error_code", "message"),
        }
        payload_fields = {
            "metadata",
            "init_segment",
            "segment",
            "buffered_bytes",
            "buffered_duration",
            "error_code",
            "message",
            "last_safe_position",
        }
        required_fields = set(required.get(self.type, ()))
        if any(getattr(self, field) is None for field in required_fields):
            raise ValueError(f"{self.type} event is missing its required payload")
        allowed_fields = required_fields
        if self.type == "terminal_error":
            allowed_fields.add("last_safe_position")
        if any(getattr(self, field) is not None for field in payload_fields - allowed_fields):
            raise ValueError(f"{self.type} event has an unexpected payload")
        return self
