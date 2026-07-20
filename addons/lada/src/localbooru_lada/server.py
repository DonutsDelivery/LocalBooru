import array
import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path

from .adapter import LadaFrameSource, RestoredFrame
from .constants import DEFAULT_BUFFER_COUNT, LADA_REVISION, MAX_MESSAGE_BYTES, PROTOCOL_VERSION
from .pool import FramePool, StaleLease
from .protocol import ProtocolError, decode_message, encode_message
from .session import SessionController


@dataclass(frozen=True)
class ServerConfig:
    nonce: str
    backend: str
    detection_model_path: str
    restoration_model_path: str
    fp16: bool = True
    max_clip_length: int = 180

    @classmethod
    def load(cls, path: Path) -> "ServerConfig":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls(**value)


class SidecarServer:
    def __init__(self, connection: socket.socket, config: ServerConfig):
        self._connection = connection
        self._config = config
        self._send_lock = threading.Lock()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._controller = SessionController(self._create_source)
        self._pool = None
        self._producer = None
        self._source_path = None

    def _create_source(self, start_ns: int):
        return LadaFrameSource(
            source_path=self._source_path,
            start_ns=start_ns,
            device=self._config.backend,
            detection_model_path=self._config.detection_model_path,
            restoration_model_path=self._config.restoration_model_path,
            fp16=self._config.fp16,
            max_clip_length=self._config.max_clip_length,
        )

    def _send(self, message: dict, fds: list[int] | None = None) -> None:
        ancillary = []
        if fds:
            ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
        with self._send_lock:
            self._connection.sendmsg([encode_message(message)], ancillary)

    def _start_producer(self) -> None:
        self._producer = threading.Thread(target=self._produce, name="lada-frame-producer", daemon=True)
        self._producer.start()

    def _stop_producer(self) -> None:
        self._stop.set()
        self._controller.stop()
        if self._producer is not None:
            self._producer.join(timeout=5)
            self._producer = None
        self._stop.clear()

    def _produce(self) -> None:
        generation = self._controller.generation
        source = self._controller.restorer
        try:
            first = next(source)
            self._ensure_pool(first)
            self._send(
                {
                    "type": "ready",
                    "generation": generation,
                    "width": first.width,
                    "height": first.height,
                    "pixel_format": "BGR",
                    "stride": first.stride,
                    "backend": self._config.backend,
                    "protocol": PROTOCOL_VERSION,
                    "upstream_revision": LADA_REVISION,
                }
            )
            self._publish(first, generation)
            for frame in source:
                if self._stop.is_set() or generation != self._controller.generation:
                    return
                while self._pause.is_set() and not self._stop.wait(0.05):
                    pass
                if self._stop.is_set():
                    return
                self._publish(frame, generation)
            self._send({"type": "eos", "generation": generation})
        except StopIteration:
            self._send({"type": "eos", "generation": generation})
        except Exception as error:
            if not self._stop.is_set():
                self._send(
                    {
                        "type": "error",
                        "generation": generation,
                        "code": "restoration_failed",
                        "recoverable": True,
                        "message": str(error)[:512],
                    }
                )

    def _ensure_pool(self, frame: RestoredFrame) -> None:
        required = frame.stride * frame.height
        if self._pool is None:
            self._pool = FramePool(buffer_count=DEFAULT_BUFFER_COUNT, buffer_capacity=required)
            descriptors = [
                {"buffer_id": item["buffer_id"], "capacity": item["capacity"]}
                for item in self._pool.descriptors
            ]
            self._send(
                {
                    "type": "buffers",
                    "generation": self._controller.generation,
                    "buffers": descriptors,
                },
                [item["fd"] for item in self._pool.descriptors],
            )
        elif self._pool.descriptors[0]["capacity"] < required:
            raise RuntimeError("restored frame dimensions exceed the negotiated shared buffers")
        else:
            self._pool.reset(generation=self._controller.generation)

    def _publish(self, frame: RestoredFrame, generation: int) -> None:
        lease = self._pool.acquire(generation=generation)
        size = self._pool.write(lease, frame.data)
        self._send(
            {
                "type": "frame",
                "generation": generation,
                "sequence": lease.sequence,
                "buffer_id": lease.buffer_id,
                "width": frame.width,
                "height": frame.height,
                "stride": frame.stride,
                "size": size,
                "pts_ns": frame.pts_ns,
                "duration_ns": frame.duration_ns,
            }
        )

    def _handle(self, message: dict) -> bool:
        kind = message["type"]
        if kind == "hello":
            if message.get("nonce") != self._config.nonce:
                raise ProtocolError("session nonce does not match")
            self._send(
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "role": "lada_sidecar",
                    "upstream_revision": LADA_REVISION,
                }
            )
        elif kind == "open":
            source = Path(message.get("source_path", ""))
            if not source.is_absolute() or not source.is_file():
                raise ProtocolError("source_path must be an existing absolute file")
            self._source_path = str(source)
            started = self._controller.open(start_ns=int(message.get("start_ns", 0)))
            self._send(started)
            self._start_producer()
        elif kind == "seek":
            self._stop.set()
            started = self._controller.seek(
                start_ns=int(message["start_ns"]),
                request_id=int(message["request_id"]),
            )
            if self._pool is not None:
                self._pool.reset(generation=started["generation"])
            if self._producer is not None:
                self._producer.join(timeout=5)
                self._producer = None
            self._stop.clear()
            self._send(started)
            self._start_producer()
        elif kind == "release":
            self._controller.accept_generation(message["generation"])
            if self._pool is None:
                raise ProtocolError("no shared frame pool is active")
            try:
                self._pool.release(
                    int(message["buffer_id"]),
                    int(message["sequence"]),
                    generation=message["generation"],
                )
            except StaleLease as error:
                raise ProtocolError(str(error)) from error
        elif kind == "pause":
            self._controller.accept_generation(message["generation"])
            self._pause.set()
        elif kind == "resume":
            self._controller.accept_generation(message["generation"])
            self._pause.clear()
        elif kind == "stop":
            return False
        else:
            raise ProtocolError(f"unsupported client message: {kind}")
        return True

    def run(self) -> None:
        try:
            while True:
                data = self._connection.recv(MAX_MESSAGE_BYTES + 1)
                if not data:
                    break
                try:
                    if not self._handle(decode_message(data)):
                        break
                except ProtocolError as error:
                    generation = self._controller.generation
                    self._send(
                        {
                            "type": "error",
                            "generation": generation,
                            "code": "protocol_error",
                            "recoverable": False,
                            "message": str(error),
                        }
                    )
        finally:
            self._stop_producer()
            if self._pool is not None:
                self._pool.close()
            self._connection.close()
