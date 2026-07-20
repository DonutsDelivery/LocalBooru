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
    producer_join_timeout_seconds: float = 5.0

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
        self._pool_lock = threading.Lock()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._authenticated = False
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

    def _retire_pool(self) -> None:
        with self._pool_lock:
            pool = self._pool
            self._pool = None
            if pool is not None:
                pool.close()

    def _stop_producer(self) -> None:
        self._stop.set()
        self._controller.stop()
        self._retire_pool()
        if self._producer is not None:
            self._producer.join(timeout=self._config.producer_join_timeout_seconds)
            if not self._producer.is_alive():
                self._producer = None

    def _generation_is_active(self, generation: int) -> bool:
        return not self._stop.is_set() and generation == self._controller.generation

    def _send_eos(self, generation: int) -> None:
        if self._generation_is_active(generation):
            self._send({"type": "eos", "generation": generation})

    def _produce(self) -> None:
        generation = self._controller.generation
        source = self._controller.restorer
        try:
            first = next(source)
            if self._stop.is_set() or generation != self._controller.generation:
                return
            pool = self._create_pool(first, generation)
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
            self._publish(first, generation, pool)
            for frame in source:
                if self._stop.is_set() or generation != self._controller.generation:
                    return
                while self._pause.is_set() and not self._stop.wait(0.05):
                    pass
                if self._stop.is_set():
                    return
                self._publish(frame, generation, pool)
            self._send_eos(generation)
        except StopIteration:
            self._send_eos(generation)
        except Exception as error:
            if self._generation_is_active(generation):
                self._send(
                    {
                        "type": "error",
                        "generation": generation,
                        "code": "restoration_failed",
                        "recoverable": True,
                        "message": str(error)[:512],
                    }
                )

    def _create_pool(self, frame: RestoredFrame, generation: int) -> FramePool:
        required = frame.stride * frame.height
        with self._pool_lock:
            if self._stop.is_set() or generation != self._controller.generation:
                raise StaleLease(f"generation {generation} is stale")
            if self._pool is not None:
                raise RuntimeError("a shared frame pool is already active")
            pool = FramePool(buffer_count=DEFAULT_BUFFER_COUNT, buffer_capacity=required)
            self._pool = pool
            descriptors = [
                {"buffer_id": item["buffer_id"], "capacity": item["capacity"]}
                for item in pool.descriptors
            ]
            self._send(
                {
                    "type": "buffers",
                    "generation": generation,
                    "buffers": descriptors,
                },
                [item["fd"] for item in pool.descriptors],
            )
            return pool

    def _publish(self, frame: RestoredFrame, generation: int, pool: FramePool) -> None:
        lease = pool.acquire(generation=generation)
        size = pool.write(lease, frame.data)
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

    def _quiesce_for_seek(self) -> None:
        self._stop.set()
        source = self._controller.restorer
        if source is not None:
            source.stop()
        self._retire_pool()
        if self._producer is not None:
            self._producer.join(timeout=self._config.producer_join_timeout_seconds)
            if self._producer.is_alive():
                raise ProtocolError("previous frame producer did not stop before seek")
            self._producer = None

    def _handle(self, message: dict) -> bool:
        kind = message["type"]
        if kind != "hello" and not self._authenticated:
            raise ProtocolError("hello authentication is required before commands")
        if kind == "hello":
            if self._authenticated:
                raise ProtocolError("session is already authenticated")
            if message.get("nonce") != self._config.nonce:
                raise ProtocolError("session nonce does not match")
            if message.get("protocol") != PROTOCOL_VERSION:
                raise ProtocolError("protocol version does not match")
            if message.get("role") != "coordinator":
                raise ProtocolError("hello role must be coordinator")
            self._authenticated = True
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
            self._controller.accept_generation(message["generation"])
            self._quiesce_for_seek()
            self._stop.clear()
            started = self._controller.seek(
                start_ns=int(message["start_ns"]),
                request_id=int(message["request_id"]),
            )
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
            self._connection.close()
