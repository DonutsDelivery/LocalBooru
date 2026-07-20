import array
import mmap
import os
import socket
import threading
import time
from dataclasses import dataclass

from localbooru_lada.adapter import RestoredFrame
from localbooru_lada.constants import PROTOCOL_VERSION
from localbooru_lada.protocol import decode_message, encode_message
from localbooru_lada.server import ServerConfig, SidecarServer
from localbooru_lada.session import SessionController


@dataclass
class FakeSource:
    start_ns: int
    stopped: bool = False
    emitted: bool = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.emitted:
            raise StopIteration
        self.emitted = True
        return RestoredFrame(
            data=bytes([self.start_ns % 256]) * 12,
            width=2,
            height=2,
            stride=6,
            pts_ns=self.start_ns,
            duration_ns=40_000_000,
        )


@dataclass
class SlowStoppingSource(FakeSource):
    def __next__(self):
        if not self.emitted:
            return super().__next__()
        time.sleep(0.08)
        raise StopIteration


def _recv(connection):
    data, ancillary, _, _ = connection.recvmsg(65536, socket.CMSG_SPACE(3 * array.array("i").itemsize))
    fds = []
    for level, kind, payload in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array.array("i")
            values.frombytes(payload[: len(payload) - (len(payload) % values.itemsize)])
            fds.extend(values)
    return decode_message(data), fds


def test_server_negotiates_buffers_and_publishes_timestamped_frame(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    server_socket, client_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server = SidecarServer(
        server_socket,
        ServerConfig(
            nonce="secret",
            backend="cuda",
            detection_model_path="detect.pt",
            restoration_model_path="restore.pth",
        ),
    )
    server._controller = SessionController(FakeSource)
    thread = threading.Thread(target=server.run)
    thread.start()

    client_socket.send(encode_message({
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "role": "coordinator",
        "nonce": "secret",
    }))
    hello, _ = _recv(client_socket)
    assert hello["role"] == "lada_sidecar"

    client_socket.send(encode_message({
        "type": "open",
        "source_path": str(source),
        "start_ns": 5_000,
    }))
    started, _ = _recv(client_socket)
    buffers, fds = _recv(client_socket)
    ready, _ = _recv(client_socket)
    frame, _ = _recv(client_socket)

    assert started["generation"] == 1
    assert buffers["type"] == "buffers"
    assert len(buffers["buffers"]) == 3
    assert len(fds) == 3
    assert ready["backend"] == "cuda"
    assert frame["pts_ns"] == 5_000
    assert frame["size"] == 12

    client_socket.send(encode_message({
        "type": "release",
        "generation": 1,
        "buffer_id": frame["buffer_id"],
        "sequence": frame["sequence"],
    }))
    eos, _ = _recv(client_socket)
    assert eos["type"] == "eos"

    client_socket.send(encode_message({"type": "stop"}))
    thread.join(timeout=2)
    assert not thread.is_alive()
    client_socket.close()
    for fd in fds:
        os.close(fd)


def test_server_rejects_open_before_authenticated_hello(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    server_socket, client_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client_socket.settimeout(2)
    server = SidecarServer(
        server_socket,
        ServerConfig(
            nonce="secret",
            backend="cuda",
            detection_model_path="detect.pt",
            restoration_model_path="restore.pth",
        ),
    )
    server._controller = SessionController(FakeSource)
    thread = threading.Thread(target=server.run)
    thread.start()

    client_socket.send(encode_message({
        "type": "open",
        "source_path": str(source),
        "start_ns": 0,
    }))
    error, _ = _recv(client_socket)
    assert error["type"] == "error"
    assert error["code"] == "protocol_error"
    assert "authentication" in error["message"]
    assert server._controller.generation == 0

    client_socket.send(encode_message({
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "role": "coordinator",
        "nonce": "secret",
    }))
    hello, _ = _recv(client_socket)
    assert hello["type"] == "hello"
    client_socket.send(encode_message({"type": "stop"}))
    thread.join(timeout=2)
    assert not thread.is_alive()
    client_socket.close()


def test_seek_uses_fresh_buffers_without_mutating_old_generation(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    server_socket, client_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client_socket.settimeout(2)
    server = SidecarServer(
        server_socket,
        ServerConfig(
            nonce="secret",
            backend="cuda",
            detection_model_path="detect.pt",
            restoration_model_path="restore.pth",
        ),
    )
    server._controller = SessionController(FakeSource)
    thread = threading.Thread(target=server.run)
    thread.start()

    client_socket.send(encode_message({
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "role": "coordinator",
        "nonce": "secret",
    }))
    _recv(client_socket)
    client_socket.send(encode_message({
        "type": "open",
        "source_path": str(source),
        "start_ns": 5_000,
    }))
    _recv(client_socket)
    _, old_fds = _recv(client_socket)
    _recv(client_socket)
    old_frame, _ = _recv(client_socket)
    _recv(client_socket)

    old_mapping = mmap.mmap(old_fds[old_frame["buffer_id"]], old_frame["size"], access=mmap.ACCESS_READ)
    old_bytes = old_mapping[:]

    client_socket.send(encode_message({
        "type": "seek",
        "generation": 1,
        "request_id": 7,
        "start_ns": 5_001,
    }))
    started, _ = _recv(client_socket)
    _, new_fds = _recv(client_socket)
    _recv(client_socket)
    new_frame, _ = _recv(client_socket)
    _recv(client_socket)

    assert started["generation"] == 2
    assert new_frame["generation"] == 2
    assert old_mapping[:] == old_bytes
    assert old_mapping[:] != bytes([5_001 % 256]) * 12

    client_socket.send(encode_message({"type": "stop"}))
    thread.join(timeout=2)
    assert not thread.is_alive()
    old_mapping.close()
    client_socket.close()
    for fd in old_fds + new_fds:
        os.close(fd)


def test_seek_timeout_never_emits_stale_eos_or_starts_concurrent_generation(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    server_socket, client_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client_socket.settimeout(2)
    server = SidecarServer(
        server_socket,
        ServerConfig(
            nonce="secret",
            backend="cuda",
            detection_model_path="detect.pt",
            restoration_model_path="restore.pth",
            producer_join_timeout_seconds=0.01,
        ),
    )
    server._controller = SessionController(
        lambda start_ns: SlowStoppingSource(start_ns) if start_ns == 5_000 else FakeSource(start_ns)
    )
    thread = threading.Thread(target=server.run)
    thread.start()

    client_socket.send(encode_message({
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "role": "coordinator",
        "nonce": "secret",
    }))
    _recv(client_socket)
    client_socket.send(encode_message({
        "type": "open",
        "source_path": str(source),
        "start_ns": 5_000,
    }))
    _recv(client_socket)
    _, old_fds = _recv(client_socket)
    _recv(client_socket)
    _recv(client_socket)

    seek = {
        "type": "seek",
        "generation": 1,
        "request_id": 8,
        "start_ns": 5_001,
    }
    client_socket.send(encode_message(seek))
    error, _ = _recv(client_socket)
    assert error["type"] == "error"
    assert "did not stop" in error["message"]
    assert server._controller.generation == 1

    time.sleep(0.1)
    client_socket.settimeout(0.02)
    try:
        _recv(client_socket)
    except TimeoutError:
        pass
    else:
        raise AssertionError("stopped generation must not emit a late EOS")

    client_socket.settimeout(2)
    client_socket.send(encode_message(seek))
    started, _ = _recv(client_socket)
    _, new_fds = _recv(client_socket)
    _recv(client_socket)
    frame, _ = _recv(client_socket)
    _recv(client_socket)
    assert started["generation"] == 2
    assert frame["generation"] == 2

    client_socket.send(encode_message({"type": "stop"}))
    thread.join(timeout=2)
    assert not thread.is_alive()
    client_socket.close()
    for fd in old_fds + new_fds:
        os.close(fd)
