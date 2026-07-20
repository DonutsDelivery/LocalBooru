import array
import os
import socket
import threading
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
            data=b"\x01\x02\x03" * 4,
            width=2,
            height=2,
            stride=6,
            pts_ns=self.start_ns,
            duration_ns=40_000_000,
        )


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
