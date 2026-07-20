from localbooru_lada.protocol import PROTOCOL_VERSION, ProtocolError, decode_message, encode_message
from localbooru_lada.session import SessionController


class FakeRestorer:
    def __init__(self, start_ns):
        self.start_ns = start_ns
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_protocol_round_trip_and_required_generation():
    message = {
        "type": "frame",
        "generation": 7,
        "sequence": 12,
        "buffer_id": 1,
        "pts_ns": 5_000,
        "duration_ns": 40_000_000,
    }
    assert decode_message(encode_message(message)) == message

    try:
        decode_message(b'{"type":"frame","sequence":1}')
    except ProtocolError as error:
        assert "generation" in str(error)
    else:
        raise AssertionError("frame without generation must fail")
    try:
        decode_message(b'{"type":"seek","request_id":2,"start_ns":0}')
    except ProtocolError as error:
        assert "generation" in str(error)
    else:
        raise AssertionError("seek without generation must fail")


def test_seek_stops_old_restorer_and_increments_generation():
    created = []

    def factory(start_ns):
        restorer = FakeRestorer(start_ns)
        created.append(restorer)
        return restorer

    controller = SessionController(factory)
    opened = controller.open(start_ns=10)
    sought = controller.seek(start_ns=99, request_id=4)

    assert PROTOCOL_VERSION == 1
    assert opened == {"type": "generation_started", "request_id": 0, "generation": 1, "start_ns": 10}
    assert sought == {"type": "generation_started", "request_id": 4, "generation": 2, "start_ns": 99}
    assert created[0].stopped is True
    assert created[1].stopped is False


def test_stale_generation_commands_are_rejected():
    controller = SessionController(FakeRestorer)
    controller.open(start_ns=0)
    controller.seek(start_ns=50, request_id=1)

    try:
        controller.accept_generation(1)
    except ProtocolError as error:
        assert "stale generation" in str(error)
    else:
        raise AssertionError("stale generation must fail")
