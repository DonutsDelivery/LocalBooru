import json

from .constants import MAX_MESSAGE_BYTES, PROTOCOL_VERSION


class ProtocolError(ValueError):
    pass


_GENERATION_MESSAGES = {
    "ready",
    "buffers",
    "frame",
    "release",
    "pause",
    "resume",
    "seek",
    "eos",
    "error",
}
_ALLOWED_MESSAGES = _GENERATION_MESSAGES | {
    "hello",
    "open",
    "seek",
    "generation_started",
    "stop",
}


def validate_message(message: dict) -> dict:
    if not isinstance(message, dict):
        raise ProtocolError("protocol message must be an object")
    message_type = message.get("type")
    if message_type not in _ALLOWED_MESSAGES:
        raise ProtocolError(f"unsupported message type: {message_type}")
    if message_type in _GENERATION_MESSAGES and not isinstance(message.get("generation"), int):
        raise ProtocolError(f"{message_type} message requires an integer generation")
    if message_type == "hello" and message.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("incompatible protocol version")
    return message


def encode_message(message: dict) -> bytes:
    validate_message(message)
    encoded = json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolError("protocol message exceeds the size limit")
    return encoded


def decode_message(data: bytes) -> dict:
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError("protocol message exceeds the size limit")
    try:
        message = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid JSON protocol message") from error
    return validate_message(message)
