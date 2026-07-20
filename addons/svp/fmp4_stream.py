import io
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Optional


MAX_BOX_SIZE = 256 * 1024 * 1024


@dataclass(frozen=True)
class Mp4Part:
    kind: str
    data: bytes
    decode_time: Optional[int] = None


@dataclass(frozen=True)
class Mp4VideoTrack:
    track_id: int
    timescale: int
    codec: str


def read_box(stream: BinaryIO) -> Optional[tuple[str, bytes]]:
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise ValueError("truncated MP4 box header")
    size = int.from_bytes(header[:4], "big")
    box_type = header[4:8].decode("latin1")
    header_size = 8
    if size == 1:
        extended = stream.read(8)
        if len(extended) != 8:
            raise ValueError("truncated extended MP4 box header")
        size = int.from_bytes(extended, "big")
        header += extended
        header_size = 16
    elif size == 0:
        payload = stream.read()
        return box_type, header + payload
    if size < header_size or size > MAX_BOX_SIZE:
        raise ValueError(f"invalid MP4 box size: {size}")
    payload = stream.read(size - header_size)
    if len(payload) != size - header_size:
        raise ValueError(f"truncated MP4 {box_type} box")
    return box_type, header + payload


def iter_fragmented_mp4(stream: BinaryIO, video_track_id: Optional[int] = None) -> Iterator[Mp4Part]:
    init = bytearray()
    fragment = bytearray()
    fragment_decode_time = None
    saw_fragment = False

    while True:
        box = read_box(stream)
        if box is None:
            break
        box_type, data = box
        if box_type == "moof":
            if not saw_fragment:
                if not init:
                    raise ValueError("fragmented MP4 has no initialization segment")
                yield Mp4Part("init", bytes(init))
                saw_fragment = True
            elif fragment:
                yield Mp4Part("media", bytes(fragment), fragment_decode_time)
            fragment = bytearray(data)
            fragment_decode_time = (
                find_fragment_decode_time(data, video_track_id)
                if video_track_id is not None
                else None
            )
        elif not saw_fragment:
            init.extend(data)
        elif box_type == "mfra":
            break
        else:
            fragment.extend(data)

    if fragment:
        yield Mp4Part("media", bytes(fragment), fragment_decode_time)


def parse_video_track(init_segment: bytes) -> Mp4VideoTrack:
    moov = next((payload for box_type, payload in _child_boxes(init_segment) if box_type == "moov"), None)
    if moov is None:
        raise ValueError("initialization segment has no moov box")
    for box_type, trak in _child_boxes(moov):
        if box_type != "trak":
            continue
        track_id = _track_id(trak)
        mdia = next((payload for child_type, payload in _child_boxes(trak) if child_type == "mdia"), None)
        if mdia is None or _handler_type(mdia) != "vide":
            continue
        timescale = _media_timescale(mdia)
        codec = _avc_codec(init_segment)
        return Mp4VideoTrack(track_id=track_id, timescale=timescale, codec=codec)
    raise ValueError("initialization segment has no video track")


def find_fragment_decode_time(moof_box: bytes, video_track_id: int) -> int:
    moof = _box_payload(moof_box)
    for box_type, traf in _child_boxes(moof):
        if box_type != "traf":
            continue
        track_id = None
        decode_time = None
        for child_type, payload in _child_boxes(traf):
            if child_type == "tfhd" and len(payload) >= 8:
                track_id = int.from_bytes(payload[4:8], "big")
            elif child_type == "tfdt" and len(payload) >= 8:
                version = payload[0]
                width = 8 if version == 1 else 4
                decode_time = int.from_bytes(payload[4 : 4 + width], "big")
        if track_id == video_track_id and decode_time is not None:
            return decode_time
    raise ValueError("fragment has no decode time for the video track")


def _child_boxes(data: bytes) -> Iterator[tuple[str, bytes]]:
    stream = io.BytesIO(data)
    while stream.tell() < len(data):
        box = read_box(stream)
        if box is None:
            return
        box_type, full_box = box
        yield box_type, _box_payload(full_box)


def _box_payload(box: bytes) -> bytes:
    size = int.from_bytes(box[:4], "big")
    header_size = 16 if size == 1 else 8
    return box[header_size:]


def _track_id(trak: bytes) -> int:
    tkhd = next((payload for box_type, payload in _child_boxes(trak) if box_type == "tkhd"), None)
    if tkhd is None:
        raise ValueError("track has no tkhd box")
    version = tkhd[0]
    offset = 20 if version == 1 else 12
    return int.from_bytes(tkhd[offset : offset + 4], "big")


def _handler_type(mdia: bytes) -> str:
    hdlr = next((payload for box_type, payload in _child_boxes(mdia) if box_type == "hdlr"), None)
    if hdlr is None or len(hdlr) < 12:
        raise ValueError("media track has no handler")
    return hdlr[8:12].decode("latin1")


def _media_timescale(mdia: bytes) -> int:
    mdhd = next((payload for box_type, payload in _child_boxes(mdia) if box_type == "mdhd"), None)
    if mdhd is None:
        raise ValueError("media track has no mdhd box")
    version = mdhd[0]
    offset = 20 if version == 1 else 12
    timescale = int.from_bytes(mdhd[offset : offset + 4], "big")
    if timescale <= 0:
        raise ValueError("video track has an invalid timescale")
    return timescale


def _avc_codec(init_segment: bytes) -> str:
    marker = init_segment.find(b"avcC")
    if marker < 0 or marker + 8 > len(init_segment):
        raise ValueError("initialization segment has no AVC configuration")
    profile, compatibility, level = init_segment[marker + 5 : marker + 8]
    return f"avc1.{profile:02x}{compatibility:02x}{level:02x}"
