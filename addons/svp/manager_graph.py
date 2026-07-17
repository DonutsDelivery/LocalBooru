import base64
import hashlib
import hmac
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from session_protocol import SessionGraph

MAX_MANAGER_GRAPH_BYTES = 8 * 1024 * 1024


class ManagerGraphUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CapturedManagerGraph:
    revision: int
    path: Path
    sha256: str
    contents: bytes


def trusted_snapshot_root() -> Path:
    configured = os.environ.get("LOCALBOORU_SVP_SNAPSHOT_ROOT")
    if not configured:
        raise ManagerGraphUnavailable("SVP Manager snapshot root is not configured")
    root_path = Path(configured)
    if not root_path.is_absolute() or root_path.is_symlink():
        raise ManagerGraphUnavailable("SVP Manager snapshot root is invalid")
    try:
        root = root_path.resolve(strict=True)
        metadata = root.stat()
    except OSError as error:
        raise ManagerGraphUnavailable("SVP Manager snapshot root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ManagerGraphUnavailable("SVP Manager snapshot root is not a directory")
    if os.name != "nt":
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ManagerGraphUnavailable("SVP Manager snapshot root is not private")
    return root


def load_manager_snapshot(graph: SessionGraph, root: Path | None = None) -> CapturedManagerGraph:
    trusted_root = root or trusted_snapshot_root()
    candidate = Path(graph.snapshot_path)
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise ManagerGraphUnavailable("SVP Manager snapshot is unavailable") from error
    if not candidate.is_absolute() or parent != trusted_root:
        raise ManagerGraphUnavailable("SVP Manager snapshot is outside the trusted root")
    try:
        link_metadata = candidate.lstat()
    except OSError as error:
        raise ManagerGraphUnavailable("SVP Manager snapshot is unavailable") from error
    if stat.S_ISLNK(link_metadata.st_mode) or not stat.S_ISREG(link_metadata.st_mode):
        raise ManagerGraphUnavailable("SVP Manager snapshot is not a regular file")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ManagerGraphUnavailable("SVP Manager snapshot could not be opened") from error
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ManagerGraphUnavailable("SVP Manager snapshot is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            contents = stream.read(MAX_MANAGER_GRAPH_BYTES + 1)
    finally:
        os.close(descriptor)

    if len(contents) > MAX_MANAGER_GRAPH_BYTES:
        raise ManagerGraphUnavailable("SVP Manager snapshot is too large")
    digest = hashlib.sha256(contents).hexdigest()
    if not hmac.compare_digest(digest, graph.snapshot_sha256):
        raise ManagerGraphUnavailable("SVP Manager snapshot hash does not match")
    return CapturedManagerGraph(graph.revision, candidate, digest, contents)


def generate_manager_snapshot_stdin_script(
    graph: CapturedManagerGraph,
    width: int,
    height: int,
    fps_num: int,
    fps_den: int,
    num_frames: int,
) -> str:
    encoded_graph = base64.b64encode(graph.contents).decode("ascii")
    frame_size = width * height * 3 // 2
    chroma_width = width // 2
    chroma_height = height // 2
    return f'''import base64
import ctypes
import sys
import vapoursynth as vs
core = vs.core

WIDTH = {width}
HEIGHT = {height}
FPS_NUM = {fps_num}
FPS_DEN = {fps_den}
NUM_FRAMES = {num_frames}
FRAME_SIZE = {frame_size}
CHROMA_WIDTH = {chroma_width}
CHROMA_HEIGHT = {chroma_height}
stdin = sys.stdin.buffer
next_frame = 0
last_frame = None

video_in = core.std.BlankClip(
    width=WIDTH,
    height=HEIGHT,
    format=vs.YUV420P8,
    length=max(NUM_FRAMES, 1),
    fpsnum=FPS_NUM,
    fpsden=FPS_DEN,
)

def read_exact(size):
    data = bytearray()
    while len(data) < size:
        chunk = stdin.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)

def write_plane(frame, plane, src, width, height):
    stride = frame.get_stride(plane)
    ptr = frame.get_write_ptr(plane)
    pos = 0
    for y in range(height):
        ctypes.memmove(ptr.value + y * stride, src[pos:pos + width], width)
        pos += width

def source_frame(n, f):
    global next_frame, last_frame
    if n < next_frame and last_frame is not None:
        return last_frame
    while next_frame < n:
        skipped = read_exact(FRAME_SIZE)
        if len(skipped) < FRAME_SIZE:
            break
        next_frame += 1
    raw = read_exact(FRAME_SIZE)
    if len(raw) < FRAME_SIZE:
        if last_frame is not None:
            return last_frame
        raw = raw + bytes(FRAME_SIZE - len(raw))
    out = f.copy()
    y_size = WIDTH * HEIGHT
    uv_size = CHROMA_WIDTH * CHROMA_HEIGHT
    write_plane(out, 0, raw[:y_size], WIDTH, HEIGHT)
    write_plane(out, 1, raw[y_size:y_size + uv_size], CHROMA_WIDTH, CHROMA_HEIGHT)
    write_plane(out, 2, raw[y_size + uv_size:y_size + uv_size * 2], CHROMA_WIDTH, CHROMA_HEIGHT)
    next_frame = n + 1
    last_frame = out
    return out

video_in = core.std.ModifyFrame(video_in, video_in, source_frame)
video_in_dw = WIDTH
video_in_dh = HEIGHT
container_fps = FPS_NUM / FPS_DEN
display_fps = 0.0
display_res = [WIDTH, HEIGHT]
user_data = ""
__file__ = {str(graph.path)!r}
_manager_graph = base64.b64decode({encoded_graph!r})
exec(compile(_manager_graph, __file__, "exec"), globals())
'''
