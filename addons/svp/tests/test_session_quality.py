import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (  # noqa: E402
    _session_decode_size,
    processing_session_metadata,
    start_processing_session,
)
from fmp4_processor import Fmp4SessionProcessor  # noqa: E402
from processing_session import ProcessingSessionRegistry  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from session_protocol import OpenSessionRequest, SessionGraph  # noqa: E402


def graph_body():
    return {
        "kind": "manager_snapshot",
        "revision": 1,
        "snapshot_path": "/tmp/manager.vpy",
        "snapshot_sha256": "0" * 64,
    }


def open_body(**extra):
    return {
        "protocol_version": 1,
        "generation": 1,
        "file_path": "/media/clip.mp4",
        "graph": graph_body(),
        **extra,
    }


class SessionRequestTests(unittest.TestCase):
    def test_target_resolution_is_optional(self):
        self.assertIsNone(OpenSessionRequest.model_validate(open_body()).target_resolution)

    def test_target_resolution_is_parsed(self):
        request = OpenSessionRequest.model_validate(open_body(target_resolution="720p"))
        self.assertEqual(request.target_resolution, "720p")

    def test_empty_target_resolution_is_rejected(self):
        with self.assertRaises(ValidationError):
            OpenSessionRequest.model_validate(open_body(target_resolution=""))


class SessionDecodeSizeTests(unittest.TestCase):
    INFO = {"width": 3840, "height": 2160}

    def test_preset_downscales_before_interpolation(self):
        self.assertEqual(_session_decode_size("720p", self.INFO), (1280, 720))

    def test_missing_unknown_and_original_keep_source_size(self):
        for value in (None, "original", "bogus"):
            self.assertEqual(_session_decode_size(value, self.INFO), (3840, 2160))

    def test_never_upscales(self):
        self.assertEqual(_session_decode_size("1080p", {"width": 640, "height": 360}), (640, 360))


class DecodeCommandTests(unittest.TestCase):
    @staticmethod
    def command(decode_size):
        processor = Fmp4SessionProcessor.__new__(Fmp4SessionProcessor)
        processor.session = SimpleNamespace(file_path="/media/clip.mp4")
        processor.ffmpeg_executable = "ffmpeg"
        processor.decode_size = decode_size
        return processor._decode_command(12.5)

    def test_reduced_size_is_scaled_in_the_decode_stage(self):
        command = self.command((1280, 720))
        self.assertIn("scale=1280:720:flags=lanczos", command)
        self.assertEqual(command[command.index("-vf") - 2:command.index("-vf")], ["-i", "/media/clip.mp4"])
        self.assertLess(command.index("-vf"), command.index("-f"))

    def test_unchanged_size_adds_no_scaler(self):
        self.assertNotIn("-vf", self.command(None))


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("vspipe"),
    "FFmpeg, ffprobe, and vspipe are required",
)
class SessionGraphResolutionTests(unittest.TestCase):
    def test_graph_is_probed_and_run_at_the_requested_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24",
                    "-t", "0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
                ],
                check=True,
            )
            manager_script = (
                b"import vapoursynth as vs\n"
                b"vs.core.std.Interleave([video_in, video_in], extend=True).set_output()\n"
            )
            manager_path = root / "manager.vpy"
            manager_path.write_bytes(manager_script)
            previous_root = os.environ.get("LOCALBOORU_SVP_SNAPSHOT_ROOT")
            os.environ["LOCALBOORU_SVP_SNAPSHOT_ROOT"] = str(root)
            self.addCleanup(
                lambda: os.environ.pop("LOCALBOORU_SVP_SNAPSHOT_ROOT", None)
                if previous_root is None
                else os.environ.__setitem__("LOCALBOORU_SVP_SNAPSHOT_ROOT", previous_root)
            )
            graph = SessionGraph(
                kind="manager_snapshot",
                revision=1,
                snapshot_path=str(manager_path),
                snapshot_sha256=hashlib.sha256(manager_script).hexdigest(),
            )

            def request(target_resolution):
                return OpenSessionRequest(
                    protocol_version=1,
                    generation=1,
                    file_path=str(source),
                    graph=graph,
                    target_resolution=target_resolution,
                )

            original = processing_session_metadata(request(None))
            self.assertEqual((original.width, original.height), (1280, 720))
            reduced = request("480p")
            metadata = processing_session_metadata(reduced)
            self.assertEqual((metadata.width, metadata.height), (854, 480))

            registry = ProcessingSessionRegistry(root / "sessions")
            session = registry.open(reduced, metadata)
            processor = start_processing_session(session)
            try:
                self.assertEqual(processor.decode_size, (854, 480))
            finally:
                processor.stop()

            full = registry.open(request(None), original)
            processor = start_processing_session(full)
            try:
                self.assertIsNone(processor.decode_size)
            finally:
                processor.stop()


if __name__ == "__main__":
    unittest.main()
