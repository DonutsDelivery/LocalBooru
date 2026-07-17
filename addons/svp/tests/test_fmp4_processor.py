import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import processing_session_metadata, start_processing_session  # noqa: E402
from processing_session import ProcessingSessionRegistry  # noqa: E402
from session_protocol import OpenSessionRequest, SessionGraph  # noqa: E402

@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("vspipe"),
    "FFmpeg, ffprobe, and vspipe are required",
)
class Fmp4SessionProcessorTests(unittest.TestCase):
    # AC: @svp-bounded-stream ac-bounded-buffer
    # AC: @svp-bounded-stream ac-av-timeline
    def test_manager_graph_pipeline_publishes_its_distinct_bounded_muxed_fmp4_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000",
                    "-t",
                    "1.5",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    str(source),
                ],
                check=True,
            )
            manager_script = (
                b"import vapoursynth as vs\n"
                b"vs.core.std.Interleave([video_in, video_in], extend=True).set_output()\n"
            )
            manager_path = root / "distinct-manager.vpy"
            manager_path.write_bytes(manager_script)
            previous_root = os.environ.get("LOCALBOORU_SVP_SNAPSHOT_ROOT")
            os.environ["LOCALBOORU_SVP_SNAPSHOT_ROOT"] = str(root)
            self.addCleanup(
                lambda: os.environ.pop("LOCALBOORU_SVP_SNAPSHOT_ROOT", None)
                if previous_root is None
                else os.environ.__setitem__("LOCALBOORU_SVP_SNAPSHOT_ROOT", previous_root)
            )
            request = OpenSessionRequest(
                protocol_version=1,
                generation=1,
                file_path=str(source),
                graph=SessionGraph(
                    kind="manager_snapshot",
                    revision=1,
                    snapshot_path=str(manager_path),
                    snapshot_sha256=hashlib.sha256(manager_script).hexdigest(),
                ),
                max_buffer_bytes=4 * 1024 * 1024,
                max_buffer_duration=2.0,
            )
            metadata = processing_session_metadata(request)
            self.assertEqual(metadata.width, 160)
            self.assertEqual(metadata.height, 90)
            self.assertGreater(metadata.output_fps, 47.0)
            self.assertLess(metadata.output_fps, 49.0)
            registry = ProcessingSessionRegistry(root / "sessions")
            session = registry.open(request, metadata)
            processor = start_processing_session(session)

            output = bytearray()
            cursor = -1
            deadline = time.time() + 20
            terminal_error = None
            ended = False
            while time.time() < deadline and not ended and terminal_error is None:
                for event in session.events_after(cursor):
                    cursor = event.event_sequence
                    if event.type == "init_ready":
                        output.extend(
                            session.read_owned_file(event.generation, event.init_segment.filename)
                        )
                        session.acknowledge_init(event.generation)
                    elif event.type == "segment_ready":
                        output.extend(
                            session.read_owned_file(event.generation, event.segment.filename)
                        )
                        session.acknowledge(event.generation, event.segment.sequence)
                    elif event.type == "terminal_error":
                        terminal_error = event.message
                    elif event.type == "ended":
                        ended = True
                time.sleep(0.02)

            processor.stop()
            self.assertIsNone(terminal_error)
            self.assertTrue(ended)
            self.assertGreater(len(output), 0)
            self.assertLessEqual(session.buffered_bytes, request.max_buffer_bytes)
            self.assertLessEqual(session.buffered_duration, request.max_buffer_duration)

            assembled = root / "assembled.mp4"
            assembled.write_bytes(output)
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,codec_type,avg_frame_rate,duration",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(assembled),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            import json

            info = json.loads(probe.stdout)
            streams = {stream["codec_type"]: stream for stream in info["streams"]}
            self.assertEqual(streams["video"]["codec_name"], "h264")
            self.assertEqual(streams["audio"]["codec_name"], "aac")
            output_rate = float(Fraction(streams["video"]["avg_frame_rate"]))
            self.assertGreater(output_rate, 47.0)
            self.assertLess(output_rate, 49.0)
            self.assertAlmostEqual(float(info["format"]["duration"]), 1.5, delta=0.1)
            registry.stop_all()


if __name__ == "__main__":
    unittest.main()
