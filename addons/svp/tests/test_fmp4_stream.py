import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fmp4_stream import (  # noqa: E402
    find_fragment_decode_time,
    iter_fragmented_mp4,
    parse_video_track,
    read_box,
)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class FragmentedMp4Tests(unittest.TestCase):
    def test_parser_splits_playable_init_and_monotonic_media_fragments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "fixture.mp4"
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
                    "-t",
                    "1.5",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "12",
                    "-keyint_min",
                    "12",
                    "-sc_threshold",
                    "0",
                    "-movflags",
                    "frag_keyframe+empty_moov+default_base_moof",
                    "-f",
                    "mp4",
                    str(output),
                ],
                check=True,
            )

            parts = list(iter_fragmented_mp4(io.BytesIO(output.read_bytes())))
            self.assertEqual(parts[0].kind, "init")
            self.assertGreaterEqual(len(parts), 3)
            track = parse_video_track(parts[0].data)
            self.assertGreater(track.track_id, 0)
            self.assertGreater(track.timescale, 0)
            self.assertTrue(track.codec.startswith("avc1."))

            decode_times = [
                find_fragment_decode_time(part.data, track.track_id)
                for part in parts[1:]
            ]
            self.assertEqual(decode_times, sorted(decode_times))
            self.assertEqual(len(decode_times), len(set(decode_times)))

    def test_parser_rejects_truncated_box(self):
        with self.assertRaises(ValueError):
            read_box(io.BytesIO(b"\x00\x00\x00\x10moofshort"))


if __name__ == "__main__":
    unittest.main()
