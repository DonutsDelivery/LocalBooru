import ctypes
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as svp_app  # noqa: E402


@unittest.skipUnless(os.name == 'posix', 'legacy stderr polling uses POSIX pipes')
class LegacyStderrTests(unittest.TestCase):
    def make_stream(self, payload):
        read_fd, write_fd = os.pipe()
        pipe = os.fdopen(read_fd, 'rb')
        with patch.object(svp_app, 'check_nvenc', return_value=False):
            stream = svp_app.SVPStream('/synthetic/unused.mp4')
        stream._decode_proc = SimpleNamespace(stderr=pipe)
        os.write(write_fd, payload)
        self.addCleanup(pipe.close)
        return stream, write_fd

    def assert_drain_does_not_wait_for_writer(self, final):
        stream, write_fd = self.make_stream(b'frame=1\r')
        # Closing later makes the old blocking read fail deterministically
        # instead of hanging the test suite indefinitely.
        close_writer = threading.Timer(0.3, os.close, (write_fd,))
        close_writer.start()
        self.addCleanup(close_writer.join)
        start = time.monotonic()
        stream._drain_stderr(final=final)
        elapsed = time.monotonic() - start
        self.assertEqual(stream._decode_stderr, b'frame=1\r')
        self.assertLess(elapsed, 0.1, 'live stderr must not block HLS HTTP service')

    def test_partial_progress_line_never_blocks_event_loop(self):
        self.assert_drain_does_not_wait_for_writer(False)

    def test_final_drain_does_not_wait_for_other_live_pipeline_stages(self):
        self.assert_drain_does_not_wait_for_writer(True)

    def test_stderr_tail_stays_bounded_and_preserves_failure_detail(self):
        stream, write_fd = self.make_stream(b'x' * 1000 + b' encoder failed')
        os.close(write_fd)
        stream._decode_stderr = b'previous' * 2000
        stream._drain_stderr(final=True)
        self.assertEqual(len(stream._decode_stderr), 12000)
        self.assertTrue(stream._decode_stderr.endswith(b' encoder failed'))


class StdinPlaneCopyTests(unittest.TestCase):
    def test_generated_writer_copies_contiguous_and_padded_planes(self):
        script = svp_app.generate_vspipe_stdin_script(1920, 1080, 24, 1, 1, 60)
        writer = script.split('def write_plane', 1)[1].split('\ndef source_frame', 1)[0]
        namespace = {'ctypes': ctypes}
        exec('def write_plane' + writer, namespace)
        for stride in (4, 8):
            with self.subTest(stride=stride):
                storage = ctypes.create_string_buffer(b'_' * (stride * 3), stride * 3)
                frame = SimpleNamespace(
                    get_stride=lambda _plane: stride,
                    get_write_ptr=lambda _plane: ctypes.c_void_p(ctypes.addressof(storage)),
                )
                with patch.object(ctypes, 'memmove', wraps=ctypes.memmove) as copy:
                    namespace['write_plane'](frame, 0, b'abcdefghijkl', 4, 3)
                self.assertEqual(copy.call_count, 1 if stride == 4 else 3)
                self.assertEqual(storage.raw[:4], b'abcd')
                self.assertEqual(storage.raw[stride:stride + 4], b'efgh')
                self.assertEqual(storage.raw[2 * stride:2 * stride + 4], b'ijkl')
                if stride > 4:
                    self.assertEqual(storage.raw[4:stride], b'_' * (stride - 4))


if __name__ == '__main__':
    unittest.main()
