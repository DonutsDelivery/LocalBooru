import ctypes
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


class WindowsStderrTests(unittest.TestCase):
    def windows_readiness(self, peek):
        self.enterContext(patch.object(svp_app, '_is_windows', return_value=True))
        self.enterContext(patch.object(svp_app.select, 'select', side_effect=OSError('Windows pipes unsupported')))
        self.enterContext(patch.dict(sys.modules, {'msvcrt': SimpleNamespace(get_osfhandle=lambda _fd: 0x100000001)}))
        self.enterContext(patch.object(ctypes, 'WinDLL', return_value=SimpleNamespace(PeekNamedPipe=peek), create=True))

    def test_final_exited_process_keeps_error_tail_without_select(self):
        payload = b'FFmpeg synthetic fatal detail'
        pipe = io.BytesIO(payload)
        pipe.fileno = lambda: 42

        def peek(_handle, _buffer, _size, _read, available, _remaining):
            available._obj.value = len(payload) - pipe.tell()
            return 1

        self.windows_readiness(Mock(side_effect=peek))
        stream = object.__new__(svp_app.SVPStream)
        stream._decode_proc = SimpleNamespace(stderr=pipe, poll=lambda: 1)
        stream._vspipe_proc = stream._ffmpeg_proc = None
        stream._decode_stderr = b''
        stream._drain_stderr(final=True)
        self.assertEqual(stream._decode_stderr, payload)

    def test_live_pipe_with_no_ready_bytes_is_never_read(self):
        peek = Mock(return_value=1)  # available starts at zero
        self.windows_readiness(peek)
        pipe = Mock()
        stream = object.__new__(svp_app.SVPStream)
        stream._decode_proc = SimpleNamespace(stderr=pipe, poll=lambda: None)
        stream._vspipe_proc = stream._ffmpeg_proc = None
        stream._decode_stderr = b''
        stream._drain_stderr(final=True)
        pipe.read1.assert_not_called()
        self.assertEqual(stream._decode_stderr, b'')

    def test_failed_or_closed_windows_pipe_has_no_ready_bytes(self):
        self.windows_readiness(Mock(return_value=0))
        self.assertEqual(svp_app._stderr_ready_bytes(Mock()), 0)

    def test_windows_ready_size_is_bounded_and_preserves_64_bit_handle(self):
        def peek(handle, _buffer, _size, _read, available, _remaining):
            self.assertEqual(handle, 0x100000001)
            available._obj.value = 10000
            return 1

        self.windows_readiness(Mock(side_effect=peek))
        self.assertEqual(svp_app._stderr_ready_bytes(Mock()), 4096)


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
