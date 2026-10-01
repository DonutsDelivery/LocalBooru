import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as svp_app


class NvencAvailabilityTests(unittest.TestCase):
    def tearDown(self):
        for stream in list(svp_app._active_streams.values()):
            stream.stop()

    def test_unusable_nvenc_selects_software_encoder(self):
        # Compiled encoder listings do not establish device/driver availability.
        failed = subprocess.CompletedProcess([], 1, b'', b'Cannot load libcuda.so.1')
        with patch.object(svp_app, 'get_ffmpeg_path', return_value='/synthetic/ffmpeg'), \
                patch.object(svp_app.subprocess, 'run', return_value=failed) as run:
            stream = svp_app.SVPStream('/synthetic/video.mp4')
        self.assertFalse(stream.use_nvenc)
        command = run.call_args.args[0]
        self.assertIn('color=size=64x64:rate=1', command)
        self.assertIn('h264_nvenc', command)
        self.assertNotIn('-encoders', command)

    def test_working_encoder_remains_enabled(self):
        with patch.object(svp_app, 'get_ffmpeg_path', return_value='/synthetic/ffmpeg'), \
                patch.object(svp_app.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)):
            self.assertTrue(svp_app.check_nvenc())

    def test_probe_timeout_selects_software(self):
        with patch.object(svp_app, 'get_ffmpeg_path', return_value='/synthetic/ffmpeg'), \
                patch.object(svp_app.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffmpeg', 5)):
            self.assertFalse(svp_app.check_nvenc())


if __name__ == '__main__':
    unittest.main()
