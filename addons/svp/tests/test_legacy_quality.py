import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as svp_app


class LegacyQualityTests(unittest.TestCase):
    def setUp(self):
        self.info = {
            'success': True, 'width': 1280, 'height': 720, 'src_fps': 60,
            'duration': 5, 'src_fps_num': 60, 'src_fps_den': 1,
            'num_frames': 300, 'has_audio': False,
        }

    def test_matching_cadence_still_prepares_requested_lower_resolution(self):
        with tempfile.TemporaryDirectory(prefix='dmc-synthetic-svp-quality-') as root, \
                patch.object(svp_app, 'check_nvenc', return_value=False), \
                patch.object(svp_app, 'get_video_info', return_value=self.info), \
                patch.object(svp_app.tempfile, 'mkdtemp', return_value=root):
            stream = svp_app.SVPStream('/synthetic/video.mp4', target_resolution=(854, 480))
            try:
                script = stream._prepare()
                self.assertIsNotNone(script)
                self.assertIsNone(stream.error)
                text = script.read_text()
                self.assertIn('WIDTH = 854', text)
                self.assertIn('HEIGHT = 480', text)
                self.assertIn('FPS_NUM = 60', text)
                self.assertIn('smooth = clip', text)
                self.assertNotIn('core.svp2.SmoothFps(', text)
            finally:
                stream.stop()

    def test_matching_cadence_without_quality_change_still_skips_interpolation(self):
        with patch.object(svp_app, 'check_nvenc', return_value=False), \
                patch.object(svp_app, 'get_video_info', return_value=self.info):
            stream = svp_app.SVPStream('/synthetic/video.mp4')
            try:
                self.assertIsNone(stream._prepare())
                self.assertIn('already near target', stream.error)
            finally:
                stream.stop()

    def test_requested_downscale_produces_hls_at_matching_cadence(self):
        with tempfile.TemporaryDirectory(prefix='dmc-synthetic-svp-cfr-') as root:
            source = Path(root) / 'synthetic.mp4'
            generated = subprocess.run([
                'ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=1280x720:rate=60', '-t', '5', '-an',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', str(source),
            ], capture_output=True, timeout=30)
            self.assertEqual(generated.returncode, 0, generated.stderr.decode(errors='replace'))

            async def run():
                stream = svp_app.SVPStream(str(source), target_fps=60, target_resolution=(854, 480))
                try:
                    self.assertTrue(await stream.start(), stream.error)
                    self.assertTrue(await stream.wait_for_ready(timeout=15), stream.error)
                    probed = subprocess.run([
                        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height,avg_frame_rate', '-of', 'json',
                        str(stream.hls_dir / 'segment_000.ts'),
                    ], capture_output=True, timeout=10)
                    self.assertEqual(probed.returncode, 0, probed.stderr.decode(errors='replace'))
                    info = json.loads(probed.stdout)['streams'][0]
                    self.assertEqual((info['width'], info['height']), (854, 480))
                    self.assertEqual(info['avg_frame_rate'], '60/1')
                finally:
                    await stream.stop_async()

            asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
