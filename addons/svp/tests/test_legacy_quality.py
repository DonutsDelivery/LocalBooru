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


class LegacyDownscaleBeforeSvpTests(unittest.TestCase):
    def info(self, width, height, fps=30):
        return {
            'success': True, 'width': width, 'height': height, 'src_fps': fps,
            'duration': 5, 'src_fps_num': fps, 'src_fps_den': 1,
            'num_frames': fps * 5, 'has_audio': False,
        }

    def prepare(self, width, height, target_resolution):
        with tempfile.TemporaryDirectory(prefix='dmc-synthetic-svp-downscale-') as root, \
                patch.object(svp_app, 'check_nvenc', return_value=False), \
                patch.object(svp_app, 'get_video_info', return_value=self.info(width, height)), \
                patch.object(svp_app.tempfile, 'mkdtemp', return_value=root):
            stream = svp_app.SVPStream(
                '/synthetic/video.mp4', target_fps=60, target_resolution=target_resolution)
            try:
                script = stream._prepare()
                self.assertIsNotNone(script)
                return stream, script.read_text(), stream._decode_command()
            finally:
                stream.stop()

    def test_4k_source_is_scaled_before_the_raw_pipe_so_svp_sees_reduced_frames(self):
        stream, script, decode = self.prepare(3840, 2160, (1280, 720))
        self.assertIn('WIDTH = 1280', script)
        self.assertIn('HEIGHT = 720', script)
        self.assertNotIn('WIDTH = 3840', script)
        # Motion analysis runs on the reduced frame size, not the 4K source.
        self.assertIn('core.svp2.SmoothFps_NVOF(', script)
        self.assertIn('src_width=1280', script)
        self.assertEqual(decode[decode.index('-vf') + 1], 'scale=1280:720:flags=lanczos')
        # The scaler is an output option of the decode stage, ahead of the rawvideo pipe.
        self.assertLess(decode.index('-i'), decode.index('-vf'))
        self.assertLess(decode.index('-vf'), decode.index('rawvideo'))

    def test_downscale_keeps_aspect_ratio_inside_the_requested_box(self):
        stream, script, decode = self.prepare(2880, 2160, (854, 480))
        self.assertIn('WIDTH = 640', script)
        self.assertIn('HEIGHT = 480', script)
        self.assertIn('scale=640:480', ' '.join(decode))

    def test_source_inside_the_requested_box_is_never_upscaled(self):
        stream, script, decode = self.prepare(1280, 720, (1920, 1080))
        self.assertIn('WIDTH = 1280', script)
        self.assertIn('HEIGHT = 720', script)
        self.assertNotIn('-vf', decode)

    def test_original_quality_keeps_source_resolution_without_a_scaler(self):
        stream, script, decode = self.prepare(3840, 2160, None)
        self.assertIn('WIDTH = 3840', script)
        self.assertIn('HEIGHT = 2160', script)
        self.assertNotIn('-vf', decode)

    def test_fit_resolution_returns_even_dimensions_within_bounds(self):
        self.assertEqual(svp_app.fit_resolution(3840, 2160, (1280, 720)), (1280, 720))
        self.assertEqual(svp_app.fit_resolution(1280, 720, (854, 480)), (854, 480))
        self.assertEqual(svp_app.fit_resolution(2160, 3840, (1280, 720)), (404, 720))
        self.assertEqual(svp_app.fit_resolution(640, 360, (3840, 2160)), (640, 360))
        self.assertEqual(svp_app.fit_resolution(1281, 721, (3840, 2160)), (1280, 720))


if __name__ == '__main__':
    unittest.main()
