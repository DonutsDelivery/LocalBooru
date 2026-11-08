import asyncio
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as svp_app


class HlsOutputArgsTests(unittest.TestCase):
    def test_segments_are_one_gop_cut_only_on_keyframes(self):
        args = svp_app.hls_output_args(60, '/synthetic/segment_%03d.ts', '/synthetic/stream.m3u8')
        self.assertEqual(args[args.index('-hls_time') + 1], '2')
        self.assertEqual(args[args.index('-g') + 1], '120')
        self.assertEqual(args[args.index('-keyint_min') + 1], '60')
        flags = args[args.index('-hls_flags') + 1].split('+')
        self.assertEqual(flags, ['append_list'])
        self.assertEqual(args[-1], '/synthetic/stream.m3u8')

    def test_gop_follows_target_fps_so_segment_equals_gop(self):
        for fps in (24, 30, 48, 60):
            args = svp_app.hls_output_args(fps, 's_%03d.ts', 'p.m3u8')
            gop_frames = int(args[args.index('-g') + 1])
            self.assertEqual(gop_frames / fps, svp_app.HLS_SEGMENT_SECONDS)


class BitrateCapTests(unittest.TestCase):
    def cap(self, height):
        args = svp_app.bitrate_cap_args(height)
        return args[args.index('-maxrate') + 1], args[args.index('-bufsize') + 1]

    def test_cap_tracks_output_height_at_1_5x_the_advertised_label(self):
        self.assertEqual(self.cap(480), ('6000k', '12000k'))
        self.assertEqual(self.cap(720), ('12000k', '24000k'))
        self.assertEqual(self.cap(1080), ('20000k', '40000k'))
        self.assertEqual(self.cap(1440), ('30000k', '60000k'))

    def test_heights_between_presets_use_the_next_ceiling(self):
        self.assertEqual(self.cap(360), ('6000k', '12000k'))
        self.assertEqual(self.cap(576), ('12000k', '24000k'))
        self.assertEqual(self.cap(1081), ('30000k', '60000k'))

    def test_oversized_output_has_a_finite_ceiling(self):
        self.assertEqual(self.cap(2160), ('45000k', '90000k'))


class PipelineHlsOutputTests(unittest.TestCase):
    def test_pipeline_emits_keyframe_aligned_two_second_segments_with_capped_bitrate(self):
        with tempfile.TemporaryDirectory(prefix='dmc-synthetic-svp-hls-') as root:
            source = Path(root) / 'synthetic.mp4'
            generated = subprocess.run([
                'ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=1280x720:rate=60', '-t', '7', '-an',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '1', str(source),
            ], capture_output=True, timeout=30)
            self.assertEqual(generated.returncode, 0, generated.stderr.decode(errors='replace'))

            commands = []
            real_popen = subprocess.Popen

            def recording_popen(command, *args, **kwargs):
                commands.append(list(command))
                return real_popen(command, *args, **kwargs)

            def first_video_packet_flags(segment):
                probed = subprocess.run([
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'packet=flags', '-of', 'csv=p=0', str(segment),
                ], capture_output=True, timeout=10)
                return probed.stdout.decode().splitlines()[0]

            async def run():
                stream = svp_app.SVPStream(str(source), target_fps=60, target_resolution=(854, 480))
                try:
                    self.assertTrue(await stream.start(), stream.error)
                    self.assertTrue(await stream.wait_for_ready(timeout=15), stream.error)
                    for _ in range(100):
                        if '#EXT-X-ENDLIST' in stream.playlist_path.read_text():
                            break
                        await asyncio.sleep(0.3)
                    playlist = stream.playlist_path.read_text()
                    segments = re.findall(r'^(segment_\d+\.ts)$', playlist, re.MULTILINE)
                    return playlist, [first_video_packet_flags(stream.hls_dir / name) for name in segments]
                finally:
                    await stream.stop_async()

            with patch.object(svp_app.subprocess, 'Popen', side_effect=recording_popen):
                playlist, first_packet_flags = asyncio.run(run())

            durations = [float(v) for v in re.findall(r'#EXTINF:([\d.]+)', playlist)]
            self.assertGreaterEqual(len(durations), 3, playlist)
            for duration in durations[:-1]:
                self.assertAlmostEqual(duration, 2.0, places=2, msg=playlist)
            self.assertIn('#EXT-X-TARGETDURATION:2', playlist)
            # Every segment must open on a keyframe so it decodes standalone.
            self.assertEqual(len(first_packet_flags), len(durations))
            for flags in first_packet_flags:
                self.assertTrue(flags.startswith('K'), first_packet_flags)

            encode = next(c for c in commands if '-hls_time' in c)
            self.assertEqual(encode[encode.index('-maxrate') + 1], '6000k')
            self.assertEqual(encode[encode.index('-bufsize') + 1], '12000k')
            self.assertNotIn('split_by_time', ' '.join(encode))


if __name__ == '__main__':
    unittest.main()
