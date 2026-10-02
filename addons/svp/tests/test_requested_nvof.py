import io
import sys
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as svp_app


class Node:
    def __init__(self, width=1920, height=1080):
        self.width, self.height = width, height
        self.fps = Fraction(24, 1)
        self.output = False

    def set_output(self):
        self.output = True


class RequestedNvofTests(unittest.TestCase):
    def execute(self, *, available=True, error=None, **options):
        calls = []
        clip = Node(options.pop('width', 1920), options.pop('height', 1080))
        result = Node()

        def super_call(source, params):
            calls.append(('super', params))
            return {'clip': source, 'data': 'super-data'}

        def analyse(*args):
            calls.append(('analyse', args[-1]))
            return {'clip': clip, 'data': 'vectors-data'}

        def smooth(*args, **kwargs):
            calls.append(('classic', args[-1], kwargs))
            return result

        def nvof(source, params, **kwargs):
            calls.append(('nvof', params, kwargs))
            if error:
                raise RuntimeError(error)
            return result

        def resize(source, **kwargs):
            calls.append(('resize', kwargs))
            return Node(kwargs['width'], kwargs['height'])

        svp2 = types.SimpleNamespace(SmoothFps=smooth)
        if available:
            svp2.SmoothFps_NVOF = nvof
        core = types.SimpleNamespace(
            std=types.SimpleNamespace(LoadPlugin=lambda _: None,
                BlankClip=lambda **_: clip, ModifyFrame=lambda *args: clip),
            resize=types.SimpleNamespace(Bicubic=resize),
            svp1=types.SimpleNamespace(Super=super_call, Analyse=analyse), svp2=svp2)
        module = types.SimpleNamespace(core=core, YUV420P8=1)
        script = svp_app.generate_vspipe_stdin_script(
            clip.width, clip.height, 24, 1, 288, 60, **options)
        with patch.dict(sys.modules, {'vapoursynth': module}), \
                patch.object(sys, 'stdin', types.SimpleNamespace(buffer=io.BytesIO())):
            exec(compile(script, '<synthetic-svp>', 'exec'), {})
        self.assertTrue(result.output or clip.output)
        return calls

    # AC: @svp-platform-routing ac-nondesktop-route
    def test_requested_nvof_uses_real_engine_and_balanced_grid(self):
        calls = self.execute()
        self.assertEqual([c[0] for c in calls], ['resize', 'nvof'])
        self.assertEqual(calls[0][1], {'width': 480, 'height': 268,
            'src_width': 1920, 'src_height': 1072})
        self.assertIn('rate:{num:60,den:1,abs:true}', calls[1][1])
        self.assertIn('algo:23,mask:{area:100}', calls[1][1])
        self.assertEqual(calls[1][2]['fps'], 24)
        self.assertEqual((calls[1][2]['src'].width, calls[1][2]['src'].height), (1920, 1080))

    def test_each_selected_motion_grid_preserves_shader_and_mask(self):
        for preset, block in [('fast', 32), ('balanced', 16), ('quality', 8),
                              ('max', 8), ('animation', 32), ('film', 16)]:
            with self.subTest(preset=preset):
                calls = self.execute(preset=preset, shader=13, artifact_masking=50)
                self.assertEqual(calls[0][1]['width'], 1920 // block * 4)
                self.assertEqual(calls[0][1]['height'], 1080 // block * 4)
                self.assertIn('algo:13,mask:{area:50}', calls[1][1])

    def test_small_source_never_samples_beyond_its_dimensions(self):
        calls = self.execute(width=8, height=6)
        self.assertEqual(calls[0][1], {'width': 8, 'height': 6,
                                     'src_width': 8, 'src_height': 6})

    def test_disabled_nvof_preserves_classic_cpu_path(self):
        calls = self.execute(use_nvof=False)
        self.assertEqual([c[0] for c in calls], ['super', 'analyse', 'classic'])
        self.assertIn('gpu:0', calls[0][1])
        self.assertIn('gpu:0', calls[1][1])
        self.assertNotIn('gpuid:', calls[2][1])

    def test_custom_parameters_still_execute_classic_exact_contract(self):
        for name, value in [('custom_super', '{gpu:1,pel:4}'),
                            ('custom_analyse', '{gpu:1,block:{w:8,h:8}}'),
                            ('custom_smooth', '{algo:21,rate:{num:48,den:1}}')]:
            with self.subTest(name=name):
                calls = self.execute(**{name: value})
                self.assertEqual([c[0] for c in calls], ['super', 'analyse', 'classic'])
                index = {'custom_super': 0, 'custom_analyse': 1, 'custom_smooth': 2}[name]
                self.assertEqual(calls[index][1], value)

    def test_unavailable_plugin_function_has_explicit_fallback_error(self):
        with self.assertRaisesRegex(RuntimeError, 'NVOF unavailable'):
            self.execute(available=False)

    def test_gpu_initialization_failure_has_explicit_fallback_error(self):
        with self.assertRaisesRegex(RuntimeError, 'NVOF unavailable: synthetic GPU failure'):
            self.execute(error='synthetic GPU failure')

    def test_manager_graph_is_not_replaced(self):
        script = 'smooth = clip\n'
        # Exercise actual generated graph rather than inspecting source text.
        calls = self.execute(graph_script=script)
        self.assertEqual(calls, [])

    def test_nvof_failure_classifier_is_narrow(self):
        for error in ['NVOF unavailable: API missing',
                      'SVSmoothFps: unable to init NVOF - code 5',
                      'SVSmoothFps: NVOF runtime error 2',
                      'unable to init GPU-based renderer']:
            self.assertTrue(svp_app.is_gpu_renderer_error(error))
        self.assertFalse(svp_app.is_gpu_renderer_error('file NVOF.mp4 not found'))


class RequestedNvofFallbackTests(unittest.IsolatedAsyncioTestCase):
    # AC: @svp-platform-routing ac-nondesktop-route
    async def test_requested_engine_failure_retries_cpu_with_explicit_receipt(self):
        streams = []

        class Stream:
            def __init__(self, **options):
                self.use_nvof = options['use_nvof']
                self.error = 'NVOF unavailable: synthetic unsupported GPU' if self.use_nvof else None
                self.stream_id = 'synthetic-fallback'
                self._duration, self._width, self._height = 12, 1920, 1080
                self.stopped = False
                streams.append(self)

            async def start(self):
                return True

            async def wait_for_ready(self, timeout):
                return not self.use_nvof

            def stop(self):
                self.stopped = True

        async def json_request():
            return {'file_path': '/synthetic/video.mp4', 'use_nvof': True}

        with patch.object(svp_app, 'stop_all_streams', return_value=None), \
                patch.object(svp_app.os.path, 'exists', return_value=True), \
                patch.object(svp_app, 'check_vspipe', return_value=True), \
                patch.object(svp_app, 'get_ffmpeg_path', return_value='/synthetic/ffmpeg'), \
                patch.object(svp_app, 'check_svp_plugins', return_value=True), \
                patch.object(svp_app, 'SVPStream', Stream):
            receipt = await svp_app.play(types.SimpleNamespace(json=json_request))
        self.assertEqual([s.use_nvof for s in streams], [True, False])
        self.assertTrue(streams[0].stopped)
        self.assertTrue(receipt['success'])
        self.assertTrue(receipt['nvof_fallback'])
        self.assertEqual(receipt['stream_id'], 'synthetic-fallback')


if __name__ == '__main__':
    unittest.main()
