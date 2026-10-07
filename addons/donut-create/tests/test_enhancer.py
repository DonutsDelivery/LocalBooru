"""Managed enhancer checks use disposable files and synthetic briefs only."""
import asyncio
import hashlib
import io
import json
import os
import tarfile
import urllib.error
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

import enhancer as module
from enhancer import Enhancer, parse_result, messages
from installer import SetupCancelled, SetupError
from test_installer import Opener


class OutputTests(unittest.TestCase):
    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    def test_both_inputs_are_distinct_and_valid_json_is_required(self):
        inputs = messages("A clockwork owl.", "The owl looks left, eyes open.")
        self.assertEqual(json.loads(inputs[1]["content"]), {
            "image_brief": "A clockwork owl.", "face_brief": "The owl looks left, eyes open."})
        result = {"main_prompt": "A clockwork owl looks left, eyes open.", "face_prompt": "The owl looks left, eyes open."}
        self.assertEqual(parse_result(json.dumps(result), "stop", True), result)

    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    def test_expanded_scene_and_identity_only_face_are_not_shortened_or_combined(self):
        result = {
            "main_prompt": "An owl beside a wooden cabin, with visible wood grain and softly lit timber textures, gentle natural light and a quiet atmosphere, photographed with a natural perspective.",
            "face_prompt": "The owl's face.",
        }
        self.assertEqual(parse_result(json.dumps(result), "stop", True), result)
        self.assertEqual(parse_result('<think>synthetic reasoning</think>\n' + json.dumps(result), "stop", True), result)

    # AC: @create-managed-prompt-enhancer ac-owned-private
    def test_malformed_truncated_and_empty_supplied_face_are_rejected(self):
        for text, finish, face in [('{"main_prompt":"owl"}', "stop", False),
                                   ('```json\n{}\n```', "stop", False),
                                   ('{"main_prompt":"owl","face_prompt":""}', "length", False),
                                   ('{"main_prompt":"owl","face_prompt":""}', "stop", True),
                                   ('{"main_prompt":"%load:/secret%","face_prompt":""}', "stop", False)]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_result(text, finish, face)


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dmc-enhancer-test-")
        self.enhancer = Enhancer(Path(self.temporary.name))

    async def asyncTearDown(self):
        await self.enhancer.close()
        self.temporary.cleanup()

    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_immediate_cancel_before_task_starts(self):
        with patch.object(self.enhancer, 'ready', return_value=True), patch.object(self.enhancer, 'infer', new=AsyncMock()) as infer:
            job = self.enhancer.start('owner', 'enhance', 'minicpm', 'Synthetic owl.', '')
            response = await self.enhancer.cancel('owner', job['id'])
            self.assertEqual(response['job']['state'], 'cancelled')
            infer.assert_not_awaited()

    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_other_studio_cannot_read_or_cancel_job(self):
        gate = asyncio.Event()
        async def inference(*args):
            await gate.wait()
            return {"main_prompt": "Synthetic owl.", "face_prompt": ""}
        with patch.object(self.enhancer, "ready", return_value=True), patch.object(self.enhancer, "infer", side_effect=inference):
            job = self.enhancer.start("owner", "enhance", "minicpm", "Synthetic owl.", "")
            await asyncio.sleep(0)
            foreign = self.enhancer.status("other")
            self.assertNotIn("job", foreign)
            self.assertTrue(foreign["busy"])
            with self.assertRaises(PermissionError):
                await self.enhancer.cancel("other", job["id"])
            await self.enhancer.cancel("owner", job["id"])
            self.assertEqual(self.enhancer.status("owner")["job"]["state"], "cancelled")
            self.assertFalse(list(Path(self.temporary.name).rglob("*.json")))

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_completion_and_shutdown_release_owned_work(self):
        with patch.object(self.enhancer, "ready", return_value=True), patch.object(self.enhancer, "infer", new=AsyncMock(return_value={"main_prompt": "Synthetic owl.", "face_prompt": ""})):
            self.enhancer.start("owner", "enhance", "minicpm", "Synthetic owl.", "")
            await self.enhancer.task
            self.assertEqual(self.enhancer.status("owner")["job"]["result"]["main_prompt"], "Synthetic owl.")
            self.assertFalse(self.enhancer.status("other")["busy"])
            await self.enhancer.close()
            self.assertIsNone(self.enhancer.process)

    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_cancel_during_native_launch_reaps_the_owned_child(self):
        spawned = asyncio.Event()
        release = asyncio.Event()
        process = unittest.mock.Mock(returncode=None)
        async def wait():
            process.returncode = 0
        process.wait = AsyncMock(side_effect=wait)
        async def spawn(*args, **kwargs):
            spawned.set()
            await release.wait()
            return process
        with patch.object(module, '_read_json', return_value={'server': 'llama-server'}), patch.object(module.asyncio, 'create_subprocess_exec', side_effect=spawn):
            task = asyncio.create_task(self.enhancer.infer('minicpm', 'Synthetic owl.', ''))
            await spawned.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        process.terminate.assert_called_once()
        self.assertIsNone(self.enhancer.process)

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    async def test_inference_contacts_only_owned_loopback_and_rejects_context_overflow(self):
        client_type = httpx.AsyncClient
        for token_count in (5, 16384):
            requests = []
            process = unittest.mock.Mock(returncode=None)
            process.wait = AsyncMock(return_value=0)
            def respond(request):
                requests.append(request)
                self.assertEqual(request.url.host, '127.0.0.1')
                self.assertTrue(request.headers['Authorization'].startswith('Bearer '))
                route = request.url.path
                data = {} if request.method == 'GET' else json.loads(request.content)
                if route == '/health':
                    return httpx.Response(200, json={})
                if route == '/apply-template':
                    self.assertEqual(json.loads(data['messages'][1]['content'])['face_brief'], 'The owl looks left.')
                    return httpx.Response(200, json={'prompt': 'synthetic formatted chat'})
                if route == '/tokenize':
                    return httpx.Response(200, json={'tokens': [1] * token_count})
                self.assertEqual(route, '/v1/chat/completions')
                return httpx.Response(200, json={'choices': [{'finish_reason':'stop','message':{'content':json.dumps({'main_prompt':'A synthetic owl looks left.','face_prompt':'The owl looks left.'})}}]})
            def client(**options):
                options['transport'] = httpx.MockTransport(respond)
                return client_type(**options)
            with patch.dict(os.environ, {'LLAMA_ARG_RPC': 'unselected.invalid:1234', 'LLAMA_ARG_LOG_FILE': '/synthetic/log', 'HF_TOKEN': 'synthetic-secret'}), patch.object(module, '_read_json', return_value={'server':'llama-server'}), patch.object(module.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=process)) as spawn, patch.object(module.httpx, 'AsyncClient', side_effect=client):
                if token_count == 5:
                    result = await self.enhancer.infer('minicpm', 'Synthetic owl.', 'The owl looks left.')
                    self.assertEqual(result['face_prompt'], 'The owl looks left.')
                else:
                    with self.assertRaisesRegex(ValueError, 'context'):
                        await self.enhancer.infer('minicpm', 'Synthetic owl.', 'The owl looks left.')
                    self.assertNotIn('/v1/chat/completions', [r.url.path for r in requests])
                environment = spawn.call_args.kwargs['env']
                self.assertFalse(any(name.startswith('LLAMA_') for name in environment))
                self.assertNotIn('HF_TOKEN', environment)
            process.terminate.assert_called_once()
            self.assertIsNone(self.enhancer.process)

    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_request_preserves_full_briefs_and_returns_exact_model_output(self):
        image = 'An owl beside a wooden cabin, no other animals, dim blue light. Literal trigger SYNTH_OWL and the sign says "Keep out—".'
        face = "The owl's face."
        output = {"main_prompt": image + ' Fine wood grain and soft feather textures remain visible in the dim light.', "face_prompt": face}
        client_type = httpx.AsyncClient
        conversations = []
        process = unittest.mock.Mock(returncode=None)
        process.wait = AsyncMock(return_value=0)
        def respond(request):
            data = {} if request.method == 'GET' else json.loads(request.content)
            if request.url.path == '/health':
                return httpx.Response(200, json={})
            if request.url.path == '/apply-template':
                conversations.append(data['messages'])
                return httpx.Response(200, json={'prompt': 'synthetic formatted chat'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1] * 20})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            conversations.append(data['messages'])
            self.assertFalse(data['cache_prompt'])
            self.assertFalse(data['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(data['response_format']['json_schema']['schema']['required'], ['main_prompt', 'face_prompt'])
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]})
        def client(**options):
            options['transport'] = httpx.MockTransport(respond)
            return client_type(**options)
        with patch.object(module, '_read_json', return_value={'server': 'llama-server'}), patch.object(module.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=process)), patch.object(module.httpx, 'AsyncClient', side_effect=client):
            result = await self.enhancer.infer('qwen', image, face)
        self.assertEqual(result, output)
        self.assertEqual(conversations, [messages(image, face), messages(image, face)])
        for conversation in conversations:
            self.assertEqual(json.loads(conversation[1]['content']), {'image_brief': image, 'face_brief': face})
        self.assertIsNone(self.enhancer.process)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dmc-enhancer-download-')
        self.enhancer = Enhancer(Path(self.temporary.name))
        self.enhancer.state_dir.mkdir()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            member = tarfile.TarInfo('bin/llama-server')
            member.mode = 0o755
            data = b'synthetic executable'
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        self.archive = stream.getvalue()
        self.weights = b'synthetic gguf weights, long enough for partial downloads'
        def item(data, filename):
            return {'filename': filename, 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data), 'url': 'https://synthetic.invalid/' + filename}
        self.runtime = item(self.archive, 'cpu.tar.gz')
        self.model = {**item(self.weights, 'synthetic.gguf'), 'name': 'Synthetic', 'quantization': 'Q6_K'}
        self.catalog = patch.dict(module.CATALOG, models={'minicpm': self.model})
        self.catalog.start()
        self.enhancer.runtime = self.runtime
        owner = self
        class Downloads:
            def open(self, request, timeout):
                return owner.runtime_opener.open(request, timeout) if request.full_url.endswith('.tar.gz') else owner.model_opener.open(request, timeout)
        self.runtime_opener = Opener(self.archive)
        self.model_opener = Opener(self.weights)
        self.enhancer._opener = Downloads()

    def tearDown(self):
        self.catalog.stop()
        self.temporary.cleanup()

    # AC: @create-managed-prompt-enhancer ac-managed-models
    def test_verified_activation_resume_and_offline_repair(self):
        self.model_opener.callback = self.enhancer._cancel.set
        with self.assertRaises(SetupCancelled):
            self.enhancer.install('minicpm')
        self.assertFalse(self.enhancer.ready('minicpm'))
        self.assertFalse(self.enhancer.model_path('minicpm').exists())
        self.enhancer._cancel.clear()
        self.model_opener.callback = None
        self.enhancer.install('minicpm')
        self.assertTrue(self.enhancer.ready('minicpm'))
        self.assertEqual(self.enhancer.model_path('minicpm').read_bytes(), self.weights)
        self.assertEqual(self.model_opener.requests[-1].get_header('Range'), 'bytes=8-')
        server = self.enhancer.runtime_root / 'bin/llama-server'
        if os.name != 'nt':
            self.assertTrue(os.access(server, os.X_OK))
        requests = len(self.runtime_opener.requests) + len(self.model_opener.requests)
        self.enhancer.install('minicpm')
        self.assertEqual(len(self.runtime_opener.requests) + len(self.model_opener.requests), requests)
        server.write_bytes(b'changed')
        self.assertFalse(self.enhancer.ready('minicpm'))

    # AC: @create-managed-prompt-enhancer ac-managed-models
    def test_invalid_checksum_never_activates_weights(self):
        self.model_opener = Opener(b'x' * len(self.weights))
        with self.assertRaises(SetupError):
            self.enhancer.install('minicpm')
        self.assertFalse(self.enhancer.ready('minicpm'))
        self.assertFalse(self.enhancer.model_path('minicpm').exists())

    # AC: @create-managed-prompt-enhancer ac-managed-models
    def test_archive_cannot_escape_owned_runtime(self):
        archive = Path(self.temporary.name) / 'unsafe.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            member = tarfile.TarInfo('../escaped')
            member.size = 3
            bundle.addfile(member, io.BytesIO(b'bad'))
        with self.assertRaises(SetupError):
            self.enhancer.unpack(archive, self.enhancer.runtime_root)
        self.assertFalse((Path(self.temporary.name) / 'escaped').exists())

    # AC: @create-managed-prompt-enhancer ac-managed-models
    def test_low_disk_stops_before_download_and_provider_errors_are_actionable(self):
        with patch.object(module.shutil, 'disk_usage', return_value=unittest.mock.Mock(free=0)):
            with self.assertRaisesRegex(SetupError, 'free disk space'):
                self.enhancer.install('minicpm')
        self.assertFalse(self.model_opener.requests)
        self.assertFalse(self.runtime_opener.requests)
        self.enhancer._opener = unittest.mock.Mock()
        self.enhancer._opener.open.side_effect = urllib.error.HTTPError(self.runtime['url'], 403, 'denied', {}, None)
        with self.assertRaisesRegex(SetupError, 'download provider'):
            self.enhancer.install('minicpm')


if __name__ == "__main__":
    unittest.main()
