"""Resident helper lifecycle uses synthetic HTTP replies and owned fake children."""
import asyncio
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

import enhancer as module
from enhancer import Enhancer, inference_seed, messages


class ResidentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='dmc-enhancer-resident-')
        self.enhancer = Enhancer(Path(self.temporary.name))
        self.processes = []
        self.commands = []
        self.requests = []
        self.failure = None
        self.health_status = 200
        self.completion_gate = None
        self.completion_started = asyncio.Event()
        client_type = httpx.AsyncClient

        async def respond(request):
            self.requests.append(request)
            if request.url.path == '/health':
                return httpx.Response(self.health_status, json={})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': 'synthetic chat'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1, 2]})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            self.completion_started.set()
            if self.completion_gate:
                await self.completion_gate.wait()
            if self.failure == 'http':
                return httpx.Response(500)
            if self.failure == 'malformed':
                return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'invalid'}}]})
            data = json.loads(request.content)
            brief = json.loads(data['messages'][1]['content'])
            result = {'main_prompt': brief['image_brief'], 'face_prompt': brief['face_brief']}
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]})

        def client(**options):
            self.assertFalse(options['trust_env'])
            options['transport'] = httpx.MockTransport(respond)
            return client_type(**options)

        async def spawn(*args, **options):
            # Switching must reap the previous helper before starting another.
            self.assertTrue(all(process.returncode is not None for process in self.processes))
            self.commands.append((args, options))
            process = Mock(returncode=None)
            async def wait():
                process.returncode = 0
                return 0
            process.wait = AsyncMock(side_effect=wait)
            self.processes.append(process)
            return process

        self.native_spawn = spawn
        self.patches = [
            patch.object(self.enhancer, 'ready', return_value=True),
            patch.object(module, '_read_json', return_value={'server': 'llama-server'}),
            patch.object(module.asyncio, 'create_subprocess_exec', side_effect=spawn),
            patch.object(module.httpx, 'AsyncClient', side_effect=client),
        ]
        for replacement in self.patches:
            replacement.start()

    async def asyncTearDown(self):
        await self.enhancer.close()
        for replacement in reversed(self.patches):
            replacement.stop()
        self.assertTrue(all(process.returncode is not None for process in self.processes))
        self.temporary.cleanup()

    async def run_job(self, owner, model='qwen', brief='Synthetic owl.', seed=0):
        job = self.enhancer.start(owner, 'enhance', model, brief, '', seed)
        await self.enhancer.task
        return job

    # AC: @create-managed-prompt-enhancer ac-resident-model
    # AC: @create-managed-prompt-enhancer ac-per-run-seed
    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_same_model_reuses_one_cpu_helper_with_private_seeded_requests(self):
        seeds = (42, 42, 43, 2**53 - 1)
        briefs = ('Synthetic owl.', 'Synthetic owl.', 'Synthetic fox.', 'Synthetic cabin.')
        with patch.dict(os.environ, {'LLAMA_ARG_RPC': 'synthetic.invalid:1234', 'HF_TOKEN': 'synthetic-token'}):
            for index, (brief, seed) in enumerate(zip(briefs, seeds)):
                owner = 'owner' if index < 2 else 'other'
                await self.run_job(owner, brief=brief, seed=seed)
                self.assertEqual(self.enhancer.status(owner)['job']['result']['main_prompt'], brief)
                self.assertEqual(self.enhancer.status(owner)['loaded_model'], 'qwen')
        self.assertEqual(len(self.processes), 1)
        self.processes[0].terminate.assert_not_called()
        completions = [request for request in self.requests if request.url.path == '/v1/chat/completions']
        payloads = [json.loads(request.content) for request in completions]
        self.assertEqual([data['seed'] for data in payloads], [inference_seed(seed) for seed in seeds])
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual([data['messages'] for data in payloads], [messages(brief, '') for brief in briefs])
        self.assertTrue(all(data['cache_prompt'] is False for data in payloads))
        self.assertEqual(len({str(request.url) for request in completions}), 1)
        self.assertEqual(len({request.headers['Authorization'] for request in self.requests}), 1)
        self.assertTrue(all(request.url.host == '127.0.0.1' for request in self.requests))
        command, options = self.commands[0]
        for flag, value in (('--host', '127.0.0.1'), ('--n-gpu-layers', '0'), ('--device', 'none'), ('--parallel', '1')):
            self.assertEqual(command[command.index(flag) + 1], value)
        for flag in ('--no-op-offload', '--no-kv-offload', '--log-disable'):
            self.assertIn(flag, command)
        self.assertNotIn('LLAMA_ARG_RPC', options['env'])
        self.assertNotIn('HF_TOKEN', options['env'])
        self.assertEqual(options['stdout'], asyncio.subprocess.DEVNULL)
        self.assertEqual(options['stderr'], asyncio.subprocess.DEVNULL)
        foreign = self.enhancer.status('third')
        self.assertNotIn('job', foreign)
        self.assertNotIn('Synthetic', json.dumps(foreign))

    # AC: @create-managed-prompt-enhancer ac-resident-model
    async def test_switch_reaps_old_model_before_spawning_new_helper(self):
        await self.run_job('owner')
        await self.run_job('owner', 'minicpm')
        self.assertEqual(len(self.processes), 2)
        self.processes[0].terminate.assert_called_once()
        self.processes[1].terminate.assert_not_called()
        self.assertEqual(self.enhancer.status('owner')['loaded_model'], 'minicpm')
        keys = [command[command.index('--api-key') + 1] for command, _ in self.commands]
        self.assertNotEqual(keys[0], keys[1])

    # AC: @create-managed-prompt-enhancer ac-resident-model
    async def test_crashed_helper_is_not_reported_loaded_or_reused(self):
        await self.run_job('owner')
        self.processes[0].returncode = 1
        self.assertIsNone(self.enhancer.status('owner')['loaded_model'])
        await self.run_job('owner')
        self.assertEqual(len(self.processes), 2)
        self.assertEqual(self.enhancer.status('owner')['loaded_model'], 'qwen')

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_repair_stops_helper_before_touching_runtime_files(self):
        await self.run_job('owner')
        def install(model):
            self.assertIsNone(self.enhancer.process)
            self.assertEqual(self.processes[0].returncode, 0)
        with patch.object(self.enhancer, 'install', side_effect=install) as repair:
            self.enhancer.start('owner', 'install', 'qwen')
            await self.enhancer.task
        repair.assert_called_once_with('qwen')
        self.assertEqual(self.enhancer.status('owner')['job']['state'], 'complete')
        self.assertIsNone(self.enhancer.status('owner')['loaded_model'])
        await self.run_job('owner')
        self.assertEqual(len(self.processes), 2)

    # AC: @create-managed-prompt-enhancer ac-resident-model
    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_failure_discards_helper_and_retry_starts_fresh(self):
        for failure in ('http', 'malformed'):
            with self.subTest(failure=failure):
                await self.run_job('owner')
                failed_process = self.processes[-1]
                self.failure = failure
                await self.run_job('owner', brief='Synthetic failed brief.')
                job = self.enhancer.status('owner')['job']
                self.assertEqual(job['state'], 'error')
                self.assertNotIn('result', job)
                self.assertNotIn('Synthetic failed brief.', job['error'])
                self.assertIsNone(self.enhancer.process)
                self.assertIsNone(self.enhancer.status('owner')['loaded_model'])
                failed_process.terminate.assert_called_once()
                self.failure = None
        await self.run_job('owner')
        self.assertEqual(len(self.processes), 3)

    # AC: @create-managed-prompt-enhancer ac-resident-model
    async def test_unhealthy_resident_receives_no_new_brief_and_is_released(self):
        await self.run_job('owner')
        self.requests.clear()
        self.health_status = 503
        with patch.object(module.asyncio, 'sleep', new=AsyncMock()):
            await self.run_job('owner', brief='Synthetic fox.')
        self.assertEqual(self.enhancer.status('owner')['job']['state'], 'error')
        self.assertTrue(all(request.url.path == '/health' for request in self.requests))
        self.processes[0].terminate.assert_called_once()
        self.assertIsNone(self.enhancer.process)
        self.assertIsNone(self.enhancer.status('owner')['loaded_model'])
        self.health_status = 200
        await self.run_job('owner')
        self.assertEqual(len(self.processes), 2)

    # AC: @create-managed-prompt-enhancer ac-resident-model
    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_active_cancel_reaps_resident_helper_and_retry_loads_again(self):
        await self.run_job('owner')
        self.completion_gate = asyncio.Event()
        self.completion_started.clear()
        job = self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic fox.')
        await self.completion_started.wait()
        with self.assertRaises(PermissionError):
            await self.enhancer.cancel('other', job['id'])
        self.processes[0].terminate.assert_not_called()
        await self.enhancer.cancel('owner', job['id'])
        self.assertIsNone(self.enhancer.process)
        self.assertEqual(self.enhancer.status('owner')['job']['state'], 'cancelled')
        self.completion_gate = None
        await self.run_job('owner')
        self.assertEqual(len(self.processes), 2)

    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_old_completed_job_cancel_does_not_stop_another_studios_worker(self):
        old = await self.run_job('owner')
        self.completion_gate = asyncio.Event()
        self.completion_started.clear()
        self.enhancer.start('other', 'enhance', 'qwen', 'Synthetic fox.')
        await self.completion_started.wait()
        await self.enhancer.cancel('owner', old['id'])
        self.processes[0].terminate.assert_not_called()
        self.assertEqual(self.enhancer.status('other')['job']['state'], 'running')
        self.completion_gate.set()
        await self.enhancer.task
        self.assertEqual(self.enhancer.status('other')['job']['result']['main_prompt'], 'Synthetic fox.')

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_immediate_and_completed_cancel_release_current_resident_helper(self):
        completed = await self.run_job('owner')
        await self.enhancer.cancel('owner', completed['id'])
        self.assertIsNone(self.enhancer.process)
        await self.run_job('owner')
        early = self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic fox.')
        await self.enhancer.cancel('owner', early['id'])
        self.assertEqual(self.enhancer.status('owner')['job']['state'], 'cancelled')
        self.assertIsNone(self.enhancer.process)
        self.assertEqual(len(self.processes), 2)

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_close_reaps_idle_resident_and_clears_owned_results(self):
        await self.run_job('owner')
        await self.enhancer.close()
        self.processes[0].terminate.assert_called_once()
        self.assertIsNone(self.enhancer.process)
        self.assertIsNone(self.enhancer.status('owner')['loaded_model'])
        self.assertFalse(self.enhancer.jobs)
        with self.assertRaises(RuntimeError):
            self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic owl.')

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_close_cancels_active_request_and_releases_helper(self):
        await self.run_job('owner')
        self.completion_gate = asyncio.Event()
        self.completion_started.clear()
        self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic fox.')
        await self.completion_started.wait()
        await self.enhancer.close()
        self.processes[0].terminate.assert_called_once()
        self.assertIsNone(self.enhancer.process)
        self.assertFalse(self.enhancer.jobs)

    # AC: @create-managed-prompt-enhancer ac-resident-model
    async def test_cancel_during_switch_finishes_teardown_before_a_new_load(self):
        await self.run_job('owner')
        old = self.processes[0]
        stopping, release = asyncio.Event(), asyncio.Event()
        async def wait():
            stopping.set()
            await release.wait()
            old.returncode = 0
            return 0
        old.wait.side_effect = wait
        job = self.enhancer.start('owner', 'enhance', 'minicpm', 'Synthetic fox.')
        await stopping.wait()
        first_cancel = asyncio.create_task(self.enhancer.cancel('owner', job['id']))
        await asyncio.sleep(0)
        second_cancel = asyncio.create_task(self.enhancer.cancel('owner', job['id']))
        await asyncio.sleep(0)
        self.assertEqual(len(self.processes), 1)
        with self.assertRaises(RuntimeError):
            self.enhancer.start('other', 'enhance', 'qwen', 'Synthetic cabin.')
        release.set()
        await asyncio.gather(first_cancel, second_cancel)
        self.assertEqual(old.returncode, 0)
        self.assertIsNone(self.enhancer.process)
        self.assertIsNone(self.enhancer._stop_task)
        await self.run_job('other', 'minicpm')
        self.assertEqual(len(self.processes), 2)

    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_new_job_waits_for_completed_job_discard_to_reap_idle_helper(self):
        job = await self.run_job('owner')
        old = self.processes[0]
        stopping, release = asyncio.Event(), asyncio.Event()
        async def wait():
            stopping.set()
            await release.wait()
            old.returncode = 0
            return 0
        old.wait.side_effect = wait
        discard = asyncio.create_task(self.enhancer.cancel('owner', job['id']))
        await stopping.wait()
        self.enhancer.start('other', 'enhance', 'qwen', 'Synthetic fox.')
        await asyncio.sleep(0)
        self.assertEqual(len(self.processes), 1)
        release.set()
        await discard
        await self.enhancer.task
        self.assertEqual(len(self.processes), 2)
        self.assertEqual(self.enhancer.status('other')['job']['state'], 'complete')

    # AC: @create-managed-prompt-enhancer ac-resident-model
    # AC: @create-managed-prompt-enhancer ac-owned-private
    async def test_repeated_cancel_during_native_spawn_reaps_child_and_never_loads(self):
        launching, release = asyncio.Event(), asyncio.Event()
        async def delayed_spawn(*command, **options):
            launching.set()
            await release.wait()
            return await self.native_spawn(*command, **options)
        with patch.object(module.asyncio, 'create_subprocess_exec', side_effect=delayed_spawn):
            job = self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic owl.')
            await launching.wait()
            self.assertIsNone(self.enhancer.status('owner')['loaded_model'])
            first = asyncio.create_task(self.enhancer.cancel('owner', job['id']))
            await asyncio.sleep(0)
            second = asyncio.create_task(self.enhancer.cancel('owner', job['id']))
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)
        self.assertEqual(len(self.processes), 1)
        self.processes[0].terminate.assert_called_once()
        self.assertIsNone(self.enhancer.process)
        self.assertEqual(self.enhancer.status('owner')['job']['state'], 'cancelled')
        self.assertFalse(self.requests)


class SyntheticProcessTests(unittest.IsolatedAsyncioTestCase):
    # AC: @create-managed-prompt-enhancer ac-resident-model
    # AC: @create-managed-prompt-enhancer ac-offline-resources
    async def test_actual_owned_subprocess_reuse_switch_crash_failure_and_close(self):
        # A disposable HTTP helper exercises the native lifecycle without model
        # files, normal application state, external servers or prompt logs.
        with tempfile.TemporaryDirectory(prefix='dmc-enhancer-child-') as temporary:
            enhancer = Enhancer(Path(temporary))
            enhancer.runtime_root.mkdir(parents=True)
            server = enhancer.runtime_root / 'synthetic-server.py'
            server.write_text(textwrap.dedent('''
                import json
                import sys
                from http.server import BaseHTTPRequestHandler, HTTPServer

                def option(name):
                    return sys.argv[sys.argv.index(name) + 1]

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass

                    def reply(self, payload, code=200):
                        body = json.dumps(payload).encode()
                        self.send_response(code)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                    def authorized(self):
                        if self.headers.get('Authorization') != 'Bearer ' + option('--api-key'):
                            self.reply({}, 401)
                            return False
                        return True

                    def do_GET(self):
                        if self.authorized():
                            self.reply({})

                    def do_POST(self):
                        if not self.authorized():
                            return
                        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                        if self.path == '/apply-template':
                            self.reply({'prompt': 'synthetic chat'})
                        elif self.path == '/tokenize':
                            self.reply({'tokens': [1, 2]})
                        else:
                            assert data['cache_prompt'] is False
                            brief = json.loads(data['messages'][1]['content'])
                            result = {'main_prompt': brief['image_brief'] + ' Seed ' + str(data['seed']),
                                      'face_prompt': brief['face_brief']}
                            content = 'invalid' if brief['image_brief'] == 'Synthetic failure.' else json.dumps(result)
                            self.reply({'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})

                HTTPServer((option('--host'), int(option('--port'))), Handler).serve_forever()
                '''), encoding='utf-8')
            native_spawn = asyncio.create_subprocess_exec
            children = []
            async def spawn(*command, **options):
                self.assertTrue(all(child.returncode is not None for child in children))
                child = await native_spawn(sys.executable, *command, **options)
                children.append(child)
                return child
            try:
                with patch.object(module, '_read_json', return_value={'server': server.name}), patch.object(module.asyncio, 'create_subprocess_exec', side_effect=spawn):
                    first = await enhancer.infer('qwen', 'Synthetic owl.', '', 42)
                    process = enhancer.process
                    self.assertEqual(first['main_prompt'], 'Synthetic owl. Seed 42')
                    second = await enhancer.infer('qwen', 'Synthetic fox.', '', 43)
                    self.assertEqual(second['main_prompt'], 'Synthetic fox. Seed 43')
                    self.assertIs(enhancer.process, process)
                    self.assertEqual(len(children), 1)
                    await enhancer.infer('minicpm', 'Synthetic cabin.', '', 44)
                    self.assertIsNotNone(process.returncode)
                    self.assertEqual(len(children), 2)
                    crashed = enhancer.process
                    crashed.kill()
                    await crashed.wait()
                    self.assertIsNone(enhancer.status('owner')['loaded_model'])
                    await enhancer.infer('minicpm', 'Synthetic tree.', '', 45)
                    self.assertEqual(len(children), 3)
                    failed = enhancer.process
                    with self.assertRaises(ValueError):
                        await enhancer.infer('minicpm', 'Synthetic failure.', '', 46)
                    self.assertIsNotNone(failed.returncode)
                    self.assertIsNone(enhancer.process)
                    await enhancer.infer('minicpm', 'Synthetic bridge.', '', 47)
                    self.assertEqual(len(children), 4)
                    await enhancer.close()
                    self.assertIsNone(enhancer.process)
                    self.assertTrue(all(child.returncode is not None for child in children))
            finally:
                await enhancer.close()
