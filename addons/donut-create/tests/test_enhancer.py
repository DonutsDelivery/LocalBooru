"""Managed enhancer checks use disposable files and synthetic briefs only."""
import asyncio
import hashlib
import io
import importlib.util
import json
import os
import tarfile
import urllib.error
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

import enhancer as module
from enhancer import Enhancer, parse_result, messages
from installer import SetupCancelled, SetupError
from test_installer import Opener


class OutputTests(unittest.TestCase):
    # AC: @create-multiple-face-prompts ac-character-prompts
    def test_relative_two_axis_plan_keeps_arbitrary_stable_character_order(self):
        plan = [
            {'character': 'Son', 'position': {'left_of': ['Daughter'], 'above': []}, 'face_prompt': 'The son smiles.'},
            {'character': 'Mother', 'position': {'left_of': [], 'above': ['Daughter']}, 'face_prompt': "The mother's face."},
            {'character': 'Father', 'position': {'left_of': ['Mother'], 'above': ['Son']}, 'face_prompt': "The father's face."},
            {'character': 'Daughter', 'position': {'left_of': [], 'above': []}, 'face_prompt': "The daughter's face."},
        ]
        result = {'main_prompt': 'Father above son, mother above daughter; father left of mother and son left of daughter.', 'face_prompt': '', 'face_prompts': plan}
        self.assertEqual(parse_result(json.dumps(result), 'stop', True), result)
        # An edge in each direction across different axes is not an axis cycle.
        cross_axis = [
            {'character': 'Father', 'position': {'left_of': ['Son'], 'above': []}, 'face_prompt': "The father's face."},
            {'character': 'Son', 'position': {'left_of': [], 'above': ['Father']}, 'face_prompt': "The son's face."},
        ]
        self.assertEqual(parse_result(json.dumps({**result, 'face_prompts': cross_axis}), 'stop')['face_prompts'], cross_axis)

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_relative_references_and_axis_cycles_are_rejected(self):
        plan = [
            {'character': 'Father', 'position': {'left_of': ['Mother'], 'above': ['Son']}, 'face_prompt': "The father's face."},
            {'character': 'Mother', 'position': {'left_of': [], 'above': []}, 'face_prompt': "The mother's face."},
            {'character': 'Son', 'position': {'left_of': [], 'above': []}, 'face_prompt': "The son's face."},
        ]
        for index, position in (
            (0, {'left_of': ['Unknown'], 'above': []}),
            (0, {'left_of': ['mother'], 'above': []}),
            (0, {'left_of': ['Father'], 'above': []}),
            (0, {'left_of': ['Mother', 'Mother'], 'above': []}),
            (0, {'left_of': [True], 'above': []}),
            (0, {'left_of': []}),
            (0, {'left_of': [], 'above': [], 'extra': []}),
            (0, {'left_of': 'Mother', 'above': []}),
            (0, {'left_of': [], 'above': None}),
            (0, {'left_of': ['Mother'] * 16, 'above': []}),
            (0, {'column': 1, 'row': 1}),
            (1, {'left_of': ['Father'], 'above': []}),
            (2, {'left_of': [], 'above': ['Father']}),
            (2, 'right'),
        ):
            with self.subTest(index=index, position=position), self.assertRaises(ValueError):
                modified = [dict(record) for record in plan]
                modified[index]['position'] = position
                parse_result(json.dumps({'main_prompt': 'Three family members.', 'face_prompt': '', 'face_prompts': modified}), 'stop')
        cycle = [{**record, 'position': {'left_of': [plan[(index + 1) % 3]['character']], 'above': []}}
                 for index, record in enumerate(plan)]
        with self.assertRaises(ValueError):
            parse_result(json.dumps({'main_prompt': 'Three family members.', 'face_prompt': '', 'face_prompts': cycle}), 'stop')

    # AC: @create-multiple-face-prompts ac-character-prompts
    # AC: @create-multiple-face-prompts ac-single-face-compatible
    def test_ordered_face_plan_and_legacy_single_face_are_preserved(self):
        plan = [
            {'character': 'Older pilot', 'position': 'left', 'face_prompt': 'The older pilot smiles, eyes open.'},
            {'character': 'Younger pilot', 'position': 'right', 'face_prompt': 'The younger pilot frowns, eyes closed.'},
        ]
        result = {'main_prompt': 'Two pilots stand together, older on the left and younger on the right.', 'face_prompt': '', 'face_prompts': plan}
        self.assertEqual(parse_result(json.dumps(result), 'stop', True), result)
        single = {'main_prompt': 'One pilot looks left.', 'face_prompt': 'The pilot looks left.', 'face_prompts': []}
        self.assertEqual(parse_result(json.dumps(single), 'stop', True), single)
        legacy = {key: single[key] for key in ('main_prompt', 'face_prompt')}
        self.assertEqual(parse_result(json.dumps(legacy), 'stop', True), legacy)
        ordinal = [{**record, 'position': position} for record, position in zip(plan, ('1st', '2nd'))]
        self.assertEqual(parse_result(json.dumps({**result, 'face_prompts': ordinal}), 'stop')['face_prompts'], ordinal)
        maximum = [{'character': f'Character {index + 1}', 'position': position, 'face_prompt': f'Character {index + 1} smiles.'}
                   for index, position in enumerate(module.FACE_ORDINALS)]
        self.assertEqual(len(maximum), 16)
        self.assertEqual(parse_result(json.dumps({**result, 'face_prompts': maximum}), 'stop')['face_prompts'], maximum)

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_ambiguous_face_plan_can_remain_empty_in_new_contract(self):
        unresolved = {'main_prompt': 'Two pilots, both facing the viewer.', 'face_prompt': '', 'face_prompts': []}
        self.assertEqual(parse_result(json.dumps(unresolved), 'stop'), unresolved)
        with self.assertRaisesRegex(ValueError, 'resolve'):
            parse_result(json.dumps(unresolved), 'stop', True)

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_invalid_or_unsupported_face_order_is_rejected(self):
        left = {'character': 'Left twin', 'position': 'left', 'face_prompt': 'The left twin smiles.'}
        right = {'character': 'Right twin', 'position': 'right', 'face_prompt': 'The right twin frowns.'}
        valid = {'main_prompt': 'Two twins, smiling twin left and frowning twin right.', 'face_prompt': '', 'face_prompts': [left, right]}
        for patch_result in (
            {'face_prompts': None}, {'face_prompts': {}}, {'face_prompts': [left]},
            {'face_prompt': 'Both twins smile.'},
            {'face_prompts': [right, left]},
            {'face_prompts': [left, {**right, 'position': 'left'}]},
            {'face_prompts': [left, {**right, 'position': '2nd'}]},
            {'face_prompts': [{**left, 'position': 'looking left'}, right]},
            {'face_prompts': [{**left, 'position': '17th'}, right]},
            {'face_prompts': [left, {**right, 'character': ' LEFT TWIN '}]},
            {'face_prompts': [left, {**right, 'character': ''}]},
            {'face_prompts': [left, {**right, 'character': 'x' * 201}]},
            {'face_prompts': [left, {**right, 'face_prompt': ''}]},
            {'face_prompts': [left, {**right, 'face_prompt': 'x' * 8001}]},
            {'face_prompts': [left, {**right, 'face_prompt': '%load:/synthetic%'}]},
            {'face_prompts': [left, {**right, 'extra': 'unselected'}]},
            {'face_prompts': [left, 'not a character record']},
            {'face_prompts': [{**left, 'character': f'Twin {index}', 'position': f'{index}th'} for index in range(17)]},
            {'face_prompts': [left, right], 'other': 'unselected'},
        ):
            with self.subTest(patch=patch_result), self.assertRaises(ValueError):
                parse_result(json.dumps({**valid, **patch_result}), 'stop')

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

    # AC: @create-managed-prompt-enhancer ac-per-run-seed
    async def test_job_threads_execution_seed_and_reports_sampling_seed(self):
        with patch.object(self.enhancer, 'ready', return_value=True), patch.object(self.enhancer, 'infer', new=AsyncMock(return_value={'main_prompt': 'Synthetic owl.', 'face_prompt': ''})) as infer:
            for seed in (0, 12, 0xffffffff, 2**53 - 1):
                job = self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic owl.', '', seed=seed)
                await self.enhancer.task
                infer.assert_awaited_with('qwen', 'Synthetic owl.', '', seed)
                self.assertEqual(job['seed'], seed)
                self.assertEqual(job['inference_seed'], seed % 0xffffffff)
                self.assertEqual(self.enhancer.status('owner')['job']['seed'], seed)
            self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic owl.', '')
            await self.enhancer.task
            infer.assert_awaited_with('qwen', 'Synthetic owl.', '', 0)

    # AC: @create-managed-prompt-enhancer ac-per-run-seed
    async def test_invalid_seed_is_rejected_before_creating_job(self):
        with patch.object(self.enhancer, 'ready', return_value=True):
            for seed in (True, False, -1, 1.5, 1.0, '7', None, 2**53):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    self.enhancer.start('owner', 'enhance', 'qwen', 'Synthetic owl.', '', seed=seed)
                self.assertIsNone(self.enhancer.task)
                self.assertFalse(self.enhancer.jobs)

    # AC: @create-managed-prompt-enhancer ac-per-run-seed
    # AC: @create-managed-prompt-enhancer ac-creative-instructions
    async def test_actual_runtime_request_uses_repeatable_seeded_sampling_and_creative_briefs(self):
        image = 'One owl. Choose a setting and vary the lighting; no other animals. Trigger SYNTH_OWL; a sign reads "Keep out—".'
        face = 'The owl: choose its gaze, keep both eyes open and its beak closed.'
        output = {'main_prompt': 'One owl in a moonlit courtyard, looking left with both eyes open and its beak closed. No other animals. SYNTH_OWL; a sign reads "Keep out—".', 'face_prompt': 'The owl looks left, both eyes open and its beak closed.'}
        requests = []
        processes = []
        client_type = httpx.AsyncClient
        def respond(request):
            self.assertEqual(request.url.host, '127.0.0.1')
            data = {} if request.method == 'GET' else json.loads(request.content)
            if request.url.path == '/health':
                return httpx.Response(200, json={})
            if request.url.path == '/apply-template':
                return httpx.Response(200, json={'prompt': 'synthetic formatted chat'})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'tokens': [1] * 20})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            requests.append(data)
            self.assertEqual(json.loads(data['messages'][1]['content']), {'image_brief': image, 'face_brief': face})
            schema = data['response_format']['json_schema']['schema']
            self.assertEqual(schema['required'], ['main_prompt', 'face_prompt', 'face_prompts'])
            self.assertEqual(schema['properties']['face_prompts']['maxItems'], 16)
            record = schema['properties']['face_prompts']['items']
            self.assertEqual(record['required'], ['character', 'position', 'face_prompt'])
            self.assertFalse(record['additionalProperties'])
            self.assertEqual(record['properties']['character']['maxLength'], 200)
            self.assertEqual(record['properties']['face_prompt']['maxLength'], 8000)
            positions = record['properties']['position']['oneOf']
            self.assertEqual(positions[0]['enum'][:3], ['left', 'center', 'right'])
            self.assertEqual(positions[0]['enum'][-1], '16th')
            self.assertEqual(positions[1]['required'], ['left_of', 'above'])
            self.assertFalse(positions[1]['additionalProperties'])
            for axis in ('left_of', 'above'):
                self.assertEqual(positions[1]['properties'][axis]['maxItems'], 15)
                self.assertTrue(positions[1]['properties'][axis]['uniqueItems'])
                self.assertEqual(positions[1]['properties'][axis]['items']['maxLength'], 200)
            self.assertGreaterEqual(data['temperature'], 0.6)
            self.assertFalse(data['cache_prompt'])
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]})
        def client(**options):
            options['transport'] = httpx.MockTransport(respond)
            return client_type(**options)
        async def spawn(*args, **kwargs):
            process = unittest.mock.Mock(returncode=None)
            process.wait = AsyncMock(return_value=0)
            processes.append(process)
            return process
        with patch.object(module, '_read_json', return_value={'server': 'llama-server'}), patch.object(module.asyncio, 'create_subprocess_exec', side_effect=spawn), patch.object(module.httpx, 'AsyncClient', side_effect=client):
            for seed in (42, 42, 43, 0xffffffff, 2**53 - 1):
                self.assertEqual(await self.enhancer.infer('qwen', image, face, seed), output)
        # This proves identical sampling parameters, not identical real-model text.
        self.assertEqual(requests[0], requests[1])
        self.assertNotEqual(requests[1]['seed'], requests[2]['seed'])
        self.assertEqual([request['seed'] for request in requests], [42, 42, 43, 0, (2**53 - 1) % 0xffffffff])
        for process in processes:
            process.terminate.assert_called_once()
        self.assertIsNone(self.enhancer.process)

    # AC: @donut-create-plugin ac-face-prompt-routing
    # AC: @create-managed-prompt-enhancer ac-infer-empty-face
    async def test_scene_derived_face_pair_survives_owned_inference_with_empty_face_brief(self):
        # Synthetic model responses verify transport and parsing, not real-model
        # obedience to the face-selection policy. No model is downloaded here.
        # These legacy two-field outputs remain readable for compatibility;
        # current three-field policy requires coverage of every visible face.
        fixtures = [
            ('A front-facing portrait of one adult dancer. No makeup.', '',
             "The adult dancer's face, facing the viewer, without makeup."),
            ('One adult musician seated beside a piano.', '',
             "The adult musician's face."),
            ('A front-facing portrait of one owl.', ' \t ',
             "The owl's face, facing the viewer."),
            ('One adult pilot smiles, looking left, beside a red aircraft. Trigger SYNTH_PILOT.', '',
             'The adult pilot smiles, looking left.'),
            ('One adult pilot faces the viewer; one mechanic faces away with their face hidden.', '',
             "The adult pilot's face, facing the viewer."),
            ('One adult dancer faces the viewer. Choose the dancer\'s expression; keep their eyes closed.', '',
             'The adult dancer smiles, facing the viewer with eyes closed.'),
            ('Two adult dancers face the viewer; neither is the main subject.', '', ''),
            ('One adult dancer seen from behind, face entirely hidden.', '', ''),
            ('A wooden cabin, with no people or animals.', '', ''),
            ('One adult pilot seen from behind, face entirely hidden.', "The pilot's face.",
             "The pilot's face."),
            ('Two adult dancers face the viewer; neither is the main subject.', "The older dancer's face.",
             "The older dancer's face."),
            ('One adult pilot in a front-facing portrait.',
             'The pilot looks right, eyes closed, mouth closed.',
             'The pilot looks right, eyes closed, mouth closed.'),
        ]
        client_type = httpx.AsyncClient
        for image, face, expected_face in fixtures:
            with self.subTest(image=image, face=face):
                # The face override is consistent in both generated fields.
                scene = image.replace("Choose the dancer's expression; keep their eyes closed.", 'The dancer smiles with eyes closed.')
                output = {'main_prompt': scene + (' ' + face if face else ''), 'face_prompt': expected_face}
                process = unittest.mock.Mock(returncode=None)
                process.wait = AsyncMock(return_value=0)
                completion_requests = []
                def respond(request):
                    self.assertEqual(request.url.host, '127.0.0.1')
                    data = {} if request.method == 'GET' else json.loads(request.content)
                    if request.url.path == '/health':
                        return httpx.Response(200, json={})
                    if request.url.path == '/apply-template':
                        return httpx.Response(200, json={'prompt': 'synthetic formatted chat'})
                    if request.url.path == '/tokenize':
                        return httpx.Response(200, json={'tokens': [1] * 20})
                    self.assertEqual(request.url.path, '/v1/chat/completions')
                    self.assertEqual(data['messages'], messages(image, face))
                    self.assertEqual(json.loads(data['messages'][1]['content']), {'image_brief': image, 'face_brief': face})
                    self.assertEqual(data['seed'], 73)
                    completion_requests.append(data)
                    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]})
                def client(**options):
                    options['transport'] = httpx.MockTransport(respond)
                    return client_type(**options)
                with patch.object(self.enhancer, 'ready', return_value=True), patch.object(module, '_read_json', return_value={'server': 'llama-server'}), patch.object(module.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=process)), patch.object(module.httpx, 'AsyncClient', side_effect=client):
                    self.enhancer.start('owner', 'enhance', 'qwen', image, face, seed=73)
                    await self.enhancer.task
                job = self.enhancer.status('owner')['job']
                self.assertEqual(job['state'], 'complete')
                self.assertEqual(job['result'], output)
                self.assertEqual(job['seed'], 73)
                self.assertEqual(len(completion_requests), 1)
                self.assertNotIn('job', self.enhancer.status('other'))
                process.terminate.assert_called_once()
                self.assertIsNone(self.enhancer.process)

    # AC: @create-multiple-face-prompts ac-character-prompts
    # AC: @create-multiple-face-prompts ac-mismatch
    async def test_actual_http_result_retains_spatial_face_plan_and_unresolved_order_fails_closed(self):
        left = {'character': 'Alice', 'position': 'left', 'face_prompt': 'Alice smiles, looking right.'}
        right = {'character': 'Bob', 'position': 'right', 'face_prompt': 'Bob frowns, looking left.'}
        positioned = 'Bob on the right frowns, looking left; Alice on the left smiles, looking right. Trigger SYNTH_PAIR. A sign says "Keep out—".'
        family = 'Father above son, mother above daughter; father left of mother and son left of daughter. All four faces are visible.'
        fixtures = [
            (positioned, '', {'main_prompt': positioned, 'face_prompt': '', 'face_prompts': [left, right]}, 'complete'),
            (family, 'Father smiles, eyes closed.',
             {'main_prompt': family + ' Father smiles with eyes closed.', 'face_prompt': '', 'face_prompts': [
                 {'character': 'Son', 'position': {'left_of': ['Daughter'], 'above': []}, 'face_prompt': "The son's face."},
                 {'character': 'Father', 'position': {'left_of': ['Mother'], 'above': ['Son']}, 'face_prompt': 'Father smiles, eyes closed.'},
                 {'character': 'Daughter', 'position': {'left_of': [], 'above': []}, 'face_prompt': "The daughter's face."},
                 {'character': 'Mother', 'position': {'left_of': [], 'above': ['Daughter']}, 'face_prompt': "The mother's face."},
             ]}, 'complete'),
            ('Bob is on the right; Alice is on the left. Both faces are visible.', 'Alice smiles, eyes closed.',
             {'main_prompt': 'Bob is on the right; Alice is on the left, smiling with eyes closed. Both faces are visible.', 'face_prompt': '',
              'face_prompts': [{'character': 'Alice', 'position': 'left', 'face_prompt': 'Alice smiles, eyes closed.'},
                               {'character': 'Bob', 'position': 'right', 'face_prompt': "Bob's face."}]}, 'complete'),
            ('Alice and Bob pose for a portrait.', 'From left to right: Alice smiles, looking right; Bob frowns, looking left.',
             {'main_prompt': 'Alice on the left smiles, looking right; Bob on the right frowns, looking left, in a portrait.', 'face_prompt': '',
              'face_prompts': [{**left, 'position': '1st'}, {**right, 'position': '2nd'}]}, 'complete'),
            ('Alice looks left; Bob looks right. Neither has a specified position.', '',
             {'main_prompt': 'Alice looks left; Bob looks right. Neither has a specified position.', 'face_prompt': '', 'face_prompts': []}, 'complete'),
            ('Father is above son; mother is also visible, with no relationship specified for her.', '',
             {'main_prompt': 'Father is above son; mother is also visible, with no relationship specified for her.', 'face_prompt': '', 'face_prompts': []}, 'complete'),
            ('Alice and Bob pose for a portrait.', 'Alice smiles; Bob frowns. No order is specified.',
             {'main_prompt': 'Alice smiles; Bob frowns, in a portrait.', 'face_prompt': '', 'face_prompts': []}, 'error'),
            ('Alice and Bob pose together, both faces visible. Their positions are unspecified.', 'Alice smiles, eyes closed.',
             {'main_prompt': 'Alice smiles with eyes closed; Bob poses beside her, both faces visible. Their positions are unspecified.', 'face_prompt': '', 'face_prompts': []}, 'error'),
        ]
        client_type = httpx.AsyncClient
        for image, face, output, state in fixtures:
            with self.subTest(image=image, face=face):
                process = unittest.mock.Mock(returncode=None)
                process.wait = AsyncMock(return_value=0)
                requests = []
                def respond(request):
                    self.assertEqual(request.url.host, '127.0.0.1')
                    data = {} if request.method == 'GET' else json.loads(request.content)
                    if request.url.path == '/health':
                        return httpx.Response(200, json={})
                    if request.url.path == '/apply-template':
                        return httpx.Response(200, json={'prompt': 'synthetic formatted chat'})
                    if request.url.path == '/tokenize':
                        return httpx.Response(200, json={'tokens': [1] * 20})
                    self.assertEqual(request.url.path, '/v1/chat/completions')
                    self.assertEqual(data['messages'], messages(image, face))
                    self.assertEqual(json.loads(data['messages'][1]['content']), {'image_brief': image, 'face_brief': face})
                    self.assertEqual(data['seed'], 73)
                    requests.append(data)
                    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}]})
                def client(**options):
                    options['transport'] = httpx.MockTransport(respond)
                    return client_type(**options)
                with patch.object(self.enhancer, 'ready', return_value=True), patch.object(module, '_read_json', return_value={'server': 'llama-server'}), patch.object(module.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=process)), patch.object(module.httpx, 'AsyncClient', side_effect=client):
                    self.enhancer.start('owner', 'enhance', 'qwen', image, face, seed=73)
                    await self.enhancer.task
                job = self.enhancer.status('owner')['job']
                self.assertEqual(job['state'], state)
                if state == 'complete':
                    self.assertEqual(job['result'], output)
                else:
                    self.assertNotIn('result', job)
                    self.assertIn('could not resolve', job['error'])
                self.assertEqual(len(requests), 1)
                self.assertNotIn('job', self.enhancer.status('other'))
                process.terminate.assert_called_once()
                self.assertIsNone(self.enhancer.process)

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
            self.assertEqual(data['response_format']['json_schema']['schema']['required'], ['main_prompt', 'face_prompt', 'face_prompts'])
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


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from fastapi import FastAPI, Request
        spec = importlib.util.spec_from_file_location('enhancer_route_controller', module.ROOT / 'app.py')
        controller_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controller_module)
        self.temporary = tempfile.TemporaryDirectory(prefix='dmc-enhancer-route-')
        self.enhancer = Enhancer(Path(self.temporary.name))
        app = FastAPI()
        @app.post('/dmc/enhancer')
        async def enhance(request: Request):
            return await controller_module.studio_local(SimpleNamespace(enhancer=self.enhancer), {'id': 'owner'}, 'dmc/enhancer', request)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://synthetic.invalid')

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.enhancer.close()
        self.temporary.cleanup()

    # AC: @create-managed-prompt-enhancer ac-per-run-seed
    async def test_route_accepts_execution_seed_and_rejects_invalid_seed(self):
        brief = {'kind': 'enhance', 'model': 'qwen', 'image_brief': 'Synthetic owl.', 'face_brief': ''}
        with patch.object(self.enhancer, 'ready', return_value=True), patch.object(self.enhancer, 'infer', new=AsyncMock(return_value={'main_prompt': 'Synthetic owl.', 'face_prompt': ''})) as infer:
            response = await self.http.post('/dmc/enhancer', json={**brief, 'seed': 2**53 - 1})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertEqual(response.json()['job']['seed'], 2**53 - 1)
            await self.enhancer.task
            infer.assert_awaited_with('qwen', 'Synthetic owl.', '', 2**53 - 1)
            for seed in (True, -1, 1.5, 1.0, 2**53, None):
                with self.subTest(seed=seed):
                    response = await self.http.post('/dmc/enhancer', json={**brief, 'seed': seed})
                    self.assertEqual(response.status_code, 400, response.text)


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
