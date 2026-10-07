"""Opt-in real inference using one freshly downloaded, owned synthetic fixture.

Run through the host heavy-work gate with DMC_ENHANCER_INFERENCE_TEST=1.
No installed model, application profile, prompt history or shared cache is read.
"""
import os
import tempfile
import unittest
from pathlib import Path

from enhancer import Enhancer


@unittest.skipUnless(os.environ.get('DMC_ENHANCER_INFERENCE_TEST') == '1', 'opt-in model download and CPU inference')
class RealInferenceTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='dmc-enhancer-inference-')
        cls.addClassCleanup(cls.temporary.cleanup)
        Enhancer(Path(cls.temporary.name)).install('qwen')

    async def asyncSetUp(self):
        self.enhancer = Enhancer(Path(self.temporary.name))

    async def asyncTearDown(self):
        await self.enhancer.close()

    # AC: @create-managed-prompt-enhancer ac-infer-empty-face
    async def test_visible_portrait_derives_face_without_separate_facial_brief(self):
        result = await self.enhancer.infer('qwen', 'A front-facing portrait of one adult dancer. No makeup.', '', seed=73)
        self.assertIn('dancer', result['main_prompt'].lower())
        self.assertIn('dancer', result['face_prompt'].lower())
        self.assertTrue(result['face_prompt'].strip())
        for invented in ('smiling', 'eyes closed', 'green eyes', 'beard'):
            self.assertNotIn(invented, result['face_prompt'].lower())

    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    async def test_sparse_scene_gains_rendering_detail_without_invented_subjects(self):
        result = await self.enhancer.infer('qwen', 'A wooden cabin.', '')
        scene = result['main_prompt'].lower()
        self.assertIn('cabin', scene)
        self.assertTrue('wood' in scene or 'timber' in scene)
        self.assertGreaterEqual(len(scene.split()), 20)
        self.assertEqual(result['face_prompt'], '')
        for extra in ('person', 'people', 'forest', 'trees', 'furniture', 'chimney'):
            self.assertNotIn(extra, scene)

    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    async def test_sparse_animal_does_not_gain_an_action_or_facial_traits(self):
        result = await self.enhancer.infer('qwen', 'One owl.', "The owl's face.")
        scene = result['main_prompt'].lower()
        self.assertIn('owl', scene)
        self.assertGreaterEqual(len(scene.split()), 20)
        self.assertIn('owl', result['face_prompt'].lower())
        for added in ('perch', 'flying', 'branch', 'forest', 'looking', 'gazing', 'eyes open', 'eyes closed'):
            self.assertNotIn(added, scene)
        for invented in ('green', 'brown', 'yellow', 'smiling', 'open', 'closed', 'left', 'right'):
            self.assertNotIn(invented, result['face_prompt'].lower())

    # AC: @create-managed-prompt-enhancer ac-enhance-preview
    async def test_detailed_scene_and_identity_only_face_keep_supplied_facts(self):
        brief = 'One owl beside a wooden cabin. The owl looks left, both eyes open and beak closed. Dim blue light, visible wood grain, matte timber, quiet atmosphere, natural perspective. No other animals. Literal trigger SYNTH_OWL. A sign reads "Keep out—".'
        result = await self.enhancer.infer('qwen', brief, "The owl's face.")
        scene = result['main_prompt'].lower()
        for fact in ('owl', 'cabin', 'left', 'open', 'closed', 'blue', 'wood grain', 'matte', 'quiet', 'natural', 'no other animals'):
            self.assertIn(fact, scene)
        self.assertIn('SYNTH_OWL', result['main_prompt'])
        self.assertIn('"Keep out—"', result['main_prompt'])
        for invented in ('perch', 'flying', 'attached', 'hanging'):
            self.assertNotIn(invented, scene)
        self.assertIn('owl', result['face_prompt'].lower())
        for invented in ('green', 'brown', 'yellow', 'smiling', 'beard', 'makeup'):
            self.assertNotIn(invented, result['face_prompt'].lower())


if __name__ == '__main__':
    unittest.main()
