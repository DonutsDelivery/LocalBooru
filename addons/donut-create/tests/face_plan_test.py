"""Synthetic validation of the wire schema consumed by Create."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dmc_face_plan_fixture', Path(__file__).resolve().parents[1] / 'face_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FacePlanTests(unittest.TestCase):
    # AC: @create-multiple-face-prompts ac-character-prompts
    def test_consumes_ordered_and_relative_records_without_reordering(self):
        for positions in [('left', 'right'), ('1st', '2nd'),
                ({'left_of': ['Bo'], 'above': []}, {'left_of': [], 'above': []})]:
            rows = [{'character': 'Ada', 'position': positions[0], 'face_prompt': 'Ada smiles.'},
                    {'character': 'Bo', 'position': positions[1], 'face_prompt': 'Bo frowns.'}]
            self.assertEqual(module.face_prompts(json.dumps(rows)), rows)
        self.assertEqual(module.face_prompts('[]'), [])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_rejects_malformed_ambiguous_and_oversized_plans(self):
        good = [{'character': 'Ada', 'position': 'left', 'face_prompt': 'Ada smiles.'},
                {'character': 'Bo', 'position': 'right', 'face_prompt': 'Bo frowns.'}]
        invalid = [None, '', '{}', '[', json.dumps(good[:1]), 'x' * (1024 * 1024 + 1),
                   json.dumps(good * 9), '[{"character":"Ada","character":"Bo"}]']
        for field, value in [('character', ''), ('character', 'Bo'), ('face_prompt', ''),
                             ('face_prompt', 'x' * 8001), ('position', 'right'), ('position', {})]:
            changed = json.loads(json.dumps(good)); changed[0][field] = value
            invalid.append(json.dumps(changed))
        for raw in invalid:
            with self.subTest(raw=type(raw).__name__), self.assertRaises(ValueError):
                module.face_prompts(raw)

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_relative_relations_reject_unknown_self_and_cycles_on_either_axis(self):
        for axis in ['left_of', 'above']:
            for mode in ['unknown', 'self', 'cycle', 'duplicate']:
                rows = [{'character': name, 'position': {'left_of': [], 'above': []},
                         'face_prompt': name + ' smiles.'} for name in ['Ada', 'Bo']]
                rows[0]['position'][axis] = {'unknown': ['Cy'], 'self': ['Ada'],
                                             'cycle': ['Bo'], 'duplicate': ['Bo', 'Bo']}[mode]
                if mode == 'cycle': rows[1]['position'][axis] = ['Ada']
                with self.subTest(axis=axis, mode=mode), self.assertRaises(ValueError):
                    module.face_prompts(json.dumps(rows))

if __name__ == '__main__':
    unittest.main()
