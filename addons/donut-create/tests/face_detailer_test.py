"""Executable wrapper/installer fixtures; no ComfyUI, models, GPU or user data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = load("dmc_face_detailer_fixture", "dmc_face_detailer.py")
installer = load("dmc_installer_fixture", "installer.py")
ROWS = [{"character": "Ada", "position": "left", "face_prompt": "Ada smiles."},
        {"character": "Bo", "position": "right", "face_prompt": "Bo frowns."}]


class Image:
    def __init__(self, index=0, state=0):
        self.index, self.state = index, state

    def unsqueeze(self, axis):
        return self


class Mask:
    ndim = 2

    def unsqueeze(self, axis):
        result = Mask()
        result.ndim = 3
        return result


class Detector:
    def __init__(self, detections):
        self.detections, self.aux = detections, None

    def setAux(self, value):
        self.aux = value

    def detect(self, image, *args, **kwargs):
        return (100, 100), self.detections[image.index]


def segment(x, y=0, width=10, mask=1):
    return types.SimpleNamespace(bbox=(x, y, x + width, y + 10), crop_region=(x, y, x + width, y + 10),
                                 cropped_mask=mask, cropped_image="stale original crop")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.passes, self.encoded, self.legacy = [], [], []
        fixture = self
        self.module = types.ModuleType("synthetic_donut_detailer")
        class Upstream:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)}, "optional": {"edit_mode": ("BOOLEAN", {"default": False})}}

            def doit(self, image, model="model", clip="clip", vae="vae", resolution=1024, max_resolution=1280,
                     guide_size_for=False, seed=42, steps=8, cfg=1, sampler_name="er_sde", scheduler="simple",
                     positive=None, negative="negative", denoise=.2, feather=5, noise_mask=True,
                     force_inpaint=False, bbox_threshold=.21, bbox_dilation=8, bbox_crop_factor=1.5,
                     sam_detection_hint="rect-4", sam_dilation=3, sam_threshold=.93, sam_bbox_expansion=6,
                     sam_mask_hint_threshold=.7, sam_mask_hint_use_negative="False", drop_size=12,
                     bbox_detector=None, wildcard="", cycle=2, max_faces=13, sam_model_opt=None,
                     segm_detector_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=20,
                     scheduler_func_opt=None, edit_mode=False, edit_prompt="whole image",
                     edit_negative_prompt="negative text", grounding_px=1088, edit_model=None,
                     face_reference=None, vary_seed_per_face=False, turbo_mode=True,
                     face_reference_b=None, vae_damage_correction=True, vae_damage_strength=.8, **nag_options):
                fixture.legacy.append((image, nag_options))
                return ("legacy",) * 6

            @staticmethod
            def enhance_face(image, **kwargs):
                _, segs = kwargs["bbox_detector"].detect(image)
                assert len(segs) == 1
                fixture.passes.append((image.index, image.state, segs[0].bbox, kwargs))
                # Simulates a crop from the evolving full image, not seg.cropped_image.
                return Image(image.index, image.state + 1), ["crop"], ["alpha"], "mask", ["control"]
        Upstream.__module__ = self.module.__name__
        self.module.Upstream = Upstream
        self.module._ensure_impact = lambda: None
        self.module.offload_model_for_auxiliary_stage = lambda *args: None
        self.module.comfy = types.SimpleNamespace(model_management=object())
        self.module.torch = types.SimpleNamespace(is_tensor=lambda _: False, count_nonzero=lambda mask: mask,
                                                 cat=lambda values, dim=0: values)
        self.module.np = types.SimpleNamespace(count_nonzero=lambda mask: mask)
        self.module.core = types.SimpleNamespace(segs_to_combined_mask=lambda segs: Mask(),
            make_sam_mask=lambda *args: "sam", segs_bitwise_and_mask=lambda segs, mask: segs)
        self.module.reapply_edit_variance = lambda fresh, original: [[fresh[0][0], {"variance": original[0][1]["variance"]}]]
        self.module.prepare_positive_conditioning_taps = lambda model, fresh: [[fresh[0][0], {**fresh[0][1], "fusion": model}]]
        class Encoder:
            def encode(self, clip, text):
                fixture.encoded.append((clip, text))
                return ([[text, {}]],)
        self.nodes = types.ModuleType("nodes")
        self.nodes.NODE_CLASS_MAPPINGS = {"DonutFaceDetailer": Upstream}
        self.nodes.CLIPTextEncode = Encoder
        self.modules = patch.dict(sys.modules, {"nodes": self.nodes, self.module.__name__: self.module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def run_faces(self, detections=None, rows=ROWS, **kwargs):
        detections = detections or [[segment(70, width=25), segment(10)]]
        return runtime.DMCFaceDetailer().doit([Image(index) for index in range(len(detections))],
            bbox_detector=Detector(detections), positive=[["NEVER REUSE WHOLE PROMPT", {"variance": "recipe"}]],
            face_prompts_json=json.dumps(rows), **kwargs)

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_ordered_separate_conditioning_and_evolving_crops(self):
        result = self.run_faces(vary_seed_per_face=True, nag_alpha=3.4)
        self.assertEqual(len(result), 6)
        self.assertEqual(len(result[4]), 14)
        self.assertEqual([mask.ndim for mask in result[3]], [3])
        self.assertEqual([entry[2][0] for entry in self.passes], [10, 70])
        self.assertEqual([entry[1] for entry in self.passes], [0, 1])
        self.assertEqual(self.encoded, [("clip", "Ada smiles."), ("clip", "Bo frowns.")])
        for index, (_, _, _, inputs) in enumerate(self.passes):
            self.assertEqual(inputs["positive"], [[ROWS[index]["face_prompt"], {"variance": "recipe", "fusion": "model"}]])
            self.assertEqual(inputs["edit_prompt"], ROWS[index]["face_prompt"])
            self.assertEqual(inputs["seed"], 42 + index)
            self.assertEqual(inputs["resolution"], 1024 ** 2)
            for key, wanted in {"cycle": 2, "nag_alpha": 3.4, "vae_damage_strength": .8,
                                "scheduler": "simple", "noise_mask_enabled": True, "guide_size_for_bbox": False,
                                "denoise": .2, "turbo_mode": True}.items():
                self.assertEqual(inputs[key], wanted)

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_y_and_detection_index_break_equal_x_ties(self):
        same = [segment(10, 20), segment(10, 0)]
        self.run_faces([same])
        self.assertEqual([entry[2][1] for entry in self.passes], [0, 20])
        self.passes.clear()
        self.run_faces([[segment(10, 0, 10), segment(5, 0, 20)]])
        self.assertEqual([entry[2][0] for entry in self.passes], [10, 5])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_every_batch_count_is_checked_before_any_sampling_or_encoding(self):
        for detections in [[[segment(0)]], [[segment(0), segment(20), segment(50)]],
                           [[segment(0), segment(20)], [segment(0)]], [[segment(0, mask=0), segment(20)]]]:
            with self.subTest(detections=detections), self.assertRaisesRegex(ValueError, "routing mismatch"):
                self.run_faces(detections)
            self.assertEqual(self.passes, [])
            self.assertEqual(self.encoded, [])
        with self.assertRaisesRegex(ValueError, "Maximum faces"):
            self.run_faces(max_faces=1)
        self.assertEqual(self.passes, [])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_identity_edit_references_and_prompt_overrides_are_rejected(self):
        for options in ({"edit_mode": True}, {"face_reference": Image()}, {"face_reference_b": Image()},
                        {"wildcard": "replace prompt"}, {"detailer_hook": object()}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "not support"):
                self.run_faces(**options)
            self.assertEqual(self.passes, [])

    # AC: @create-multiple-face-prompts ac-spatial-matching
    def test_batch_seeds_and_geometry_are_independent(self):
        self.run_faces([[segment(50), segment(0)], [segment(80), segment(20)]], vary_seed_per_face=True)
        self.assertEqual([(entry[0], entry[1], entry[2][0], entry[3]["seed"]) for entry in self.passes],
                         [(0, 0, 0, 42), (0, 1, 50, 43), (1, 0, 20, 43), (1, 1, 80, 44)])

    # AC: @create-multiple-face-prompts ac-single-face-compatible
    def test_empty_and_single_row_delegate_without_rejecting_legacy_options(self):
        for rows in ([], ROWS[:1]):
            self.assertEqual(self.run_faces(rows=rows, edit_mode=True, face_reference=Image(), wildcard="legacy"), ("legacy",) * 6)
        self.assertEqual(len(self.legacy), 2)
        self.assertEqual(self.passes, [])
        self.assertEqual(runtime.DMCFaceDetailer.INPUT_TYPES()["optional"]["face_prompts_json"][1]["default"], "[]")

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_strict_character_and_position_schema(self):
        for invalid in ([{**ROWS[0], "position": "order"}, ROWS[1]], [ROWS[1], ROWS[0]],
                        [ROWS[0], {**ROWS[1], "position": "left"}], [ROWS[0], {**ROWS[1], "character": "ada"}],
                        [{**ROWS[0], "position": "1st"}, ROWS[1]], [{**ROWS[0], "face_prompt": ""}, ROWS[1]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime.face_prompts(json.dumps(invalid))
        self.assertEqual(len(runtime.face_prompts(json.dumps([{**row, "position": str(index + 1) + suffix}
            for index, (row, suffix) in enumerate(zip(ROWS, ["st", "nd"]))]))), 2)
        maximum = [{"character": "Character " + str(index), "position": position, "face_prompt": "\u263a" * 8000}
                   for index, position in enumerate(runtime.ORDINALS)]
        self.assertEqual(len(runtime.face_prompts(json.dumps(maximum))), 16)


class OwnedInstallerTests(unittest.TestCase):
    # AC: @create-multiple-face-prompts ac-single-face-compatible
    def test_owned_sources_activate_at_startup_and_checksums_detect_modifications(self):
        with tempfile.TemporaryDirectory(prefix="dmc-synthetic-runtime-") as directory:
            manager = installer.Installer(directory)
            manager.backend_root.mkdir(parents=True)
            installer._atomic_json(manager.backend_root / ".donut-create-source.json", {"sha256": manager.manifest["comfyui"]["sha256"]})
            restarted = installer.Installer(directory)
            self.assertTrue(restarted._owned_nodes_match())
            restarted.ensure_owned_nodes()
            source = restarted.backend_root / "custom_nodes/dmc_create/__init__.py"
            source.write_text("Synthetic user modification")
            self.assertFalse(restarted._owned_nodes_match())
            with self.assertRaisesRegex(installer.SetupError, "modified"):
                restarted.ensure_owned_nodes()
            self.assertEqual(source.read_text(), "Synthetic user modification")
            service = installer.Installer(directory)
            self.assertEqual(service._status["state"], "error")
            self.assertIn("modified", service._status["error"])

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_unowned_folder_is_preserved_and_owned_source_update_is_checked(self):
        with tempfile.TemporaryDirectory(prefix="dmc-synthetic-runtime-") as directory:
            manager = installer.Installer(directory)
            target = manager.backend_root / "custom_nodes/dmc_create"
            target.mkdir(parents=True)
            with self.assertRaisesRegex(installer.SetupError, "not owned"):
                manager.ensure_owned_nodes()
            target.rmdir()
            manager.ensure_owned_nodes()
            with tempfile.TemporaryDirectory(prefix="dmc-synthetic-source-") as source_dir:
                Path(source_dir, "dmc_face_detailer.py").write_text("Synthetic upgraded source")
                with patch.object(installer, "ROOT", Path(source_dir)):
                    manager.ensure_owned_nodes()
                    self.assertTrue(manager._owned_nodes_match())
            self.assertEqual((target / "__init__.py").read_text(), "Synthetic upgraded source")

    # AC: @create-multiple-face-prompts ac-mismatch
    def test_interrupted_owned_update_recovers_and_initial_activation_is_atomic(self):
        with tempfile.TemporaryDirectory(prefix="dmc-synthetic-runtime-") as directory:
            manager = installer.Installer(directory)
            with patch.object(installer, "_atomic_json", side_effect=OSError("Synthetic interrupted activation")):
                with self.assertRaises(OSError):
                    manager.ensure_owned_nodes()
            self.assertFalse((manager.backend_root / "custom_nodes/dmc_create").exists())
            manager.ensure_owned_nodes()
            with tempfile.TemporaryDirectory(prefix="dmc-synthetic-source-") as source_dir:
                Path(source_dir, "dmc_face_detailer.py").write_text("Synthetic upgraded source")
                with patch.object(installer, "ROOT", Path(source_dir)):
                    with patch.object(installer, "_atomic_json", side_effect=OSError("Synthetic interrupted receipt")):
                        with self.assertRaises(OSError):
                            manager.ensure_owned_nodes()
                    self.assertFalse(manager._owned_nodes_match())
                    manager.ensure_owned_nodes()
                    self.assertTrue(manager._owned_nodes_match())


if __name__ == "__main__":
    unittest.main()
