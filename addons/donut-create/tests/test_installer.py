"""Synthetic behavioral checks; fixtures stay outside the repository.

No model weights, backend packages, ComfyUI services, or GPU imports are used.
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("donut_create_installer", ROOT / "installer.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def model(data=b"synthetic model content", filename="weights.safetensors"):
    return {"folder": "diffusion_models", "filename": filename, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "url": "https://huggingface.co/synthetic/resolve/pin/" + filename}


class Response:
    def __init__(self, data, code=200, headers=None, callback=None):
        self.stream = io.BytesIO(data)
        self.code = code
        self.headers = headers or {}
        self.callback = callback

    def getcode(self):
        return self.code

    def read(self, length):
        # Small chunks let tests exercise cancellation without large fixtures.
        data = self.stream.read(min(length, 8))
        if data and self.callback:
            self.callback()
        return data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Opener:
    def __init__(self, data, ignore_range=False, callback=None):
        self.data = data
        self.requests = []
        self.ignore_range = ignore_range
        self.callback = callback

    def open(self, request, timeout):
        self.requests.append(request)
        offset = int(request.get_header("Range", "bytes=0-")[6:-1])
        if offset and not self.ignore_range:
            return Response(self.data[offset:], 206, {"Content-Range": f"bytes {offset}-{len(self.data)-1}/{len(self.data)}"}, self.callback)
        return Response(self.data, callback=self.callback)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="donut-create-installer-test-")
        self.root = Path(self.temporary.name)
        self.installer = module.Installer(self.root / "managed")

    def tearDown(self):
        self.installer.cancel()
        if self.installer._thread:
            self.installer._thread.join(timeout=8)
        self.temporary.cleanup()

    # AC: @donut-create-plugin ac-managed-setup
    def test_catalog_and_source_pins_are_complete_and_immutable(self):
        self.assertEqual(len(self.installer.catalog["models"]), 16)
        self.assertEqual(sum(m["size"] for m in self.installer.catalog["models"]), 63_162_364_113)
        manifest = self.installer.manifest
        self.assertEqual(manifest["comfyui"]["version"], "0.36.0")
        self.assertEqual(manifest["comfyui"]["frontend"], "1.53.6")
        for source in [manifest["comfyui"], *manifest["node_packs"], manifest["sam2"]]:
            self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(source["url"].endswith(source["commit"]))
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(source["size"], 0)
        for node in ("DonutWorkflowPanel", "DonutLatestPreview", "DonutModelDownloads", "Note"):
            self.assertNotIn(node, manifest["required_nodes"])
        for node in ("DonutVAELoader", "DonutVAEDecode", "BlehSetSamplerPreset", "Krea2EditModelPatch"):
            self.assertIn(node, manifest["required_nodes"])
        self.assertEqual(len(self.installer._models("base")), 6)
        self.assertEqual(len(self.installer._models("workflow")), 8)
        self.assertEqual(len(self.installer._models("all")), 16)
        self.installer._status["profile"] = "all"
        self.assertTrue(any(s.get("directory") == "ComfyUI-VAE-Utils" for s, _ in self.installer._sources()))

    # AC: @donut-create-plugin ac-managed-setup
    def test_status_exposes_profile_disk_and_runtime_choices(self):
        status = self.installer.status()
        self.assertEqual(status["state"], "idle")
        self.assertFalse(status["ready"])
        self.assertFalse(status["running"])
        self.assertEqual([p["id"] for p in status["catalog"]["profiles"]], ["base", "workflow", "all"])
        self.assertGreater(status["disk"]["required_bytes"], status["catalog"]["profiles"][0]["total_bytes"])
        with patch.object(module.platform, "system", return_value="Darwin"), patch.object(module.platform, "machine", return_value="arm64"):
            self.assertEqual(self.installer.supported_runtimes(), ["cpu", "mps"])
            self.assertEqual(self.installer.default_runtime(), "mps")
        with patch.object(module.platform, "system", return_value="Windows"), patch.object(module.platform, "machine", return_value="arm64"):
            self.assertEqual(self.installer.supported_runtimes(), [])

    # AC: @donut-create-plugin ac-managed-setup
    def test_bundled_graph_keeps_real_bindings_and_safe_png_metadata(self):
        workflow = json.loads((ROOT / "workflow.json").read_text())
        self.assertEqual(len(workflow["nodes"]), 25)
        self.assertEqual([len(g["nodes"]) for g in workflow["definitions"]["subgraphs"]], [19, 7, 17, 3])
        self.assertEqual(workflow["extra"]["donut_workflow"]["release"], "V5")
        self.assertNotIn("linearData", workflow["extra"])
        self.assertNotIn("anomalous_hashes", workflow["extra"])
        by_id = {n["id"]: n for graph in [workflow, *workflow["definitions"]["subgraphs"]] for n in graph["nodes"]}
        self.assertEqual(by_id[64]["widgets_values"][5], "png")
        self.assertTrue(by_id[64]["widgets_values"][11])
        self.assertFalse(by_id[64]["widgets_values"][10])
        self.assertEqual(by_id[1153]["widgets_values"][0], "DonutCreate")
        self.assertEqual(by_id[996]["widgets_values"][16], "[]")
        self.assertTrue(by_id[1126]["widgets_values"][0].endswith(by_id[56]["widgets_values"][0]))
        prompt_controls = by_id[1140]["properties"]["donut_app_controls"]["groups"][0]["controls"]
        self.assertEqual([c["path"] for c in prompt_controls], [[1138, 1125], [1138, 1126], [1138, 1127], [1138, 1178]])
        self.assertNotIn("last_image", by_id[1152]["properties"])
        self.assertNotIn("stage_prompts", by_id[1152]["properties"])
        text = (ROOT / "workflow.json").read_text()
        self.assertNotRegex(text, r"/home/|/Users/|data:image/|[A-Z]:\\\\")

    # AC: @donut-create-plugin ac-managed-setup
    def test_explicit_base_preset_changes_models_but_original_profile_keeps_enabled_lora(self):
        original = json.loads((ROOT / "workflow.json").read_text())
        bundled_base = json.loads((ROOT / "assets/base-workflow.json").read_text())
        self.installer._workflow("base")
        base = json.loads((self.installer.state_dir / "workflow.json").read_text())
        self.assertEqual(bundled_base, base)
        by_id = lambda w: {n["id"]: n for g in [w, *w["definitions"]["subgraphs"]] for n in g["nodes"]}
        self.assertEqual(by_id(base)[1122]["widgets_values"][0], "krea2_turbo_bf16.safetensors")
        self.assertEqual(by_id(base)[1124]["widgets_values"][-1], "Single model")
        self.assertEqual(by_id(base)[1055]["widgets_values"][1], "[]")
        self.assertEqual(base["links"], original["links"])
        for a, b in zip(base["definitions"]["subgraphs"], original["definitions"]["subgraphs"]):
            self.assertEqual(a["links"], b["links"])
            self.assertEqual(a["inputs"], b["inputs"])
            self.assertEqual(a["outputs"], b["outputs"])
        self.installer._workflow("workflow")
        exact = json.loads((self.installer.state_dir / "workflow.json").read_text())
        self.assertEqual(exact, original)
        self.assertTrue(json.loads(by_id(exact)[1055]["widgets_values"][1])[0]["enabled"])

    # AC: @donut-create-plugin ac-workflow-state
    def test_shipped_recipe_defaults_and_provenance(self):
        self.installer.state_dir.mkdir(parents=True, exist_ok=True)
        (self.installer.state_dir / "workflow.json").write_text('{"nodes":[],"stale":true}')
        self.assertEqual(self.installer.workflow_path, ROOT / "assets/base-workflow.json")
        self.installer._status["profile"] = "workflow"
        self.assertEqual(self.installer.workflow_path, ROOT / "workflow.json")
        for file in (ROOT / "workflow.json", ROOT / "assets/base-workflow.json"):
            workflow = json.loads(file.read_text())
            nodes = {node["id"]: node for graph in [workflow, *workflow["definitions"]["subgraphs"]] for node in graph["nodes"]}
            for node_id, expected in {
                1014: {"sampler_name": "euler", "scheduler": "simple", "alpha": 0.45, "compatibility_preset": "Rebalance"},
                993: {"sampler_name": "euler", "scheduler": "simple", "nag_enabled": True, "nag_alpha": 0.45},
                1118: {"compatibility_preset": "Rebalance", "uncensorfix_controls": "Fusion only"},
            }.items():
                node = nodes[node_id]
                named = node["widgets_values_named"]
                order = node.get("properties", {}).get("donut_widget_order") or list(named)
                for key, value in expected.items():
                    self.assertEqual(named[key], value)
                    self.assertEqual(node["widgets_values"][order.index(key)], value)
            for row in json.loads(nodes[1055]["widgets_values_named"]["slots_json"]):
                if row["lora_name"] == "krea2/Krea2_NSFW_Aesthetics_V1.safetensors":
                    self.assertEqual((row["model_weight"], row["clip_weight"]), (1, 1))
            self.assertEqual(workflow["extra"]["donut_workflow"]["controls_revision"], 1)
        provenance = self.installer.manifest["provenance"]
        self.assertEqual(provenance["workflow"]["bundled_sha256"], hashlib.sha256((ROOT / "workflow.json").read_bytes()).hexdigest())
        for asset, expected in provenance["donutui"]["browser_assets"].items():
            self.assertEqual(expected, hashlib.sha256((ROOT / asset).read_bytes()).hexdigest(), asset)

    # AC: @donut-create-plugin ac-managed-setup
    def test_exact_download_is_activated_atomically_and_verified_file_is_reused(self):
        data = b"synthetic model content"
        item = model(data)
        self.installer._opener = Opener(data)
        path = self.installer._model(item, None, 0)
        self.assertEqual(path.read_bytes(), data)
        self.assertFalse((self.installer.state_dir / "downloads" / (item["sha256"] + ".part")).exists())
        self.assertEqual(self.installer._model(item, None, 0), path)
        self.assertEqual(len(self.installer._opener.requests), 1)

    # AC: @donut-create-plugin ac-setup-recovery
    def test_completed_models_reduce_retry_disk_requirement_without_final_ready_receipt(self):
        data = b"synthetic completed model"
        item = model(data)
        self.installer.catalog["models"] = [item]
        self.installer.manifest["profiles"]["base"]["required_models"] = [{"folder": item["folder"], "filename": item["filename"]}]
        reserve = self.installer.manifest["environment_reserve_bytes"]
        self.assertEqual(self.installer._disk()["required_bytes"], reserve + len(data))
        self.installer._opener = Opener(data)
        self.installer._model(item, None, 0)
        self.assertFalse((self.installer.state_dir / "ready.json").exists())
        self.assertEqual(self.installer._disk()["required_bytes"], reserve)
        restored = module.Installer(self.installer.state_dir)
        restored.catalog = self.installer.catalog
        restored.manifest = self.installer.manifest
        self.assertEqual(restored._disk()["required_bytes"], reserve)

    # AC: @donut-create-plugin ac-setup-recovery
    def test_conflicting_managed_and_external_files_are_preserved(self):
        item = model()
        target = self.installer.backend_root / "models" / item["folder"] / item["filename"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"a different existing model")
        with self.assertRaisesRegex(module.SetupError, "conflicting model was preserved"):
            self.installer._model(item, None, 0)
        self.assertEqual(target.read_bytes(), b"a different existing model")
        target.unlink()
        external = self.root / "external-models"
        reused = external / item["folder"] / item["filename"]
        reused.parent.mkdir(parents=True)
        reused.write_bytes(b"user-owned different weights")
        with self.assertRaisesRegex(module.SetupError, "It was preserved"):
            self.installer._model(item, external, 0)
        self.assertEqual(reused.read_bytes(), b"user-owned different weights")
        self.assertFalse(target.exists())

    # AC: @donut-create-plugin ac-setup-recovery
    def test_authorized_external_model_is_read_in_place_without_copying(self):
        data = b"synthetic reused weights"
        item = model(data)
        external = self.root / "external-models"
        path = external / item["folder"] / item["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        before = path.stat().st_mtime_ns
        self.assertEqual(self.installer._model(item, external, 0), path)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertFalse((self.installer.backend_root / "models").exists())

    # AC: @donut-create-plugin ac-setup-recovery
    def test_cancel_preserves_partial_retry_resumes_and_then_activates(self):
        data = b"the synthetic weight content is deliberately longer than one chunk"
        item = model(data)
        counter = [0]
        def cancel_after_second_chunk():
            counter[0] += 1
            if counter[0] == 2:
                self.installer._cancel.set()
        self.installer._opener = Opener(data, callback=cancel_after_second_chunk)
        with self.assertRaises(module.SetupCancelled):
            self.installer._model(item, None, 0)
        partial = self.installer.state_dir / "downloads" / (item["sha256"] + ".part")
        self.assertEqual(partial.stat().st_size, 16)
        self.assertFalse((self.installer.backend_root / "models" / item["folder"] / item["filename"]).exists())
        self.installer._cancel.clear()
        self.installer._opener = Opener(data)
        path = self.installer._model(item, None, 0)
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(self.installer._opener.requests[0].get_header("Range"), "bytes=16-")

    # AC: @donut-create-plugin ac-setup-recovery
    def test_provider_ignoring_range_restarts_partial_without_corruption(self):
        data = b"synthetic content with a partial"
        item = model(data)
        cache = self.installer.state_dir / "downloads"
        cache.mkdir(parents=True)
        (cache / (item["sha256"] + ".part")).write_bytes(b"synthetic")
        self.installer._opener = Opener(data, ignore_range=True)
        self.assertEqual(self.installer._model(item, None, 0).read_bytes(), data)

    # AC: @donut-create-plugin ac-setup-recovery
    def test_checksum_mismatch_never_activates_and_removes_invalid_partial(self):
        data = b"right synthetic bytes"
        item = model(data)
        self.installer._opener = Opener(b"wrong synthetic bytes")
        with self.assertRaisesRegex(module.SetupError, "Checksum mismatch"):
            self.installer._model(item, None, 0)
        self.assertFalse((self.installer.state_dir / "downloads" / (item["sha256"] + ".part")).exists())
        self.assertFalse((self.installer.backend_root / "models" / item["folder"] / item["filename"]).exists())

    # AC: @donut-create-plugin ac-setup-recovery
    def test_redirect_drops_credentials_across_origin_and_keeps_same_origin(self):
        handler = module.SafeRedirects()
        request = urllib.request.Request("https://huggingface.co/a", headers={"Authorization": "Bearer synthetic-secret"})
        cross = handler.redirect_request(request, None, 302, "", {}, "https://cdn.example.test/model")
        same = handler.redirect_request(request, None, 302, "", {}, "https://huggingface.co/b")
        self.assertIsNone(cross.get_header("Authorization"))
        self.assertEqual(same.get_header("Authorization"), "Bearer synthetic-secret")
        with self.assertRaises(module.SetupError):
            handler.redirect_request(request, None, 302, "", {}, "http://cdn.example.test/model")

    # AC: @donut-create-plugin ac-setup-recovery
    def test_failed_auth_is_visible_and_secrets_never_enter_persistent_state(self):
        self.installer._tokens = {"hf_token": "hf_synthetic_secret", "civitai_api_key": "civitai_synthetic_secret"}
        class Denied:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, None)
        self.installer._opener = Denied()
        with self.assertRaisesRegex(module.SetupError, "Access denied.*Hugging Face"):
            self.installer._download(model())
        self.installer._update(state="error", error=self.installer._redact("hf_synthetic_secret civitai_synthetic_secret"))
        state = (self.installer.state_dir / "setup.json").read_text()
        self.assertNotIn("hf_synthetic_secret", state)
        self.assertNotIn("civitai_synthetic_secret", state)

    # AC: @donut-create-plugin ac-setup-recovery
    def test_interrupted_state_is_retryable_and_partial_is_retained(self):
        item = model()
        self.installer._update(state="installing", phase="models")
        cache = self.installer.state_dir / "downloads"
        cache.mkdir()
        partial = cache / (item["sha256"] + ".part")
        partial.write_bytes(b"synthetic")
        restored = module.Installer(self.installer.state_dir)
        self.assertEqual(restored.status()["state"], "cancelled")
        self.assertIn("Retry", restored.status()["error"])
        self.assertEqual(partial.read_bytes(), b"synthetic")

    # AC: @donut-create-plugin ac-managed-setup
    def test_preflight_failure_does_not_install_and_clears_tokens(self):
        self.installer._status.update(runtime="cpu", profile="base")
        self.installer._tokens = {"hf_token": "synthetic_secret"}
        with patch.object(self.installer, "_run"), patch.object(self.installer, "_python_command", return_value=[sys.executable]), patch.object(self.installer, "_disk", return_value={"required_bytes": 1000, "available_bytes": 0}), patch.object(self.installer, "_source") as source:
            self.installer._install("cpu", "base", None)
        self.assertEqual(self.installer.status()["state"], "error")
        self.assertIn("Insufficient free disk", self.installer.status()["error"])
        source.assert_not_called()
        self.assertEqual(self.installer._tokens, {})

    # AC: @donut-create-plugin ac-managed-setup
    def test_ready_requires_real_capability_proof_and_unchanged_required_files(self):
        self.installer._status.update(state="ready", runtime="cpu", profile="base")
        fixture = model()
        self.installer.catalog["models"] = [fixture]
        self.installer.manifest["profiles"]["base"]["required_models"] = [{"folder": fixture["folder"], "filename": fixture["filename"]}]
        self.installer.python_path.parent.mkdir(parents=True)
        self.installer.python_path.write_text("synthetic interpreter stand-in")
        self.installer._workflow("base")
        for source, target in self.installer._sources():
            target.mkdir(parents=True, exist_ok=True)
            (target / ("main.py" if target == self.installer.backend_root else "__init__.py")).write_text("# synthetic source stand-in")
            module._atomic_json(target / ".donut-create-source.json", {"sha256": source["sha256"]})
        records = {}
        for item in self.installer._models():
            path = self.root / "synthetic-models" / item["folder"] / item["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic model content")
            records[item["folder"] + "/" + item["filename"]] = {"path": str(path), "sha256": item["sha256"], "fingerprint": module._fingerprint(path)}
        receipt = {"manifest": self.installer._manifest_hash, "runtime": "cpu", "profile": "base", "nodes": self.installer.expected_capabilities()["required_nodes"], "models": records, "capability_check": False}
        module._atomic_json(self.installer.state_dir / "ready.json", receipt)
        self.assertFalse(self.installer.ready())
        receipt["capability_check"] = True
        module._atomic_json(self.installer.state_dir / "ready.json", receipt)
        self.assertTrue(self.installer.ready())
        first = next(iter(records.values()))
        Path(first["path"]).write_bytes(b"edited")
        self.assertFalse(self.installer.ready())

    # AC: @donut-create-plugin ac-setup-recovery
    def test_source_traversal_is_rejected_and_existing_source_preserved(self):
        archive = self.root / "unsafe.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("source/../../escaped")
            info.size = 4
            tar.addfile(info, io.BytesIO(b"nope"))
        source = {"commit": "0" * 40, "sha256": "1" * 64, "size": archive.stat().st_size}
        target = self.root / "extract" / "source"
        with patch.object(self.installer, "_download", return_value=archive):
            with self.assertRaisesRegex(module.SetupError, "unsafe entry"):
                self.installer._source(source, target)
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "escaped").exists())
        target.mkdir(parents=True)
        (target / "private-file.txt").write_text("synthetic existing content")
        with self.assertRaisesRegex(module.SetupError, "preserved"):
            self.installer._source(source, target)
        self.assertEqual((target / "private-file.txt").read_text(), "synthetic existing content")

    # AC: @donut-create-plugin ac-managed-setup
    def test_pinned_sam2_internal_yaml_links_are_materialized_without_symlinks(self):
        archive = self.root / "links.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            content = tarfile.TarInfo("source/configs/model.yaml")
            content.size = 11
            tar.addfile(content, io.BytesIO(b"model: test"))
            link = tarfile.TarInfo("source/model.yaml")
            link.type = tarfile.SYMTYPE
            link.linkname = "configs/model.yaml"
            tar.addfile(link)
        source = {"commit": "0" * 40, "sha256": "1" * 64, "size": archive.stat().st_size}
        target = self.root / "safe-extract"
        with patch.object(self.installer, "_download", return_value=archive):
            self.installer._source(source, target)
        self.assertEqual((target / "model.yaml").read_bytes(), b"model: test")
        self.assertFalse((target / "model.yaml").is_symlink())
        with tarfile.open(archive, "w:gz") as tar:
            link = tarfile.TarInfo("source/model.yaml")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside.yaml"
            tar.addfile(link)
        with patch.object(self.installer, "_download", return_value=archive):
            with self.assertRaisesRegex(module.SetupError, "unsafe link"):
                self.installer._source(source, self.root / "unsafe-link-extract")

    # AC: @donut-create-plugin ac-setup-recovery
    def test_invalid_resume_response_leaves_partial_unchanged(self):
        data = b"synthetic content for resume"
        item = model(data)
        cache = self.installer.state_dir / "downloads"
        cache.mkdir(parents=True)
        partial = cache / (item["sha256"] + ".part")
        partial.write_bytes(data[:5])
        class InvalidRange:
            def open(self, request, timeout):
                return Response(data[5:], 206, {"Content-Range": f"bytes 6-{len(data)-1}/{len(data)}"})
        self.installer._opener = InvalidRange()
        with self.assertRaisesRegex(module.SetupError, "invalid resume range"):
            self.installer._model(item, None, 0)
        self.assertEqual(partial.read_bytes(), data[:5])

    # AC: @donut-create-plugin ac-setup-recovery
    def test_auth_failure_in_setup_thread_keeps_profile_and_clears_supplied_tokens(self):
        item = model()
        class Denied:
            def open(self, request, timeout):
                self.request = request
                raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, None)
        opener = Denied()
        self.installer._opener = opener
        with patch.object(self.installer, "_models", return_value=[item]), patch.object(self.installer, "_run"), patch.object(self.installer, "_python_command", return_value=[sys.executable]), patch.object(self.installer, "_source"), patch.object(self.installer, "_environment"), patch.object(self.installer, "_dependencies"), patch.object(self.installer, "_disk", return_value={"required_bytes": 100, "available_bytes": 100 * 1024**3}):
            self.installer.start({"runtime": "cpu", "profile": "workflow", "hf_token": "hf_synthetic_secret", "civitai_api_key": "civitai_synthetic_secret"})
            self.installer._thread.join(timeout=5)
        self.assertEqual(self.installer._status["state"], "error")
        self.assertEqual(self.installer._status["profile"], "workflow")
        self.assertEqual(self.installer._tokens, {})
        self.assertIn("Access denied", self.installer._status["error"])
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer hf_synthetic_secret")
        state = (self.installer.state_dir / "setup.json").read_text()
        self.assertNotIn("hf_synthetic_secret", state)
        self.assertNotIn("civitai_synthetic_secret", state)
        self.assertIn("Access denied", state)

    # AC: @donut-create-plugin ac-setup-recovery
    @unittest.skipIf(os.name == "nt", "POSIX process-group assertion; Windows uses taskkill /T")
    def test_cancel_stops_owned_command_tree_in_sidecar_group(self):
        pidfile = self.root / "synthetic-child.pid"
        command = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
        failure = []
        def run():
            try:
                self.installer._run([sys.executable, "-c", command, str(pidfile)], timeout=15)
            except module.SetupCancelled:
                failure.append("cancelled")
        worker = threading.Thread(target=run)
        worker.start()
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(pidfile.exists())
        child_pid = int(pidfile.read_text())
        self.assertEqual(os.getpgid(child_pid), os.getpgrp())
        self.installer._cancel.set()
        worker.join(timeout=8)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failure, ["cancelled"])
        if Path("/proc").is_dir():
            stat = Path(f"/proc/{child_pid}/stat")
            self.assertTrue(not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z "))


if __name__ == "__main__":
    unittest.main()
