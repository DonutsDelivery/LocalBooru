"""Synthetic source updates only: no real ComfyUI, user profile or model downloads."""
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import httpx
from installer import Installer, SetupCancelled, SetupError, LEGACY_FACE_NODE_SHA256


def comfy(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# synthetic ComfyUI\n")
    (root / "comfy").mkdir(exist_ok=True)
    return root


def archive(root, label="new", entries=None):
    path = root / (label + ".tar.gz")
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in (entries or {"pack/__init__.py": label, "pack/dmc_face_detailer.py": label}).items():
            member = tarfile.TarInfo(name)
            data = content.encode()
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return path, {"directory": "donutnodes", "repo": "fixture/source", "commit": hashlib.sha1(label.encode()).hexdigest(),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size, "url": "https://fixture.invalid/pinned"}


class SourceUpdates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dmc-source-fixture-")
        self.root = Path(self.temp.name)
        self.installer = Installer(self.root / "state")
        self.comfy = comfy(self.root / "connected")
        self.target = self.comfy / "custom_nodes" / "donutnodes"
        self.old_archive, self.old = archive(self.root, "old")
        self.new_archive, self.new = archive(self.root)

    def tearDown(self):
        self.installer.cancel()
        if self.installer._thread:
            self.installer._thread.join(timeout=3)
        self.temp.cleanup()

    def install_old(self):
        with patch.object(self.installer, "_download", return_value=self.old_archive):
            self.installer._source(self.old, self.target)

    def update(self):
        with patch.object(self.installer, "_download", return_value=self.new_archive):
            self.installer._source(self.new, self.target, update=True)

    def test_source_update_replaces_verified_files_and_receipt(self):
        self.install_old()
        self.update()
        self.assertEqual((self.target / "__init__.py").read_text(), "new")
        receipt = json.loads((self.target / ".donut-create-source.json").read_text())
        self.assertEqual(receipt["commit"], self.new["commit"])
        self.assertEqual(receipt["files"], self.installer._source_files(self.target))
        self.assertFalse(list(self.target.parent.glob(".donut-source-*")))

    def test_modified_extra_unowned_and_old_receipt_sources_are_preserved(self):
        for conflict in ("modified", "extra", "unowned", "old-receipt"):
            with self.subTest(conflict=conflict):
                target = self.comfy / "custom_nodes" / conflict
                with patch.object(self.installer, "_download", return_value=self.old_archive):
                    self.installer._source(self.old, target)
                if conflict == "modified":
                    (target / "__init__.py").write_text("local changes")
                elif conflict == "extra":
                    (target / "extra.txt").write_text("keep")
                elif conflict == "unowned":
                    (target / ".donut-create-source.json").unlink()
                else:
                    (target / ".donut-create-source.json").write_text(json.dumps({"commit": self.old["commit"], "sha256": self.old["sha256"]}))
                before = {p.name: p.read_bytes() for p in target.iterdir()}
                with self.assertRaises(SetupError), patch.object(self.installer, "_download") as download:
                    self.installer._source(self.new, target, update=True)
                self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})
                download.assert_not_called()

    def test_symlink_target_and_source_are_preserved(self):
        self.install_old()
        outside = self.root / "outside.py"
        outside.write_text("keep")
        (self.target / "__init__.py").unlink()
        (self.target / "__init__.py").symlink_to(outside)
        with self.assertRaises(SetupError):
            self.update()
        self.assertEqual(outside.read_text(), "keep")
        linked = self.comfy / "custom_nodes" / "linked"
        linked.symlink_to(self.target, target_is_directory=True)
        with self.assertRaises(SetupError):
            self.installer._source(self.new, linked, update=True)

    def test_extra_files_are_rejected_before_reading_any_file_contents(self):
        self.install_old()
        (self.target / "unowned-config.txt").write_text("synthetic content that must not be read")
        with patch.object(self.installer, "_hash") as hashed, self.assertRaises(SetupError):
            self.update()
        hashed.assert_not_called()

    def test_trusted_old_archive_verifies_legacy_receipt_before_upgrade(self):
        self.install_old()
        marker = self.target / ".donut-create-source.json"
        marker.write_text(json.dumps({"commit": self.old["commit"], "sha256": self.old["sha256"]}))
        with patch.object(self.installer, "_download", side_effect=lambda item: self.old_archive if item["sha256"] == self.old["sha256"] else self.new_archive):
            self.installer._source({**self.new, "previous_revisions": [self.old]}, self.target, update=True)
        self.assertEqual((self.target / "__init__.py").read_text(), "new")

    def test_trusted_old_archive_does_not_override_local_changes(self):
        self.install_old()
        marker = self.target / ".donut-create-source.json"
        marker.write_text(json.dumps({"commit": self.old["commit"], "sha256": self.old["sha256"]}))
        (self.target / "__init__.py").write_text("local changes")
        with patch.object(self.installer, "_download", return_value=self.old_archive) as download, self.assertRaises(SetupError):
            self.installer._source({**self.new, "previous_revisions": [self.old]}, self.target, update=True)
        download.assert_called_once_with(self.old)
        self.assertEqual((self.target / "__init__.py").read_text(), "local changes")

    def test_archive_traversal_rejection_leaves_old_source(self):
        self.install_old()
        for name in ("pack/../../escape.py", "/absolute/escape.py", "pack/..\\escape.py", "pack/C:/escape.py"):
            with self.subTest(name=name):
                bad_archive, bad = archive(self.root, "bad", {"pack/__init__.py": "new", name: "escape"})
                with patch.object(self.installer, "_download", return_value=bad_archive), self.assertRaises(SetupError):
                    self.installer._source(bad, self.target, update=True)
                self.assertEqual((self.target / "__init__.py").read_text(), "old")
        self.assertFalse((self.root / "escape.py").exists())

    def test_cancel_before_activation_preserves_source(self):
        self.install_old()
        with patch.object(self.installer, "_download", side_effect=SetupCancelled("cancelled")), self.assertRaises(SetupCancelled):
            self.installer._source(self.new, self.target, update=True)
        self.assertEqual((self.target / "__init__.py").read_text(), "old")

    def test_activation_failure_rolls_back_source(self):
        self.install_old()
        original = Path.rename
        def fail_activation(path, destination):
            if path.name.startswith(".donut-source-") and not path.name.endswith("-previous") and destination == self.target:
                raise OSError("synthetic activation failure")
            return original(path, destination)
        with patch.object(Path, "rename", fail_activation), self.assertRaises(OSError):
            self.update()
        self.assertEqual((self.target / "__init__.py").read_text(), "old")

    def test_updates_serialize_with_setup_and_cancel(self):
        entered = threading.Event()
        def waiting_download(item):
            entered.set()
            self.installer._cancel.wait(3)
            self.installer._check_cancel()
            return self.new_archive
        with patch.object(self.installer, "required_donutnodes", return_value=self.new), patch.object(self.installer, "_download", side_effect=waiting_download):
            self.installer.start_node_update(self.comfy)
            self.assertTrue(entered.wait(1))
            with self.assertRaises(SetupError):
                self.installer.start_node_update(self.comfy)
            with self.assertRaises(SetupError):
                self.installer.start({"runtime": "cpu", "profile": "base"})
            self.installer.cancel()
            self.installer._thread.join(timeout=3)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.installer.status()["node_update"]["state"], "cancelled")

    def test_managed_receipt_refresh_restores_ready_after_pin_change(self):
        comfy(self.installer.backend_root)
        self.installer._status["state"] = "error"
        marker = self.installer.state_dir / "ready.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"manifest": "old-manifest", "capability_check": True}))
        with patch.object(self.installer, "required_donutnodes", return_value=self.new), patch.object(self.installer, "_download", return_value=self.new_archive), patch.object(self.installer, "_ready_files", return_value=True) as ready:
            self.installer._update_nodes(self.installer.backend_root, "error")
        ready.assert_called_once_with(ignore_manifest=True)
        self.assertEqual(self.installer._status["state"], "ready")
        self.assertEqual(json.loads(marker.read_text())["manifest"], self.installer._manifest_hash)

    def test_failed_remaining_runtime_check_does_not_refresh_receipt(self):
        comfy(self.installer.backend_root)
        marker = self.installer.state_dir / "ready.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"manifest": "old-manifest", "capability_check": True}))
        with patch.object(self.installer, "required_donutnodes", return_value=self.new), patch.object(self.installer, "_download", return_value=self.new_archive), patch.object(self.installer, "_ready_files", return_value=False):
            self.installer._update_nodes(self.installer.backend_root, "error")
        self.assertEqual(self.installer._status["state"], "error")
        self.assertEqual(json.loads(marker.read_text())["manifest"], "old-manifest")

    def test_legacy_collision_preserved_unless_receipt_and_source_match(self):
        self.install_old()
        legacy = self.comfy / "custom_nodes" / "dmc_create"
        legacy.mkdir()
        (legacy / "__init__.py").write_text("synthetic legacy")
        (legacy / ".donut-create-owned.json").write_text(json.dumps({"owner": "donut-create", "files": {"__init__.py": LEGACY_FACE_NODE_SHA256}}))
        with self.assertRaises(SetupError):
            self.installer._remove_legacy_node(self.comfy)
        self.assertTrue(legacy.exists())
        with patch.object(self.installer, "_hash", return_value=LEGACY_FACE_NODE_SHA256):
            (legacy / "extra.txt").write_text("keep")
            with self.assertRaises(SetupError):
                self.installer._remove_legacy_node(self.comfy)
            (legacy / "extra.txt").unlink()
            self.installer._remove_legacy_node(self.comfy)
        self.assertFalse(legacy.exists())

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_linked_git_checkout_is_verified_without_replacing_link(self):
        repo = self.root / "linked-source"
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        def git(*args):
            return subprocess.run(["git", *map(str, args)], check=True, capture_output=True, text=True, env=env).stdout.strip()
        git("init", repo)
        git("-C", repo, "config", "user.name", "Synthetic Fixture")
        git("-C", repo, "config", "user.email", "fixture@example.invalid")
        (repo / "__init__.py").write_text("old")
        git("-C", repo, "add", ".")
        git("-C", repo, "commit", "-m", "old")
        first = git("-C", repo, "rev-parse", "HEAD")
        (repo / "dmc_face_detailer.py").write_text("new")
        git("-C", repo, "add", ".")
        git("-C", repo, "commit", "-m", "pinned")
        pin = git("-C", repo, "rev-parse", "HEAD")
        git("-C", repo, "reset", "--hard", first)
        git("-C", repo, "remote", "add", "origin", "https://github.com/" + self.new["repo"] + ".git")
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to(repo, target_is_directory=True)
        run = self.installer._run
        def local_fetch(command, **kwargs):
            if "fetch" in command:
                command = list(command)
                command[command.index("--no-tags") + 1] = str(repo)
            return run(command, extra_env=env, **kwargs)
        with patch.object(self.installer, "required_donutnodes", return_value={**self.new, "commit": pin}), patch.object(self.installer, "_run", side_effect=local_fetch):
            self.installer._update_nodes(self.comfy, "idle")
            self.assertEqual(self.installer.status()["node_update"]["state"], "complete")
            self.assertEqual(git("-C", repo, "rev-parse", "HEAD"), pin)
            self.assertTrue(self.target.is_symlink())
            (repo / "__init__.py").write_text("local changes")
            self.installer._update_nodes(self.comfy, "idle")
            self.assertIn("local changes", self.installer.status()["node_update"]["error"])
            self.assertEqual((repo / "__init__.py").read_text(), "local changes")
            git("-C", repo, "remote", "set-url", "origin", "https://wrong.invalid/repo.git")
            with self.assertRaisesRegex(SetupError, "not a verified"):
                self.installer._safe_node_target(self.comfy, self.target)
        self.assertTrue(self.target.is_symlink())

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_unowned_linked_folder_is_preserved(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "personal.txt").write_text("keep")
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(SetupError, "not a verified"):
            self.installer._safe_node_target(self.comfy, self.target)
        self.assertTrue(self.target.is_symlink())
        self.assertEqual((outside / "personal.txt").read_text(), "keep")

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_link_retarget_during_download_stops_before_merge(self):
        repo, other = self.root / "source", self.root / "other"
        repo.mkdir(); other.mkdir()
        (repo / ".git").mkdir()
        (repo / "__init__.py").write_text("old")
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to(repo, target_is_directory=True)
        def run(command, **kwargs):
            if "rev-parse" in command:
                return str(repo)
            if "remote" in command:
                return "https://github.com/" + self.new["repo"] + ".git"
            if "fetch" in command:
                self.target.unlink()
                self.target.symlink_to(other, target_is_directory=True)
            return ""
        with patch.object(self.installer, "required_donutnodes", return_value=self.new), patch.object(self.installer, "_run", side_effect=run) as commands:
            self.installer._update_nodes(self.comfy, "idle")
        self.assertIn("changed during the download", self.installer.status()["node_update"]["error"])
        self.assertFalse(any("merge" in call.args[0] for call in commands.call_args_list))
        self.assertEqual((repo / "__init__.py").read_text(), "old")
        self.assertEqual(list(other.iterdir()), [])

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_linked_git_metadata_is_rejected_before_commands(self):
        self.target.mkdir(parents=True)
        metadata = self.root / "metadata"
        metadata.mkdir()
        (self.target / ".git").symlink_to(metadata, target_is_directory=True)
        with patch.object(self.installer, "_run") as commands, self.assertRaisesRegex(SetupError, "Git metadata is a symlink"):
            self.installer._safe_node_target(self.comfy, self.target)
        commands.assert_not_called()
        self.assertTrue((self.target / ".git").is_symlink())

    # AC: @create-multiple-face-prompts ac-existing-backend-node
    def test_cancelled_link_verification_remains_cancelled(self):
        repo = self.root / "source"
        repo.mkdir(); (repo / ".git").mkdir()
        self.target.parent.mkdir(parents=True)
        self.target.symlink_to(repo, target_is_directory=True)
        for responses in ([SetupCancelled("cancelled")], [str(repo), SetupCancelled("cancelled")]):
            with patch.object(self.installer, "_run", side_effect=responses):
                self.installer._update_nodes(self.comfy, "idle")
            self.assertEqual(self.installer.status()["node_update"]["state"], "cancelled")
            self.assertNotIn("not a verified", self.installer.status()["node_update"]["error"])

    def test_git_checkout_fast_forwards_to_pin_and_preserves_dirty_source(self):
        remote = self.root / "git-source"
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        def git(*args):
            return subprocess.run(["git", *map(str, args)], check=True, capture_output=True, text=True, env=env).stdout.strip()
        git("init", remote)
        git("-C", remote, "config", "user.name", "Synthetic Fixture")
        git("-C", remote, "config", "user.email", "fixture@example.invalid")
        (remote / "__init__.py").write_text("old")
        git("-C", remote, "add", "__init__.py")
        git("-C", remote, "commit", "-m", "first")
        first = git("-C", remote, "rev-parse", "HEAD")
        self.target.parent.mkdir(parents=True)
        git("clone", remote, self.target)
        (remote / "__init__.py").write_text("new")
        git("-C", remote, "commit", "-am", "pinned")
        pin = git("-C", remote, "rev-parse", "HEAD")
        (remote / "__init__.py").write_text("later")
        git("-C", remote, "commit", "-am", "later moving head")
        run = self.installer._run
        def local_fetch(command, **kwargs):
            if "fetch" in command:
                command = list(command)
                command[command.index("--no-tags") + 1] = str(remote)
            return run(command, extra_env=env, **kwargs)
        with patch.object(self.installer, "_run", side_effect=local_fetch):
            self.installer._update_git_nodes({**self.new, "commit": pin}, self.target)
            self.assertEqual(git("-C", self.target, "rev-parse", "HEAD"), pin)
            self.assertEqual(git("-C", self.target, "rev-parse", "HEAD~1"), first)
            (self.target / "__init__.py").write_text("local changes")
            with self.assertRaisesRegex(SetupError, "local changes"):
                self.installer._update_git_nodes({**self.new, "commit": pin}, self.target)
            (self.target / "__init__.py").write_text("new")
            (self.target / "untracked.txt").write_text("keep")
            with self.assertRaisesRegex(SetupError, "local changes"):
                self.installer._update_git_nodes({**self.new, "commit": pin}, self.target)
            self.assertEqual((self.target / "untracked.txt").read_text(), "keep")


class BackendAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dmc-controller-fixture-")
        self.root = Path(self.temp.name)
        self.installer = Installer(self.root / "state")
        self.installer.expected_capabilities = lambda: {"required_nodes": ["ExistingNode", "DMCFaceDetailer"], "required_models": []}
        self.local = comfy(self.root / "connected")
        self.controller = app.Controller(self.root / "controller", installer=self.installer,
                                         transport=httpx.MockTransport(lambda request: httpx.Response(503)))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app.create_app(self.controller)), base_url="http://controller.invalid")

    async def asyncTearDown(self):
        self.installer.cancel()
        if self.installer._thread:
            await asyncio.to_thread(self.installer._thread.join, 3)
        await self.client.aclose()
        await self.controller.close()
        self.temp.cleanup()

    def configure_local(self):
        self.controller.config = {"mode": "local", "backend_url": app.MANAGED_URL, "comfy_directory": str(self.local)}

    async def test_api_targets_connected_local_and_managed_installations(self):
        with patch.object(self.installer, "start_node_update", return_value={}) as start:
            self.configure_local()
            result = await self.client.post("/create/backend/update-nodes")
            self.assertEqual(result.status_code, 200)
            start.assert_called_once_with(self.local)
            start.reset_mock()
            self.controller.config = {"mode": "managed", "backend_url": app.MANAGED_URL}
            result = await self.client.post("/create/backend/update-nodes")
            self.assertEqual(result.status_code, 200)
            start.assert_called_once_with(self.installer.backend_root)

    async def test_url_only_running_and_owned_backends_cannot_update(self):
        with patch.object(self.installer, "start_node_update") as start:
            self.controller.config = {"mode": "existing", "backend_url": "http://remote.invalid"}
            self.assertEqual((await self.client.post("/create/backend/update-nodes")).status_code, 409)
            self.configure_local()
            with patch.object(self.controller, "backend_owned", return_value=True):
                self.assertEqual((await self.client.post("/create/backend/update-nodes")).status_code, 409)
            await self.controller.client.aclose()
            self.controller.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ExistingNode": {}})))
            self.assertEqual((await self.client.post("/create/backend/update-nodes")).status_code, 409)
            start.assert_not_called()

    async def test_optional_face_capability_status_and_host_remedy(self):
        self.controller.config = {"mode": "existing", "backend_url": "http://remote.invalid"}
        await self.controller.client.aclose()
        self.controller.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ExistingNode": {}})))
        status = await self.controller.status()
        self.assertTrue(status["backend"]["ready"])
        self.assertFalse(status["backend"]["face_node"]["registered"])
        self.assertFalse(status["backend"]["face_node"]["updateable"])
        self.assertIn("ComfyUI host", status["backend"]["face_node"]["error"])

    async def test_no_configure_or_start_during_node_update(self):
        self.configure_local()
        with patch.object(self.installer, "status", return_value={"running": True, "operation": "nodes"}):
            with self.assertRaises(app.HTTPException) as configure:
                await self.controller.configure({"mode": "managed"})
            self.assertEqual(configure.exception.status_code, 409)
            with self.assertRaises(app.HTTPException) as start:
                await self.controller.start_backend()
            self.assertEqual(start.exception.status_code, 409)

    async def test_download_does_not_block_api_event_loop_or_hold_backend_lock(self):
        self.configure_local()
        bundle, item = archive(self.root)
        entered = threading.Event()
        release = threading.Event()
        def waiting_download(item):
            entered.set()
            release.wait(3)
            self.installer._check_cancel()
            return bundle
        try:
            with patch.object(self.installer, "required_donutnodes", return_value=item), patch.object(self.installer, "_download", side_effect=waiting_download):
                response = await asyncio.wait_for(self.client.post("/create/backend/update-nodes"), timeout=1)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.assertFalse(self.controller.backend_lock.locked())
                self.assertEqual(response.json()["setup"]["operation"], "nodes")
                response = await asyncio.wait_for(self.client.post("/create/backend/start"), timeout=1)
                self.assertEqual(response.status_code, 409)
                release.set()
                await asyncio.to_thread(self.installer._thread.join, 3)
        finally:
            release.set()

    async def test_studio_link_cannot_use_management_endpoint(self):
        response = await self.client.post("/studio/" + "a" * 43 + "/create/backend/update-nodes")
        self.assertNotEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
