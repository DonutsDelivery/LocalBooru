"""Synthetic controller checks. No models, user profiles, or services are opened."""

import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx


MODULE_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("donut_create_controller", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeInstaller:
    def __init__(self, root):
        self.backend_root = root / "backend"
        self.python_path = root / "venv/python"
        self.assets_root = root / "assets"
        self.workflow_path = root / "workflow.json"
        self.backend_root.mkdir()
        self.assets_root.mkdir()
        self.workflow_path.write_text(json.dumps({"nodes": [{"type": "DonutText"}]}))
        (self.assets_root / "donut-create.js").write_text("window.DonutUI = {};")
        (self.assets_root / "donut-create.css").write_text("body {color: white}")
        self.options = None
        self.models = []

    def expected_capabilities(self):
        return {"required_nodes": ["DonutText", "DonutImageSave"], "required_models": self.models}

    def status(self):
        return {"state": "ready", "phase": "complete", "progress": 1, "runtime": "cpu", "profile": "base",
                "catalog": {"profiles": [{"id": "base"}]}, "disk": {"free_bytes": 10_000},
                "supported_runtimes": ["cpu"], "default_runtime": "cpu", "default_profile": "base"}

    def ready(self):
        return True

    def backend_environment(self):
        cache = self.backend_root / "cache"
        return {"HF_HOME": str(cache / "huggingface"), "TORCH_HOME": str(cache / "torch"), "PYTHONNOUSERSITE": "1"}

    def start(self, options):
        self.options = options
        return self.status()

    def cancel(self):
        return {"state": "cancelled"}


class FakeComfy:
    def __init__(self):
        self.requests = []
        self.pending = []
        self.running = []
        self.history = {}
        self.atomic_cancel_available = True
        self.cancel_race = False
        self.image = b"\x89PNG\r\n\x1a\nsynthetic-output"
        manifest = json.loads(MODULE_PATH.with_name("runtime.json").read_text())
        self.nodes = {name: {} for name in manifest["required_nodes"]}
        self.nodes.update({"DonutText": {}, "DonutImageSave": {}, "ExecutePython": {}})
        self.catalog = {}
        self.fail = False

    def row(self, prompt_id):
        return [1, prompt_id, {"1": {"class_type": "DonutText"}}, {"client_id": "other-client"}, ["1"]]

    async def __call__(self, request):
        content = await request.aread()
        self.requests.append((request.method, str(request.url), dict(request.headers), content))
        if self.fail:
            raise httpx.ConnectError("synthetic failure", request=request)
        path, method = request.url.path, request.method
        if path == "/object_info":
            return httpx.Response(200, json=self.nodes)
        if path.startswith("/object_info/"):
            name = path.split("/", 2)[2]
            return httpx.Response(200, json={name: self.nodes[name]} if name in self.nodes else {})
        if path.startswith("/models/"):
            return httpx.Response(200, json=self.catalog.get(path.split("/", 2)[2], []))
        if path == "/queue" and method == "GET":
            return httpx.Response(200, json={"queue_running": self.running, "queue_pending": self.pending})
        if path == "/queue" and method == "POST":
            data = json.loads(content)
            self.pending = [row for row in self.pending if row[1] not in data["delete"]]
            return httpx.Response(200, content=b"")
        if path.startswith("/history/"):
            pid = path.split("/", 2)[2]
            return httpx.Response(200, json={pid: self.history[pid]} if pid in self.history else {})
        if path == "/prompt" and method == "POST":
            data = json.loads(content)
            prompt_id = data["prompt_id"]
            self.pending.append(self.row(prompt_id))
            return httpx.Response(200, json={"prompt_id": prompt_id, "number": len(self.pending), "node_errors": {}})
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            if not self.atomic_cancel_available:
                return httpx.Response(404, json={"error": "not supported"})
            if self.cancel_race:
                self.running = [self.row("other-running-after-race")]
            prompt_id = path.split("/")[3]
            cancelled = any(row[1] == prompt_id for row in self.running)
            self.running = [row for row in self.running if row[1] != prompt_id]
            return httpx.Response(200, json={"cancelled": cancelled})
        if path.startswith("/api/jobs/"):
            return httpx.Response(200, json={"id": path.split("/")[3], "status": "pending"})
        if path == "/view":
            return httpx.Response(200, content=self.image, headers={"Content-Type": "image/png"})
        if path == "/extensions":
            return httpx.Response(200, json=["/extensions/donutnodes/donut_edit_studio.js", "/extensions/ComfyUI-Manager/manager.js"])
        if path == "/":
            return httpx.Response(200, text='<!doctype html><html><head><script type="module" src="/assets/main.js"></script></head><body></body></html>',
                                  headers={"Content-Type": "text/html"})
        if path == "/assets/main.js":
            return httpx.Response(200, text='import "/scripts/app.js"; fetch("/object_info");', headers={"Content-Type": "text/javascript"})
        if path == "/upload/image":
            fields, files = module.multipart_fields(request.headers["content-type"], content)
            return httpx.Response(200, json={"name": files["image"][0], "type": fields["type"], "subfolder": fields["subfolder"]})
        if path == "/donut/edit-studio/reference":
            return httpx.Response(200, json={"reference": "donutref:" + "a" * 64})
        return httpx.Response(200, json={"ok": True})

    def complete(self, prompt_id, *, status=None, images=None, extra=None):
        self.pending = [row for row in self.pending if row[1] != prompt_id]
        self.running = [row for row in self.running if row[1] != prompt_id]
        self.history[prompt_id] = {
            "status": status or {"completed": True, "status_str": "success", "messages": []},
            "outputs": {"8": {"images": images or [{"filename": "image.png", "subfolder": "donut-create/synthetic", "type": "output"}], **(extra or {})}},
        }


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dmc-create-controller-test-")
        self.root = Path(self.temporary.name)
        self.clock = 1000.0
        self.backend = FakeComfy()
        self.installer = FakeInstaller(self.root)
        self.controller = module.Controller(self.root / "state", self.installer,
            transport=httpx.MockTransport(self.backend), clock=lambda: self.clock, ttl=60)
        self.controller.config = {"mode": "existing", "backend_url": "http://first-comfy.invalid:8188"}
        self.app = module.create_app(self.controller)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://controller.invalid")

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.controller.close()
        self.temporary.cleanup()

    async def new_session(self):
        response = await self.http.post("/create/sessions")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["id"]

    async def submit(self, sid, *, prompt=None, extra=None):
        body = {"prompt": prompt or {"1": {"class_type": "DonutText", "inputs": {"text": "synthetic mountain"}}},
                "client_id": "untrusted-client", "prompt_id": "untrusted-prompt", **(extra or {})}
        response = await self.http.post(f"/studio/{sid}/prompt", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["prompt_id"]

    async def test_progress_and_completed_polling_do_not_rewrite_or_rescan_history(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.complete(own)
        await self.controller.reconcile(self.controller.session(sid))
        self.backend.requests.clear()
        with patch.object(self.controller, "persist") as persist:
            await self.controller.reconcile(self.controller.session(sid))
            for value in range(20):
                self.assertFalse(self.controller.observe_event(self.controller.session(sid),
                    {"type": "progress", "data": {"prompt_id": own, "value": value, "max": 20}}))
            persist.assert_not_called()
        self.assertFalse(any("/history/" in url for _, url, _, _ in self.backend.requests))

    async def test_confirmed_history_ignores_delayed_socket_lifecycle_events(self):
        for terminal in ("completed", "cancelled"):
            with self.subTest(terminal=terminal):
                sid = await self.new_session()
                own = await self.submit(sid)
                status = None if terminal == "completed" else {"status_str": "error", "completed": False,
                    "messages": [["execution_interrupted", {}]]}
                self.backend.complete(own, status=status)
                await self.controller.reconcile(self.controller.session(sid))
                for kind in ("execution_start", "executing", "execution_error", "execution_interrupted"):
                    self.assertFalse(self.controller.observe_event(self.controller.session(sid),
                        {"type": kind, "data": {"prompt_id": own, "node": "1"}}))
                await self.controller.reconcile(self.controller.session(sid))
                self.assertEqual(self.controller.session(sid)["jobs"][own]["status"], terminal)

    async def test_live_progress_does_not_write_session_files(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        with patch.object(self.controller, "persist") as persist:
            for value in range(20):
                self.assertTrue(self.controller.observe_event(self.controller.session(sid),
                    {"type": "progress", "data": {"prompt_id": own, "value": value, "max": 20}}))
            persist.assert_not_called()

    async def test_reconcile_accepts_a_submission_while_history_request_is_pending(self):
        sid = await self.new_session()
        first = await self.submit(sid)
        self.backend.pending = []
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.controller.backend_json
        async def held_history(backend, path):
            if path == "history/" + first:
                entered.set()
                await release.wait()
            return await original(backend, path)
        with patch.object(self.controller, "backend_json", side_effect=held_history):
            reading = asyncio.create_task(self.controller.reconcile(self.controller.session(sid)))
            await asyncio.wait_for(entered.wait(), 1)
            second = await self.submit(sid)
            release.set()
            await asyncio.wait_for(reading, 1)
        self.assertIn(second, self.controller.session(sid)["jobs"])

    async def test_cancel_handles_a_queued_job_promoted_to_running_before_delete(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        original = self.controller.request_backend
        async def promoted(backend, method, path, **kwargs):
            if method == "POST" and path == "queue":
                self.backend.pending = []
                self.backend.running = [self.backend.row(own)]
            return await original(backend, method, path, **kwargs)
        with patch.object(self.controller, "request_backend", side_effect=promoted):
            result = (await self.http.post(f"/create/sessions/{sid}/cancel")).json()
        self.assertEqual(result["cancelled"], [own])
        self.assertEqual(self.backend.running, [])
        self.assertTrue(any(url.endswith(f"/api/jobs/{own}/cancel") for _, url, _, _ in self.backend.requests))
        # A late authoritative completion remains recoverable after cancellation.
        self.backend.complete(own)
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(state["jobs"][0]["status"], "completed")
        self.assertEqual(len(state["outputs"]), 1)

    async def test_cancel_does_not_wait_for_completed_job_history(self):
        sid = await self.new_session()
        old = await self.submit(sid)
        own = await self.submit(sid)
        self.backend.pending = [self.backend.row(own)]
        self.backend.requests.clear()
        result = (await self.http.post(f"/create/sessions/{sid}/cancel")).json()
        self.assertEqual(result["cancelled"], [own])
        self.assertFalse(any("/history/" in url for _, url, _, _ in self.backend.requests))
        self.assertIn(old, self.controller.session(sid)["jobs"])

    async def test_workspace_polling_omits_bulk_provenance_but_output_endpoint_retains_it(self):
        response = await self.http.post("/create/sessions", json={"workspace": True})
        sid = response.json()["id"]
        workflow = {"nodes": [{"id": 1, "type": "DonutImageSave"}]}
        own = await self.submit(sid, extra={"extra_data": {"extra_pnginfo": {"workflow": workflow}}})
        self.backend.complete(own)
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        output = state["outputs"][0]
        for key in ("workflow", "prompt", "execution", "metadata"):
            self.assertNotIn(key, output)
        provenance = (await self.http.get(f"/create/sessions/{sid}/outputs/{output['id']}/provenance")).json()
        self.assertEqual(provenance["workflow"], workflow)
        self.assertIn("execution_prompt", provenance)

    async def test_interrupted_history_stays_cancelled_and_is_reconciled_once(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.complete(own, status={"status_str": "error", "completed": False,
            "messages": [["execution_interrupted", {}]]})
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(state["jobs"][0]["status"], "cancelled")
        self.backend.requests.clear()
        await self.http.get(f"/create/sessions/{sid}")
        self.assertFalse(any("/history/" in url for _, url, _, _ in self.backend.requests))

    # AC: @donut-create-plugin ac-access-boundary
    async def test_capabilities_expire_and_restore_without_repository_state(self):
        sid = await self.new_session()
        self.assertEqual(len(sid), 43)
        self.assertTrue(module.CAPABILITY.fullmatch(sid))
        saved = self.controller.state_dir / "sessions.json"
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("controller-test", saved.read_text())
        recovered = module.Controller(self.controller.state_dir, self.installer, transport=httpx.MockTransport(self.backend), clock=lambda: self.clock)
        self.assertEqual(recovered.instance_id, self.controller.instance_id)
        self.assertEqual(recovered.session(sid)["backend_url"], self.controller.session(sid)["backend_url"])
        await recovered.close()
        self.clock += 61
        for path in [f"/studio/{sid}/", f"/create/sessions/{sid}", f"/create/sessions/{sid}/outputs/unknown"]:
            self.assertEqual((await self.http.get(path)).status_code, 410)
        self.assertEqual((await self.http.get("/studio/guessed/queue")).status_code, 404)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_invalid_backend_origins_rejected_and_lan_allowed(self):
        for value in ["file:///tmp/comfy", "http://name:password@host:8188", "http://host/path", "http://host?token=x",
                      "http://host#fragment", "http://host:99999", "http://host\\elsewhere", "http://host\n"]:
            response = await self.http.post("/create/config", json={"mode": "existing", "backend_url": value})
            self.assertEqual(response.status_code, 400, value)
        self.assertEqual(module.validate_backend_url("https://192.168.1.20:8188/"), "https://192.168.1.20:8188")

    # AC: @donut-create-plugin ac-job-ownership
    async def test_session_backend_is_pinned_and_prompt_ids_and_client_id_are_owned(self):
        sid = await self.new_session()
        await self.http.post("/create/config", json={"mode": "existing", "backend_url": "http://second-comfy.invalid:8188"})
        pid = await self.submit(sid)
        request = next(entry for entry in reversed(self.backend.requests) if entry[0] == "POST" and entry[1].endswith("/prompt"))
        data = json.loads(request[3])
        self.assertEqual(data["client_id"], "donut-create-" + sid)
        self.assertNotEqual(pid, "untrusted-prompt")
        self.assertIn("first-comfy.invalid", request[1])

    # AC: @donut-create-plugin ac-workflow-state
    async def test_draft_backend_key_is_stable_per_instance_and_pinned_to_session(self):
        sid = await self.new_session()
        key = self.controller.session(sid)["backend_key"]
        second_sid = await self.new_session()
        self.assertEqual(self.controller.session(second_sid)["backend_key"], key)
        other_controller = module.Controller(self.root / "different-device-state", self.installer,
            transport=httpx.MockTransport(self.backend), clock=lambda: self.clock)
        self.assertNotEqual(other_controller.backend_key(self.controller.config["backend_url"], "existing"), key)
        await other_controller.close()
        await self.http.post("/create/config", json={"mode": "existing", "backend_url": "http://different-comfy.invalid:18010"})
        page = await self.http.get(f"/studio/{sid}/")
        self.assertIn('backendKey:"' + key + '"', page.text)
        self.assertEqual(self.controller.session(sid)["backend_key"], key)

    # AC: @donut-create-plugin ac-job-ownership
    async def test_queue_and_history_exclude_other_clients(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.pending.append(self.backend.row("someone-else"))
        queue = (await self.http.get(f"/studio/{sid}/queue")).json()
        self.assertEqual([row[1] for row in queue["queue_pending"]], [own])
        self.backend.complete(own)
        self.backend.complete("someone-else")
        history = (await self.http.get(f"/studio/{sid}/history")).json()
        self.assertEqual(list(history), [own])
        self.assertEqual((await self.http.get(f"/studio/{sid}/history/someone-else")).status_code, 404)
        jobs = (await self.http.get(f"/studio/{sid}/api/jobs")).json()
        self.assertEqual([job["id"] for job in jobs["jobs"]], [own])
        self.assertEqual(jobs["pagination"]["total"], 1)
        self.assertEqual((await self.http.get(f"/studio/{sid}/api/jobs/someone-else")).status_code, 404)
        self.assertEqual((await self.http.post(f"/studio/{sid}/api/jobs/someone-else/cancel", json={})).status_code, 403)

    # AC: @donut-create-plugin ac-job-ownership
    async def test_completed_outputs_only_from_own_history_and_type_output(self):
        sid = await self.new_session()
        own = await self.submit(sid, extra={"extra_data": {"extra_pnginfo": {"workflow": {"nodes": [{"type": "DonutText"}]}}}})
        self.backend.complete(own, images=[
            {"filename": "image.png", "subfolder": "safe", "type": "output"},
            {"filename": "temporary.png", "subfolder": "", "type": "temp"},
            {"filename": "input.png", "subfolder": "", "type": "input"},
            {"filename": "../private.png", "subfolder": "", "type": "output"},
            {"filename": "private.png", "subfolder": "../outside", "type": "output"},
        ])
        self.backend.complete("unowned")
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(state["jobs"][0]["status"], "completed")
        self.assertEqual([entry["filename"] for entry in state["outputs"]], ["image.png"])
        output = state["outputs"][0]
        self.assertEqual(output["url"], f"/api/create/output/{sid}/{output['id']}")
        self.assertEqual(output["workflow"]["nodes"][0]["type"], "DonutText")
        result = await self.http.get(f"/create/sessions/{sid}/outputs/{output['id']}")
        self.assertEqual(result.content, self.backend.image)
        self.assertEqual(result.headers["cache-control"], "private, no-store")
        view = next(entry for entry in reversed(self.backend.requests) if "/view?" in entry[1])
        self.assertIn("type=output", view[1])
        other_sid = await self.new_session()
        self.assertEqual((await self.http.get(f"/create/sessions/{other_sid}/outputs/{output['id']}")).status_code, 404)
        self.assertEqual((await self.http.get(f"/studio/{sid}/view?filename=private.png&type=output")).status_code, 404)

    # AC: @donut-create-plugin ac-job-ownership
    async def test_cancel_only_owned_pending_and_running_prompts(self):
        sid = await self.new_session()
        own_pending = await self.submit(sid)
        own_running = await self.submit(sid)
        self.backend.pending = [self.backend.row(own_pending), self.backend.row("other-pending")]
        self.backend.running = [self.backend.row(own_running)]
        result = (await self.http.post(f"/create/sessions/{sid}/cancel")).json()
        self.assertEqual(set(result["cancelled"]), {own_pending, own_running})
        self.assertEqual([row[1] for row in self.backend.pending], ["other-pending"])
        cancellations = [entry for entry in self.backend.requests if entry[0] == "POST" and (entry[1].endswith("/queue") or entry[1].endswith("/cancel"))]
        self.assertEqual(json.loads(cancellations[0][3]), {"delete": [own_pending]})
        self.assertTrue(cancellations[1][1].endswith(f"/api/jobs/{own_running}/cancel"))
        self.assertFalse(any(entry[1].endswith("/interrupt") for entry in self.backend.requests))

    # AC: @donut-create-plugin ac-job-ownership
    async def test_other_current_job_is_never_interrupted(self):
        sid = await self.new_session()
        await self.submit(sid)
        self.backend.running = [self.backend.row("other-current")]
        await self.http.post(f"/create/sessions/{sid}/cancel")
        self.assertEqual(self.backend.running[0][1], "other-current")
        self.assertFalse(any("/api/jobs/" in entry[1] or entry[1].endswith("/interrupt") for entry in self.backend.requests))

    # AC: @donut-create-plugin ac-job-ownership
    async def test_atomic_cancel_race_leaves_next_clients_job_running(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.pending = []
        self.backend.running = [self.backend.row(own)]
        self.backend.cancel_race = True
        result = (await self.http.post(f"/create/sessions/{sid}/cancel")).json()
        self.assertEqual(result["cancelled"], [])
        self.assertEqual(self.backend.running[0][1], "other-running-after-race")

    # AC: @donut-create-plugin ac-job-ownership
    async def test_older_backend_reports_scoped_interrupt_unavailable(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.pending = []
        self.backend.running = [self.backend.row(own)]
        self.backend.atomic_cancel_available = False
        result = (await self.http.post(f"/create/sessions/{sid}/cancel")).json()
        self.assertTrue(result["errors"])
        self.assertEqual(self.backend.running[0][1], own)
        self.assertFalse(any(entry[1].endswith("/interrupt") for entry in self.backend.requests))

    # AC: @donut-create-plugin ac-job-ownership
    async def test_history_execution_error_survives_reconnect_and_ignores_other_events(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        event = {"type": "execution_error", "data": {"prompt_id": "other", "exception_message": "other secret"}}
        self.assertFalse(self.controller.observe_event(self.controller.session(sid), event))
        self.backend.complete(own, status={"completed": False, "status_str": "error", "messages": [
            ["execution_error", {"exception_message": "synthetic out of memory"}]]})
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(state["jobs"][0]["status"], "error")
        self.assertEqual(state["jobs"][0]["error"], "synthetic out of memory")
        self.assertEqual(state["outputs"], [])

    # AC: @donut-create-plugin ac-access-boundary
    async def test_proxy_never_forwards_dmc_authorization_or_cookie(self):
        sid = await self.new_session()
        response = await self.http.get(f"/studio/{sid}/object_info?token=private-token",
            headers={"Authorization": "Bearer DMC-user-token", "Cookie": "private-cookie", "X-Other": "private"})
        self.assertEqual(response.status_code, 200)
        request = self.backend.requests[-1]
        self.assertNotIn("authorization", request[2])
        self.assertNotIn("cookie", request[2])
        self.assertNotIn("private-token", request[1])
        self.assertNotIn("ExecutePython", response.json())

    # AC: @donut-create-plugin ac-managed-setup
    # AC: @donut-create-plugin ac-access-boundary
    async def test_optional_model_helpers_are_discoverable_but_not_queueable(self):
        sid = await self.new_session()
        optional = {"VAEUtils_PatchWanUpscaleVAE", "SeedVR2Preprocess", "SeedVR2Conditioning",
                    "SeedVR2PostProcessing", "LoadBackgroundRemovalModel", "RemoveBackground", "SAM3_Detect"}
        self.backend.nodes.update({name: {"name": name} for name in optional})
        response = await self.http.get(f"/studio/{sid}/object_info")
        self.assertTrue(optional <= response.json().keys())
        self.assertNotIn("ExecutePython", response.json())
        for name in optional:
            with self.subTest(node=name):
                response = await self.http.get(f"/studio/{sid}/object_info/{name}")
                self.assertEqual(response.json(), {name: {"name": name}})
                queued = await self.http.post(f"/studio/{sid}/prompt", json={"prompt": {
                    "1": {"class_type": name, "inputs": {}}}})
                self.assertEqual(queued.status_code, 403)
        self.backend.nodes.pop("VAEUtils_PatchWanUpscaleVAE")
        response = await self.http.get(f"/studio/{sid}/object_info")
        self.assertNotIn("VAEUtils_PatchWanUpscaleVAE", response.json())
        response = await self.http.get(f"/studio/{sid}/object_info/VAEUtils_PatchWanUpscaleVAE")
        self.assertEqual(response.json(), {})
        self.assertEqual((await self.http.get(f"/studio/{sid}/object_info/ExecutePython")).status_code, 403)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_manager_process_filesystem_and_global_mutations_blocked(self):
        sid = await self.new_session()
        for method, route, body in [("GET", "manager/config", None), ("POST", "manager/reboot", {}),
             ("POST", "free", {}), ("POST", "history", {"clear": True}), ("POST", "queue", {"clear": True}),
             ("POST", "queue", {"delete": ["someone-else"]}), ("DELETE", "donut/loras/by-path?path=/private", None),
             ("POST", "donut/models/download", {}), ("POST", "donut/wildcards/file", {}),
             ("GET", "donut/wildcards/file?name=private", None),
             ("GET", "extensions/ComfyUI-Manager/manager.js", None), ("GET", "assets/private.py", None),
             ("GET", "userdata/%252e%252e/private", None)]:
            response = await self.http.request(method, f"/studio/{sid}/{route}", json=body)
            self.assertEqual(response.status_code, 403, (method, route, response.text))
        self.assertEqual((await self.http.get(f"/studio/{sid}/donut/loras/preview?hash=../../private&type=0")).status_code, 403)
        self.assertEqual((await self.http.get(f"/studio/{sid}/donut/loras/preview?hash={'a' * 10}&type=../private")).status_code, 403)
        response = await self.http.get(f"/studio/{sid}/donut/config")
        self.assertEqual(response.json()["civitai"]["api_key"], "")

    # AC: @donut-create-plugin ac-access-boundary
    async def test_prompt_blocks_unapproved_nodes_paths_and_file_macros(self):
        sid = await self.new_session()
        unsafe = [
            {"class_type": "ExecutePython", "inputs": {"code": "private"}},
            {"class_type": "DonutText", "inputs": {"text": "__private/file__"}},
            {"class_type": "DonutText", "inputs": {"text": "private/file*"}},
            {"class_type": "DonutText", "inputs": {"text": "{_|a}{_|b}private{_|c}{_|d}"}},
            {"class_type": "DonutText", "inputs": {"text": "%other.Text%"}},
            {"class_type": "DonutImageSave", "inputs": {"filename_prefix": "../private"}},
            {"class_type": "DonutImageSave", "inputs": {"filename_prefix": "/private"}},
            {"class_type": "LoadImage", "inputs": {"image": "other-client.png"}},
            {"class_type": "DonutEditStudio", "inputs": {"image_a": "other.png"}},
            {"class_type": "DonutEditStudio", "inputs": {"prompt": "{_|_}{_|_}private{_|_}{_|_}"}},
            {"class_type": "DonutLoRALoader", "inputs": {"slots_json": json.dumps([{"lora_name": "../private.safetensors"}])}},
        ]
        for node in unsafe:
            response = await self.http.post(f"/studio/{sid}/prompt", json={"prompt": {"1": node}})
            self.assertEqual(response.status_code, 403, node)
        split_macro = {"1": {"class_type": "StringConcatenate", "inputs": {"string_a": "_", "string_b": "_private_", "delimiter": "_"}},
                       "2": {"class_type": "DonutText", "inputs": {"text": ["1", 0]}}}
        self.assertEqual((await self.http.post(f"/studio/{sid}/prompt", json={"prompt": split_macro})).status_code, 403)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_preview_rejects_constructed_file_macros_before_backend_request(self):
        sid = await self.new_session()
        self.backend.requests.clear()
        for text in ["__private__", "{_|_}{_|_}private{_|_}{_|_}", "%other.Text%"]:
            response = await self.http.post(f"/studio/{sid}/donut/wildcards/preview", json={"text": text})
            self.assertEqual(response.status_code, 403, text)
        self.assertEqual(self.backend.requests, [])
        response = await self.http.post(f"/studio/{sid}/donut/wildcards/preview", json={"text": "a {red|blue} teapot"})
        self.assertEqual(response.status_code, 200)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_constructed_short_wildcards_are_rejected_in_all_prompt_controls(self):
        sid = await self.new_session()
        templates = [
            ("donut/wildcards/preview", lambda text: {"text": text}),
            ("prompt", lambda text: {"prompt": {"1": {"class_type": "DonutText", "inputs": {"text": text}}}}),
            ("prompt", lambda text: {"prompt": {"1": {"class_type": "DonutEditStudio", "inputs": {"prompt": text}}}}),
            ("prompt", lambda text: {"prompt": {"1": {"class_type": "DonutPromptConditioning", "inputs": {"face": text}}}}),
            ("prompt", lambda text: {"prompt": {"1": {"class_type": "DonutPromptConditioning", "inputs": {
                "prompt_sets_json": json.dumps([{"face": text, "scene": "", "negative": ""}])}}}}),
        ]
        for route, body in templates:
            with self.subTest(route=route, body=body("{private|private}{*|*}")):
                self.backend.requests.clear()
                response = await self.http.post(f"/studio/{sid}/{route}", json=body("{private|private}{*|*}"))
                self.assertEqual(response.status_code, 403, response.text)
                self.assertEqual(self.backend.requests, [])
        for route, body in templates:
            with self.subTest(valid_route=route, valid_body=body("a {red|blue} teapot")):
                response = await self.http.post(f"/studio/{sid}/{route}", json=body("a {red|blue} teapot"))
                self.assertEqual(response.status_code, 200, response.text)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_save_names_isolated_and_overwrite_disabled(self):
        sid = await self.new_session()
        await self.submit(sid, prompt={"1": {"class_type": "DonutImageSave", "inputs": {
            "filename_prefix": "album/image", "overwrite_mode": True}}})
        inputs = json.loads(self.backend.requests[-1][3])["prompt"]["1"]["inputs"]
        self.assertEqual(inputs["filename_prefix"], f"donut-create/{sid[:16]}/album/image")
        self.assertFalse(inputs["overwrite_mode"])

    # AC: @donut-create-plugin ac-save-gallery
    async def test_library_runs_stage_save_nodes_without_changing_the_workflow(self):
        sid = await self.new_session()
        workflow = {"nodes": [{"id": 1, "type": "DonutImageSave", "properties": {"dmc_final_output": True}}]}
        job = await self.submit(sid, prompt={
            "1": {"class_type": "DonutImageSave", "inputs": {"images": ["3", 0], "root": "output", "filename_prefix": "final"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "intermediate"}},
            "3": {"class_type": "DonutText", "inputs": {"text": "synthetic"}},
        }, extra={"extra_data": {"extra_pnginfo": {"workflow": workflow}}})
        sent = json.loads(self.backend.requests[-1][3])
        self.assertEqual(sent["prompt"]["1"]["inputs"]["root"], "temp")
        self.assertEqual(sent["prompt"]["2"], {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}})
        self.assertEqual(sent["extra_data"]["extra_pnginfo"]["workflow"], workflow)
        self.backend.complete(job, images=[{"filename": "final.png", "subfolder": "donut-create/synthetic", "type": "temp"}])
        self.backend.history[job]["outputs"] = {"1": self.backend.history[job]["outputs"]["8"]}
        response = await self.http.get(f"/create/sessions/{sid}")
        output = response.json()["outputs"][0]
        self.assertTrue(output["final"])
        self.assertEqual(output["type"], "temp")
        self.assertEqual(output["storage"], "temporary")
        self.assertEqual(output["workflow"], workflow)
        fetched = await self.http.get(f"/create/sessions/{sid}/outputs/{output['id']}")
        self.assertEqual(fetched.content, self.backend.image)
        self.assertIn("type=temp", self.backend.requests[-1][1])

    # AC: @donut-create-plugin ac-save-gallery
    async def test_comfy_destination_retains_permanent_sinks_and_is_recorded(self):
        sid = await self.new_session()
        workflow = {"nodes": [], "extra": {"dmc_output_destination": "comfy"}}
        job = await self.submit(sid, prompt={"8": {"class_type": "DonutImageSave", "inputs": {
            "images": ["9", 0], "root": "output", "filename_prefix": "final"}}},
            extra={"extra_data": {"extra_pnginfo": {"workflow": workflow}}})
        sent = json.loads(self.backend.requests[-1][3])
        self.assertEqual(sent["prompt"]["8"]["inputs"]["root"], "output")
        self.backend.complete(job)
        outputs = (await self.http.get(f"/create/sessions/{sid}")).json()["outputs"]
        self.assertEqual(outputs[0]["storage"], "comfy")

    # AC: @donut-create-plugin ac-job-ownership
    async def test_ordinary_temporary_previews_do_not_become_saveable_results(self):
        sid = await self.new_session()
        job = await self.submit(sid, prompt={"8": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}}})
        self.backend.complete(job, images=[{"filename": "preview.png", "subfolder": "", "type": "temp"}])
        public = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(public["outputs"], [])
        self.assertEqual(len(public["previews"]), 1)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_invalid_output_destination_is_rejected_before_queueing(self):
        sid = await self.new_session()
        for destination in ["other", [], {"path": "/private"}]:
            response = await self.http.post(f"/studio/{sid}/prompt", json={
                "prompt": {"1": {"class_type": "DonutText", "inputs": {"text": "synthetic"}}},
                "extra_data": {"extra_pnginfo": {"workflow": {"nodes": [], "extra": {"dmc_output_destination": destination}}}}})
            self.assertEqual(response.status_code, 400)
        self.assertFalse(any(method == "POST" and url.endswith("/prompt") for method, url, *_ in self.backend.requests))

    # AC: @donut-create-plugin ac-workflow-state
    async def test_v5_linked_filename_and_lora_rows_queue_safely(self):
        sid = await self.new_session()
        await self.submit(sid, prompt={
            "1": {"class_type": "DF_Text_Box", "inputs": {"Text": "a synthetic lake"}},
            "2": {"class_type": "SeedNode", "inputs": {"seed": 11}},
            "3": {"class_type": "DonutText", "inputs": {"text": ["1", 0], "seed": ["2", 0], "max_depth": 128}},
            "4": {"class_type": "DonutLoRALoader", "inputs": {"slots_json": json.dumps([
                {"id": "synthetic:1", "lora_name": "krea2/synthetic_model.safetensors", "lora_hash": "a" * 64, "enabled": True}])}},
            "5": {"class_type": "StringConcatenate", "inputs": {"string_a": "DonutCreate", "string_b": "model", "delimiter": "/"}},
            "6": {"class_type": "DonutImageSave", "inputs": {"filename_prefix": ["5", 0]}},
        })
        prompt = json.loads(self.backend.requests[-1][3])["prompt"]
        self.assertEqual(prompt["6"]["inputs"]["filename_prefix"], f"donut-create/{sid[:16]}/image")
        self.assertEqual(json.loads(prompt["4"]["inputs"]["slots_json"])[0]["lora_hash"], "")
        self.assertEqual(prompt["3"]["inputs"]["text"], ["1", 0])

    # AC: @donut-create-plugin ac-job-ownership
    async def test_late_success_history_recovers_a_missing_prompt_error(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.pending = []
        self.clock += 11
        self.assertEqual((await self.http.get(f"/create/sessions/{sid}")).json()["jobs"][0]["status"], "error")
        self.backend.complete(own)
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(state["jobs"][0]["status"], "completed")
        self.assertNotIn("error", state["jobs"][0])
        self.assertEqual(len(state["outputs"]), 1)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_uploaded_images_are_unique_scoped_inputs(self):
        sid = await self.new_session()
        response = await self.http.post(f"/studio/{sid}/upload/image",
            files={"image": ("../../other.png", b"synthetic-input", "image/png")},
            data={"subfolder": "../private", "type": "output", "overwrite": "true"})
        self.assertEqual(response.status_code, 200, response.text)
        name = response.json()["name"]
        self.assertTrue(name.startswith(f"donut-create-{sid[:16]}-"))
        self.assertNotIn("other", name)
        self.assertEqual(response.json()["type"], "input")
        fields, _ = module.multipart_fields(self.backend.requests[-1][2]["content-type"], self.backend.requests[-1][3])
        self.assertEqual(fields, {"type": "input", "subfolder": "", "overwrite": "false"})
        await self.submit(sid, prompt={"1": {"class_type": "LoadImage", "inputs": {"image": name}}})
        other = await self.new_session()
        response = await self.http.post(f"/studio/{other}/prompt", json={"prompt": {"1": {"class_type": "LoadImage", "inputs": {"image": name}}}})
        self.assertEqual(response.status_code, 403)

    # AC: @donut-create-plugin ac-workflow-state
    async def test_studio_rewrites_absolute_assets_and_injects_only_browser_runtime(self):
        sid = await self.new_session()
        prefix = f"/remote/api/create/studio/{sid}/"
        response = await self.http.get(f"/studio/{sid}/", headers={"X-DMC-Studio-Prefix": prefix})
        self.assertEqual(response.status_code, 200)
        self.assertIn(f'<base href="{prefix}">', response.text)
        self.assertIn(f'src="{prefix}assets/main.js"', response.text)
        self.assertIn(f'src="{prefix}donut-create.js"', response.text)
        self.assertIn("backendKey:", response.text)
        self.assertNotIn("loadGraphData", response.text)
        self.assertNotIn("__TAURI", response.text)
        scripts = await self.http.get(f"/studio/{sid}/assets/main.js")
        self.assertIn(f'/api/create/studio/{sid}/scripts/app.js', scripts.text)
        extensions = (await self.http.get(f"/studio/{sid}/extensions")).json()
        self.assertEqual(extensions, [f"/api/create/studio/{sid}/extensions/donutnodes/donut_edit_studio.js"])

    # AC: @donut-create-plugin ac-managed-setup
    async def test_readiness_checks_required_nodes_and_models(self):
        self.installer.models = [{"folder": "diffusion_models", "filename": "synthetic.safetensors"}]
        status = (await self.http.get("/create/status")).json()
        self.assertFalse(status["backend"]["ready"])
        self.assertEqual(status["backend"]["missing_models"], ["synthetic.safetensors"])
        self.assertEqual((await self.http.post("/create/sessions")).status_code, 409)
        self.backend.catalog["diffusion_models"] = ["synthetic.safetensors"]
        self.assertTrue((await self.http.get("/create/status")).json()["backend"]["ready"])
        self.backend.nodes.pop("DonutText")
        self.assertFalse((await self.http.get("/create/status")).json()["backend"]["ready"])

    # AC: @donut-create-plugin ac-setup-recovery
    async def test_setup_tokens_are_not_in_controller_state(self):
        self.controller.config["mode"] = "managed"
        result = await self.http.post("/create/setup", json={"runtime": "cpu", "profile": "base",
            "hf_token": "synthetic-hf-token", "civitai_api_key": "synthetic-civitai-key"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.installer.options["hf_token"], "synthetic-hf-token")
        for path in self.controller.state_dir.glob("*.json"):
            self.assertNotIn("synthetic-hf-token", path.read_text())
            self.assertNotIn("synthetic-civitai-key", path.read_text())
        status = (await self.http.get("/create/status")).json()
        self.assertEqual(status["catalog"]["profiles"][0]["id"], "base")

    # AC: @donut-create-plugin ac-job-ownership
    async def test_existing_backend_start_and_stop_do_not_touch_processes(self):
        with patch.object(module.asyncio, "create_subprocess_exec", new=AsyncMock()) as spawn:
            self.assertEqual((await self.http.post("/create/backend/start")).status_code, 409)
            self.assertEqual((await self.http.post("/create/backend/stop")).status_code, 409)
            spawn.assert_not_called()

    # AC: @donut-create-plugin ac-managed-setup
    async def test_managed_port_occupant_is_not_adopted_as_owned_backend(self):
        self.controller.config = {"mode": "managed", "backend_url": module.MANAGED_URL}
        with patch.object(module.asyncio, "create_subprocess_exec", new=AsyncMock()) as spawn:
            status = (await self.http.get("/create/status")).json()
            self.assertTrue(status["backend"]["running"])
            self.assertFalse(status["backend"]["owned"])
            self.assertFalse(status["backend"]["ready"])
            self.assertIn("18010", status["backend"]["error"])
            self.assertEqual((await self.http.post("/create/backend/start")).status_code, 409)
            self.assertEqual((await self.http.post("/create/sessions")).status_code, 409)
            spawn.assert_not_called()

    # AC: @donut-create-plugin ac-managed-setup
    async def test_managed_child_uses_its_own_interpreter_caches_and_port(self):
        self.controller.config = {"mode": "managed", "backend_url": module.MANAGED_URL}
        process = SimpleNamespace(returncode=None, terminate=lambda: None, wait=AsyncMock())
        with patch.object(self.controller, "readiness", new=AsyncMock(return_value={"running": False})), \
             patch.object(module.asyncio, "create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn, \
             patch.dict(module.os.environ, {"PYTHONPATH": "/synthetic/unrelated", "PYTHONHOME": "/synthetic/python",
                         "HF_HOME": "/synthetic/hf", "TORCH_HOME": "/synthetic/torch", "HF_TOKEN": "synthetic-secret"}, clear=True):
            response = await self.http.post("/create/backend/start")
            self.assertEqual(response.status_code, 200)
            command = spawn.call_args.args
            self.assertEqual(command[0], str(self.installer.python_path))
            self.assertEqual(command[command.index("--port") + 1], "18010")
            environment = spawn.call_args.kwargs["env"]
            self.assertNotIn("PYTHONPATH", environment)
            self.assertNotIn("PYTHONHOME", environment)
            self.assertNotIn("HF_TOKEN", environment)
            for key, value in self.installer.backend_environment().items():
                self.assertEqual(environment[key], value)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_existing_mode_requires_explicit_origin(self):
        response = await self.http.post("/create/config", json={"mode": "existing"})
        self.assertEqual(response.status_code, 400)

    # AC: @donut-create-plugin ac-job-ownership
    async def test_backend_loss_marks_missing_running_prompt_and_retains_execution_error(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.pending = []
        self.clock += 11
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        self.assertEqual(state["jobs"][0]["status"], "error")
        self.assertIn("restarted", state["jobs"][0]["error"])
        self.backend.fail = True
        response = await self.http.get(f"/create/sessions/{sid}")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.controller.session(sid)["jobs"][own]["status"], "error")

    # AC: @donut-create-plugin ac-workflow-state
    async def test_resolved_execution_metadata_is_owned_and_uses_actual_prompt(self):
        sid = await self.new_session()
        own = await self.submit(sid, prompt={
            "1": {"class_type": "SeedNode", "inputs": {"seed": 91}},
            "2": {"class_type": "DonutSampler", "inputs": {"noise_seed": ["1", 0], "steps": 8, "cfg_start": 1.2, "sampler_name": "euler"}},
        })
        self.backend.complete(own, extra={"donut_final_prompt": ["actual expanded synthetic prompt"], "donut_prompt_set": [2]})
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        output = state["outputs"][0]
        result = await self.http.get(f"/create/sessions/{sid}/outputs/{output['id']}/metadata")
        self.assertEqual(result.json(), {"prompt_id": own, "prompt": "actual expanded synthetic prompt", "prompt_source": "executed_output",
            "prompt_set": 2, "seed": 91, "steps": 8, "cfg": 1.2, "sampler": "euler"})
        self.assertNotIn("negative_prompt", result.json())
        other = await self.new_session()
        self.assertEqual((await self.http.get(f"/create/sessions/{other}/outputs/{output['id']}/metadata")).status_code, 404)

    # AC: @donut-create-plugin ac-workflow-state
    async def test_v5_seed_plan_nested_sampler_seed_is_retained_in_output_metadata(self):
        sid = await self.new_session()
        own = await self.submit(sid, prompt={
            "1138:1102": {"class_type": "SeedNode", "inputs": {"seed": 731}},
            "1138:1166": {"class_type": "DonutSeedPlan", "inputs": {
                "text_seed": 19, "sampler_seed": ["1138:1102", 0], "filename_seed": 23}},
            "1138:1176:32": {"class_type": "DonutSampler", "inputs": {
                "noise_seed": ["1138:1166", 1], "steps": 8, "cfg_start": 1.2, "sampler_name": "euler"}},
        })
        self.backend.complete(own)
        state = (await self.http.get(f"/create/sessions/{sid}")).json()
        output = state["outputs"][0]
        result = await self.http.get(f"/create/sessions/{sid}/outputs/{output['id']}/metadata")
        self.assertEqual(result.json()["seed"], 731)
        self.assertEqual(output["metadata"]["seed"], 731)
        self.assertEqual(state["jobs"][0]["prompt"]["1138:1176:32"]["inputs"]["noise_seed"], ["1138:1166", 1])

    # AC: @donut-create-plugin ac-workflow-state
    async def test_seed_plan_output_slots_and_offsets_wrap_at_json_integer_limit(self):
        for sampler_seed, expected in (
            (0, [19, 0, 2, 3, 4, "23"]),
            (9007199254740991, [19, 9007199254740991, 1, 2, 3, "23"]),
        ):
            for slot, resolved in enumerate(expected):
                with self.subTest(sampler_seed=sampler_seed, slot=slot):
                    prompt = {
                        "1138:1166": {"class_type": "DonutSeedPlan", "inputs": {
                            "text_seed": 19, "sampler_seed": sampler_seed, "filename_seed": 23}},
                        "1138:1176:32": {"class_type": "DonutSampler", "inputs": {"noise_seed": ["1138:1166", slot]}},
                    }
                    metadata = self.controller.execution_metadata({"id": "synthetic", "prompt": prompt}, {"outputs": {}})
                    self.assertEqual(metadata["seed"], resolved)
                    self.assertEqual(type(metadata["seed"]), type(resolved))

    # AC: @donut-create-plugin ac-workflow-state
    async def test_seed_plan_invalid_inputs_slots_and_cycles_do_not_invent_seed_metadata(self):
        defaults = {"text_seed": 19, "sampler_seed": 731, "filename_seed": 23}
        cases = [(slot, defaults) for slot in (-1, 6, "1", True)]
        cases.extend((1, {**defaults, "sampler_seed": value}) for value in (-1, 9007199254740992, 1.0, True, "731"))
        cases.extend((1, {**defaults, key: value}) for key, value in (
            ("text_seed", -1), ("filename_seed", 9007199254740992), ("sampler_seed", ["1138:1166", 1])))
        cases.append((1, {"text_seed": 19, "sampler_seed": 731}))
        for slot, inputs in cases:
            with self.subTest(slot=slot, inputs=inputs):
                prompt = {
                    "1138:1166": {"class_type": "DonutSeedPlan", "inputs": inputs},
                    "1138:1176:32": {"class_type": "DonutSampler", "inputs": {"noise_seed": ["1138:1166", slot]}},
                }
                metadata = self.controller.execution_metadata({"id": "synthetic", "prompt": prompt}, {"outputs": {}})
                self.assertNotIn("seed", metadata)

    # AC: @donut-create-plugin ac-job-ownership
    async def test_websocket_filters_other_jobs_and_scopes_binary_preview(self):
        sid = await self.new_session()
        own = await self.submit(sid)
        self.backend.pending = []
        self.backend.running = [self.backend.row(own)]
        messages = [
            json.dumps({"type": "execution_start", "data": {"prompt_id": "other"}}),
            json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 50}}}}),
            json.dumps({"type": "execution_start", "data": {"prompt_id": own}}),
            b"synthetic-own-preview",
            json.dumps({"type": "progress", "data": {"prompt_id": "other", "value": 7, "max": 8}}),
            json.dumps({"type": "progress", "data": {"prompt_id": own, "value": 3, "max": 8}}),
            json.dumps({"type": "execution_error", "data": {"prompt_id": own, "exception_message": "synthetic failure"}}),
            b"other-preview-after-own-job",
        ]

        class Upstream:
            def __aiter__(self):
                self.messages = iter(messages)
                return self

            async def __anext__(self):
                await asyncio.sleep(0)
                try:
                    return next(self.messages)
                except StopIteration:
                    raise StopAsyncIteration from None

        class Socket:
            def __init__(self):
                self.sent = []
                self.accepted = False
                self.closed = None

            async def accept(self):
                self.accepted = True

            async def receive(self):
                await asyncio.Event().wait()

            async def send_text(self, value):
                self.sent.append(json.loads(value))

            async def send_bytes(self, value):
                self.sent.append(value)

            async def close(self, **kwargs):
                self.closed = kwargs

        captured_urls = []

        @asynccontextmanager
        async def connect(url, **kwargs):
            captured_urls.append((url, kwargs))
            yield Upstream()

        endpoint = next(route.endpoint for route in self.app.routes if route.path == "/studio/{sid}/ws")
        socket = Socket()
        with patch.object(module.websockets, "connect", connect):
            await asyncio.wait_for(endpoint(socket, sid), timeout=2)
        self.assertTrue(socket.accepted)
        self.assertEqual(socket.closed, None)
        self.assertEqual([entry for entry in socket.sent if isinstance(entry, bytes)], [b"synthetic-own-preview"])
        events = [entry for entry in socket.sent if isinstance(entry, dict)]
        self.assertEqual([entry["type"] for entry in events], ["status", "execution_start", "progress", "execution_error"])
        self.assertEqual(events[0]["data"]["status"]["exec_info"]["queue_remaining"], 1)
        self.assertEqual(self.controller.session(sid)["jobs"][own]["status"], "error")
        self.assertIn("first-comfy.invalid", captured_urls[0][0])
        self.assertIn("clientId=donut-create-" + sid, captured_urls[0][0])
        self.assertEqual(captured_urls[0][1]["proxy"], None)

    # AC: @donut-create-plugin ac-access-boundary
    async def test_websocket_expired_capability_rejected_before_connect(self):
        sid = await self.new_session()
        self.clock += 61
        close = AsyncMock()
        socket = type("Socket", (), {"close": close})()
        endpoint = next(route.endpoint for route in self.app.routes if route.path == "/studio/{sid}/ws")
        with patch.object(module.websockets, "connect") as connect:
            await endpoint(socket, sid)
            connect.assert_not_called()
        self.assertEqual(close.call_args.kwargs["code"], 1008)


if __name__ == "__main__":
    unittest.main()
