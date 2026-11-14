"""Optional owned llama.cpp helper. Briefs/results live only in bounded memory."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import platform
import re
import shutil
import socket
import tarfile
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path

import httpx
from installer import VerifiedDownloads, SetupError, SetupCancelled, download_opener, _atomic_json, _read_json, _relative_path, _fingerprint

ROOT = Path(__file__).resolve().parent
CATALOG = json.loads((ROOT / "enhancer_models.json").read_text(encoding="utf-8"))
SYSTEM_PROMPT = (ROOT / "enhancer_prompt.txt").read_text(encoding="utf-8")
MAX_BRIEF = 16000
MAX_FACE = 8000
MAX_STYLE = 4000
FACE_POSITIONS = ("left", "center", "right")
FACE_ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th",
                 "9th", "10th", "11th", "12th", "13th", "14th", "15th", "16th")


def inference_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**53:
        raise ValueError("The enhancer seed must be a nonnegative JavaScript-safe integer.")
    # The pinned llama.cpp sampler uses uint32 seeds; UINT32_MAX requests a
    # random seed. Fold wider image seeds deterministically without that sentinel.
    # This finite mapping can collide; retain the original seed on the job.
    return seed % 0xffffffff


def messages(image_brief, face_brief, style_brief=""):
    brief = {"image_brief": image_brief, "face_brief": face_brief}
    if style_brief.strip():
        brief["style_brief"] = style_brief
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(brief, ensure_ascii=False)}]


def carry_style(text, style):
    # The style is a fixed fact of every prompt it conditions. Small models can
    # drop literal wording, so restore it in front rather than lose it silently.
    squash = lambda value: " ".join(value.split()).casefold()
    if not style.strip() or not text.strip() or squash(text).startswith(squash(style)):
        return text
    return style.strip() + " " + text


def _valid_relative_positions(plan):
    labels = {record["character"] for record in plan}
    for record in plan:
        position = record["position"]
        if not isinstance(position, dict) or set(position) != {"left_of", "above"}:
            return False
        for references in position.values():
            if (not isinstance(references, list) or len(references) > 15
                    or not all(isinstance(label, str) and label in labels and label != record["character"] for label in references)
                    or len(references) != len(set(references))):
                return False
    for axis in ("left_of", "above"):
        edges = {record["character"]: record["position"][axis] for record in plan}
        visiting, complete = set(), set()
        def visit(label):
            if label in visiting:
                return False
            if label in complete:
                return True
            visiting.add(label)
            if not all(visit(target) for target in edges[label]):
                return False
            visiting.remove(label)
            complete.add(label)
            return True
        if not all(visit(label) for label in labels):
            return False
    return True


def parse_result(text, finish_reason, supplied_face=False, style=""):
    if finish_reason != "stop" or not isinstance(text, str):
        raise ValueError("The model did not finish its response. Try again; your original prompts are unchanged.")
    # MiniCPM's ChatML template can return its complete reasoning before JSON.
    text = re.sub(r"\A\s*<think>[\s\S]*?</think>\s*", "", text, count=1)
    try:
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Duplicate response field")
                value[key] = item
            return value
        result = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        raise ValueError("The model returned an invalid prompt pair. Try again; your original prompts are unchanged.") from None
    def valid_text(value, limit, required=False):
        return (isinstance(value, str) and len(value) <= limit and (not required or bool(value.strip()))
                and not re.search(r"%(?:load|file)\s*:", value, re.I))
    if (not isinstance(result, dict) or set(result) not in ({"main_prompt", "face_prompt"}, {"main_prompt", "face_prompt", "face_prompts"})
            or not valid_text(result["main_prompt"], 32000, True)
            or not valid_text(result["face_prompt"], 32000)):
        raise ValueError("The model returned an incomplete or unsafe prompt pair. Try again; your original prompts are unchanged.")
    if "face_prompts" in result:
        plan = result["face_prompts"]
        valid = isinstance(plan, list) and len(plan) <= 16 and len(plan) != 1
        labels, positions = [], []
        if valid:
            for item in plan:
                if (not isinstance(item, dict) or set(item) != {"character", "position", "face_prompt"}
                        or not valid_text(item["character"], 200, True)
                        or not valid_text(item["face_prompt"], MAX_FACE, True)):
                    valid = False
                    break
                labels.append(item["character"].strip().casefold())
                positions.append(item["position"])
            valid = valid and len(labels) == len(set(labels))
            if valid and positions:
                if all(isinstance(position, str) for position in positions):
                    families = [family for family in (FACE_POSITIONS, FACE_ORDINALS) if all(position in family for position in positions)]
                    valid = bool(families)
                    if valid:
                        order = [families[0].index(position) for position in positions]
                        valid = all(first < second for first, second in zip(order, order[1:]))
                else:
                    valid = _valid_relative_positions(plan)
                valid = valid and not result["face_prompt"].strip()
        if not valid:
            raise ValueError("The model returned an invalid spatial face plan. Try again; your original prompts are unchanged.")
        if supplied_face and not plan and not result["face_prompt"].strip():
            raise ValueError("The model could not resolve the supplied facial descriptions. Specify which characters are left of or above others, or give left-to-right face order and retry; your original prompts are unchanged.")
    elif supplied_face and not result["face_prompt"].strip():
        raise ValueError("The model returned an incomplete or unsafe prompt pair. Try again; your original prompts are unchanged.")
    if style.strip():
        if not valid_text(style, MAX_STYLE):
            raise ValueError("The style is oversized or contains unsafe file macros. Edit it and retry; your original prompts are unchanged.")
        result = {**result, "main_prompt": carry_style(result["main_prompt"], style),
                  "face_prompt": carry_style(result["face_prompt"], style)}
        if "face_prompts" in result:
            result["face_prompts"] = [{**row, "face_prompt": carry_style(row["face_prompt"], style)} for row in result["face_prompts"]]
        if (len(result["main_prompt"]) > MAX_BRIEF or len(result["face_prompt"]) > MAX_FACE
                or any(len(row["face_prompt"]) > MAX_FACE for row in result.get("face_prompts", []))):
            raise ValueError("The prompts plus the style exceed the length limits. Shorten the style and retry; your original prompts are unchanged.")
    return result


def runtime_key():
    system = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}.get(platform.system(), "unsupported")
    machine = platform.machine().lower()
    architecture = "arm64" if machine in {"arm64", "aarch64"} else "x64" if machine in {"x86_64", "amd64"} else machine
    return system + "-" + architecture


class Enhancer(VerifiedDownloads):
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir) / "enhancer"
        self.key = runtime_key()
        self.runtime = CATALOG["runtimes"].get(self.key)
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._tokens = {}
        self._opener = download_opener()
        self._status = {}
        self.jobs = {}
        self.current = None
        self.task = None
        self.process = None
        self._loaded_model = None
        self._port = None
        self._api_key = None
        self._stop_task = None
        self.closing = False

    @property
    def runtime_root(self):
        return self.state_dir / (CATALOG["runtime_version"] + "-" + self.key)

    def model_path(self, model):
        return self.state_dir / "models" / CATALOG["models"][model]["filename"]

    def _update(self, **values):
        with self._lock:
            self._status.update(values)

    def ready(self, model):
        receipt = _read_json(self.state_dir / (model + "-ready.json"), {})
        expected = {"runtime_sha256": self.runtime["sha256"] if self.runtime else None,
                    "model_sha256": CATALOG["models"][model]["sha256"]}
        if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("runtime_files"):
            return False
        try:
            return (_fingerprint(self.model_path(model)) == receipt["model_file"]
                    and all(_fingerprint(self.runtime_root / name) == fingerprint for name, fingerprint in receipt["runtime_files"].items()))
        except (OSError, KeyError):
            return False

    def status(self, owner):
        # Each studio can see shared availability, but never another studio's text/job ID.
        result = {"supported": self.runtime is not None, "default_model": CATALOG["default_model"],
                  "device": "cpu", "busy": self.task is not None and not self.task.done(),
                  "loaded_model": self._loaded_model if self.process is not None and self.process.returncode is None else None,
                  "models": [{"id": key, "name": model["name"], "size": model["size"],
                              "quantization": model["quantization"], "ready": self.ready(key)}
                             for key, model in CATALOG["models"].items()]}
        with self._lock:
            result["setup"] = copy.deepcopy(self._status)
        if owner in self.jobs:
            result["job"] = copy.deepcopy(self.jobs[owner])
        return result

    def start(self, owner, kind, model, image_brief="", face_brief="", seed=0, style_brief=""):
        if self.closing or self.task is not None and not self.task.done():
            raise RuntimeError("The enhancer is busy. Wait for the current operation to finish.")
        if model not in CATALOG["models"] or kind not in {"install", "enhance"}:
            raise ValueError("Choose an available enhancer model and action.")
        if not self.runtime:
            raise ValueError("A managed enhancer runtime is not available for this platform.")
        if (not isinstance(image_brief, str) or not isinstance(face_brief, str) or not isinstance(style_brief, str)
                or len(image_brief) > MAX_BRIEF or len(face_brief) > MAX_FACE or len(style_brief) > MAX_STYLE):
            raise ValueError("The main brief must fit 16,000 characters, the face brief 8,000 and the style 4,000.")
        sampling_seed = inference_seed(seed)
        if kind == "enhance" and (not image_brief.strip() or not self.ready(model)):
            raise ValueError("Write an image brief and install the selected model first.")
        if len(self.jobs) >= 32:
            self.jobs.pop(next(iter(self.jobs)))
        job = {"id": uuid.uuid4().hex, "kind": kind, "model": model, "state": "running"}
        if kind == "enhance":
            job.update(seed=seed, inference_seed=sampling_seed)
        self.jobs[owner] = job
        self.current = (owner, job)
        self._cancel.clear()
        self._status = {}
        self.task = asyncio.create_task(self._run(job, model, image_brief, face_brief, seed, style_brief))
        return copy.deepcopy(job)

    async def _run(self, job, model, image_brief, face_brief, seed, style_brief=""):
        try:
            if job["kind"] == "install":
                await self.stop_process()
                await asyncio.to_thread(self.install, model)
            else:
                job["result"] = await self.infer(model, image_brief, face_brief, seed, style_brief)
            job["state"] = "complete"
        except (asyncio.CancelledError, SetupCancelled):
            job["state"] = "cancelled"
            job.pop("result", None)
        except (ValueError, SetupError) as error:
            job.update(state="error", error=str(error))
        except Exception:
            # Native/HTTP errors can contain request bodies. Never expose or log them.
            job.update(state="error", error="The local enhancer could not complete this action. Retry or repair its installation.")

    async def cancel(self, owner, job_id):
        job = self.jobs.get(owner)
        if not job or job["id"] != job_id:
            raise PermissionError("Enhancer operation not found in this studio.")
        if self.current == (owner, job) and self.task is not None and not self.task.done():
            task = self.task
            self._cancel.set()
            if job["kind"] == "enhance":
                task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        # A completed job can discard its idle helper; an older job must never
        # stop the helper now serving another studio's newer operation.
        if self.current == (owner, job):
            await self.stop_process()
        # Explicit cancellation/discard also removes completed text from memory.
        job.pop("result", None)
        job["state"] = "cancelled"
        return self.status(owner)

    async def close(self):
        self.closing = True
        if self.current and self.task is not None and not self.task.done():
            await self.cancel(self.current[0], self.current[1]["id"])
        await self.stop_process()
        self.jobs.clear()

    def unpack(self, archive, target):
        """Extract verified release binaries without trusting archive paths or links."""
        stage = Path(tempfile.mkdtemp(prefix=".runtime-", dir=self.state_dir))
        try:
            links = []
            if self.runtime["filename"].endswith(".zip"):
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        self._check_cancel()
                        path = stage / _relative_path(member.filename.rstrip("/"))
                        if member.is_dir():
                            path.mkdir(parents=True, exist_ok=True)
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.open(member) as source, path.open("xb") as destination:
                                shutil.copyfileobj(source, destination)
            else:
                with tarfile.open(archive, "r:gz") as bundle:
                    for member in bundle:
                        self._check_cancel()
                        name = member.name.removeprefix("./").rstrip("/")
                        if not name:
                            continue
                        path = stage / _relative_path(name)
                        if member.isdir():
                            path.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.extractfile(member) as source, path.open("xb") as destination:
                                shutil.copyfileobj(source, destination)
                            path.chmod(0o755 if member.mode & 0o111 else 0o644)
                        elif member.issym():
                            referent = (path.parent / member.linkname).resolve()
                            if not referent.is_relative_to(stage.resolve()):
                                raise SetupError("The runtime archive contains an unsafe link.")
                            links.append((path, referent))
                        else:
                            raise SetupError("The runtime archive contains an unsafe entry.")
                while links:
                    remaining = []
                    for path, referent in links:
                        if referent.is_file():
                            path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(referent, path)
                        else:
                            remaining.append((path, referent))
                    if len(remaining) == len(links):
                        raise SetupError("The runtime archive contains an unresolved link.")
                    links = remaining
            executable = "llama-server.exe" if os.name == "nt" else "llama-server"
            servers = list(stage.rglob(executable))
            if len(servers) != 1:
                raise SetupError("The runtime archive did not contain one llama-server executable.")
            self._check_cancel()
            if target.exists():
                shutil.rmtree(target)  # Only this version's owned, inactive runtime is replaced.
            stage.rename(target)
            return servers[0].relative_to(stage).as_posix()
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def install(self, model):
        item = CATALOG["models"][model]
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._update(phase="download", total_bytes=self.runtime["size"] + item["size"], progress=0)
        runtime_receipt = _read_json(self.runtime_root / "receipt.json", {})
        valid_runtime = runtime_receipt.get("sha256") == self.runtime["sha256"]
        try:
            valid_runtime = valid_runtime and bool(runtime_receipt.get("files")) and all(
                _fingerprint(self.runtime_root / name) == fingerprint for name, fingerprint in runtime_receipt["files"].items())
        except OSError:
            valid_runtime = False
        target = self.model_path(model)
        valid_model = self._matches(target, item)
        def remaining(source):
            partial = self.state_dir / 'downloads' / (source['sha256'] + '.part')
            return source['size'] - min(partial.stat().st_size if partial.is_file() else 0, source['size'])
        required = (0 if valid_model else remaining(item)) + (0 if valid_runtime else remaining(self.runtime) + 256 * 1024 * 1024)
        if shutil.disk_usage(self.state_dir).free < required:
            raise SetupError(f"Enhancer setup needs another {required / 1e9:.2f} GB of free disk space. Verified files and partial downloads were preserved.")
        if not valid_runtime:
            archive = self._download(self.runtime, model_progress=True)
            server = self.unpack(archive, self.runtime_root)
            runtime_receipt = {"sha256": self.runtime["sha256"], "server": server,
                               "files": {path.relative_to(self.runtime_root).as_posix(): _fingerprint(path)
                                         for path in self.runtime_root.rglob("*") if path.is_file()}}
            _atomic_json(self.runtime_root / "receipt.json", runtime_receipt)
            archive.unlink(missing_ok=True)
        self._update(phase="verify", current_file=item["filename"])
        if not valid_model:
            self._update(phase="download")
            part = self._download(item, completed=self.runtime["size"], model_progress=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._check_cancel()
            os.replace(part, target)
        self._check_cancel()
        _atomic_json(self.state_dir / (model + "-ready.json"), {
            "runtime_sha256": self.runtime["sha256"], "model_sha256": item["sha256"],
            "model_file": _fingerprint(target), "runtime_files": runtime_receipt["files"], "server": runtime_receipt["server"]})
        self._update(phase="complete", progress=1, downloaded_bytes=self._status["total_bytes"])

    async def stop_process(self):
        cleanup = self._stop_task
        if cleanup is None:
            process, self.process = self.process, None
            self._loaded_model = self._port = self._api_key = None
            if process is None or process.returncode is not None:
                return
            async def reap():
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
            cleanup = asyncio.create_task(reap())
            self._stop_task = cleanup
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                if cleanup.cancelled():
                    raise
                # A second Cancel request must not abandon native teardown.
                cancelled = True
        if self._stop_task is cleanup:
            self._stop_task = None
        if cancelled:
            raise asyncio.CancelledError

    async def _start_process(self, model):
        receipt = _read_json(self.state_dir / (model + "-ready.json"))
        executable = self.runtime_root / receipt["server"]
        key = uuid.uuid4().hex + uuid.uuid4().hex
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = [str(executable), "-m", str(self.model_path(model)), "--host", "127.0.0.1", "--port", str(port),
                   "--api-key", key, "--ctx-size", "16384", "--parallel", "1", "--n-gpu-layers", "0",
                   "--device", "none", "--no-op-offload", "--no-kv-offload",
                   "--threads", str(min(4, os.cpu_count() or 1)), "--log-disable", "--no-webui", "--no-context-shift"]
        # An existing llama installation's environment must not enable remote
        # models, RPC, log files, adapters, or other unselected runtime options.
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith('LLAMA_') and name != 'HF_TOKEN'}
        library_key = 'DYLD_LIBRARY_PATH' if platform.system() == 'Darwin' else 'LD_LIBRARY_PATH'
        if os.name != 'nt':
            environment[library_key] = str(executable.parent) + (os.pathsep + environment[library_key] if environment.get(library_key) else '')
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(*command, cwd=executable.parent,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, env=environment))
        cancelled = False
        while True:
            try:
                self.process = await asyncio.shield(spawn)
                break
            except asyncio.CancelledError:
                if spawn.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        self._port, self._api_key = port, key

    async def infer(self, model, image_brief, face_brief, seed=0, style_brief=""):
        sampling_seed = inference_seed(seed)
        # Keep only this add-on's healthy CPU helper. Requests carry fresh briefs
        # and seeds; no shared model server, GPU unload, cache file, or text logs.
        try:
            if self.process is None or self.process.returncode is not None or self._loaded_model != model:
                await self.stop_process()
                await self._start_process(model)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{self._port}", trust_env=False,
                    headers={"Authorization": "Bearer " + self._api_key}, timeout=600) as client:
                for _ in range(600):
                    if self.process.returncode is not None:
                        raise ValueError("The managed enhancer runtime could not start on this computer. Repair it or check platform compatibility.")
                    try:
                        health = await client.get("/health", timeout=1)
                        if health.status_code == 200:
                            self._loaded_model = model
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.2)
                else:
                    raise ValueError("The managed enhancer model took too long to load. Try again.")
                conversation = messages(image_brief, face_brief, style_brief)
                # Measure the actual templated input; never let a long brief silently lose facts.
                template = await client.post('/apply-template', json={"messages": conversation,
                    "chat_template_kwargs": {"enable_thinking": False}})
                template.raise_for_status()
                tokens = await client.post('/tokenize', json={"content": template.json()["prompt"],
                    "add_special": True, "parse_special": True})
                tokens.raise_for_status()
                if len(tokens.json()["tokens"]) + 4096 + 64 > 16384:
                    raise ValueError("These prompts exceed the model's context. Shorten them and retry; nothing has been truncated or applied.")
                response = await client.post("/v1/chat/completions", json={
                    "messages": conversation, "seed": sampling_seed, "temperature": 0.7, "top_p": 0.9,
                    "max_tokens": 4096, "stream": False, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_schema", "json_schema": {"name": "prompt_pair", "strict": True, "schema": {
                        "type": "object", "properties": {"main_prompt": {"type": "string", "maxLength": 32000},
                            "face_prompt": {"type": "string", "maxLength": 32000},
                            "face_prompts": {"type": "array", "maxItems": 16, "items": {
                                "type": "object", "properties": {"character": {"type": "string", "minLength": 1, "maxLength": 200},
                                    "position": {"oneOf": [{"type": "string", "enum": list(FACE_POSITIONS + FACE_ORDINALS)},
                                        {"type": "object", "properties": {axis: {"type": "array", "maxItems": 15, "uniqueItems": True,
                                            "items": {"type": "string", "minLength": 1, "maxLength": 200}} for axis in ("left_of", "above")},
                                         "required": ["left_of", "above"], "additionalProperties": False}]},
                                    "face_prompt": {"type": "string", "minLength": 1, "maxLength": MAX_FACE}},
                                "required": ["character", "position", "face_prompt"], "additionalProperties": False}}},
                        "required": ["main_prompt", "face_prompt", "face_prompts"], "additionalProperties": False}}}})
                if not response.is_success:
                    raise ValueError("The local model could not process this brief. Try a shorter brief or retry; nothing has been applied.")
                choice = response.json()["choices"][0]
                return parse_result(choice["message"].get("content"), choice.get("finish_reason"), bool(face_brief.strip()), style_brief)
        except BaseException:
            await self.stop_process()
            raise
