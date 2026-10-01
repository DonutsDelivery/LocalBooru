"""Donut Create controller. The DMC API authorizes management requests.

Studio links are short lived capabilities, with an immutable ComfyUI backend
and a ledger of prompts submitted through that link. No user library is opened
by this process; completed images are streamed from ComfyUI's history/view API.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import mimetypes
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask


MANAGED_URL = "http://127.0.0.1:18010"
SESSION_TTL = 24 * 60 * 60
MAX_BODY = 32 * 1024 * 1024
MAX_PROMPT = 4 * 1024 * 1024
CAPABILITY = re.compile(r"^[A-Za-z0-9_-]{43}$")
PROMPT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
STATIC_SUFFIXES = {".js", ".mjs", ".css", ".wasm", ".png", ".jpg", ".jpeg",
                   ".webp", ".gif", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".json"}
STATIC_ROOTS = {"assets", "scripts", "lib", "locales", "i18n", "fonts"}
EXTENSION_ROOTS = {"core", "donutnodes", "comfyui-impact-pack", "comfyui-impact-subpack",
                   "ComfyUI_essentials", "ComfyUI-VAE-Utils", "ComfyUI-bleh", "RES4LYF", "mikey_nodes",
                   "derfuu_comfyui_moddednodes", "comfyui-krea2edit", "krea2-nag",
                   "krea-seed-variance-enhancer", "rgthree-comfy"}
SAFE_EXTRA_NODES = {"SaveImage", "PreviewImage", "LoadImage", "DonutSubjectMaskPreview", "DonutToneLab"}
FILE_MACRO = re.compile(r"__|(?<![\w/*])[A-Za-z][\w/-]*\*(?![\w*])")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}


def safe_prompt_text(value: str) -> None:
    if FILE_MACRO.search(value) or re.search(r"%(?!date:[A-Za-z0-9.:-]+%)[^%\n]+%", value):
        raise HTTPException(403, "File and graph text macros are unavailable through studio links.")
    if ("_" in value or "*" in value) and ("{" in value or "%" in value):
        raise HTTPException(403, "Prompt choices must not construct file wildcard tokens.")


def validate_backend_url(value: str) -> str:
    """Deliberate LAN connections are supported; credentials and URL paths aren't."""
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError("Enter an HTTP or HTTPS ComfyUI origin.")
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError:
        raise ValueError("Invalid backend port.") from None
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.path not in {"", "/"} or parts.query or parts.fragment
            or "\\" in value or "%" in parts.netloc):
        raise ValueError("Use a ComfyUI origin without credentials, paths, queries, or fragments.")
    if port is not None and not 0 < port < 65536:
        raise ValueError("Invalid backend port.")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def safe_relative(value: str, *, empty: bool = False) -> bool:
    if not isinstance(value, str) or "\\" in value or "\0" in value:
        return False
    if value == "":
        return empty
    path = PurePosixPath(value)
    return (not path.is_absolute() and not re.match(r"^[A-Za-z]:", value)
            and all(part not in {"..", ".", ""} for part in value.split("/")))


def clean_route(rest: str) -> str:
    decoded = unquote(rest)
    if "%" in decoded or not safe_relative(decoded, empty=True):
        raise HTTPException(403, "This route is outside the studio.")
    return decoded


def state_directory() -> Path:
    explicit = os.environ.get("LOCALBOORU_CREATE_STATE_DIR")
    if explicit:
        return Path(explicit)
    packages = os.environ.get("LOCALBOORU_PACKAGES_DIR")
    if packages:
        return Path(packages) / "donut-create-runtime"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "localbooru/donut-create"


def private_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"), ensure_ascii=False)
    os.replace(temporary, path)
    path.chmod(0o600)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


def queue_ids(queue: dict[str, Any], key: str) -> set[str]:
    return {str(row[1]) for row in queue.get(key, []) if isinstance(row, (list, tuple)) and len(row) > 1}


def scrub_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: scrub_paths(child) for key, child in value.items()
                if key not in {"path", "full_path", "directory", "api_key", "token"}}
    if isinstance(value, list):
        return [scrub_paths(child) for child in value]
    return value


async def limited_body(request: Request, limit: int = MAX_BODY) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(413, "Studio request is too large.")
    return bytes(body)


def multipart_fields(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]]]:
    if not content_type.startswith("multipart/form-data;") or "\r" in content_type or "\n" in content_type:
        raise HTTPException(400, "Upload an image using multipart form data.")
    message = BytesParser(policy=email_policy).parsebytes(
        ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
    fields, files = {}, {}
    if not message.is_multipart():
        raise HTTPException(400, "Invalid image upload.")
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        if not isinstance(name, str) or name in fields or name in files:
            raise HTTPException(400, "Invalid image upload fields.")
        payload = part.get_payload(decode=True) or b""
        if filename is not None:
            files[name] = (filename, payload, part.get_content_type())
        else:
            try:
                fields[name] = payload.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(400, "Invalid image upload fields.") from None
    return fields, files


class Controller:
    def __init__(self, state_dir: Path | str, installer: Any = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 clock: Any = time.time, ttl: int = SESSION_TTL):
        self.state_dir = Path(state_dir).resolve()
        repository = Path(__file__).resolve().parents[2]
        if self.state_dir.is_relative_to(repository) or any((parent / ".git").exists() for parent in [self.state_dir, *self.state_dir.parents]):
            raise ValueError("Donut Create state must live outside the repository.")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.chmod(0o700)
        instance = read_json(self.state_dir / "instance.json", {})
        self.instance_id = instance.get("id") if isinstance(instance, dict) else None
        if not isinstance(self.instance_id, str) or not re.fullmatch(r"[a-f0-9]{32}", self.instance_id):
            self.instance_id = uuid.uuid4().hex
            private_json(self.state_dir / "instance.json", {"id": self.instance_id})
        if installer is None:
            from installer import Installer
            installer = Installer(self.state_dir)
        self.installer = installer
        self.clock = clock
        self.ttl = ttl
        self.config = read_json(self.state_dir / "config.json", {"mode": "managed", "backend_url": MANAGED_URL})
        self.config["backend_url"] = validate_backend_url(self.config["backend_url"])
        persisted = read_json(self.state_dir / "sessions.json", {})
        self.sessions = {sid: session for sid, session in persisted.items()
                         if CAPABILITY.fullmatch(sid) and isinstance(session, dict)
                         and session.get("expires_at", 0) > self.clock()}
        workspaces = read_json(self.state_dir / "workspaces.json", {})
        self.workspaces = {key: value for key, value in workspaces.items()
                           if re.fullmatch(r"[a-f0-9]{32}", key) and isinstance(value, dict)}
        self.client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                       timeout=httpx.Timeout(60.0, connect=5.0))
        self.process: asyncio.subprocess.Process | None = None
        self.backend_error: str | None = None
        self.mutation_lock = asyncio.Lock()
        self.backend_lock = asyncio.Lock()
        self.closing = False
        self.workspace_streams: dict[str, dict[str, Any]] = {}

    def persist(self) -> None:
        private_json(self.state_dir / "sessions.json", self.sessions)
        private_json(self.state_dir / "workspaces.json", self.workspaces)

    def session(self, sid: str) -> dict[str, Any]:
        if not CAPABILITY.fullmatch(sid):
            raise HTTPException(404, "Studio session not found.")
        session = self.sessions.get(sid)
        if session is None:
            raise HTTPException(404, "Studio session not found.")
        if session["expires_at"] <= self.clock():
            raise HTTPException(410, "Studio session expired. Open a new studio.")
        return session

    def allowed_nodes(self) -> set[str]:
        manifest = read_json(Path(__file__).with_name("runtime.json"), {})
        return SAFE_EXTRA_NODES | set(manifest.get("required_nodes", []))

    def discoverable_nodes(self) -> set[str]:
        catalog = read_json(Path(__file__).with_name("model_sources.json"), {})
        helpers = {name for model in catalog.get("models", [])
                   for name in model.get("requires_nodes", []) if isinstance(name, str)}
        # Donut's vetted wrappers call these helpers internally. Discovery
        # verifies optional model capabilities without widening queued graphs.
        return self.allowed_nodes() | helpers | {"DonutLoRAStack"}

    def expected_capabilities(self) -> dict[str, Any]:
        if hasattr(self.installer, "expected_capabilities"):
            return self.installer.expected_capabilities()
        manifest = read_json(Path(__file__).with_name("runtime.json"), {})
        profile = self.installer.status().get("profile") or manifest.get("default_profile", "base")
        return {"required_nodes": manifest.get("required_nodes", []),
                "required_models": manifest.get("profiles", {}).get(profile, {}).get("required_models", [])}

    async def request_backend(self, backend: str, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.client.request(method, backend + "/" + path.lstrip("/"), **kwargs)
        except httpx.HTTPError:
            raise HTTPException(502, "ComfyUI is unavailable. Check the backend and retry.") from None
        if 300 <= response.status_code < 400:
            raise HTTPException(502, "ComfyUI redirected outside the configured backend.")
        return response

    async def backend_json(self, backend: str, path: str) -> Any:
        response = await self.request_backend(backend, "GET", path)
        if not response.is_success:
            raise HTTPException(502, f"ComfyUI did not return {path.split('/')[0]} data.")
        try:
            return response.json()
        except ValueError:
            raise HTTPException(502, "ComfyUI returned invalid JSON.") from None

    async def readiness(self) -> dict[str, Any]:
        backend = self.config["backend_url"]
        result = {"running": False, "ready": False,
                  "owned": self.config["mode"] in {"managed", "local"} and self.process is not None and self.process.returncode is None,
                  "error": self.backend_error, "missing_nodes": [], "missing_models": []}
        try:
            info = await self.backend_json(backend, "object_info")
            if not isinstance(info, dict):
                raise HTTPException(502, "ComfyUI returned invalid node information.")
            result["running"] = True
            capabilities = self.expected_capabilities()
            required = set(capabilities.get("required_nodes", []))
            result["missing_nodes"] = sorted(required - set(info))
            models = capabilities.get("required_models", [])
            catalog: dict[str, Any] = {}
            for model in models:
                folder, filename = model.get("folder"), model.get("filename")
                if not isinstance(folder, str) or not isinstance(filename, str):
                    continue
                if folder not in catalog:
                    catalog[folder] = await self.backend_json(backend, "models/" + folder)
                if filename not in catalog[folder]:
                    result["missing_models"].append(filename)
            result["ready"] = (bool(required) and not result["missing_nodes"] and not result["missing_models"])
            if self.config["mode"] in {"managed", "local"}:
                result["ready"] = result["ready"] and result["owned"]
                if self.config["mode"] == "managed":
                    result["ready"] = result["ready"] and self.installer.ready()
                if not result["owned"]:
                    result["error"] = "Port 18010 is occupied by a backend this controller does not own."
            if not required:
                result["error"] = "The bundled v5 workflow is unavailable. Reinstall Donut Create."
            elif not result["ready"] and not result["error"]:
                result["error"] = "The backend is missing required workflow nodes or models."
        except HTTPException as error:
            result["error"] = self.backend_error or (None if result["owned"] else error.detail)
            if self.process is not None and self.process.returncode is not None:
                result["error"] = f"ComfyUI exited with code {self.process.returncode}. Check the installation and restart it."
        result["controllable"] = self.config["mode"] in {"managed", "local"}
        result["state"] = "running" if result["running"] else "starting" if result["owned"] else "stopped"
        return result

    async def status(self) -> dict[str, Any]:
        setup = self.installer.status()
        return {**self.config, "backend": await self.readiness(), "setup": setup,
                **{key: setup[key] for key in ("catalog", "disk", "default_runtime", "supported_runtimes", "default_profile") if key in setup},
                "session_ttl_seconds": self.ttl}

    def local_installation(self, options: dict[str, Any]) -> tuple[Path, Path]:
        folder = options.get("comfy_directory")
        if not isinstance(folder, str) or not folder.strip():
            raise HTTPException(400, "Choose the ComfyUI folder containing main.py.")
        root = Path(folder).expanduser().resolve()
        if not (root / "main.py").is_file() or not (root / "comfy").is_dir():
            raise HTTPException(400, "Choose a ComfyUI installation containing main.py and the comfy folder.")
        interpreter = options.get("python_executable")
        if interpreter:
            python = Path(interpreter).expanduser().absolute()
        else:
            relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
            candidates = [base / env / relative for base in (root, root.parent) for env in ("venv", ".venv")]
            candidates += [base / "python_embeded/python.exe" for base in (root, root.parent)]
            python = next((path for path in candidates if path.is_file()), None)
        if python is None or not python.is_file() or not os.access(python, os.X_OK) or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?", python.name, re.I):
            raise HTTPException(400, "Select the Python executable used by this ComfyUI installation.")
        return root, python

    async def configure(self, options: dict[str, Any]) -> dict[str, Any]:
        async with self.backend_lock:
            if self.closing:
                raise HTTPException(503, "The creator add-on is shutting down.")
            if options["mode"] not in {"managed", "existing", "local"}:
                raise HTTPException(400, "Choose a supported backend mode.")
            if self.process is not None and self.process.returncode is None:
                raise HTTPException(409, "Stop ComfyUI before changing its backend configuration.")
            if options["mode"] == "existing" and not options.get("backend_url"):
                raise HTTPException(400, "Enter the existing ComfyUI backend origin explicitly.")
            try:
                backend = validate_backend_url(options.get("backend_url") or MANAGED_URL)
            except ValueError as error:
                raise HTTPException(400, str(error)) from None
            config = {"mode": options["mode"], "backend_url": backend if options["mode"] == "existing" else MANAGED_URL}
            if options["mode"] == "local":
                root, python = self.local_installation(options)
                config.update(comfy_directory=str(root), python_executable=str(python))
            self.config = config
            private_json(self.state_dir / "config.json", self.config)
        return await self.status()

    async def start_backend(self) -> dict[str, Any]:
        async with self.backend_lock:
            await self._start_backend()
        return await self.status()

    async def _start_backend(self) -> None:
        if self.closing:
            raise HTTPException(503, "The creator add-on is shutting down.")
        if self.config["mode"] not in {"managed", "local"}:
            raise HTTPException(409, "Register a local ComfyUI installation to control it, or use managed ComfyUI.")
        if self.process is not None and self.process.returncode is None:
            return
        if self.config["mode"] == "managed":
            if not self.installer.ready():
                raise HTTPException(409, "Finish managed setup before starting ComfyUI.")
            root, python = Path(self.installer.backend_root), Path(self.installer.python_path)
        else:
            root, python = self.local_installation(self.config)
        if (await self.readiness())["running"]:
            raise HTTPException(409, "Port 18010 is occupied by a backend this controller does not own.")
        command = [str(python), str(root / "main.py"), "--listen", "127.0.0.1", "--port", "18010"]
        if self.config["mode"] == "managed" and self.installer.status().get("runtime") == "cpu":
            command.append("--cpu")
        environment = {key: value for key, value in os.environ.items()
                       if not any(word in key.upper() for word in ("TOKEN", "API_KEY", "PASSWORD", "SECRET"))}
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        if self.config["mode"] == "managed":
            environment.update(self.installer.backend_environment())
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            self.process = await asyncio.create_subprocess_exec(*command, cwd=str(root), env=environment,
                                                               stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            self.backend_error = None
        except OSError:
            self.backend_error = "ComfyUI could not start. Check its installation and Python executable."
            raise HTTPException(502, self.backend_error) from None

    async def _stop_backend(self) -> None:
        if self.config["mode"] not in {"managed", "local"}:
            raise HTTPException(409, "This URL-only backend is controlled by its host application.")
        process = self.process
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self.process = None
        self.backend_error = None

    async def stop_backend(self) -> dict[str, Any]:
        async with self.backend_lock:
            await self._stop_backend()
        return await self.status()

    async def restart_backend(self) -> dict[str, Any]:
        async with self.backend_lock:
            await self._stop_backend()
            await self._start_backend()
        return await self.status()

    async def close(self) -> None:
        # Stop only our child, including when shutdown races with a restart.
        self.closing = True
        async with self.backend_lock:
            if self.config["mode"] in {"managed", "local"}:
                await self._stop_backend()
        stream_tasks = [stream["task"] for stream in self.workspace_streams.values()]
        for task in stream_tasks:
            task.cancel()
        await asyncio.gather(*stream_tasks, return_exceptions=True)
        await self.client.aclose()

    def workspace_events(self, session: dict[str, Any]) -> tuple[dict[str, Any], asyncio.Queue]:
        """One upstream client per shared workspace, with bounded per-device delivery."""
        sid = session["id"]
        stream = self.workspace_streams.get(sid)
        if stream is None:
            stream = {"subscribers": set()}
            self.workspace_streams[sid] = stream
            stream["task"] = asyncio.create_task(self.relay_workspace_events(session, stream))
        subscriber: asyncio.Queue = asyncio.Queue(maxsize=64)
        stream["subscribers"].add(subscriber)
        return stream, subscriber

    async def relay_workspace_events(self, session: dict[str, Any], stream: dict[str, Any]) -> None:
        sid = session["id"]
        parts = urlsplit(session["backend_url"])
        backend = urlunsplit(("wss" if parts.scheme == "https" else "ws", parts.netloc, "/ws",
                              urlencode({"clientId": "donut-create-" + sid}), ""))

        def broadcast(message):
            for subscriber in tuple(stream["subscribers"]):
                if subscriber.full():
                    # A slow device reconnects and restores state via scoped HTTP polling.
                    stream["subscribers"].discard(subscriber)
                    while not subscriber.empty():
                        subscriber.get_nowait()
                    subscriber.put_nowait(None)
                else:
                    subscriber.put_nowait(message)

        try:
            async with websockets.connect(backend, proxy=None, max_size=MAX_BODY, open_timeout=5) as upstream:
                queue = await self.reconcile(session)
                active = next(iter(queue_ids(queue, "queue_running") & set(session["jobs"])), None)
                while True:
                    self.session(sid)
                    try:
                        message = await asyncio.wait_for(upstream.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        if not stream["subscribers"]:
                            return
                        continue
                    if isinstance(message, bytes):
                        if active in session["jobs"] and session["jobs"][active]["status"] == "running":
                            broadcast(message)
                        continue
                    try:
                        event = json.loads(message)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "status":
                        queue = await self.reconcile(session)
                        own = self.owned_queue(session, queue)
                        event = {"type": "status", "data": {"sid": "donut-create-" + sid,
                                 "status": {"exec_info": {"queue_remaining": sum(len(rows) for rows in own.values())}}}}
                        active = next(iter(queue_ids(queue, "queue_running") & set(session["jobs"])), None)
                    else:
                        if not self.observe_event(session, event):
                            continue
                        if event.get("type") == "execution_start":
                            active = event["data"]["prompt_id"]
                        if event.get("type") in {"execution_error", "execution_interrupted", "execution_success"} or (event.get("type") == "executing" and event["data"].get("node") is None):
                            active = None
                    broadcast(json.dumps(event))
        except (HTTPException, OSError, httpx.HTTPError, websockets.WebSocketException):
            pass  # Each device reconnects; scoped polling remains authoritative.
        finally:
            broadcast(None)
            if self.workspace_streams.get(sid) is stream:
                self.workspace_streams.pop(sid)

    async def create_session(self, workspace: bool = False) -> dict[str, Any]:
        readiness = await self.readiness()
        if not readiness["ready"]:
            raise HTTPException(409, readiness.get("error") or "ComfyUI is not ready for this workflow.")
        backend_key = self.backend_key(self.config["backend_url"], self.config["mode"])
        previous = self.workspaces.get(backend_key, {}) if workspace else {}
        shared = self.sessions.get(previous.get("session_id"))
        if (workspace and shared and shared.get("workspace")
                and shared.get("backend_key") == backend_key
                and shared["expires_at"] > self.clock()
                and (len(shared["jobs"]) < 256 or any(job["status"] in {"queued", "running"} for job in shared["jobs"].values()))):
            shared["expires_at"] = self.clock() + self.ttl
            self.persist()
            return self.public_session(shared)
        sid = secrets.token_urlsafe(32)
        self.sessions[sid] = {"id": sid, "backend_url": self.config["backend_url"],
                              "backend_key": backend_key, "workspace": workspace,
                              "profile": self.installer.status().get("profile") or "base",
                              "created_at": self.clock(), "expires_at": self.clock() + self.ttl,
                              "jobs": {}, "outputs": {}, "previews": {}, "uploads": [], "references": [],
                              "masks": [], "settings": {}, "userdata": {}}
        if workspace:
            session = self.sessions[sid]
            session["uploads"] = list(previous.get("uploads", []))
            session["references"] = list(previous.get("references", []))
            session["masks"] = list(previous.get("masks", []))
            session["latest_run_revision"] = previous.get("revision", 0)
            session["latest_run_workflow"] = previous.get("workflow")
            self.workspaces[backend_key] = {**previous, "session_id": sid}
        self.persist()
        return self.public_session(self.sessions[sid])

    def backend_key(self, backend: str, mode: str) -> str:
        identity = json.dumps([self.instance_id, mode, backend], separators=(",", ":"))
        return hashlib.sha256(identity.encode()).hexdigest()[:32]

    def public_session(self, session: dict[str, Any]) -> dict[str, Any]:
        sid = session["id"]
        return {"id": sid, "backend_url": session["backend_url"],
                "studio_path": f"/api/create/studio/{sid}/", "expires_at": session["expires_at"],
                "workspace": session.get("workspace", False), "latest_run_revision": session.get("latest_run_revision", 0),
                "jobs": [{key: value for key, value in job.items() if not session.get("workspace") or key not in {"prompt", "workflow"}}
                         for job in session["jobs"].values()], "outputs": list(session["outputs"].values()),
                "previews": list(session.get("previews", {}).values())}

    def workflow_property(self, job: dict[str, Any], node_id: str, name: str) -> Any:
        workflow = job.get("workflow") or {}
        definitions = workflow.get("definitions")
        subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
        definitions_by_id = {str(graph.get("id")): graph for graph in subgraphs or []
                             if isinstance(graph, dict)} if isinstance(subgraphs, list) else {}
        graph = workflow
        path = str(node_id).split(":")
        for index, part in enumerate(path):
            nodes = graph.get("nodes") if isinstance(graph, dict) else None
            matches = [item for item in nodes if isinstance(item, dict) and str(item.get("id")) == part] if isinstance(nodes, list) else []
            if len(matches) != 1:
                return None
            item = matches[0]
            if index == len(path) - 1:
                properties = item.get("properties")
                return properties.get(name) if isinstance(properties, dict) else None
            graph = definitions_by_id.get(str(item.get("type")))
            if graph is None:
                return None
        return None

    def final_output(self, job: dict[str, Any], node_id: str) -> bool:
        if str(node_id) in job.get("save_nodes", {}):
            return job["save_nodes"][str(node_id)]
        node = job.get("prompt", {}).get(str(node_id), {})
        if node.get("class_type") not in {"SaveImage", "DonutImageSave"}:
            return False
        marker = self.workflow_property(job, node_id, "dmc_final_output")
        if isinstance(marker, bool):
            return marker
        # Older v5 graphs have a dedicated final Donut save and an ordinary
        # SaveImage for the intermediate result.
        return node.get("class_type") == "DonutImageSave"

    def record_previews(self, session: dict[str, Any], prompt_id: str, node_id: str,
                        output: dict[str, Any]) -> None:
        if prompt_id not in session["jobs"]:
            return
        previews = session.setdefault("previews", {})
        for image in output.get("images", []):
            if (not isinstance(image, dict) or image.get("type") != "temp"
                    or not safe_relative(image.get("filename", ""))
                    or "/" in image["filename"]
                    or PurePosixPath(image["filename"]).suffix.lower() not in IMAGE_SUFFIXES
                    or not safe_relative(image.get("subfolder", ""), empty=True)):
                continue
            filename, subfolder = image["filename"], image.get("subfolder", "")
            pid = hashlib.sha256(json.dumps([prompt_id, node_id, filename, subfolder]).encode()).hexdigest()[:32]
            previews.pop(pid, None)
            previews[pid] = {"id": pid, "prompt_id": prompt_id, "node_id": str(node_id),
                             "filename": filename, "subfolder": subfolder, "type": "temp"}
            while len(previews) > 128:
                previews.pop(next(iter(previews)))

    def owned_queue(self, session: dict[str, Any], queue: dict[str, Any]) -> dict[str, Any]:
        return {key: [row for row in queue.get(key, [])
                      if isinstance(row, (list, tuple)) and len(row) > 1 and row[1] in session["jobs"]]
                for key in ("queue_running", "queue_pending")}

    def record_history(self, session: dict[str, Any], prompt_id: str, record: Any) -> None:
        if prompt_id not in session["jobs"] or not isinstance(record, dict):
            return
        job = session["jobs"][prompt_id]
        status = record.get("status", {})
        messages = status.get("messages", []) if isinstance(status, dict) else []
        failed = False
        for entry in messages:
            if isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[0] in {"execution_error", "execution_interrupted"}:
                failed = True
                job["status"] = "cancelled" if entry[0] == "execution_interrupted" else "error"
                detail = entry[1] if isinstance(entry[1], dict) else {}
                job["error"] = str(detail.get("exception_message") or detail.get("exception_type") or "Generation failed.")[:2000]
        if isinstance(status, dict) and status.get("status_str") == "error":
            failed = True
            job["status"] = "error"
            job.setdefault("error", "ComfyUI reported an execution error.")
        completed = (status.get("completed") is True or status.get("status_str") == "success"
                     or (not status and bool(record.get("outputs")))) if isinstance(status, dict) else False
        if not completed or failed:
            return
        job["status"] = "completed"
        job.pop("error", None)
        job["completed_at"] = job.get("completed_at", self.clock())
        job["execution"] = {"status": status, "prompt_id": prompt_id}
        metadata = self.execution_metadata(job, record)
        for node_id, output in record.get("outputs", {}).items():
            if not isinstance(output, dict):
                continue
            self.record_masks(session, output)
            self.record_previews(session, prompt_id, str(node_id), output)
            for image in output.get("images", []):
                if (not isinstance(image, dict) or image.get("type") not in {"output", "temp"}
                        or (image.get("type") == "temp" and str(node_id) not in job.get("save_nodes", {}))
                        or not safe_relative(image.get("filename", ""))
                        or "/" in image["filename"]
                        or PurePosixPath(image["filename"]).suffix.lower() not in IMAGE_SUFFIXES
                        or not safe_relative(image.get("subfolder", ""), empty=True)):
                    continue
                filename, subfolder = image["filename"], image.get("subfolder", "")
                oid = hashlib.sha256(json.dumps([prompt_id, node_id, filename, subfolder]).encode()).hexdigest()[:32]
                session["outputs"][oid] = {"id": oid, "prompt_id": prompt_id, "filename": filename,
                    "subfolder": subfolder, "type": image["type"], "node_id": str(node_id),
                    "storage": "comfy" if image["type"] == "output" else "temporary",
                    "final": self.final_output(job, str(node_id)),
                    "media_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                    "url": f"/api/create/output/{session['id']}/{oid}",
                    "workflow": job.get("workflow"), "prompt": job.get("prompt"), "execution": job["execution"],
                    "metadata": metadata}

    def execution_metadata(self, job: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {"prompt_id": job["id"]}
        for node_id, output in record.get("outputs", {}).items():
            if not isinstance(output, dict):
                continue
            if self.workflow_property(job, str(node_id), "dmc_prompt_role") == "face":
                continue
            for source, target in (("donut_final_prompt", "prompt"), ("donut_final_negative_prompt", "negative_prompt")):
                values = output.get(source)
                if isinstance(values, list) and values and isinstance(values[0], str):
                    metadata[target] = values[0]
                    metadata[target + "_source"] = "executed_output"
            if "prompt" not in metadata and job.get("prompt", {}).get(str(node_id), {}).get("class_type") == "DonutPromptConditioning":
                values = output.get("text")
                if isinstance(values, list) and values and isinstance(values[0], str):
                    metadata["prompt"] = values[0]
                    metadata["prompt_source"] = "executed_output"
            values = output.get("donut_prompt_set")
            if isinstance(values, list) and values and isinstance(values[0], int):
                metadata["prompt_set"] = values[0]
        prompt = job.get("prompt", {})

        def scalar(value: Any, seen: set[str] | None = None) -> Any:
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, list) and len(value) == 2:
                seen = set() if seen is None else set(seen)
                node_id = str(value[0])
                if node_id in seen:
                    return None
                seen.add(node_id)
                source = prompt.get(node_id, {})
                slot = value[1]
                inputs = source.get("inputs", {})
                if source.get("class_type") == "SeedNode" and type(slot) is int and slot == 0:
                    return scalar(inputs.get("seed"), seen)
                if source.get("class_type") == "DonutSeedPlan" and type(slot) is int and 0 <= slot <= 5:
                    seeds = {key: scalar(inputs.get(key), seen)
                             for key in ("text_seed", "sampler_seed", "filename_seed")}
                    if any(type(seed) is not int or not 0 <= seed < 2**53 for seed in seeds.values()):
                        return None
                    if slot == 0:
                        return seeds["text_seed"]
                    if slot == 5:
                        return str(seeds["filename_seed"])
                    # Match DonutSeedPlan.generate: base, upscale 1/2, face.
                    return (seeds["sampler_seed"] + (0, 2, 3, 4)[slot - 1]) % 2**53
            return None

        for node in prompt.values():
            if node.get("class_type") != "DonutSampler":
                continue
            inputs = node.get("inputs", {})
            for source, target in (("seed", "seed"), ("noise_seed", "seed"), ("steps", "steps"),
                                   ("cfg", "cfg"), ("cfg_start", "cfg"), ("sampler_name", "sampler")):
                value = scalar(inputs.get(source))
                if value is not None:
                    metadata[target] = value
            break
        return metadata

    def record_masks(self, session: dict[str, Any], output: dict[str, Any]) -> None:
        for mask in output.get("donut_subject_mask", []):
            value = mask.get("mask") if isinstance(mask, dict) else None
            if isinstance(value, str) and re.fullmatch(r"donutmask:[a-f0-9]{64}", value) and value not in session["masks"]:
                session["masks"].append(value)

    async def reconcile(self, session: dict[str, Any]) -> dict[str, Any]:
        queue = await self.backend_json(session["backend_url"], "queue")
        if not isinstance(queue, dict):
            raise HTTPException(502, "ComfyUI returned an invalid queue.")
        running, pending = queue_ids(queue, "queue_running"), queue_ids(queue, "queue_pending")
        for prompt_id, job in session["jobs"].items():
            if prompt_id in running:
                job["status"] = "running"
                job["last_seen_at"] = self.clock()
            elif prompt_id in pending:
                job["status"] = "queued"
                job["last_seen_at"] = self.clock()
            else:
                history = await self.backend_json(session["backend_url"], "history/" + prompt_id)
                record = history.get(prompt_id) if isinstance(history, dict) else None
                if record is not None:
                    self.record_history(session, prompt_id, record)
                elif job["status"] in {"queued", "running"} and self.clock() - job.get("last_seen_at", job["created_at"]) > 10:
                    job["status"] = "error"
                    job["error"] = "The backend no longer reports this prompt. It may have restarted or cleared history."
        self.persist()
        return queue

    def validate_prompt(self, session: dict[str, Any], body: dict[str, Any]) -> None:
        prompt = body.get("prompt")
        if not isinstance(prompt, dict) or not prompt or len(prompt) > 512:
            raise HTTPException(400, "Provide a valid workflow prompt.")
        allowed = self.allowed_nodes()
        def check_text(value: Any) -> None:
            if isinstance(value, str) and FILE_MACRO.search(value):
                raise HTTPException(403, "File wildcard expansion is unavailable through studio links.")
            if isinstance(value, dict):
                for child in value.values():
                    check_text(child)
            if isinstance(value, list):
                for child in value:
                    check_text(child)
        check_text(body)

        def source_text(value: Any, seen: set[str] | None = None) -> str:
            if isinstance(value, str):
                return value
            if not isinstance(value, list) or len(value) != 2:
                return ""
            source_id = str(value[0])
            seen = set() if seen is None else set(seen)
            if source_id in seen or source_id not in prompt:
                raise HTTPException(400, "Invalid text input connection.")
            seen.add(source_id)
            source = prompt[source_id]
            if source.get("class_type") not in {"StringConcatenate", "DF_Text_Box", "DonutText", "DonutPromptConditioning"}:
                raise HTTPException(403, "Use the bundled text controls for prompt text.")
            inputs = source.get("inputs", {})
            if source["class_type"] == "DonutPromptConditioning":
                if type(value[1]) is not int or value[1] != 0:
                    raise HTTPException(403, "Use the main prompt text output for face instructions.")
                face, scene = (source_text(inputs.get(key, ""), seen) for key in ("face", "scene"))
                variants_json = inputs.get("prompt_sets_json", "[]")
                if not isinstance(variants_json, str):
                    raise HTTPException(403, "Use literal prompt variants in the studio.")
                try:
                    variants = json.loads(variants_json or "[]")
                except ValueError:
                    raise HTTPException(400, "Invalid prompt variants.") from None
                if not isinstance(variants, list) or any(not isinstance(row, dict) for row in variants):
                    raise HTTPException(400, "Invalid prompt variants.")
                if variants:
                    selected_index = inputs.get("prompt_set_index", 0)
                    if type(selected_index) is not int:
                        raise HTTPException(403, "Use a literal prompt variant index in the studio.")
                    position = max(0, selected_index - 1) % (len(variants) + 1)
                    if position:
                        row = variants[position - 1]
                        face, scene = row.get("face", face), row.get("scene", scene)
                        if not isinstance(face, str) or not isinstance(scene, str):
                            raise HTTPException(400, "Prompt variant fields must be text.")
                full = face + source_text(inputs.get("separator", ""), seen) + scene
                safe_prompt_text(full)
                return full
            if source["class_type"] == "DF_Text_Box":
                return source_text(inputs.get("Text", inputs.get("text", "")), seen)
            if source["class_type"] == "StringConcatenate":
                return (source_text(inputs.get("string_a", ""), seen)
                        + source_text(inputs.get("delimiter", ""), seen)
                        + source_text(inputs.get("string_b", ""), seen))
            separator = source_text(inputs.get("separator", ""), seen)
            return separator.join(source_text(inputs[key], seen) for key in ("prefix", "text", "suffix") if key in inputs)

        for node in prompt.values():
            if not isinstance(node, dict) or node.get("class_type") not in allowed:
                raise HTTPException(403, "This studio supports the bundled image workflow nodes.")
            inputs = node.get("inputs", {})
            if not isinstance(inputs, dict):
                raise HTTPException(400, "Invalid workflow node inputs.")
            if node["class_type"] == "DonutText":
                separator = source_text(inputs.get("separator", ""))
                combined = separator.join(source_text(inputs[key]) for key in ("prefix", "text", "suffix") if key in inputs)
                safe_prompt_text(combined)
            if node["class_type"] == "DonutEditStudio":
                safe_prompt_text(source_text(inputs.get("prompt")))
            if node["class_type"] == "DonutPromptConditioning":
                for key in ("face", "scene", "negative", "edit_negative"):
                    safe_prompt_text(source_text(inputs.get(key)))
                if "prompt_sets_json" in inputs and not isinstance(inputs["prompt_sets_json"], str):
                    raise HTTPException(403, "Use literal prompt variants in the studio.")
                if "prompt_set_index" in inputs and type(inputs["prompt_set_index"]) is not int:
                    raise HTTPException(403, "Use a literal prompt variant index in the studio.")
                if isinstance(inputs.get("prompt_sets_json"), str):
                    try:
                        variants = json.loads(inputs["prompt_sets_json"])
                    except ValueError:
                        raise HTTPException(400, "Invalid prompt variants.") from None
                    if not isinstance(variants, list) or any(not isinstance(row, dict) for row in variants):
                        raise HTTPException(400, "Invalid prompt variants.")
                    for row in variants:
                        for value in row.values():
                            if isinstance(value, str):
                                safe_prompt_text(value)
            for key, value in inputs.items():
                path_input = any(word in key.lower() for word in ("path", "filename", "model_name", "lora_name", "vae_name", "ckpt_name", "unet_name", "mask_b_model"))
                # DonutSeedPlan consumes an INT here, not a filesystem location.
                # v5 links it to the same SeedNode used by text and sampling.
                if node["class_type"] == "DonutSeedPlan" and key == "filename_seed":
                    path_input = False
                if key == "filename_prefix" and node["class_type"] in {"DonutImageSave", "SaveImage"} and isinstance(value, list):
                    # v5 derives its filename from a model-name output. A linked
                    # name cannot be checked until execution, so give the save
                    # node a literal studio-owned name before it is queued.
                    inputs[key] = value = "image"
                if path_input and isinstance(value, list):
                    raise HTTPException(403, "Use literal model and image names in the studio workflow.")
                if isinstance(value, str) and path_input:
                    if value and not safe_relative(value):
                        raise HTTPException(403, "Workflow paths must stay inside the backend model and image folders.")
                if key in {"reference_a", "reference_b", "image_a", "image_b"} and isinstance(value, str) and value:
                    if value not in session["references"]:
                        raise HTTPException(403, "Upload references through this studio.")
                if key in {"reference_a", "reference_b", "image_a", "image_b", "filename_prefix"} and isinstance(value, list):
                    raise HTTPException(403, "Use studio controls for image references and output names.")
                if key == "mask_b_data" and isinstance(value, str) and value:
                    try:
                        mask = json.loads(value).get("mask")
                    except (ValueError, AttributeError):
                        raise HTTPException(400, "Invalid saved studio mask.") from None
                    if mask not in session["masks"]:
                        raise HTTPException(403, "The saved mask is outside this studio.")
                if key in {"auto_download", "download_missing"}:
                    inputs[key] = False
                if key == "slots_json" and isinstance(value, str):
                    try:
                        slots = json.loads(value)
                    except ValueError:
                        raise HTTPException(400, "Invalid LoRA slot controls.") from None
                    if not isinstance(slots, list) or any(not isinstance(slot, dict) or not safe_relative(slot.get("lora_name", "None")) for slot in slots):
                        raise HTTPException(403, "LoRA slots must use backend model names.")
                    for slot in slots:
                        # The upstream loader auto-downloads only when a saved
                        # hash is present. Managed setup owns all installation.
                        slot["lora_hash"] = ""
                    inputs[key] = json.dumps(slots, separators=(",", ":"))
            if node["class_type"] == "LoadImage" and inputs.get("image") not in session["uploads"]:
                raise HTTPException(403, "Upload input images through this studio.")
            if node["class_type"] in {"SaveImage", "DonutImageSave"}:
                prefix = inputs.get("filename_prefix", "ComfyUI")
                if not isinstance(prefix, str) or not safe_relative(prefix, empty=True):
                    raise HTTPException(403, "Invalid image filename prefix.")
                inputs["filename_prefix"] = f"donut-create/{session['id'][:16]}/{prefix or 'image'}"
                if node["class_type"] == "DonutImageSave":
                    inputs["overwrite_mode"] = False

    async def queue_prompt(self, session: dict[str, Any], body: dict[str, Any]) -> httpx.Response:
        if len(session["jobs"]) >= 256:
            raise HTTPException(409, "This studio has reached its prompt limit. Open a new studio.")
        self.validate_prompt(session, body)
        extra = body.get("extra_data")
        metadata = extra.get("extra_pnginfo") if isinstance(extra, dict) else None
        workflow = metadata.get("workflow") if isinstance(metadata, dict) else None
        if workflow is not None and (not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list)
                                     or len(json.dumps(workflow).encode()) > MAX_PROMPT):
            raise HTTPException(400, "Provide a valid workflow with the generation request.")
        workflow_extra = workflow.get("extra") if isinstance(workflow, dict) else None
        destination = workflow_extra.get("dmc_output_destination", "preview") if isinstance(workflow_extra, dict) else "preview"
        if not isinstance(destination, str) or destination not in {"preview", "comfy"}:
            raise HTTPException(400, "Choose a valid output destination.")
        save_nodes = {}
        for node_id, node in body["prompt"].items():
            if node["class_type"] not in {"SaveImage", "DonutImageSave"}:
                continue
            save_nodes[str(node_id)] = self.final_output({"prompt": body["prompt"], "workflow": workflow}, str(node_id))
            if node["class_type"] == "DonutImageSave":
                node["inputs"]["root"] = "output" if destination == "comfy" else "temp"
                node["inputs"]["show_previews"] = True
            if destination != "comfy":
                if node["class_type"] == "SaveImage":
                    node["class_type"] = "PreviewImage"
                    node["inputs"] = {"images": node["inputs"]["images"]} if "images" in node["inputs"] else {}
        body["client_id"] = "donut-create-" + session["id"]
        body["prompt_id"] = str(uuid.uuid4())
        response = await self.request_backend(session["backend_url"], "POST", "prompt", json=body)
        if response.is_success:
            try:
                prompt_id = response.json()["prompt_id"]
            except (ValueError, KeyError, TypeError):
                raise HTTPException(502, "ComfyUI accepted a prompt without returning its ID.") from None
            if not isinstance(prompt_id, str) or not PROMPT_ID.fullmatch(prompt_id):
                raise HTTPException(502, "ComfyUI returned an invalid prompt ID.")
            if any(prompt_id in entry["jobs"] for entry in self.sessions.values()):
                raise HTTPException(502, "ComfyUI returned a prompt ID already owned by a studio.")
            session["jobs"][prompt_id] = {"id": prompt_id, "status": "queued", "created_at": self.clock(),
                "prompt": body["prompt"], "workflow": workflow, "save_nodes": save_nodes}
            if workflow is not None:
                session["latest_run_revision"] = session.get("latest_run_revision", 0) + 1
                session["latest_run_workflow"] = workflow
                if session.get("workspace"):
                    key = session["backend_key"]
                    if self.workspaces.get(key, {}).get("session_id") == session["id"]:
                        self.workspaces[key] = {"session_id": session["id"], "revision": session["latest_run_revision"],
                                                "workflow": workflow, "uploads": list(session["uploads"]),
                                                "references": list(session["references"]),
                                                "masks": list(session["masks"])}
            self.persist()
            if workflow is not None:
                # Acknowledge this exact accepted run, before another device can submit.
                response = httpx.Response(response.status_code, json={**response.json(),
                    "dmc_workflow_revision": session["latest_run_revision"]})
        return response

    async def cancel(self, session: dict[str, Any], requested: set[str] | None = None) -> dict[str, Any]:
        async with self.mutation_lock:
            queue = await self.reconcile(session)
            owned = set(session["jobs"]) if requested is None else set(session["jobs"]) & requested
            pending = queue_ids(queue, "queue_pending") & owned
            running = queue_ids(queue, "queue_running") & owned
            cancelled, errors = [], []
            if pending:
                response = await self.request_backend(session["backend_url"], "POST", "queue", json={"delete": sorted(pending)})
                if not response.is_success:
                    raise HTTPException(502, "ComfyUI could not remove this studio's queued prompts.")
                cancelled.extend(sorted(pending))
            for prompt_id in sorted(running):
                # ComfyUI's jobs API performs the ownership check atomically.
                response = await self.request_backend(session["backend_url"], "POST", f"api/jobs/{prompt_id}/cancel", json={})
                if response.status_code == 404:
                    errors.append("This backend lacks prompt-specific interruption. Its running prompt was left unchanged.")
                elif not response.is_success:
                    errors.append("ComfyUI could not cancel the running studio prompt.")
                elif response.json().get("cancelled"):
                    cancelled.append(prompt_id)
            for prompt_id in cancelled:
                session["jobs"][prompt_id]["status"] = "cancelled"
            self.persist()
            return {"cancelled": cancelled, "errors": errors, "session": self.public_session(session)}

    async def output_response(self, session: dict[str, Any], oid: str) -> StreamingResponse:
        await self.reconcile(session)
        output = session["outputs"].get(oid)
        if output is None or session["jobs"].get(output["prompt_id"], {}).get("status") != "completed":
            raise HTTPException(404, "Completed studio output not found.")
        request = self.client.build_request("GET", session["backend_url"] + "/view", params={
            "filename": output["filename"], "subfolder": output["subfolder"], "type": output.get("type", "output")})
        try:
            response = await self.client.send(request, stream=True)
        except httpx.HTTPError:
            raise HTTPException(502, "ComfyUI output is unavailable.") from None
        if not response.is_success:
            await response.aclose()
            raise HTTPException(502, "ComfyUI no longer serves this completed output.")
        return StreamingResponse(response.aiter_bytes(), media_type=output["media_type"],
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
            background=BackgroundTask(response.aclose))

    def observe_event(self, session: dict[str, Any], event: dict[str, Any]) -> bool:
        data = event.get("data", {})
        if not isinstance(data, dict):
            return False
        prompt_id = data.get("prompt_id")
        if prompt_id not in session["jobs"]:
            return False
        job = session["jobs"][prompt_id]
        kind = event.get("type")
        if kind in {"execution_error", "execution_interrupted"}:
            job["status"] = "cancelled" if kind == "execution_interrupted" else "error"
            job["error"] = str(data.get("exception_message") or "Generation interrupted.")[:2000]
        elif kind == "execution_start" or (kind == "executing" and data.get("node") is not None):
            job["status"] = "running"
            job["last_seen_at"] = self.clock()
        elif kind == "executed" and isinstance(data.get("output"), dict):
            self.record_masks(session, data["output"])
            self.record_previews(session, prompt_id, str(data.get("node", "")), data["output"])
        self.persist()
        return True


class ConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    backend_url: str | None = None
    comfy_directory: str | None = None
    python_executable: str | None = None


class SetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: str
    profile: str
    hf_token: str | None = None
    civitai_api_key: str | None = None
    model_root: str | None = None


async def call_installer(installer: Any, method: str, *args: Any) -> Any:
    result = getattr(installer, method)(*args)
    return await result if inspect.isawaitable(result) else result


def studio_prefix(request: Request, sid: str) -> str:
    value = request.headers.get("x-dmc-studio-prefix", f"/api/create/studio/{sid}/")
    if (not value.startswith("/") or value.startswith("//") or not value.endswith("/")
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+/", value)
            or ".." in value.split("/") or f"/studio/{sid}/" not in value):
        raise HTTPException(400, "Invalid studio prefix.")
    return value


def rewrite_asset(text: str, prefix: str, media_type: str) -> str:
    # Rewrite resource locations, never arbitrary JS strings or regex literals.
    # Fetch/XHR/WebSocket URLs are scoped separately by the bootstrap adapter.
    location = r"(?P<quote>[\"'`])/(?P<path>(?!/)[^\"'`\s<>]*)(?P=quote)"
    if "javascript" in media_type:
        pattern = r"(?P<head>\bfrom\s*|\bimport\s*(?:\(\s*)?|\bnew\s+URL\s*\(\s*)" + location
    elif "text/html" in media_type:
        pattern = r"(?P<head>\b(?:src|href)\s*=\s*)" + location
    else:
        pattern = None
    if pattern:
        text = re.sub(pattern, lambda match: match["head"] + match["quote"]
                      + ("/" + match["path"] if match["path"].startswith(prefix.lstrip("/")) else prefix + match["path"])
                      + match["quote"], text)
    if "text/css" in media_type or "text/html" in media_type:
        text = re.sub(r"url\(\s*(?P<quote>[\"']?)/(?P<path>(?!/)[^)\s\"']+)(?P=quote)\s*\)",
                      lambda match: "url(" + match["quote"] + prefix + match["path"] + match["quote"] + ")", text)
    return text


def bootstrap(prefix: str, sid: str, backend_key: str, profile: str) -> str:
    """Adapt ComfyUI URLs before its modules execute; load only browser DonutUI."""
    return """<script>(function(){'use strict';
const prefix=PREFIX, sid=SID;
window.__DMC_CREATE__={prefix,sessionId:sid,backendKey:BACKEND_KEY,profile:PROFILE};
function scoped(raw){const url=new URL(String(raw),location.href);
 if(url.origin===location.origin && !url.pathname.startsWith(prefix)){
  let path=url.pathname;
  while(path.startsWith('/api/') && !path.startsWith(prefix))path=path.slice(4);
  url.pathname=path.startsWith(prefix)?path:prefix+path.replace(/^\\//,''); }
 return url.href;}
const originalFetch=window.fetch.bind(window);
window.fetch=function(input,options){
 if(input instanceof Request){const url=scoped(input.url);return originalFetch(new Request(url,input),options);}
 return originalFetch(scoped(input),options);};
const NativeSocket=window.WebSocket;
window.WebSocket=class extends NativeSocket{constructor(url,protocols){
 const endpoint=new URL(String(url),location.href); if(endpoint.host===location.host){
  endpoint.pathname=prefix+'ws'; endpoint.searchParams.set('clientId','donut-create-'+sid);}
 super(endpoint.href,protocols);}};
const originalOpen=XMLHttpRequest.prototype.open;
XMLHttpRequest.prototype.open=function(method,url,...args){return originalOpen.call(this,method,scoped(url),...args);};
})();</script><link rel="stylesheet" href="PREFIX_ESCdonut-create.css"><script src="PREFIX_ESCdonut-create.js" defer></script>""".replace("PREFIX_ESC", prefix).replace("PREFIX", json.dumps(prefix)).replace("SID", json.dumps(sid)).replace("BACKEND_KEY", json.dumps(backend_key)).replace("PROFILE", json.dumps(profile))


def allowed_route(method: str, route: str) -> bool:
    if method in {"GET", "HEAD"}:
        if route in {"", "index.html", "favicon.ico", "materialdesignicons.min.css", "user.css", "object_info", "embeddings", "models", "experiment/models", "extensions", "features", "system_stats", "prompt", "queue", "history", "view", "settings", "users", "userdata", "donut/config", "api/jobs", "dmc/workflow"}:
            return True
        if re.fullmatch(r"(?:object_info|models|history)/[A-Za-z0-9_.-]+", route):
            return True
        if re.fullmatch(r"api/jobs/[A-Za-z0-9_-]+", route):
            return True
        if route in {"donut/models/status", "donut/loras/list", "donut/loras/info", "donut/loras/analyze", "donut/loras/preview", "donut/lora/get_hash", "donut/styles/by_category", "donut/styles/hierarchy", "donut/locations/hierarchy", "donut/wildcards", "donut/edit-studio/subject-mask-model"}:
            return True
        if re.fullmatch(r"donut/edit-studio/(?:reference|subject-mask)/[a-f0-9]{64}", route):
            return True
        root = route.split("/", 1)[0]
        if root in STATIC_ROOTS and PurePosixPath(route).suffix.lower() in STATIC_SUFFIXES:
            return True
        parts = route.split("/")
        if root == "extensions" and len(parts) >= 3 and parts[1] in EXTENSION_ROOTS and PurePosixPath(route).suffix.lower() in STATIC_SUFFIXES:
            return True
        if route.startswith("userdata/") or route.startswith("settings/"):
            return True
    if method == "POST" and route in {"prompt", "queue", "interrupt", "upload/image", "settings", "donut/wildcards/preview", "donut/edit-studio/reference", "donut/edit-studio/subject-mask"}:
        return True
    if method == "POST" and (route == "api/jobs/cancel" or re.fullmatch(r"api/jobs/[A-Za-z0-9_-]+/cancel", route)):
        return True
    return method in {"POST", "PUT", "DELETE"} and (route.startswith("userdata/") or route.startswith("settings/"))


def query_params(route: str, request: Request) -> dict[str, str]:
    allowed = {
        "view": {"filename", "subfolder", "type", "preview", "channel", "rand"},
        "donut/loras/list": {"include_meta"}, "donut/loras/info": {"name"},
        "donut/loras/analyze": {"name"}, "donut/lora/get_hash": {"name"},
        "donut/loras/preview": {"hash", "type", "t"}, "donut/styles/by_category": {"category"},
        "donut/edit-studio/subject-mask-model": {"name"},
        "userdata": {"dir", "recurse", "full_info", "split"}, "history": {"max_items", "offset"},
    }.get(route, set())
    return {key: value for key, value in request.query_params.items() if key in allowed}


async def studio_local(controller: Controller, session: dict[str, Any], route: str, request: Request) -> Response | None:
    if route == "dmc/workflow" and request.method in {"GET", "HEAD"}:
        if set(request.query_params) - {"revision_only"}:
            raise HTTPException(400, "Invalid workflow query.")
        result = {"revision": session.get("latest_run_revision", 0)}
        if request.query_params.get("revision_only") != "1":
            result["workflow"] = session.get("latest_run_workflow")
        return JSONResponse(result)
    if route == "experiment/models" and request.method in {"GET", "HEAD"}:
        # The optional model-manager API exposes host folders. ComfyUI supports
        # a 404 fallback; workflow model selectors use the scoped /models API.
        return JSONResponse([], status_code=404)
    if route == "user.css" and request.method in {"GET", "HEAD"}:
        return Response(session["userdata"].get("user.css", ""), media_type="text/css")
    if route == "users":
        return JSONResponse({"storage": "server", "multi_user": False, "migrated": False})
    if route == "donut/config":
        return JSONResponse({"civitai": {"api_key": "", "auto_lookup": False, "download_previews": False, "prefer_sfw": True}})
    if route == "userdata":
        return JSONResponse(list(session["userdata"]))
    if route == "settings" or route.startswith("settings/"):
        key = route.removeprefix("settings/")
        if request.method in {"GET", "HEAD"}:
            return JSONResponse(session["settings"] if route == "settings" else session["settings"].get(key))
        if request.method == "DELETE":
            session["settings"].pop(key, None)
            controller.persist()
            return JSONResponse({})
        try:
            data = json.loads(await limited_body(request, MAX_PROMPT))
        except ValueError:
            raise HTTPException(400, "Invalid studio settings JSON.") from None
        if route == "settings" and isinstance(data, dict):
            session["settings"].update(data)
        else:
            session["settings"][key] = data
        controller.persist()
        return JSONResponse({})
    if route.startswith("userdata/"):
        key = route[len("userdata/"):]
        if not safe_relative(key):
            raise HTTPException(403, "Invalid studio document name.")
        if request.method in {"GET", "HEAD"}:
            if key not in session["userdata"]:
                raise HTTPException(404, "Studio document not found.")
            return Response(session["userdata"][key], media_type="application/json")
        if request.method == "DELETE":
            session["userdata"].pop(key, None)
        else:
            body = await limited_body(request, MAX_PROMPT)
            if len(body) > MAX_PROMPT or len(session["userdata"]) >= 128:
                raise HTTPException(413, "Studio document storage limit reached.")
            try:
                session["userdata"][key] = body.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(400, "Studio documents must contain UTF-8 text.") from None
        controller.persist()
        return JSONResponse({})
    return None


def create_app(controller: Controller | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.controller is None:
            application.state.controller = Controller(state_directory())
        try:
            yield
        finally:
            await application.state.controller.close()

    application = FastAPI(title="Donut Create", lifespan=lifespan)
    application.state.controller = controller

    def current() -> Controller:
        return application.state.controller

    @application.get("/health")
    async def health():
        return {"status": "ok", "addon": "donut-create"}

    @application.get("/create/status")
    async def status():
        return await current().status()

    @application.post("/create/config")
    async def configure(body: ConfigRequest):
        if body.mode not in {"managed", "existing", "local"}:
            raise HTTPException(400, "Choose managed, local installation, or URL connection mode.")
        return await current().configure(body.model_dump())

    @application.post("/create/setup")
    async def setup(body: SetupRequest):
        if current().config["mode"] != "managed":
            raise HTTPException(409, "Switch to managed mode to install the backend.")
        if body.runtime not in {"cuda", "cpu", "mps"} or body.profile not in {"base", "workflow", "all"}:
            raise HTTPException(400, "Choose a supported runtime and model profile.")
        return await call_installer(current().installer, "start", body.model_dump(exclude_none=True))

    @application.post("/create/cancel-setup")
    async def cancel_setup():
        return await call_installer(current().installer, "cancel")

    @application.post("/create/backend/start")
    async def start_backend():
        return await current().start_backend()

    @application.post("/create/backend/stop")
    async def stop_backend():
        return await current().stop_backend()

    @application.post("/create/backend/restart")
    async def restart_backend():
        return await current().restart_backend()

    @application.post("/create/sessions")
    async def create_session(request: Request):
        raw = await limited_body(request, 1024)
        try:
            options = json.loads(raw) if raw else {}
        except ValueError:
            raise HTTPException(400, "Invalid studio options.") from None
        if not isinstance(options, dict) or set(options) - {"workspace"} or not isinstance(options.get("workspace", False), bool):
            raise HTTPException(400, "Choose a valid studio workspace.")
        return await current().create_session(workspace=options.get("workspace", False))

    @application.get("/create/sessions/{sid}")
    async def get_session(sid: str):
        session = current().session(sid)
        await current().reconcile(session)
        if session.get("workspace") and session["expires_at"] - current().clock() < current().ttl / 2:
            session["expires_at"] = current().clock() + current().ttl
            current().persist()
        return current().public_session(session)

    @application.post("/create/sessions/{sid}/cancel")
    async def cancel_session(sid: str):
        return await current().cancel(current().session(sid))

    @application.get("/create/sessions/{sid}/outputs/{oid}")
    async def get_output(sid: str, oid: str):
        return await current().output_response(current().session(sid), oid)

    @application.get("/create/sessions/{sid}/outputs/{oid}/metadata")
    async def get_output_metadata(sid: str, oid: str):
        ctrl = current()
        session = ctrl.session(sid)
        await ctrl.reconcile(session)
        output = session["outputs"].get(oid)
        if output is None or session["jobs"].get(output["prompt_id"], {}).get("status") != "completed":
            raise HTTPException(404, "Completed studio output not found.")
        return output.get("metadata", {})

    @application.get("/create/sessions/{sid}/outputs/{oid}/provenance")
    async def get_output_provenance(sid: str, oid: str):
        ctrl = current()
        session = ctrl.session(sid)
        await ctrl.reconcile(session)
        output = session["outputs"].get(oid)
        if output is None or session["jobs"].get(output["prompt_id"], {}).get("status") != "completed":
            raise HTTPException(404, "Completed studio output not found.")
        return {**output.get("metadata", {}), "workflow": output.get("workflow"),
                "execution_prompt": output.get("prompt"), "execution": output.get("execution")}

    @application.api_route("/studio/{sid}/", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    @application.api_route("/studio/{sid}/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def studio(sid: str, request: Request, rest: str = ""):
        ctrl = current()
        session = ctrl.session(sid)
        route = clean_route(rest)
        prefix = studio_prefix(request, sid)
        # ComfyUI's fileURL may prepend its base to an already scoped extension
        # URL before the browser adapter has initialized. Accept only duplicate
        # prefixes for this exact capability, never another studio's link.
        markers = {prefix.lstrip("/"), f"api/create/studio/{sid}/"}
        while any(route.startswith(marker) for marker in markers):
            route = next(route[len(marker):] for marker in markers if route.startswith(marker))
        # Current ComfyUI's API helper adds /api to older route names.
        if route.startswith("api/") and not route.startswith("api/jobs"):
            route = route[4:]
        if route in {"workflow.json", "donut-create.js", "donut-create.css"} and request.method in {"GET", "HEAD"}:
            path = Path(ctrl.installer.workflow_path) if route == "workflow.json" else Path(ctrl.installer.assets_root) / route
            if not path.is_file():
                raise HTTPException(503, "The bundled studio assets are unavailable. Reinstall Donut Create.")
            return Response(path.read_bytes(), media_type={"workflow.json": "application/json", "donut-create.js": "text/javascript", "donut-create.css": "text/css"}[route],
                            headers={"Cache-Control": "private, no-store"})
        if not allowed_route(request.method, route):
            raise HTTPException(403, "This backend route is outside the creation studio.")
        local = await studio_local(ctrl, session, route, request)
        if local is not None:
            return local
        body = await limited_body(request, MAX_PROMPT if route == "prompt" else MAX_BODY)
        if request.method == "POST" and route == "prompt":
            try:
                data = json.loads(body)
            except ValueError:
                raise HTTPException(400, "Invalid workflow JSON.") from None
            if not isinstance(data, dict):
                raise HTTPException(400, "Invalid workflow JSON.")
            response = await ctrl.queue_prompt(session, data)
        elif request.method == "POST" and route in {"queue", "interrupt"}:
            requested = None
            if route == "queue":
                try:
                    data = json.loads(body)
                except ValueError:
                    raise HTTPException(400, "Invalid queue request.") from None
                if not isinstance(data, dict) or data.get("clear") or not isinstance(data.get("delete"), list):
                    raise HTTPException(403, "Only this studio's queued prompt IDs may be removed.")
                if not all(isinstance(value, str) and value in session["jobs"] for value in data["delete"]):
                    raise HTTPException(403, "The prompt is outside this studio.")
                requested = set(data["delete"])
            return JSONResponse(await ctrl.cancel(session, requested))
        elif request.method == "POST" and (route == "api/jobs/cancel" or route.endswith("/cancel") and route.startswith("api/jobs/")):
            if route == "api/jobs/cancel":
                try:
                    data = json.loads(body)
                except ValueError:
                    raise HTTPException(400, "Invalid job cancellation request.") from None
                values = data.get("job_ids") if isinstance(data, dict) else None
            else:
                values = [route.split("/")[2]]
            if not isinstance(values, list) or not all(isinstance(value, str) and value in session["jobs"] for value in values):
                raise HTTPException(403, "The prompt is outside this studio.")
            result = await ctrl.cancel(session, set(values))
            return JSONResponse({"cancelled": bool(result["cancelled"]), "errors": result["errors"]})
        elif route == "api/jobs" or route.startswith("api/jobs/"):
            requested_id = route.split("/")[2] if route != "api/jobs" else None
            if requested_id is not None and requested_id not in session["jobs"]:
                raise HTTPException(404, "Studio prompt not found.")
            jobs = []
            for prompt_id in ([requested_id] if requested_id else session["jobs"]):
                response = await ctrl.request_backend(session["backend_url"], "GET", "api/jobs/" + prompt_id)
                if response.is_success:
                    jobs.append(response.json())
            if requested_id:
                if not jobs:
                    raise HTTPException(404, "Studio prompt not found.")
                return JSONResponse(jobs[0])
            return JSONResponse({"jobs": jobs, "pagination": {"offset": 0, "limit": len(jobs), "total": len(jobs), "has_more": False}})
        elif route == "queue":
            queue = await ctrl.reconcile(session)
            return JSONResponse(ctrl.owned_queue(session, queue))
        elif route == "prompt":
            queue = await ctrl.reconcile(session)
            owned = ctrl.owned_queue(session, queue)
            return JSONResponse({"exec_info": {"queue_remaining": sum(len(rows) for rows in owned.values())}})
        elif route == "history" or route.startswith("history/"):
            requested_id = route.split("/", 1)[1] if "/" in route else None
            if requested_id is not None and requested_id not in session["jobs"]:
                raise HTTPException(404, "Studio prompt not found.")
            records = {}
            for prompt_id in ([requested_id] if requested_id else session["jobs"]):
                history = await ctrl.backend_json(session["backend_url"], "history/" + prompt_id)
                if isinstance(history, dict) and prompt_id in history:
                    records[prompt_id] = history[prompt_id]
                    ctrl.record_history(session, prompt_id, history[prompt_id])
            ctrl.persist()
            return JSONResponse(records)
        else:
            params = query_params(route, request)
            if route == "donut/loras/preview" and (not re.fullmatch(r"[a-fA-F0-9]{10,64}", params.get("hash", "")) or params.get("type", "collage") not in {"collage", "0", "1", "2", "3"}):
                raise HTTPException(403, "Invalid model preview identifier.")
            if route == "donut/wildcards/preview":
                try:
                    data = json.loads(body)
                except ValueError:
                    raise HTTPException(400, "Invalid prompt preview.") from None
                if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                    raise HTTPException(400, "Provide prompt text for the preview.")
                safe_prompt_text(data["text"])
            if route == "view":
                await ctrl.reconcile(session)
                output = next((entry for entry in session["outputs"].values()
                               if entry["filename"] == params.get("filename")
                               and entry["subfolder"] == params.get("subfolder", "")
                               and params.get("type", "output") == entry.get("type", "output")), None)
                if output is None:
                    preview = any(entry["filename"] == params.get("filename")
                                  and entry["subfolder"] == params.get("subfolder", "")
                                  and params.get("type") == "temp"
                                  for entry in session.get("previews", {}).values())
                    uploaded = (params.get("type") == "input" and params.get("filename") in session["uploads"]
                                and not params.get("subfolder"))
                    if not preview and not uploaded:
                        raise HTTPException(404, "This image is outside the studio.")
                else:
                    return await ctrl.output_response(session, output["id"])
            if route.startswith("donut/edit-studio/reference/"):
                if "donutref:" + route.rsplit("/", 1)[1] not in session["references"]:
                    raise HTTPException(404, "Studio reference not found.")
            if route.startswith("donut/edit-studio/subject-mask/"):
                if "donutmask:" + route.rsplit("/", 1)[1] not in session["masks"]:
                    raise HTTPException(404, "Studio mask not found.")
            for key in {"name", "filename", "subfolder"} & params.keys():
                if not safe_relative(params[key], empty=key == "subfolder"):
                    raise HTTPException(403, "Invalid model or image name.")
            headers = {key: value for key, value in request.headers.items()
                       if key.lower() in {"content-type", "accept"}}
            if request.method == "POST" and route in {"upload/image", "donut/edit-studio/reference", "donut/edit-studio/subject-mask"}:
                fields, files = multipart_fields(request.headers.get("content-type", ""), body)
                if route == "upload/image":
                    if set(files) != {"image"}:
                        raise HTTPException(400, "Choose an input image.")
                    original, content, content_type = files["image"]
                    suffix = PurePosixPath(original).suffix.lower()
                    if suffix not in IMAGE_SUFFIXES:
                        raise HTTPException(400, "Choose a supported image format.")
                    files = {"image": (f"donut-create-{sid[:16]}-{secrets.token_hex(8)}{suffix}", content, content_type)}
                    fields = {"type": "input", "subfolder": "", "overwrite": "false"}
                elif route == "donut/edit-studio/reference":
                    if set(files) != {"image"}:
                        raise HTTPException(400, "Choose a reference image.")
                    fields = {}
                else:
                    if set(files) != {"mask"} or fields.get("reference") not in session["references"]:
                        raise HTTPException(403, "Upload a mask for a reference in this studio.")
                    fields = {"reference": fields["reference"]}
                response = await ctrl.request_backend(session["backend_url"], "POST", route, data=fields, files=files)
            else:
                response = await ctrl.request_backend(session["backend_url"], request.method, route,
                                                      content=body, params=params, headers=headers)
        media_type = response.headers.get("content-type", "application/octet-stream")
        content = response.content
        if response.is_success and "application/json" in media_type:
            try:
                data = response.json()
            except ValueError:
                raise HTTPException(502, "ComfyUI returned invalid JSON.") from None
            if route == "extensions":
                data = [path for path in data if isinstance(path, str) and allowed_route("GET", clean_route(path.lstrip("/")))]
                data = [prefix + path.lstrip("/") for path in data]
            elif route == "object_info":
                data = {key: value for key, value in data.items() if key in ctrl.discoverable_nodes()}
            elif route.startswith("object_info/") and route.split("/", 1)[1] not in ctrl.discoverable_nodes():
                raise HTTPException(403, "This node is outside the studio workflow.")
            elif route == "upload/image":
                name = data.get("name")
                if not safe_relative(name) or data.get("type") != "input" or data.get("subfolder"):
                    raise HTTPException(502, "ComfyUI returned an invalid studio upload.")
                session["uploads"].append(name)
                ctrl.persist()
            elif route == "donut/edit-studio/reference" and isinstance(data.get("reference"), str):
                session["references"].append(data["reference"])
                ctrl.persist()
            elif route == "donut/edit-studio/subject-mask" and isinstance(data.get("mask"), str):
                session["masks"].append(data["mask"])
                ctrl.persist()
            content = json.dumps(scrub_paths(data)).encode()
        elif any(kind in media_type for kind in ("text/html", "javascript", "text/css")):
            text = rewrite_asset(content.decode("utf-8"), prefix, media_type)
            # New ComfyUI frontends expose compatibility modules through a
            # global namespace populated by their asynchronous main bundle.
            # Do not evaluate those shims before that namespace is ready.
            if "javascript" in media_type and route in {"scripts/app.js", "scripts/api.js", "scripts/defaultGraph.js"}:
                namespaces = re.findall(r"window\.comfyAPI\.([A-Za-z_][A-Za-z0-9_]*)", text)
                if namespaces:
                    text = "await new Promise((resolve,reject)=>{const started=Date.now();const check=()=>{" \
                        + "if(" + "&&".join("window.comfyAPI?." + name for name in sorted(set(namespaces))) \
                        + "){resolve();return;}if(Date.now()-started>60000){reject(new Error('ComfyUI did not initialize its browser API.'));return;}setTimeout(check,100);};check();});\n" + text
            if "text/html" in media_type and route in {"", "index.html"}:
                text = re.sub(r"<base\b[^>]*>", "", text, flags=re.IGNORECASE)
                injection = f'<base href="{prefix}">' + bootstrap(prefix, sid, session.get("backend_key") or ctrl.backend_key(session["backend_url"], "existing"), session.get("profile", "base"))
                if re.search(r"<head\b[^>]*>", text, flags=re.IGNORECASE):
                    text = re.sub(r"(<head\b[^>]*>)", lambda match: match[1] + injection, text, count=1, flags=re.IGNORECASE)
                else:
                    text = injection + text
            content = text.encode()
        return Response(content if request.method != "HEAD" else b"", status_code=response.status_code,
                        media_type=media_type, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

    @application.websocket("/studio/{sid}/ws")
    async def studio_socket(socket: WebSocket, sid: str):
        ctrl = current()
        try:
            session = ctrl.session(sid)
        except HTTPException:
            await socket.close(code=1008, reason="Studio session is invalid or expired.")
            return
        if session.get("workspace"):
            await socket.accept()
            stream, subscriber = ctrl.workspace_events(session)

            async def deliver_workspace_events():
                while True:
                    message = await subscriber.get()
                    if message is None:
                        await socket.close(code=1011, reason="ComfyUI connection unavailable; reconnect to restore job state.")
                        return
                    if isinstance(message, bytes):
                        await socket.send_bytes(message)
                    else:
                        await socket.send_text(message)

            async def watch_workspace_device():
                while True:
                    remaining = max(0, session["expires_at"] - ctrl.clock())
                    try:
                        event = await asyncio.wait_for(socket.receive(), timeout=min(remaining, 30))
                        if event["type"] == "websocket.disconnect":
                            return
                    except asyncio.TimeoutError:
                        if ctrl.clock() >= session["expires_at"]:
                            await socket.close(code=1008, reason="Studio session expired.")
                            return

            tasks = [asyncio.create_task(deliver_workspace_events()), asyncio.create_task(watch_workspace_device())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                stream["subscribers"].discard(subscriber)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            return
        parts = urlsplit(session["backend_url"])
        backend = urlunsplit(("wss" if parts.scheme == "https" else "ws", parts.netloc, "/ws",
                              urlencode({"clientId": "donut-create-" + sid}), ""))
        await socket.accept()
        try:
            async with websockets.connect(backend, proxy=None, max_size=MAX_BODY, open_timeout=5) as upstream:
                active: str | None = None
                queue = await ctrl.reconcile(session)
                own_running = queue_ids(queue, "queue_running") & set(session["jobs"])
                if own_running:
                    active = next(iter(own_running))

                async def expire_or_disconnect():
                    while True:
                        remaining = max(0, session["expires_at"] - ctrl.clock())
                        try:
                            event = await asyncio.wait_for(socket.receive(), timeout=min(remaining, 30))
                            if event["type"] == "websocket.disconnect":
                                return
                            # The ComfyUI socket carries server events; clients cannot send commands.
                        except asyncio.TimeoutError:
                            if ctrl.clock() >= session["expires_at"]:
                                await socket.close(code=1008, reason="Studio session expired.")
                                return

                async def relay():
                    nonlocal active
                    async for message in upstream:
                        ctrl.session(sid)
                        if isinstance(message, bytes):
                            if active in session["jobs"] and session["jobs"][active]["status"] == "running":
                                await socket.send_bytes(message)
                            continue
                        try:
                            event = json.loads(message)
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("type") == "status":
                            queue = await ctrl.reconcile(session)
                            own = ctrl.owned_queue(session, queue)
                            event = {"type": "status", "data": {"sid": "donut-create-" + sid,
                                     "status": {"exec_info": {"queue_remaining": sum(len(rows) for rows in own.values())}}}}
                            active = next(iter(queue_ids(queue, "queue_running") & set(session["jobs"])), None)
                        else:
                            if not ctrl.observe_event(session, event):
                                continue
                            if event.get("type") == "execution_start":
                                active = event["data"]["prompt_id"]
                            if event.get("type") in {"execution_error", "execution_interrupted", "execution_success"} or (event.get("type") == "executing" and event["data"].get("node") is None):
                                active = None
                        await socket.send_text(json.dumps(event))

                tasks = [asyncio.create_task(relay()), asyncio.create_task(expire_or_disconnect())]
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except HTTPException as error:
            with contextlib.suppress(RuntimeError):
                await socket.close(code=1008 if error.status_code in {404, 410} else 1011,
                                   reason="Studio session expired." if error.status_code == 410 else "ComfyUI connection unavailable.")
        except (OSError, httpx.HTTPError, websockets.WebSocketException):
            with contextlib.suppress(RuntimeError):
                await socket.close(code=1011, reason="ComfyUI connection unavailable; reconnect to restore job state.")
        except WebSocketDisconnect:
            pass

    return application


app = create_app()


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Donut Create controller")
    parser.add_argument("--port", type=int, default=18009)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
