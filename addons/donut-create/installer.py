"""Isolated, resumable Donut Create setup. Only the standard library is needed.

The sidecar's small environment runs this installer. ComfyUI and all of its
dependencies live in a separate backend environment under the add-on state root.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent


class SetupError(RuntimeError):
    pass


class SetupCancelled(SetupError):
    pass


class SafeRedirects(urllib.request.HTTPRedirectHandler):
    """Provider credentials never follow a redirect to another origin."""

    def redirect_request(self, request, response, code, message, headers, url):
        before, after = urllib.parse.urlsplit(request.full_url), urllib.parse.urlsplit(url)
        if after.scheme not in ("https", "http") or (before.scheme == "https" and after.scheme != "https"):
            raise SetupError("The download provider returned an unsafe redirect.")
        redirected = super().redirect_request(request, response, code, message, headers, url)
        if redirected and (before.scheme, before.netloc) != (after.scheme, after.netloc):
            redirected.remove_header("Authorization")
        return redirected


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(p in ("..", ".") for p in path.parts) or ":" in value or "\0" in value:
        raise SetupError("The model catalog contains an unsafe destination.")
    return Path(*path.parts)


def _fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class Installer:
    def __init__(self, state_dir):
        self.state_dir = Path(state_dir).resolve()
        self.assets_root = ROOT / "assets"
        self.manifest = _read_json(ROOT / "runtime.json")
        self.catalog = _read_json(ROOT / "model_sources.json")
        if not self.manifest or not self.catalog:
            raise SetupError("The Donut Create runtime manifest is missing or invalid.")
        self._manifest_hash = hashlib.sha256((ROOT / "runtime.json").read_bytes()).hexdigest()
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread = None
        self._process = None
        self._process_children = set()
        self._tokens = {}
        self._opener = urllib.request.build_opener(SafeRedirects())
        self._status = _read_json(self.state_dir / "setup.json", {})
        self._status.setdefault("state", "idle")
        self._status.setdefault("phase", "idle")
        self._status.setdefault("runtime", self.default_runtime())
        self._status.setdefault("profile", self.manifest["default_profile"])
        self._status.setdefault("progress", 0)
        self._status.setdefault("downloaded_bytes", 0)
        self._status.setdefault("total_bytes", 0)
        self._status.setdefault("error", None)
        if self._status["state"] in ("installing", "cancelling"):
            self._status.update(state="cancelled", phase="interrupted", error="Setup was interrupted. Retry to reuse verified files and resume downloads.")
        if self._status["state"] == "ready" and not self.ready():
            self._status.update(state="error", phase="verify", error="The managed runtime or a required model changed. Run setup again to verify it.")

    @property
    def backend_root(self):
        return self.state_dir / "backend" / "ComfyUI"

    @property
    def python_path(self):
        environment = self.state_dir / "backend" / "venv"
        return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    @property
    def workflow_path(self):
        installed = self.state_dir / "workflow.json"
        if installed.is_file():
            return installed
        return self.assets_root / "base-workflow.json" if self._status.get("profile") == "base" else ROOT / "workflow.json"

    def backend_environment(self):
        cache = self.state_dir / "backend" / "cache"
        return {"HF_HOME": str(cache / "huggingface"), "TORCH_HOME": str(cache / "torch"), "PYTHONNOUSERSITE": "1"}

    def supported_runtimes(self):
        system, machine = platform.system(), platform.machine()
        result = []
        for name, item in self.manifest["runtimes"].items():
            machines = item.get("platform_machines", {}).get(system, item["machines"])
            if system in item["systems"] and machine in machines:
                result.append(name)
        return result

    def default_runtime(self):
        supported = self.supported_runtimes()
        if platform.system() == "Darwin" and "mps" in supported:
            return "mps"
        return "cpu" if "cpu" in supported else (supported[0] if supported else None)

    def _models(self, profile=None):
        selected = self.manifest["profiles"][profile or self._status["profile"]]["required_models"]
        keys = {(m["folder"], m["filename"]) for m in selected}
        models = [m for m in self.catalog["models"] if (m["folder"], m["filename"]) in keys]
        if len(models) != len(keys):
            raise SetupError("The selected profile references models missing from the verified catalog.")
        for model in models:
            _relative_path(model["folder"] + "/" + model["filename"])
            if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]) or int(model["size"]) <= 0:
                raise SetupError("The model catalog has an invalid size or checksum.")
        return models

    def expected_capabilities(self):
        models = self._models()
        nodes = set(self.manifest["required_nodes"])
        for model in models:
            nodes.update(model.get("requires_nodes", []))
        return {"required_nodes": sorted(nodes), "required_models": [{"folder": m["folder"], "filename": m["filename"]} for m in models]}

    def _disk(self, models=None):
        models = models or self._models()
        probe = self.state_dir
        while not probe.exists():
            probe = probe.parent
        receipt = _read_json(self.state_dir / "ready.json", {})
        known = {**receipt.get("models", {}), **_read_json(self.state_dir / "verified-models.json", {})}
        missing = 0
        for model in models:
            key = model["folder"] + "/" + model["filename"]
            item = known.get(key)
            if item:
                try:
                    allowed = {str(self.backend_root / "models" / _relative_path(key))}
                    if self._status.get("model_root"):
                        allowed.add(str(Path(self._status["model_root"]) / _relative_path(key)))
                    if item.get("path") in allowed and item.get("sha256") == model["sha256"] and item["fingerprint"]["size"] == model["size"] and _fingerprint(item["path"]) == item["fingerprint"]:
                        continue
                except OSError:
                    pass
            partial = self.state_dir / "downloads" / (model["sha256"] + ".part")
            missing += max(0, model["size"] - (partial.stat().st_size if partial.is_file() else 0))
        return {"required_bytes": missing + self.manifest["environment_reserve_bytes"], "available_bytes": shutil.disk_usage(probe).free}

    def status(self):
        with self._lock:
            result = copy.deepcopy(self._status)
        result["running"] = bool(self._thread and self._thread.is_alive())
        result["ready"] = self.ready()
        result["supported_runtimes"] = self.supported_runtimes()
        result["default_runtime"] = self.default_runtime()
        result["default_profile"] = self.manifest["default_profile"]
        result["catalog"] = {"profiles": [{"id": key, **value} for key, value in self.manifest["profiles"].items()], "total_models": len(self.catalog["models"]), "total_bytes": sum(m["size"] for m in self.catalog["models"])}
        result["disk"] = result.get("disk", self._disk()) if result["running"] else self._disk()
        return result

    def _update(self, **values):
        with self._lock:
            self._status.update(values)
            _atomic_json(self.state_dir / "setup.json", self._status)

    def start(self, options=None):
        options = dict(options or {})
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status()
            runtime = options.get("runtime") or self.default_runtime()
            profile = options.get("profile", self.manifest["default_profile"])
            if runtime not in self.supported_runtimes():
                raise SetupError("This runtime is unavailable on this OS and architecture. Select one of: " + ", ".join(self.supported_runtimes()))
            if profile not in self.manifest["profiles"]:
                raise SetupError("Unknown model profile. Select base, workflow, or all.")
            model_root = options.get("model_root")
            if model_root:
                model_root = Path(model_root).expanduser().resolve()
                if not model_root.is_dir():
                    raise SetupError("The model folder does not exist. Choose the existing ComfyUI models folder.")
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._cancel.clear()
            self._tokens = {"hf_token": str(options.get("hf_token") or ""), "civitai_api_key": str(options.get("civitai_api_key") or "")}
            models = self._models(profile)
            self._update(state="installing", phase="preflight", runtime=runtime, profile=profile, model_root=str(model_root) if model_root else None, progress=0, error=None, downloaded_bytes=0, total_bytes=sum(m["size"] for m in models), completed_files=0, total_files=len(models), current_file=None, missing_nodes=[], missing_models=[])
            self._thread = threading.Thread(target=self._install, args=(runtime, profile, model_root), daemon=True, name="donut-create-setup")
            self._thread.start()
        return self.status()

    def cancel(self):
        if self._thread and self._thread.is_alive():
            self._cancel.set()
            self._update(state="cancelling")
            self._stop_process()
        return self.status()

    def ready(self):
        return self._status.get("state") == "ready" and self._ready_files()

    def _ready_files(self):
        receipt = _read_json(self.state_dir / "ready.json")
        if not receipt or receipt.get("manifest") != self._manifest_hash or receipt.get("runtime") != self._status.get("runtime") or receipt.get("profile") != self._status.get("profile"):
            return False
        if not self.python_path.is_file() or not (self.backend_root / "main.py").is_file() or not self.workflow_path.is_file():
            return False
        for model in self._models():
            item = receipt.get("models", {}).get(model["folder"] + "/" + model["filename"])
            if not item or item.get("sha256") != model["sha256"] or item.get("fingerprint", {}).get("size") != model["size"]:
                return False
            try:
                if _fingerprint(item["path"]) != item["fingerprint"]:
                    return False
            except OSError:
                return False
        capabilities = self.expected_capabilities()
        if set(capabilities["required_nodes"]) - set(receipt.get("nodes", [])):
            return False
        if set(m["folder"] + "/" + m["filename"] for m in capabilities["required_models"]) - set(receipt.get("models", {})):
            return False
        for source, target in self._sources():
            if _read_json(target / ".donut-create-source.json", {}).get("sha256") != source["sha256"] or not (target / "__init__.py" if target != self.backend_root else target / "main.py").is_file():
                return False
        return receipt.get("capability_check") is True

    def _check_cancel(self):
        if self._cancel.is_set():
            raise SetupCancelled("Setup cancelled. Retry resumes partial downloads and reuses verified files.")

    def _redact(self, text):
        for token in self._tokens.values():
            if token:
                text = text.replace(token, "[redacted]")
        return re.sub(r"(?i)(?:bearer\s+|(?:token|api[_-]?key)=)[^\s&]+", "[redacted]", text)

    def _hash(self, path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                self._check_cancel()
                digest.update(chunk)
        return digest.hexdigest()

    def _matches(self, path, item):
        return Path(path).is_file() and Path(path).stat().st_size == item["size"] and self._hash(path) == item["sha256"]

    def _auth(self, url):
        host = urllib.parse.urlsplit(url).hostname
        token = self._tokens.get("hf_token", "") if host == "huggingface.co" else self._tokens.get("civitai_api_key", "") if host == "civitai.com" else ""
        return {"Authorization": "Bearer " + token} if token else {}

    def _download(self, item, completed=0, model_progress=False):
        """Download to a hash-addressed partial; verify before any activation."""
        cache = self.state_dir / "downloads"
        cache.mkdir(parents=True, exist_ok=True)
        part = cache / (item["sha256"] + ".part")
        name = item.get("filename", item.get("directory", "source archive"))
        if part.exists() and part.stat().st_size > item["size"]:
            part.unlink()
        for attempt in range(3):
            self._check_cancel()
            offset = part.stat().st_size if part.is_file() else 0
            if offset == item["size"]:
                if self._matches(part, item):
                    return part
                part.unlink()
                raise SetupError("Checksum mismatch for " + name + ". The invalid partial was removed; Retry downloads it again.")
            headers = {"User-Agent": "DonutCreate/1", **self._auth(item["url"])}
            if offset:
                headers["Range"] = "bytes=" + str(offset) + "-"
            try:
                request = urllib.request.Request(item["url"], headers=headers)
                with self._opener.open(request, timeout=20) as response:
                    code = response.getcode()
                    content_range = response.headers.get("Content-Range", "")
                    if code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if not match or int(match[1]) != offset or int(match[3]) != item["size"]:
                            raise SetupError("The provider returned an invalid resume range for " + name + ".")
                    elif code == 200:
                        offset = 0  # A server may ignore Range; safely restart the partial.
                    else:
                        raise SetupError("Unexpected download response for " + name + ".")
                    updated = 0
                    with part.open("ab" if offset else "wb") as handle:
                        while True:
                            self._check_cancel()
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            offset += len(chunk)
                            if offset > item["size"]:
                                part.unlink(missing_ok=True)
                                raise SetupError("Download exceeds the catalog size for " + name + ".")
                            handle.write(chunk)
                            if model_progress and time.monotonic() - updated > 0.2:
                                updated = time.monotonic()
                                total = self._status["total_bytes"]
                                self._update(current_file=name, downloaded_bytes=completed + offset, progress=0.35 + 0.55 * (completed + offset) / max(1, total))
                        handle.flush()
                        os.fsync(handle.fileno())
                if self._matches(part, item):
                    return part
                if part.stat().st_size == item["size"]:
                    part.unlink()
                    raise SetupError("Checksum mismatch for " + name + ". The invalid partial was removed; Retry downloads it again.")
                raise OSError("The download ended before the expected byte count.")
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    provider = "Hugging Face" if urllib.parse.urlsplit(item["url"]).hostname == "huggingface.co" else "Civitai"
                    raise SetupError("Access denied for " + name + ". Check the " + provider + " token and your account's model permissions, then Retry.") from None
                if error.code not in (408, 429, 500, 502, 503, 504):
                    raise SetupError("Download failed for " + name + " (HTTP " + str(error.code) + "). Retry after the provider is available.") from None
            except (OSError, urllib.error.URLError) as error:
                if attempt == 2:
                    raise SetupError("Download failed for " + name + ": " + self._redact(str(error)) + ". Partial bytes remain for Retry.") from None
            if attempt == 2:
                raise SetupError("The provider repeatedly refused " + name + ". Partial bytes remain for Retry.")
            self._cancel.wait(2 ** attempt)
        raise SetupError("Download failed for " + name)

    def _sources(self):
        result = [(self.manifest["comfyui"], self.backend_root)]
        selected = {m["folder"] + "/" + m["filename"] for m in self._models()}
        for pack in self.manifest["node_packs"]:
            if not pack["optional"] or selected.intersection(pack.get("required_for_models", [])):
                result.append((pack, self.backend_root / "custom_nodes" / pack["directory"]))
        return result

    def _source(self, item, target):
        if target.exists():
            if _read_json(target / ".donut-create-source.json", {}).get("sha256") == item["sha256"]:
                return
            raise SetupError("An existing source folder conflicts with the pinned runtime: " + target.name + ". It was preserved. Move it aside before Retry.")
        archive = self._download(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".donut-source-", dir=target.parent))
        try:
            with tarfile.open(archive, "r:gz") as bundle:
                members = bundle.getmembers()
                if not members:
                    raise SetupError("The source archive is empty.")
                prefix = PurePosixPath(members[0].name).parts[0]
                links = []
                for member in members:
                    self._check_cancel()
                    parts = PurePosixPath(member.name).parts
                    if PurePosixPath(member.name).is_absolute() or not parts or parts[0] != prefix or any(p in ("..", ".") or ":" in p for p in parts) or member.islnk() or not (member.isdir() or member.isfile() or member.issym()):
                        raise SetupError("The source archive contains an unsafe entry.")
                    relative = Path(*parts[1:])
                    path = stage / relative
                    if member.issym():
                        link = PurePosixPath(member.linkname)
                        if link.is_absolute() or any(":" in part for part in link.parts):
                            raise SetupError("The source archive contains an unsafe link.")
                        referent = (path.parent / Path(*link.parts)).resolve()
                        if not referent.is_relative_to(stage.resolve()):
                            raise SetupError("The source archive contains an unsafe link.")
                        links.append((path, referent))
                    elif member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as source, path.open("xb") as destination:
                            shutil.copyfileobj(source, destination)
                # SAM2 contains four internal YAML symlinks. Materializing only
                # verified in-tree files preserves them on Windows as well.
                for path, referent in links:
                    if not referent.is_file() or path.exists():
                        raise SetupError("The source archive contains an unresolved link.")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(referent, path)
            _atomic_json(stage / ".donut-create-source.json", {"commit": item["commit"], "sha256": item["sha256"]})
            self._check_cancel()
            stage.rename(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _stop_process(self, force=False):
        process = self._process
        if not process:
            return
        try:
            if os.name == "nt":
                if process.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
            else:
                # Setup children share the sidecar group so add-on shutdown
                # also kills them. Cancellation only targets this command tree.
                rows = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=5)
                parents = {}
                for row in rows.stdout.splitlines():
                    fields = row.split()
                    if len(fields) == 2 and all(field.isdigit() for field in fields):
                        parents[int(fields[0])] = int(fields[1])
                owned = {process.pid}
                while True:
                    descendants = {pid for pid, parent in parents.items() if parent in owned}
                    if descendants <= owned:
                        break
                    owned.update(descendants)
                self._process_children.update(owned - {process.pid})
                for pid in [*self._process_children, process.pid]:
                    try:
                        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill() if force else process.terminate()
            pass

    def _run(self, command, cwd=None, timeout=1800, extra_env=None):
        self._check_cancel()
        environment = dict(os.environ)
        for key in list(environment):
            if any(word in key.upper() for word in ("TOKEN", "API_KEY", "PASSWORD", "SECRET")) or key.startswith("PIP_") or key in ("PYTHONPATH", "PYTHONHOME"):
                environment.pop(key, None)
        environment.update(self.backend_environment())
        environment.update(PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CONFIG_FILE=os.devnull, PIP_CACHE_DIR=str(self.state_dir / "backend" / "cache" / "pip"), SAM2_BUILD_CUDA="0")
        environment.update(extra_env or {})
        # Temporary output stays outside the repository and is removed, including
        # pip logs. No provider tokens are placed in commands or subprocess envs.
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT)
            self._process = process
            self._process_children = set()
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if self._cancel.wait(0.2):
                        self._stop_process()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self._stop_process(force=True)
                            process.wait()
                        self._stop_process(force=True)
                        self._check_cancel()
                    if time.monotonic() - started > timeout:
                        self._stop_process()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self._stop_process(force=True)
                        raise SetupError("A runtime setup command timed out. Retry after checking available disk space and network access.")
                self._check_cancel()
                output.seek(0, os.SEEK_END)
                length = output.tell()
                output.seek(max(0, length - 12000))
                text = output.read().decode("utf-8", errors="replace")
                if process.returncode:
                    raise SetupError("Runtime setup failed:\n" + self._redact(text[-6000:]))
                return text
            finally:
                if process.poll() is None:
                    self._stop_process()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._stop_process(force=True)
                        process.wait()
                self._process = None
                self._process_children.clear()

    def _python_command(self):
        if (3, 12) <= sys.version_info[:2] < (3, 13):
            return [sys.executable]
        candidate = shutil.which("python3.12")
        if candidate:
            return [candidate]
        if os.name == "nt" and shutil.which("py"):
            return [shutil.which("py"), "-3.12"]
        raise SetupError("Install Python 3.12, then Retry. The managed ComfyUI environment is isolated from the add-on's interpreter.")

    def _environment(self, runtime):
        root = self.python_path.parent.parent
        marker = root / ".donut-create-env.json"
        if root.exists() and _read_json(marker, {}).get("runtime") != runtime:
            raise SetupError("The existing managed Python environment uses a different runtime or is incomplete. It was preserved. Move the backend/venv folder aside before switching runtime or Retry.")
        if not root.exists():
            # Write ownership before venv creation so interrupted setup is retryable.
            root.mkdir(parents=True)
            _atomic_json(marker, {"runtime": runtime})
        if not self.python_path.is_file():
            self._run([*self._python_command(), "-m", "venv", root])
        self._run([self.python_path, "-c", "import sys; assert sys.version_info[:2] == (3,12), 'Python 3.12 is required'"])
        self._run([self.python_path, "-m", "pip", "install", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "--upgrade", "pip", "setuptools", "wheel"])

    def _dependencies(self, runtime):
        item = self.manifest["runtimes"][runtime]
        index = item.get("darwin_index_url", item["index_url"]) if platform.system() == "Darwin" else item["index_url"]
        self._run([self.python_path, "-m", "pip", "install", "--only-binary=:all:", "--index-url", index, *item["packages"]])
        constraints = self.state_dir / "backend" / "constraints.txt"
        constraints.write_text("\n".join([*item["packages"], "comfyui-frontend-package==" + self.manifest["comfyui"]["frontend"]]) + "\n", encoding="utf-8")
        requirements = []
        needs_sam2 = False
        for _, root in self._sources():
            path = root / "requirements.txt"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped == "git+https://github.com/facebookresearch/sam2":
                    needs_sam2 = True
                elif stripped and not stripped.startswith("#"):
                    if "git+" in stripped or stripped.startswith(("-", "http:")):
                        raise SetupError("An upstream requirement is unpinned or unsafe: " + stripped)
                    requirements.append(stripped)
        requirements_path = self.state_dir / "backend" / "requirements.txt"
        requirements_path.write_text("\n".join(dict.fromkeys(requirements)) + "\n", encoding="utf-8")
        self._run([self.python_path, "-m", "pip", "install", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "-c", constraints, "-r", requirements_path])
        if needs_sam2:
            sam2_root = self.state_dir / "backend" / "sam2"
            self._source(self.manifest["sam2"], sam2_root)
            self._run([self.python_path, "-m", "pip", "install", "--no-build-isolation", "--index-url", "https://pypi.org/simple", "-c", constraints, sam2_root])
        self._run([self.python_path, "-m", "pip", "check"])

    def _model(self, item, model_root, completed):
        relative = _relative_path(item["folder"] + "/" + item["filename"])
        target = self.backend_root / "models" / relative
        if target.exists():
            if self._matches(target, item):
                self._record_model(item, target)
                return target
            raise SetupError("A conflicting model was preserved: " + relative.as_posix() + ". Move that file aside before Retry.")
        if model_root:
            existing = model_root / relative
            if existing.is_file():
                if self._matches(existing, item):
                    self._record_model(item, existing)
                    return existing
                raise SetupError("The selected existing model differs from the catalog: " + relative.as_posix() + ". It was preserved. Choose another folder or move it aside before Retry.")
        part = self._download(item, completed, model_progress=True)
        self._check_cancel()
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hard-link activation is atomic and refuses an existing destination.
        # The hash-addressed partial is always on the same managed filesystem.
        try:
            os.link(part, target)
        except FileExistsError:
            if not self._matches(target, item):
                raise SetupError("A model appeared during setup and conflicts with the catalog. It was preserved.") from None
        part.unlink(missing_ok=True)
        self._record_model(item, target)
        return target

    def _record_model(self, item, path):
        record = {"path": str(path), "sha256": item["sha256"], "fingerprint": _fingerprint(path)}
        inventory = _read_json(self.state_dir / "verified-models.json", {})
        inventory[item["folder"] + "/" + item["filename"]] = record
        _atomic_json(self.state_dir / "verified-models.json", inventory)
        return record

    def _workflow(self, profile):
        workflow = _read_json(ROOT / "workflow.json")
        if profile == "base":
            for graph in [workflow, *workflow["definitions"]["subgraphs"]]:
                for node in graph["nodes"]:
                    if node["id"] == 1122:
                        node["widgets_values"][0] = "krea2_turbo_bf16.safetensors"
                        node["title"] = "Krea2 base model · neutral starter"
                    if node["id"] == 1124:
                        node["widgets_values"][-1] = "Single model"
                    if node["id"] == 1055:
                        node["widgets_values"][1] = "[]"
            workflow["extra"]["donut_workflow"]["name"] = "Donut Create · neutral starter v5"
        _atomic_json(self.state_dir / "workflow.json", workflow)

    def _capabilities(self, runtime):
        expected = self.expected_capabilities()["required_nodes"]
        # This imports the real installed engine and all node packs without
        # listening on a port or loading weights. GPU capability is checked too.
        script = """
import asyncio, json, sys
import torch
runtime, expected = sys.argv[1], json.loads(sys.argv[2])
if runtime == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable. Check your NVIDIA driver or select CPU.')
if runtime == 'mps' and not torch.backends.mps.is_available():
    raise RuntimeError('Metal/MPS is unavailable on this Mac. Select CPU.')
sys.argv = ['donut-create-check', '--cpu'] if runtime == 'cpu' else ['donut-create-check']
import comfy.options
comfy.options.enable_args_parsing()
import nodes, server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server.PromptServer(loop)
loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
registered = sorted(set(expected).intersection(nodes.NODE_CLASS_MAPPINGS))
missing = sorted(set(expected) - set(registered))
print('DONUT_CREATE_CAPABILITIES=' + json.dumps({'nodes':registered, 'missing':missing}))
"""
        output = self._run([self.python_path, "-c", script, runtime, json.dumps(expected)], cwd=self.backend_root, timeout=180)
        rows = [line.split("=", 1)[1] for line in output.splitlines() if line.startswith("DONUT_CREATE_CAPABILITIES=")]
        if not rows:
            raise SetupError("The real ComfyUI capability check did not return a result.")
        result = json.loads(rows[-1])
        if result["missing"]:
            self._update(missing_nodes=result["missing"])
            raise SetupError("Required node types are missing: " + ", ".join(result["missing"]))
        return result["nodes"]

    def _frontend_prefix(self):
        # Read package files without importing GPU backend modules.
        script = """
import importlib.metadata, json
distribution = importlib.metadata.distribution('comfyui-frontend-package')
files = [distribution.locate_file(f) for f in distribution.files if str(f).endswith('.js') and '/assets/api-' in str(f)]
if not files or not any('api_base' in p.read_text(encoding='utf-8') for p in files):
    raise RuntimeError('The installed ComfyUI frontend lacks scoped API base support.')
print(json.dumps({'frontend':distribution.version, 'prefix_support':True}))
"""
        self._run([self.python_path, "-c", script], timeout=30)

    def _install(self, runtime, profile, model_root):
        try:
            models = self._models(profile)
            self._run([*self._python_command(), "-c", "import sys; assert sys.version_info[:2] == (3,12), 'Install Python 3.12, then Retry.'"], timeout=30)
            # Verify reusable files before disk accounting, including completed
            # downloads from a cancelled setup that has no final ready receipt.
            reused = {}
            for item in models:
                key = item["folder"] + "/" + item["filename"]
                relative = _relative_path(key)
                candidates = [self.backend_root / "models" / relative]
                if model_root:
                    candidates.append(model_root / relative)
                for path in candidates:
                    if not path.exists():
                        continue
                    self._update(current_file="Checking " + item["filename"])
                    if not self._matches(path, item):
                        raise SetupError("A conflicting existing model was preserved: " + key + ". Move the file aside before Retry or choose a different models folder.")
                    reused[key] = path
                    self._record_model(item, path)
                    break
            disk = self._disk(models)
            self._update(disk=disk)
            if disk["available_bytes"] < disk["required_bytes"]:
                raise SetupError("Insufficient free disk space: setup requires " + str(disk["required_bytes"]) + " bytes including the environment reserve; " + str(disk["available_bytes"]) + " bytes are available.")
            self._update(phase="source", progress=0.05)
            for item, target in self._sources():
                self._update(current_file=item.get("directory", "ComfyUI"))
                self._source(item, target)
            self._update(phase="environment", progress=0.15, current_file=None)
            self._environment(runtime)
            self._update(phase="dependencies", progress=0.2)
            self._dependencies(runtime)
            self._update(phase="models", progress=0.35)
            completed, records = 0, {}
            for count, item in enumerate(models):
                self._update(current_file=item["filename"])
                key = item["folder"] + "/" + item["filename"]
                path = reused.get(key) or self._model(item, model_root, completed)
                records[key] = {"path": str(path), "sha256": item["sha256"], "fingerprint": _fingerprint(path)}
                completed += item["size"]
                self._update(downloaded_bytes=completed, completed_files=count + 1, progress=0.35 + 0.55 * completed / self._status["total_bytes"])
            paths = self.backend_root / "extra_model_paths.yaml"
            if model_root:
                # JSON is valid YAML. ComfyUI reads this owned configuration on
                # startup; external models are read in place and never modified.
                _atomic_json(paths, {"donut_create_reuse": {"base_path": str(model_root), **{m["folder"]: m["folder"] for m in models}}})
            elif paths.exists():
                paths.unlink()
            self._workflow(profile)
            self._update(phase="verify", progress=0.92, current_file=None)
            self._frontend_prefix()
            nodes = self._capabilities(runtime)
            self._check_cancel()
            _atomic_json(self.state_dir / "ready.json", {"manifest": self._manifest_hash, "runtime": runtime, "profile": profile, "nodes": nodes, "models": records, "capability_check": True})
            if not self._ready_files():
                raise SetupError("The final runtime verification failed. Retry to recheck required files.")
            self._update(state="ready", phase="ready", progress=1, error=None)
        except SetupCancelled as error:
            self._update(state="cancelled", error=str(error))
        except Exception as error:
            self._update(state="error", error=self._redact(str(error)))
        finally:
            self._tokens.clear()
