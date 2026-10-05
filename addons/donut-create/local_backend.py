# SPDX-License-Identifier: GPL-3.0-only
"""Identify and control an explicitly selected, same-user local ComfyUI process."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

try:
    import psutil
except ImportError:
    psutil = None

PROCESS_ERRORS = (psutil.Error, OSError) if psutil else (OSError,)
DISCOVERY_ERRORS = PROCESS_ERRORS + (subprocess.SubprocessError,)


class DiscoveryError(Exception):
    pass


ENVIRONMENT_KEYS = {
    "PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES", "CUDA_HOME", "ROCM_HOME", "TORCH_HOME",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
}


def local_port(origin):
    parts = urlsplit(origin)
    host = parts.hostname or ""
    try:
        local = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host.lower() == "localhost"
    if not local:
        raise DiscoveryError("Automatic management requires a localhost backend on the computer running the DMC add-on.")
    return parts.port or (443 if parts.scheme == "https" else 80)


def listener_pids(port):
    if psutil is None:
        raise DiscoveryError("Update or reinstall the Donut Create add-on to install its local process discovery dependency.")
    # macOS restricts psutil's system socket table to root; lsof can inspect this user's listeners.
    if sys.platform == "darwin":
        lsof = shutil.which("lsof") or ("/usr/sbin/lsof" if Path("/usr/sbin/lsof").is_file() else None)
        if not lsof:
            raise DiscoveryError("Local discovery needs lsof on macOS; select the installation manually instead.")
        result = subprocess.run([lsof, "-nP", "-a", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                                capture_output=True, text=True, timeout=10)
        return {int(line) for line in result.stdout.splitlines() if line.isdecimal()}
    return {connection.pid for connection in psutil.net_connections(kind="tcp")
            if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port and connection.pid}


def same_user(process):
    if process.username().casefold() != psutil.Process().username().casefold():
        raise DiscoveryError("The listening process belongs to another user; it cannot be managed by this DMC instance.")


def fingerprint(command):
    return hashlib.sha256(json.dumps(command, ensure_ascii=False).encode()).hexdigest()


def discover(origin):
    """Read only. No process is stopped or started during discovery/adoption."""
    port = local_port(origin)
    try:
        pids = listener_pids(port)
        if len(pids) != 1:
            raise DiscoveryError("Could not identify one local ComfyUI listener on this port. Select the installation manually.")
        process = psutil.Process(pids.pop())
        same_user(process)
        command = process.cmdline()
        cwd = Path(process.cwd())
        # Accept a direct Python script invocation, never a shell, -c command or -m wrapper.
        index = 1
        while index < len(command) and command[index].startswith("-"):
            argument = command[index]
            if argument in {"-", "-c", "-m"}:
                break
            if argument == "--":
                index += 1
                break
            index += 2 if argument in {"-W", "-X", "--check-hash-based-pycs"} else 1
        if index >= len(command) or Path(command[index]).name != "main.py":
            raise DiscoveryError("The listener is not a direct ComfyUI main.py process. Select its installation manually.")
        script = Path(command[index])
        script = (script if script.is_absolute() else cwd / script).resolve()
        if not script.is_file() or not (script.parent / "comfy").is_dir():
            raise DiscoveryError("The listening process does not point to a valid ComfyUI installation.")
        try:
            environment = process.environ()
        except psutil.AccessDenied:
            environment = {}
        executable = Path(process.exe())
        if not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?", executable.name, re.I):
            raise DiscoveryError("The listener is not a Python process.")
        # Preserve the venv symlink/path: resolving it to system Python loses the installed packages.
        argument = Path(command[0])
        if argument.is_absolute():
            python = argument
        elif len(argument.parts) > 1:
            python = cwd / argument
        else:
            found = shutil.which(command[0], path=environment.get("PATH")) if environment.get("PATH") else None
            python = Path(found) if found else None
            if python is None:
                relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
                candidates = [base / relative for base in
                              [Path(environment[key]) for key in ("VIRTUAL_ENV", "CONDA_PREFIX") if environment.get(key)]]
                candidates += [base / env / relative for base in (script.parent, script.parent.parent)
                               for env in ("venv", ".venv")]
                candidates = [p for p in candidates if p.is_file() and p.resolve() == executable.resolve()]
                if len(candidates) > 1:
                    raise DiscoveryError("More than one Python environment matches. Select its executable manually.")
                if not candidates and not environment:
                    raise DiscoveryError("The running Python environment could not be read. Select its executable manually.")
                python = candidates[0] if candidates else executable
        python = python.absolute()
        if not python.is_file() or not os.access(python, os.X_OK) or python.resolve() != executable.resolve():
            raise DiscoveryError("The original Python executable could not be verified. Select it manually.")
        identity = {"pid": process.pid, "started_at": process.create_time(), "command_hash": fingerprint(command)}
        profile = {"command": [str(python), *command[1:index], str(script), *command[index + 1:]],
                   "cwd": str(cwd), "environment": {key: value for key, value in environment.items() if key in ENVIRONMENT_KEYS}}
        return {"comfy_directory": str(script.parent), "python_executable": str(python),
                "launch_profile": profile, "process_identity": identity, "detected_local": True}
    except DiscoveryError:
        raise
    except DISCOVERY_ERRORS as error:
        raise DiscoveryError("The local listener could not be inspected. Select the ComfyUI installation manually.") from error


def restore(identity):
    if psutil is None or not isinstance(identity, dict):
        return None
    try:
        process = psutil.Process(identity["pid"])
        same_user(process)
        if (process.create_time() != identity["started_at"] or
                fingerprint(process.cmdline()) != identity["command_hash"] or
                not process.is_running() or process.status() == psutil.STATUS_ZOMBIE):
            return None
        return process
    except (psutil.Error, DiscoveryError, KeyError, TypeError, ValueError):
        return None


def alive(process):
    try:
        return process is not None and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except PROCESS_ERRORS:
        return False


def stop(identity):
    process = restore(identity)
    if process is None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            process.kill()  # psutil validates PID/creation-time identity before signalling.
            process.wait(timeout=10)
    except psutil.NoSuchProcess:
        pass
    except psutil.Error as error:
        raise DiscoveryError("The matched ComfyUI process could not be stopped by this user.") from error


def child_identity(pid):
    if psutil is None:
        return None
    try:
        process = psutil.Process(pid)
        return {"pid": pid, "started_at": process.create_time(), "command_hash": fingerprint(process.cmdline())}
    except psutil.Error:
        return None
