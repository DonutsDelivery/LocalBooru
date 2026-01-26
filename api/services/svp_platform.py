"""
Cross-platform SVP path resolution and environment utilities.

Centralizes all platform-specific logic for SVP plugin paths,
VapourSynth source filter locations, system Python detection,
and clean subprocess environments.

Supports Linux, Windows, and macOS (Intel + ARM).
"""
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, List


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_macos_arm() -> bool:
    return _is_macos() and platform.machine() == "arm64"


# ── SVP plugin paths ────────────────────────────────────────────────────

_SVP_PLUGIN_SEARCH_DIRS: List[str] = []

if _is_windows():
    _SVP_PLUGIN_SEARCH_DIRS = [
        os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                     "SVP 4", "plugins64"),
        os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                     "SVP 4", "plugins64"),
    ]
elif _is_macos():
    _SVP_PLUGIN_SEARCH_DIRS = [
        "/Applications/SVP 4 Mac.app/Contents/Resources/plugins",
    ]
else:  # Linux
    _SVP_PLUGIN_SEARCH_DIRS = [
        "/opt/svp/plugins",
        "/usr/lib/svp/plugins",
    ]


def get_svp_plugin_path() -> Optional[str]:
    """Return the first existing SVP plugin directory, or None.

    Checks the ``LOCALBOORU_SVP_PLUGIN_PATH`` env var first, then
    searches platform-specific default locations.
    """
    env_override = os.environ.get("LOCALBOORU_SVP_PLUGIN_PATH")
    if env_override and os.path.isdir(env_override):
        return env_override

    for candidate in _SVP_PLUGIN_SEARCH_DIRS:
        if os.path.isdir(candidate):
            return candidate

    return None


def get_svp_plugin_names() -> Tuple[str, str]:
    """Return ``(flow1_filename, flow2_filename)`` for the current platform."""
    if _is_windows():
        return ("svpflow1_vs.dll", "svpflow2_vs.dll")
    elif _is_macos():
        return ("libsvpflow1_vs64.dylib", "libsvpflow2_vs64.dylib")
    else:
        return ("libsvpflow1.so", "libsvpflow2.so")


def get_svp_plugin_full_paths(plugin_dir: Optional[str] = None) -> Tuple[str, str]:
    """Return absolute paths to svpflow1 and svpflow2 plugins.

    Uses forward slashes on all platforms (Python/VapourSynth accept them
    on Windows too).

    Args:
        plugin_dir: Explicit plugin directory.  Falls back to
                    :func:`get_svp_plugin_path` then the legacy Linux default.
    """
    base = plugin_dir or get_svp_plugin_path() or "/opt/svp/plugins"
    flow1, flow2 = get_svp_plugin_names()
    # Forward slashes everywhere – Python and VS handle them on Windows
    return (
        Path(base, flow1).as_posix(),
        Path(base, flow2).as_posix(),
    )


# ── VapourSynth source filter paths ─────────────────────────────────────

def get_source_filter_paths() -> List[Tuple[str, str, str]]:
    """Return a list of ``(abs_path, result_key, vs_namespace)`` for source filters.

    Each entry represents a VapourSynth source-filter plugin to try loading.
    """
    env_dir = os.environ.get("LOCALBOORU_VS_PLUGIN_PATH")
    if env_dir and os.path.isdir(env_dir):
        ext = ".dll" if _is_windows() else ".dylib" if _is_macos() else ".so"
        return [
            (str(Path(env_dir) / f"ffms2{ext}"), "ffms2_available", "ffms2"),
            (str(Path(env_dir) / f"libvslsmashsource{ext}"), "lsmas_available", "lsmas"),
        ]

    if _is_windows():
        appdata = os.environ.get("APPDATA", "")
        base = os.path.join(appdata, "VapourSynth", "plugins64") if appdata else ""
        ext = ".dll"
    elif _is_macos_arm():
        base = "/opt/homebrew/lib/vapoursynth"
        ext = ".dylib"
    elif _is_macos():
        base = "/usr/local/lib/vapoursynth"
        ext = ".dylib"
    else:  # Linux
        base = "/usr/lib/vapoursynth"
        ext = ".so"

    if not base:
        return []

    return [
        (str(Path(base) / f"bestsource{ext}"), "bestsource_available", "bs"),
        (str(Path(base) / f"ffms2{ext}"),      "ffms2_available",      "ffms2"),
        (str(Path(base) / f"lsmas{ext}"),      "lsmas_available",      "lsmas"),
    ]


# ── System Python ────────────────────────────────────────────────────────

def get_system_python() -> str:
    """Return a path to the system Python interpreter (outside any venv).

    On Windows falls back to ``sys.executable`` since there's typically no
    separate ``python3`` binary.
    """
    vs_python = os.environ.get("LOCALBOORU_VS_PYTHON")
    if vs_python and os.path.isfile(vs_python):
        return vs_python

    if _is_windows():
        found = shutil.which("python3") or shutil.which("python")
        return found or sys.executable
    else:
        # Prefer python3 on Unix; fall back to sys.executable
        found = shutil.which("python3")
        return found or sys.executable


# ── Clean subprocess environment ─────────────────────────────────────────

def get_clean_env() -> dict:
    """Return a minimal environment dict for running vspipe / VapourSynth.

    VapourSynth's vspipe can fail inside a Python venv due to env-var
    conflicts.  This strips down to only what's needed per-platform.
    """
    if _is_windows():
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\WINDOWS"),
            "TEMP": os.environ.get("TEMP", os.environ.get("TMP", "")),
            "USERPROFILE": os.environ.get("USERPROFILE", ""),
            "APPDATA": os.environ.get("APPDATA", ""),
            "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        }
    elif _is_macos():
        home = os.environ.get("HOME", "/tmp")
        path_parts = ["/usr/bin", "/bin", "/usr/local/bin"]
        if _is_macos_arm():
            path_parts.append("/opt/homebrew/bin")
        env = {
            "PATH": ":".join(path_parts),
            "HOME": home,
            "USER": os.environ.get("USER", "user"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        }
    else:
        # Linux
        home = os.environ.get("HOME", "/tmp")
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": home,
            "USER": os.environ.get("USER", "user"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "DISPLAY": os.environ.get("DISPLAY", ":0"),
        }
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            env["XDG_RUNTIME_DIR"] = xdg
        else:
            try:
                env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
            except AttributeError:
                pass  # os.getuid() doesn't exist on Windows

    # Forward bundled tool env vars into subprocess environment (all platforms)
    for key in ("LOCALBOORU_VS_PYTHON", "LOCALBOORU_SVP_PLUGIN_PATH",
                "LOCALBOORU_VS_PLUGIN_PATH", "LOCALBOORU_PACKAGED"):
        val = os.environ.get(key)
        if val:
            env[key] = val

    return env
