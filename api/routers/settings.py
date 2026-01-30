"""
Settings router - app configuration and optional features
Uses JSON file for settings to avoid database migration issues
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
import os
import json
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_data_dir, get_db, directory_db_manager
from ..models import WatchDirectory, Image, DirectoryImage

router = APIRouter()


def get_packages_dir() -> Path:
    """Get persistent packages directory that survives updates.

    Uses LOCALBOORU_PACKAGES_DIR env var if set by Electron,
    otherwise falls back to data_dir/packages.
    """
    # Check for Electron-provided path first (ensures consistency)
    if os.environ.get('LOCALBOORU_PACKAGES_DIR'):
        packages_dir = Path(os.environ['LOCALBOORU_PACKAGES_DIR'])
    else:
        packages_dir = get_data_dir() / 'packages'
    packages_dir.mkdir(parents=True, exist_ok=True)
    return packages_dir


def get_venv_dir() -> Path:
    """Get the age detection virtual environment directory."""
    # Use data_dir for persistence across app updates
    if os.environ.get('LOCALBOORU_PACKAGES_DIR'):
        # Put venv alongside packages dir
        venv_dir = Path(os.environ['LOCALBOORU_PACKAGES_DIR']).parent / 'age_detection_venv'
    else:
        venv_dir = get_data_dir() / 'age_detection_venv'
    return venv_dir


def get_venv_python() -> Path:
    """Get the Python executable path for the venv."""
    venv_dir = get_venv_dir()
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"


def get_venv_site_packages() -> Path:
    """Get the site-packages directory for the venv."""
    venv_dir = get_venv_dir()
    if sys.platform == "win32":
        return venv_dir / "Lib" / "site-packages"
    else:
        # Find the python version directory
        lib_dir = venv_dir / "lib"
        if lib_dir.exists():
            for item in lib_dir.iterdir():
                if item.name.startswith("python"):
                    return item / "site-packages"
        # Fallback - construct from current Python version
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        return venv_dir / "lib" / version / "site-packages"


def ensure_packages_in_path():
    """Add venv site-packages to Python path for imports."""
    # First try venv site-packages (preferred)
    venv_site = get_venv_site_packages()
    if venv_site.exists():
        venv_site_str = str(venv_site)
        if venv_site_str not in sys.path:
            sys.path.insert(0, venv_site_str)
            print(f"[AgeDetection] Added venv site-packages to path: {venv_site_str}", flush=True)
        return

    # Fallback to legacy packages directory
    packages_dir = get_packages_dir()
    packages_str = str(packages_dir)
    if packages_str not in sys.path:
        sys.path.insert(0, packages_str)


# Settings file path
def get_settings_file() -> Path:
    return get_data_dir() / 'settings.json'


def load_settings() -> dict:
    """Load settings from JSON file"""
    settings_file = get_settings_file()
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def save_settings(settings: dict):
    """Save settings to JSON file"""
    settings_file = get_settings_file()
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)


def get_setting(key: str, default: str = None) -> Optional[str]:
    """Get a setting value"""
    settings = load_settings()
    return settings.get(key, default)


def set_setting(key: str, value: str):
    """Set a setting value"""
    settings = load_settings()
    settings[key] = value
    save_settings(settings)


# Settings keys
AGE_DETECTION_ENABLED = "age_detection_enabled"
AGE_DETECTION_INSTALLED = "age_detection_installed"
AGE_DETECTION_INSTALLING = "age_detection_installing"
AGE_DETECTION_INSTALL_PROGRESS = "age_detection_install_progress"

# Network settings defaults
# Different default ports: portable=8791, system=8790
# This allows running both simultaneously without conflicts
def get_default_local_port():
    """Get default local port based on install mode"""
    if os.environ.get('LOCALBOORU_PORTABLE_DATA'):
        return 8791  # Portable mode
    return 8790  # System install

DEFAULT_NETWORK_SETTINGS = {
    "local_network_enabled": False,
    "public_network_enabled": False,
    "local_port": get_default_local_port(),
    "public_port": 8791,
    "auth_required_level": "local_network",  # none, public, local_network, always
    "upnp_enabled": False,
    "allow_settings_local_network": False  # Allow settings/admin access from local network
}

# Optical flow interpolation settings defaults
DEFAULT_OPTICAL_FLOW_SETTINGS = {
    "enabled": False,
    "target_fps": 60,
    "use_gpu": True,
    "quality": "fast",  # svp, gpu_native, realtime, fast, balanced, quality
}

# SVP (SmoothVideo Project) interpolation settings defaults
DEFAULT_SVP_SETTINGS = {
    "enabled": False,
    "target_fps": 60,
    "preset": "balanced",  # fast, balanced, quality, max, animation, film
    # Key settings
    "use_nvof": True,           # Use NVIDIA Optical Flow
    "shader": 23,               # SVP shader/algo (1,2,11,13,21,23)
    "artifact_masking": 100,    # Artifact masking area (0-200)
    "frame_interpolation": 2,   # Frame interpolation mode (1=uniform, 2=adaptive)
    # Advanced settings (full override when set)
    "custom_super": None,
    "custom_analyse": None,
    "custom_smooth": None,
}


def get_network_settings() -> dict:
    """Get network settings with defaults"""
    settings = load_settings()
    network = settings.get("network", {})
    # Merge with defaults
    return {**DEFAULT_NETWORK_SETTINGS, **network}


def save_network_settings(network_settings: dict):
    """Save network settings"""
    settings = load_settings()
    settings["network"] = network_settings
    save_settings(settings)


def get_optical_flow_settings() -> dict:
    """Get optical flow interpolation settings with defaults"""
    settings = load_settings()
    optical_flow = settings.get("optical_flow", {})
    # Merge with defaults
    return {**DEFAULT_OPTICAL_FLOW_SETTINGS, **optical_flow}


def save_optical_flow_settings(optical_flow_settings: dict):
    """Save optical flow interpolation settings"""
    settings = load_settings()
    settings["optical_flow"] = optical_flow_settings
    save_settings(settings)


def get_svp_settings() -> dict:
    """Get SVP interpolation settings with defaults"""
    settings = load_settings()
    svp = settings.get("svp", {})
    # Merge with defaults
    return {**DEFAULT_SVP_SETTINGS, **svp}


def save_svp_settings(svp_settings: dict):
    """Save SVP interpolation settings"""
    settings = load_settings()
    settings["svp"] = svp_settings
    save_settings(settings)


def check_age_detection_deps() -> dict:
    """Check if age detection dependencies are installed"""
    # Ensure persistent packages directory is in path
    ensure_packages_in_path()

    is_windows = sys.platform == "win32"

    deps = {
        "torch": False,
        "transformers": False,
        "ultralytics": False,
        "timm": False,
        "mivolo": False,
        "insightface": False,
    }

    # Catch all exceptions - imports can throw ImportError, OSError, RuntimeError, etc.
    try:
        import torch
        deps["torch"] = True
    except Exception as e:
        if isinstance(e, OSError) and is_windows:
            deps["torch_error"] = "Missing Visual C++ Redistributable. Install from: https://aka.ms/vs/17/release/vc_redist.x64.exe"

    try:
        import transformers
        deps["transformers"] = True
    except Exception:
        pass

    try:
        import ultralytics
        deps["ultralytics"] = True
    except Exception:
        pass

    try:
        import timm
        deps["timm"] = True
    except Exception:
        pass

    try:
        import mivolo
        deps["mivolo"] = True
    except Exception:
        pass

    try:
        import insightface
        deps["insightface"] = True
    except Exception:
        pass

    return deps


def are_required_deps_installed() -> bool:
    """Check if required (non-optional) dependencies are installed"""
    deps = check_age_detection_deps()
    # insightface is optional - OpenCV fallback is available
    required = ["torch", "transformers", "ultralytics", "timm", "mivolo"]
    return all(deps.get(r, False) for r in required)


@router.get("")
async def get_all_settings():
    """Get all app settings"""
    deps = check_age_detection_deps()
    installed = are_required_deps_installed()
    progress = get_setting(AGE_DETECTION_INSTALL_PROGRESS, "")

    # Clear stale "failed" progress message if required deps are actually installed
    if installed and "failed" in progress.lower():
        if deps.get("insightface", False):
            progress = "Installation complete!"
        else:
            progress = "Installation complete (using OpenCV fallback)"
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, progress)

    return {
        "age_detection": {
            "enabled": get_setting(AGE_DETECTION_ENABLED, "false") == "true",
            "installed": installed,
            "installing": get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
            "install_progress": progress,
            "dependencies": deps
        }
    }


class AgeDetectionToggle(BaseModel):
    enabled: bool


@router.post("/age-detection/toggle")
async def toggle_age_detection(data: AgeDetectionToggle):
    """Enable/disable age detection (requires dependencies to be installed)"""
    if data.enabled:
        if not are_required_deps_installed():
            deps = check_age_detection_deps()
            return {
                "success": False,
                "error": "Required dependencies not installed. Install them first.",
                "dependencies": deps
            }

    set_setting(AGE_DETECTION_ENABLED, "true" if data.enabled else "false")
    return {"success": True, "enabled": data.enabled}


def patch_mivolo_for_timm_compat(packages_dir: Path):
    """Patch mivolo to work with timm 0.9.x+ API changes.

    MiVOLO was written for older timm (~0.6.x-0.8.x). Newer timm versions have:
    - Renamed remap_checkpoint to remap_state_dict (with swapped args)
    - Moved split_model_name_tag from _pretrained to _registry
    - Added pos_drop_rate parameter to VOLO (breaks positional args)
    """
    mivolo_dir = packages_dir / "mivolo" / "model"
    if not mivolo_dir.exists():
        return

    patched_any = False

    # Patch create_timm_model.py
    create_timm_path = mivolo_dir / "create_timm_model.py"
    if create_timm_path.exists():
        try:
            content = create_timm_path.read_text(encoding="utf-8")
            modified = False

            # Fix remap_checkpoint import
            old_import = "from timm.models._helpers import load_state_dict, remap_checkpoint"
            new_import = """from timm.models._helpers import load_state_dict
try:
    from timm.models._helpers import remap_checkpoint
except ImportError:
    # timm 0.9.x renamed remap_checkpoint to remap_state_dict with swapped args
    from timm.models._helpers import remap_state_dict
    def remap_checkpoint(model, state_dict, allow_reshape=True):
        return remap_state_dict(state_dict, model, allow_reshape)"""

            if old_import in content:
                content = content.replace(old_import, new_import)
                modified = True

            # Fix split_model_name_tag import location
            if "from timm.models._pretrained import PretrainedCfg, split_model_name_tag" in content:
                content = content.replace(
                    "from timm.models._pretrained import PretrainedCfg, split_model_name_tag",
                    "from timm.models._pretrained import PretrainedCfg"
                )
                content = content.replace(
                    "from timm.models._registry import is_model, model_entrypoint",
                    "from timm.models._registry import is_model, model_entrypoint, split_model_name_tag"
                )
                modified = True

            if modified:
                create_timm_path.write_text(content, encoding="utf-8")
                print("[AgeDetection] Patched mivolo/create_timm_model.py for timm compatibility", flush=True)
                patched_any = True
        except Exception as e:
            print(f"[AgeDetection] Failed to patch create_timm_model.py: {e}", flush=True)

    # Patch mivolo_model.py - fix super().__init__() positional args
    model_path = mivolo_dir / "mivolo_model.py"
    if model_path.exists():
        try:
            content = model_path.read_text(encoding="utf-8")

            old_super = """super().__init__(
            layers,
            img_size,
            in_chans,
            num_classes,
            global_pool,
            patch_size,
            stem_hidden_dim,
            embed_dims,
            num_heads,
            downsamples,
            outlook_attention,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            post_layers,
            use_aux_head,
            use_mix_token,
            pooling_scale,
        )"""

            new_super = """super().__init__(
            layers,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            patch_size=patch_size,
            stem_hidden_dim=stem_hidden_dim,
            embed_dims=embed_dims,
            num_heads=num_heads,
            downsamples=downsamples,
            outlook_attention=outlook_attention,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            post_layers=post_layers,
            use_aux_head=use_aux_head,
            use_mix_token=use_mix_token,
            pooling_scale=pooling_scale,
        )"""

            if old_super in content:
                content = content.replace(old_super, new_super)
                model_path.write_text(content, encoding="utf-8")
                print("[AgeDetection] Patched mivolo/mivolo_model.py for timm compatibility", flush=True)
                patched_any = True
        except Exception as e:
            print(f"[AgeDetection] Failed to patch mivolo_model.py: {e}", flush=True)

    return patched_any


def create_venv_if_needed() -> bool:
    """Create the age detection venv if it doesn't exist.

    Returns True if venv is ready, False if creation failed.
    """
    import venv

    venv_dir = get_venv_dir()
    venv_python = get_venv_python()

    # Check if venv already exists and is functional
    if venv_python.exists():
        try:
            result = subprocess.run(
                [str(venv_python), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"[AgeDetection] Venv already exists at {venv_dir}", flush=True)
                return True
        except Exception:
            pass

    # Create new venv
    print(f"[AgeDetection] Creating venv at {venv_dir}...", flush=True)
    set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Creating virtual environment...")

    try:
        # Create venv with pip included
        venv.create(str(venv_dir), with_pip=True, clear=True)

        # Verify it works
        if not venv_python.exists():
            print(f"[AgeDetection] Venv Python not found at {venv_python}", flush=True)
            return False

        result = subprocess.run(
            [str(venv_python), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"[AgeDetection] Venv Python not functional: {result.stderr}", flush=True)
            return False

        # Upgrade pip in the venv
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Upgrading pip...")
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            timeout=120
        )

        print(f"[AgeDetection] Venv created successfully", flush=True)
        return True

    except Exception as e:
        print(f"[AgeDetection] Failed to create venv: {e}", flush=True)
        return False


def install_age_detection_deps_sync():
    """Synchronous function to install age detection dependencies into a venv."""
    try:
        set_setting(AGE_DETECTION_INSTALLING, "true")
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Starting installation...")

        is_windows = sys.platform == "win32"
        is_macos = sys.platform == "darwin"

        # On Windows, check for VC++ Redistributable (required for PyTorch)
        if is_windows:
            try:
                import ctypes
                try:
                    ctypes.CDLL("vcruntime140.dll")
                except OSError:
                    set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installing Visual C++ Redistributable...")
                    print("[AgeDetection] Installing VC++ Redistributable...", flush=True)

                    import urllib.request
                    import tempfile

                    vc_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
                    vc_path = Path(tempfile.gettempdir()) / "vc_redist.x64.exe"

                    urllib.request.urlretrieve(vc_url, vc_path)

                    result = subprocess.run(
                        [str(vc_path), "/install", "/quiet", "/norestart"],
                        capture_output=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        print("[AgeDetection] VC++ Redistributable installed", flush=True)
                    else:
                        print(f"[AgeDetection] VC++ install returned {result.returncode}", flush=True)

                    try:
                        vc_path.unlink()
                    except:
                        pass
            except Exception as e:
                print(f"[AgeDetection] VC++ check/install error: {e}", flush=True)

        # Create venv if needed
        if not create_venv_if_needed():
            set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Failed to create virtual environment")
            return

        venv_python = get_venv_python()
        venv_site_packages = get_venv_site_packages()

        # Add venv to path so we can check for already-installed packages
        ensure_packages_in_path()

        # Install packages one by one for progress tracking
        # Using venv means pip will auto-select the correct wheel for the platform/Python version
        packages = [
            ("numpy", "numpy<2"),  # numpy<2 required for insightface compatibility
            ("torch", "torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
            ("transformers", "transformers"),
            ("ultralytics", "ultralytics"),
            ("timm", "timm"),
            ("mivolo", "https://github.com/WildChlamydia/MiVOLO/archive/refs/heads/main.zip --no-deps"),
            # insightface - pip will auto-select correct wheel for platform/Python version
            # On Windows/Linux x86_64: uses pre-built wheels from PyPI
            # On macOS/ARM: may compile from source or use available wheel
            ("insightface", "insightface"),
        ]

        for name, package in packages:
            # Check if already installed (reload sys.path first)
            ensure_packages_in_path()
            try:
                __import__(name)
                print(f"[AgeDetection] {name} already installed, skipping", flush=True)
                continue
            except OSError:
                # OSError = installed but DLL issues (missing VC++ etc)
                print(f"[AgeDetection] {name} installed but has DLL issues, skipping", flush=True)
                continue
            except ImportError:
                pass

            set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Installing {name}...")
            print(f"[AgeDetection] Installing {name} to venv...", flush=True)

            try:
                # Install using venv's pip (auto-selects correct wheels)
                cmd = [str(venv_python), "-m", "pip", "install"] + package.split()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout per package
                )

                if result.returncode != 0:
                    print(f"[AgeDetection] Failed to install {name}: {result.stderr}", flush=True)
                    # Don't fail completely for optional packages like insightface
                    if name == "insightface":
                        print(f"[AgeDetection] InsightFace optional, continuing with OpenCV fallback", flush=True)
                    else:
                        set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Failed to install {name}")
                else:
                    print(f"[AgeDetection] Installed {name}", flush=True)

            except subprocess.TimeoutExpired:
                print(f"[AgeDetection] Timeout installing {name}", flush=True)
                if name != "insightface":
                    set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Timeout installing {name}")
            except Exception as e:
                print(f"[AgeDetection] Error installing {name}: {e}", flush=True)

        # Apply runtime patches for mivolo/timm compatibility
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Applying compatibility patches...")
        patch_mivolo_for_timm_compat(venv_site_packages)

        # Reload path and check final status
        ensure_packages_in_path()
        deps = check_age_detection_deps()
        required = ["torch", "transformers", "ultralytics", "timm", "mivolo"]
        required_installed = all(deps.get(r, False) for r in required)

        if required_installed:
            set_setting(AGE_DETECTION_INSTALLED, "true")
            if deps.get("insightface", False):
                set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installation complete!")
            else:
                set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installation complete (using OpenCV fallback)")
        else:
            missing = [k for k in required if not deps.get(k, False)]
            set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Some packages failed: {', '.join(missing)}")

    except Exception as e:
        print(f"[AgeDetection] Installation error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Error: {str(e)}")
    finally:
        set_setting(AGE_DETECTION_INSTALLING, "false")


@router.post("/age-detection/install")
async def install_age_detection():
    """Start installing age detection dependencies (runs in background)"""
    import threading

    # Check if already installing
    if get_setting(AGE_DETECTION_INSTALLING, "false") == "true":
        return {"success": False, "error": "Installation already in progress"}

    # Check if required deps already installed
    if are_required_deps_installed():
        return {"success": True, "message": "Required dependencies already installed"}

    # Set installing flag BEFORE starting thread (avoid race condition)
    set_setting(AGE_DETECTION_INSTALLING, "true")
    set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Starting installation...")

    # Start background installation in a thread
    thread = threading.Thread(target=install_age_detection_deps_sync, daemon=True)
    thread.start()

    return {
        "success": True,
        "installing": True,
        "message": "Installation started in background. This may take several minutes."
    }


@router.get("/age-detection/status")
async def get_age_detection_status():
    """Get current age detection installation status"""
    try:
        deps = check_age_detection_deps()

        return {
            "enabled": get_setting(AGE_DETECTION_ENABLED, "false") == "true",
            "installed": are_required_deps_installed(),
            "installing": get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
            "progress": get_setting(AGE_DETECTION_INSTALL_PROGRESS, ""),
            "dependencies": deps
        }
    except Exception as e:
        import traceback
        print(f"[Settings] Error in age-detection/status: {e}")
        traceback.print_exc()
        # Return safe defaults so app remains usable
        return {
            "enabled": False,
            "installed": False,
            "installing": False,
            "progress": f"Error checking status: {str(e)}",
            "dependencies": {}
        }


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models")
async def get_models_status():
    """Get status of all ML models"""
    from ..services.model_downloader import get_all_models_status
    return {
        "models": get_all_models_status()
    }


class ModelDownloadRequest(BaseModel):
    model_name: str


@router.post("/models/download")
async def download_model(request: ModelDownloadRequest):
    """Start downloading a model"""
    import asyncio
    from ..services.model_downloader import (
        MODELS, is_model_available, download_model as do_download,
        get_download_progress
    )

    model_name = request.model_name

    if model_name not in MODELS:
        return {"success": False, "error": f"Unknown model: {model_name}"}

    if is_model_available(model_name):
        return {"success": True, "message": "Model already available"}

    # Check if already downloading
    progress = get_download_progress(model_name)
    if progress and progress.get("status") == "downloading":
        return {
            "success": True,
            "message": "Download already in progress",
            "progress": progress
        }

    # Start download in background
    async def download_task():
        try:
            await do_download(model_name)
        except Exception as e:
            print(f"[Models] Download failed for {model_name}: {e}")

    asyncio.create_task(download_task())

    return {
        "success": True,
        "message": "Download started",
        "model": model_name
    }


@router.get("/models/{model_name}/progress")
async def get_model_progress(model_name: str):
    """Get download progress for a model"""
    from ..services.model_downloader import (
        get_download_progress, is_model_available
    )

    # URL decode the model name (slashes are encoded)
    model_name = model_name.replace("%2F", "/")

    if is_model_available(model_name):
        return {
            "status": "complete",
            "available": True,
            "percent": 100
        }

    progress = get_download_progress(model_name)
    if progress:
        return progress

    return {
        "status": "not_started",
        "available": False,
        "percent": 0
    }


# =============================================================================
# Data Migration Endpoints
# =============================================================================

# Track ongoing migration state
_migration_state = {
    "running": False,
    "progress": None,
    "result": None
}


@router.get("/migration")
async def get_migration_info_endpoint():
    """Get information about current mode and migration options."""
    from ..migration import get_migration_info
    info = await get_migration_info()

    # Include current migration state
    info["migration_running"] = _migration_state["running"]
    if _migration_state["progress"]:
        info["migration_progress"] = {
            "phase": _migration_state["progress"].phase,
            "percent": _migration_state["progress"].percent,
            "current_file": _migration_state["progress"].current_file,
            "files_copied": _migration_state["progress"].files_copied,
            "total_files": _migration_state["progress"].total_files,
        }
    if _migration_state["result"]:
        info["last_result"] = {
            "success": _migration_state["result"].success,
            "error": _migration_state["result"].error,
            "files_copied": _migration_state["result"].files_copied,
            "bytes_copied": _migration_state["result"].bytes_copied,
        }

    return info


class MigrationRequest(BaseModel):
    mode: str  # "system_to_portable" or "portable_to_system"
    directory_ids: Optional[list[int]] = None  # Selective migration: which watch directories to include


@router.get("/migration/directories")
async def get_migration_directories(mode: str):
    """Get watch directories available for selective migration.

    Returns list of directories with metadata (path, image count, size).
    """
    from ..migration import get_watch_directories_for_migration, MigrationMode

    try:
        migration_mode = MigrationMode(mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}. Must be 'system_to_portable' or 'portable_to_system'",
            "directories": []
        }

    directories = await get_watch_directories_for_migration(migration_mode)

    return {
        "success": True,
        "directories": directories,
        "total_count": len(directories),
        "total_images": sum(d["image_count"] for d in directories),
        "total_thumbnail_size": sum(d["thumbnail_size"] for d in directories)
    }


@router.post("/migration/validate")
async def validate_migration(request: MigrationRequest):
    """Validate migration can proceed (dry run).

    If directory_ids is provided, validates selective migration.
    If directory_ids is None or empty, validates full migration.
    """
    from ..migration import (
        migrate_data, migrate_data_selective, MigrationMode,
        get_migration_paths, calculate_selective_migration_size
    )

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "valid": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Use selective migration if directory_ids provided
    if request.directory_ids is not None and len(request.directory_ids) > 0:
        result = await migrate_data_selective(mode, request.directory_ids, dry_run=True)

        # Also get image count for selected directories
        try:
            source, _ = get_migration_paths(mode)
            _, _, thumb_bytes = calculate_selective_migration_size(source, request.directory_ids)
        except:
            thumb_bytes = 0

        return {
            "valid": result.success,
            "error": result.error,
            "source_path": result.source_path,
            "dest_path": result.dest_path,
            "files_to_copy": result.files_copied,
            "bytes_to_copy": result.bytes_copied,
            "size_mb": round(result.bytes_copied / 1024 / 1024, 1) if result.bytes_copied else 0,
            "thumbnail_size_mb": round(thumb_bytes / 1024 / 1024, 1) if thumb_bytes else 0,
            "selective": True,
            "directory_count": len(request.directory_ids)
        }
    else:
        result = await migrate_data(mode, dry_run=True)

        return {
            "valid": result.success,
            "error": result.error,
            "source_path": result.source_path,
            "dest_path": result.dest_path,
            "files_to_copy": result.files_copied,
            "bytes_to_copy": result.bytes_copied,
            "size_mb": round(result.bytes_copied / 1024 / 1024, 1) if result.bytes_copied else 0,
            "selective": False
        }


@router.post("/migration/start")
async def start_migration(request: MigrationRequest):
    """Start data migration (runs in background).

    If directory_ids is provided, performs selective migration.
    If directory_ids is None or empty, performs full migration.
    """
    import asyncio
    from ..migration import migrate_data, migrate_data_selective, MigrationMode
    from ..services.events import migration_events, MigrationEventType

    if _migration_state["running"]:
        return {"success": False, "error": "Migration already in progress"}

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Determine if selective migration
    is_selective = request.directory_ids is not None and len(request.directory_ids) > 0
    directory_ids = request.directory_ids if is_selective else []

    # First validate
    if is_selective:
        validation = await migrate_data_selective(mode, directory_ids, dry_run=True)
    else:
        validation = await migrate_data(mode, dry_run=True)

    if not validation.success:
        return {"success": False, "error": validation.error}

    # Reset state
    _migration_state["running"] = True
    _migration_state["progress"] = None
    _migration_state["result"] = None

    def progress_callback(progress):
        _migration_state["progress"] = progress
        # Broadcast progress via SSE (fire and forget)
        asyncio.create_task(migration_events.broadcast(
            MigrationEventType.PROGRESS,
            {
                "phase": progress.phase,
                "percent": round(progress.percent, 1),
                "current_file": progress.current_file,
                "files_copied": progress.files_copied,
                "total_files": progress.total_files,
                "bytes_copied": progress.bytes_copied,
                "total_bytes": progress.total_bytes,
                "error": progress.error
            }
        ))

    async def run_migration():
        try:
            # Broadcast start event
            await migration_events.broadcast(MigrationEventType.STARTED, {
                "mode": mode.value,
                "selective": is_selective,
                "directory_count": len(directory_ids) if is_selective else None
            })

            if is_selective:
                result = await migrate_data_selective(mode, directory_ids, progress_callback=progress_callback)
            else:
                result = await migrate_data(mode, progress_callback=progress_callback)

            _migration_state["result"] = result

            # Broadcast completion/error event
            if result.success:
                await migration_events.broadcast(MigrationEventType.COMPLETED, {
                    "files_copied": result.files_copied,
                    "bytes_copied": result.bytes_copied,
                    "source_path": result.source_path,
                    "dest_path": result.dest_path,
                    "selective": is_selective
                })
            else:
                await migration_events.broadcast(MigrationEventType.ERROR, {
                    "error": result.error,
                    "files_copied": result.files_copied
                })
        except Exception as e:
            from ..migration import MigrationResult
            _migration_state["result"] = MigrationResult(
                success=False,
                mode=mode,
                source_path="",
                dest_path="",
                files_copied=0,
                bytes_copied=0,
                error=str(e)
            )
            await migration_events.broadcast(MigrationEventType.ERROR, {"error": str(e)})
        finally:
            _migration_state["running"] = False

    # Start background task
    asyncio.create_task(run_migration())

    return {
        "success": True,
        "selective": is_selective,
        "directory_count": len(directory_ids) if is_selective else None,
        "message": "Migration started. Subscribe to /api/settings/migration/events for real-time progress."
    }


@router.get("/migration/status")
async def get_migration_status():
    """Get current migration progress."""
    from ..migration import ImportResult

    response = {
        "running": _migration_state["running"],
        "progress": None,
        "result": None
    }

    if _migration_state["progress"]:
        p = _migration_state["progress"]
        response["progress"] = {
            "phase": p.phase,
            "percent": round(p.percent, 1),
            "current_file": p.current_file,
            "files_copied": p.files_copied,
            "total_files": p.total_files,
            "bytes_copied": p.bytes_copied,
            "total_bytes": p.total_bytes,
            "error": p.error
        }

    if _migration_state["result"]:
        r = _migration_state["result"]
        result_data = {
            "success": r.success,
            "mode": r.mode.value if hasattr(r.mode, 'value') else r.mode,
            "source_path": r.source_path,
            "dest_path": r.dest_path,
            "files_copied": r.files_copied,
            "bytes_copied": r.bytes_copied,
            "error": r.error
        }

        # Add import-specific fields if this was an import operation
        if isinstance(r, ImportResult):
            result_data["import"] = True
            result_data["directories_imported"] = r.directories_imported
            result_data["images_imported"] = r.images_imported
            result_data["images_skipped"] = r.images_skipped
            result_data["tags_created"] = r.tags_created
            result_data["tags_reused"] = r.tags_reused
        else:
            result_data["import"] = False

        response["result"] = result_data

    return response


@router.post("/migration/cleanup")
async def cleanup_migration(request: MigrationRequest):
    """Clean up partially copied data from a failed migration.

    Use this if migration failed and you want to remove incomplete data
    from the destination before retrying.
    """
    from ..migration import cleanup_partial_migration, get_migration_paths, MigrationMode
    from pathlib import Path

    if _migration_state["running"]:
        return {"success": False, "error": "Migration is currently running"}

    try:
        mode = MigrationMode(request.mode)
        _, dest = get_migration_paths(mode)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    success, message = cleanup_partial_migration(dest)
    return {"success": success, "message": message}


@router.post("/migration/delete-source")
async def delete_migration_source(request: MigrationRequest):
    """Delete source data after successful migration.

    WARNING: This permanently deletes data. Only use after verifying
    migration completed successfully.
    """
    from ..migration import delete_source_data, verify_migration, MigrationMode

    if _migration_state["running"]:
        return {"success": False, "error": "Migration is currently running"}

    try:
        mode = MigrationMode(request.mode)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # First verify migration succeeded
    verified, issues = await verify_migration(mode)
    if not verified:
        return {
            "success": False,
            "error": "Migration verification failed. Cannot delete source.",
            "issues": issues
        }

    success, message = await delete_source_data(mode)
    return {"success": success, "message": message}


@router.post("/migration/verify")
async def verify_migration_endpoint(request: MigrationRequest):
    """Verify that migration completed successfully."""
    from ..migration import verify_migration, MigrationMode

    try:
        mode = MigrationMode(request.mode)
    except ValueError as e:
        return {"success": False, "error": str(e), "issues": []}

    success, issues = await verify_migration(mode)
    return {"success": success, "issues": issues}


@router.get("/migration/events")
async def migration_events_stream():
    """Server-Sent Events stream for real-time migration progress.

    Events:
    - migration_started: Migration has begun
    - migration_progress: Progress update (phase, percent, current_file, etc.)
    - migration_completed: Migration finished successfully
    - migration_error: Migration failed with error
    """
    from fastapi.responses import StreamingResponse
    from ..services.events import migration_events

    async def event_generator():
        # Send initial connection message
        yield "data: {\"type\": \"connected\"}\n\n"
        # Stream events
        async for event in migration_events.subscribe():
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# Directory Import Endpoints (import into existing database)
# =============================================================================

@router.post("/migration/import/validate")
async def validate_import_endpoint(request: MigrationRequest):
    """Validate import can proceed (dry run).

    Import differs from migration:
    - Import ADDS directories to an existing destination database
    - Migration REQUIRES an empty destination

    directory_ids is required for import.
    """
    from ..migration import (
        import_directories, validate_import, calculate_import_size,
        MigrationMode, get_migration_paths
    )

    if not request.directory_ids or len(request.directory_ids) == 0:
        return {
            "valid": False,
            "error": "directory_ids is required for import"
        }

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "valid": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Validate import
    errors = validate_import(mode, request.directory_ids)
    if errors:
        return {
            "valid": False,
            "error": "; ".join(errors)
        }

    # Calculate size
    try:
        source, dest = get_migration_paths(mode)
        total_files, total_bytes, images_to_import, images_to_skip, total_db_records, total_tags = calculate_import_size(
            source, request.directory_ids, dest
        )
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    return {
        "valid": True,
        "error": None,
        "source_path": str(source),
        "dest_path": str(dest),
        "files_to_copy": total_files,
        "bytes_to_copy": total_bytes,
        "size_mb": round(total_bytes / 1024 / 1024, 1) if total_bytes else 0,
        "images_to_import": images_to_import,
        "images_to_skip": images_to_skip,
        "directory_count": len(request.directory_ids),
        "total_db_records": total_db_records,
        "total_tags": total_tags
    }


@router.post("/migration/import/start")
async def start_import(request: MigrationRequest):
    """Start directory import (runs in background).

    Import differs from migration:
    - Import ADDS directories to an existing destination database
    - Migration REQUIRES an empty destination

    directory_ids is required for import.
    """
    import asyncio
    from ..migration import import_directories, validate_import, MigrationMode, ImportResult
    from ..services.events import migration_events, MigrationEventType

    if _migration_state["running"]:
        return {"success": False, "error": "Migration/import already in progress"}

    if not request.directory_ids or len(request.directory_ids) == 0:
        return {
            "success": False,
            "error": "directory_ids is required for import"
        }

    try:
        mode = MigrationMode(request.mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: {request.mode}. Must be 'system_to_portable' or 'portable_to_system'"
        }

    # Validate
    errors = validate_import(mode, request.directory_ids)
    if errors:
        return {"success": False, "error": "; ".join(errors)}

    directory_ids = request.directory_ids

    # Reset state
    _migration_state["running"] = True
    _migration_state["progress"] = None
    _migration_state["result"] = None

    def progress_callback(progress):
        _migration_state["progress"] = progress
        # Broadcast progress via SSE
        asyncio.create_task(migration_events.broadcast(
            MigrationEventType.PROGRESS,
            {
                "phase": progress.phase,
                "percent": round(progress.percent, 1),
                "current_file": progress.current_file,
                "files_copied": progress.files_copied,
                "total_files": progress.total_files,
                "bytes_copied": progress.bytes_copied,
                "total_bytes": progress.total_bytes,
                "error": progress.error
            }
        ))

    async def run_import():
        try:
            # Broadcast start event
            await migration_events.broadcast(MigrationEventType.STARTED, {
                "mode": mode.value,
                "import": True,
                "directory_count": len(directory_ids)
            })

            result = await import_directories(mode, directory_ids, progress_callback=progress_callback)
            _migration_state["result"] = result

            # Broadcast completion/error event
            if result.success:
                await migration_events.broadcast(MigrationEventType.COMPLETED, {
                    "import": True,
                    "directories_imported": result.directories_imported,
                    "images_imported": result.images_imported,
                    "images_skipped": result.images_skipped,
                    "tags_created": result.tags_created,
                    "tags_reused": result.tags_reused,
                    "files_copied": result.files_copied,
                    "bytes_copied": result.bytes_copied,
                    "source_path": result.source_path,
                    "dest_path": result.dest_path
                })
            else:
                await migration_events.broadcast(MigrationEventType.ERROR, {
                    "error": result.error,
                    "files_copied": result.files_copied
                })
        except Exception as e:
            _migration_state["result"] = ImportResult(
                success=False,
                mode=mode,
                source_path="",
                dest_path="",
                directories_imported=0,
                images_imported=0,
                images_skipped=0,
                tags_created=0,
                tags_reused=0,
                files_copied=0,
                bytes_copied=0,
                error=str(e)
            )
            await migration_events.broadcast(MigrationEventType.ERROR, {"error": str(e)})
        finally:
            _migration_state["running"] = False

    # Start background task
    asyncio.create_task(run_import())

    return {
        "success": True,
        "import": True,
        "directory_count": len(directory_ids),
        "message": "Import started. Subscribe to /api/settings/migration/events for real-time progress."
    }


# =============================================================================
# Optical Flow Interpolation Endpoints
# =============================================================================

@router.get("/optical-flow")
async def get_optical_flow_config():
    """Get optical flow interpolation configuration and backend status"""
    import traceback
    import logging
    logger = logging.getLogger(__name__)

    config = get_optical_flow_settings()

    try:
        from ..services.optical_flow import get_backend_status
        backend = get_backend_status()
    except Exception as e:
        logger.error(f"Failed to get backend status: {e}\n{traceback.format_exc()}")
        backend = {"error": str(e), "any_backend_available": False}

    return {
        **config,
        "backend": backend
    }


class OpticalFlowConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    target_fps: Optional[int] = None
    use_gpu: Optional[bool] = None
    quality: Optional[str] = None  # svp, gpu_native, realtime, fast, balanced, quality


@router.post("/optical-flow")
async def update_optical_flow_config(config: OpticalFlowConfigUpdate):
    """Update optical flow interpolation configuration"""
    current = get_optical_flow_settings()

    # Update only provided fields
    if config.enabled is not None:
        current["enabled"] = config.enabled
    if config.target_fps is not None:
        # Clamp to valid range
        current["target_fps"] = max(15, min(120, config.target_fps))
    if config.use_gpu is not None:
        current["use_gpu"] = config.use_gpu
    if config.quality is not None:
        # Validate quality preset
        if config.quality in ("svp", "gpu_native", "realtime", "fast", "balanced", "quality"):
            current["quality"] = config.quality

    save_optical_flow_settings(current)

    return {"success": True, **current}


class InterpolationPlayRequest(BaseModel):
    file_path: str
    start_position: float = 0.0  # Seek position in seconds
    quality_preset: Optional[str] = None  # Quality preset name


@router.post("/optical-flow/play")
async def play_video_interpolated(request: InterpolationPlayRequest):
    """
    Start interpolated video stream via HLS.

    Returns the stream URL that can be used with hls.js.
    """
    from ..services.optical_flow_stream import create_interpolated_stream
    from ..services.optical_flow import get_backend_status

    config = get_optical_flow_settings()
    backend = get_backend_status()

    if not config["enabled"]:
        return {"success": False, "error": "Optical flow interpolation is not enabled"}

    if not backend["any_backend_available"]:
        return {"success": False, "error": "No interpolation backend available. Install OpenCV or PyTorch."}

    # Check if file exists
    if not os.path.exists(request.file_path):
        return {"success": False, "error": "File not found"}

    try:
        # Parse quality preset
        import subprocess
        # Get source video resolution using ffprobe
        source_resolution = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                request.file_path
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    source_resolution = (int(parts[0]), int(parts[1]))
        except Exception as e:
            print(f"[OpticalFlow] Could not detect source resolution: {e}")

        quality_settings = parse_quality_preset(request.quality_preset, source_resolution)

        stream = await create_interpolated_stream(
            video_path=request.file_path,
            target_fps=config["target_fps"],
            use_gpu=config["use_gpu"] and backend["cuda_available"],
            quality=config.get("quality", "fast"),
            wait_for_buffer=True,
            min_segments=2,
            target_bitrate=quality_settings["bitrate"],
            target_resolution=quality_settings["resolution"],
            start_position=request.start_position,
        )

        if stream:
            return {
                "success": True,
                "stream_id": stream.stream_id,
                "stream_url": f"/api/settings/optical-flow/stream/{stream.stream_id}/stream.m3u8",
                "source_resolution": {"width": source_resolution[0], "height": source_resolution[1]} if source_resolution else None,
                "message": f"Interpolated stream started at {config['target_fps']} fps"
            }
        else:
            return {"success": False, "error": "Failed to start interpolated stream"}

    except Exception as e:
        return {"success": False, "error": f"Stream error: {str(e)}"}


@router.post("/optical-flow/stop")
async def stop_interpolated_stream():
    """Stop the active interpolated stream."""
    from ..services.optical_flow_stream import stop_all_streams

    stop_all_streams()
    return {"success": True, "message": "Stream stopped"}


from fastapi.responses import FileResponse, Response


@router.get("/optical-flow/stream/{stream_id}/{filename:path}")
async def serve_hls_file(stream_id: str, filename: str):
    """Serve HLS playlist or segment files for the interpolated stream."""
    from ..services.optical_flow_stream import get_active_stream

    stream = get_active_stream(stream_id)
    if not stream:
        return Response(content="Stream not found", status_code=404)

    file_path = stream.get_file_path(filename)
    if not file_path:
        return Response(content="File not found", status_code=404)

    # Determine content type
    if filename.endswith('.m3u8'):
        media_type = 'application/vnd.apple.mpegurl'
    elif filename.endswith('.ts'):
        media_type = 'video/mp2t'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Access-Control-Allow-Origin': '*'
        }
    )


# =============================================================================
# SVP (SmoothVideo Project) Interpolation Endpoints
# =============================================================================

@router.get("/svp")
async def get_svp_config():
    """Get SVP interpolation configuration and availability status"""
    from ..services.svp_stream import get_svp_status, SVP_PRESETS, SVP_ALGORITHMS, SVP_BLOCK_SIZES, SVP_PEL_OPTIONS, SVP_MASK_AREA

    config = get_svp_settings()
    status = get_svp_status()

    return {
        **config,
        "status": status,
        "presets": {
            name: {"name": preset["name"], "description": preset["description"]}
            for name, preset in SVP_PRESETS.items()
        },
        "options": {
            "algorithms": SVP_ALGORITHMS,
            "block_sizes": SVP_BLOCK_SIZES,
            "pel_options": SVP_PEL_OPTIONS,
            "mask_area": SVP_MASK_AREA,
        }
    }


class SVPConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    target_fps: Optional[int] = None
    preset: Optional[str] = None  # fast, balanced, quality, max, animation, film
    use_nvof: Optional[bool] = None  # Use NVIDIA Optical Flow
    shader: Optional[int] = None  # SVP shader/algo (1,2,11,13,21,23)
    artifact_masking: Optional[int] = None  # Artifact masking area (0-200)
    frame_interpolation: Optional[int] = None  # Frame interpolation mode (1=uniform, 2=adaptive)
    custom_super: Optional[str] = None
    custom_analyse: Optional[str] = None
    custom_smooth: Optional[str] = None


@router.post("/svp")
async def update_svp_config(config: SVPConfigUpdate):
    """Update SVP interpolation configuration"""
    from ..services.svp_stream import SVP_PRESETS

    current = get_svp_settings()

    # Update only provided fields
    if config.enabled is not None:
        current["enabled"] = config.enabled
    if config.target_fps is not None:
        # Clamp to valid range
        current["target_fps"] = max(15, min(144, config.target_fps))
    if config.preset is not None:
        # Validate preset
        if config.preset in SVP_PRESETS:
            current["preset"] = config.preset
    if config.use_nvof is not None:
        current["use_nvof"] = config.use_nvof
    if config.shader is not None:
        # Validate shader value
        if config.shader in [1, 2, 11, 13, 21, 23]:
            current["shader"] = config.shader
    if config.artifact_masking is not None:
        # Clamp to valid range
        current["artifact_masking"] = max(0, min(200, config.artifact_masking))
    if config.frame_interpolation is not None:
        if config.frame_interpolation in [1, 2]:
            current["frame_interpolation"] = config.frame_interpolation
    if config.custom_super is not None:
        current["custom_super"] = config.custom_super if config.custom_super else None
    if config.custom_analyse is not None:
        current["custom_analyse"] = config.custom_analyse if config.custom_analyse else None
    if config.custom_smooth is not None:
        current["custom_smooth"] = config.custom_smooth if config.custom_smooth else None

    save_svp_settings(current)

    return {"success": True, **current}


# Quality preset definitions for streaming
QUALITY_PRESETS = {
    "original": {"bitrate": None, "resolution": None},
    "1440p": {"bitrate": "30M", "resolution": (2560, 1440)},
    "1080p_enhanced": {"bitrate": "20M", "resolution": (1920, 1080)},
    "1080p": {"bitrate": "12M", "resolution": (1920, 1080)},
    "720p": {"bitrate": "8M", "resolution": (1280, 720)},
    "480p": {"bitrate": "4M", "resolution": (854, 480)},
}


def parse_quality_preset(quality_preset, source_resolution=None):
    """Parse quality preset and return bitrate and resolution settings.

    Args:
        quality_preset: Preset name (e.g., '720p_3mbps') or None for original
        source_resolution: Tuple of (width, height) for source video

    Returns:
        Dict with 'bitrate' and 'resolution' keys (either can be None)
    """
    if not quality_preset or quality_preset == 'original' or quality_preset not in QUALITY_PRESETS:
        return {"bitrate": None, "resolution": None}

    preset = QUALITY_PRESETS[quality_preset]
    bitrate = preset["bitrate"]
    resolution = preset["resolution"]

    # Don't upscale - if source resolution is smaller than preset, skip
    if source_resolution and resolution:
        src_width, src_height = source_resolution
        preset_width, preset_height = resolution
        # Check if source is smaller than preset (upscaling would be needed)
        if src_width < preset_width or src_height < preset_height:
            # Return original quality for sources smaller than preset
            return {"bitrate": None, "resolution": None}

    return {"bitrate": bitrate, "resolution": resolution}


class SVPPlayRequest(BaseModel):
    file_path: str
    start_position: float = 0.0  # Seek position in seconds
    quality_preset: Optional[str] = None  # Quality preset name


@router.post("/svp/play")
async def play_video_svp(request: SVPPlayRequest):
    """
    Start SVP-interpolated video stream via HLS.

    SVP uses VapourSynth + SVPflow plugins for high-quality
    motion-compensated frame interpolation.
    """
    from ..services.svp_stream import SVPStream, get_svp_status, stop_all_svp_streams

    # Stop any existing SVP streams before starting a new one
    stop_all_svp_streams()

    config = get_svp_settings()
    status = get_svp_status()

    # Note: We don't check config["enabled"] here - the toggle button should work
    # regardless of the global setting. We allow users to try SVP even if detection
    # fails - they'll get a runtime error if it's actually missing. This prevents
    # detection from being a blocker.

    # Check if file exists
    if not os.path.exists(request.file_path):
        return {"success": False, "error": "File not found"}

    try:
        # Parse quality preset
        import subprocess
        # Get source video resolution using ffprobe
        source_resolution = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                request.file_path
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    source_resolution = (int(parts[0]), int(parts[1]))
        except Exception as e:
            print(f"[SVP] Could not detect source resolution: {e}")

        quality_settings = parse_quality_preset(request.quality_preset, source_resolution)
        print(f"[SVP] Quality preset: {request.quality_preset}, source: {source_resolution}, settings: {quality_settings}")

        # Create SVP stream
        stream = SVPStream(
            video_path=request.file_path,
            target_fps=config["target_fps"],
            preset=config.get("preset", "balanced"),
            use_nvof=config.get("use_nvof", True),
            shader=config.get("shader", 23),
            artifact_masking=config.get("artifact_masking", 100),
            frame_interpolation=config.get("frame_interpolation", 2),
            custom_super=config.get("custom_super"),
            custom_analyse=config.get("custom_analyse"),
            custom_smooth=config.get("custom_smooth"),
            target_bitrate=quality_settings["bitrate"],
            target_resolution=quality_settings["resolution"],
            start_position=request.start_position,
        )

        # Start the stream
        success = await stream.start()

        if success:
            # Wait briefly for initial buffer, but return optimistically
            # Let the frontend HLS.js handle retry/buffering for large files
            import asyncio
            for _ in range(50):  # Wait up to 5 seconds
                if stream.playlist_ready:
                    break
                if stream.error:
                    return {"success": False, "error": stream.error}
                if not stream._running:
                    return {"success": False, "error": stream.error or "Pipeline failed to start"}
                await asyncio.sleep(0.1)

            # Return success - stream is running, frontend will handle buffering
            return {
                "success": True,
                "stream_id": stream.stream_id,
                "stream_url": f"/api/settings/svp/stream/{stream.stream_id}/stream.m3u8",
                "duration": stream._duration,
                "start_position": request.start_position,
                "source_resolution": {"width": source_resolution[0], "height": source_resolution[1]} if source_resolution else None,
                "message": f"SVP stream started at {config['target_fps']} fps with {config.get('preset', 'balanced')} preset"
            }
        else:
            return {"success": False, "error": stream.error or "Failed to start SVP stream"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"SVP stream error: {str(e)}"}


@router.post("/svp/stop")
async def stop_svp_stream():
    """Stop all active SVP streams."""
    from ..services.svp_stream import stop_all_svp_streams

    stop_all_svp_streams()
    return {"success": True, "message": "SVP streams stopped"}


# Transcoding endpoints (fallback when SVP/OpticalFlow not available)
class TranscodePlayRequest(BaseModel):
    file_path: str
    start_position: float = 0.0
    quality_preset: Optional[str] = None


@router.post("/transcode/play")
async def play_video_transcode(request: TranscodePlayRequest):
    """
    Start simple HLS transcoding via FFmpeg only (no interpolation).

    This is a fallback when SVP or OpticalFlow aren't available but
    the user wants to change quality/bitrate.
    """
    from ..services.transcode_stream import TranscodeStream, stop_all_transcode_streams

    # Stop any existing transcode streams before starting a new one
    stop_all_transcode_streams()

    # Parse quality settings
    quality_settings = parse_quality_preset(request.quality_preset)

    try:
        # Detect source resolution
        source_resolution = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                request.file_path
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    source_resolution = (int(parts[0]), int(parts[1]))
        except Exception as e:
            print(f"[Transcode] Could not detect source resolution: {e}")

        quality_settings = parse_quality_preset(request.quality_preset, source_resolution)

        # Create transcode stream
        stream = TranscodeStream(
            video_path=request.file_path,
            target_bitrate=quality_settings["bitrate"],
            target_resolution=quality_settings["resolution"],
            start_position=request.start_position,
        )

        # Start the stream
        success = await stream.start()

        if success:
            # Wait briefly for initial buffer
            import asyncio
            for _ in range(50):  # Wait up to 5 seconds
                if stream.playlist_ready:
                    break
                if stream.error:
                    return {"success": False, "error": stream.error}
                if not stream._running:
                    return {"success": False, "error": stream.error or "Encoding failed"}
                await asyncio.sleep(0.1)

            # Return success
            return {
                "success": True,
                "stream_id": stream.stream_id,
                "stream_url": f"/api/settings/transcode/stream/{stream.stream_id}/playlist.m3u8",
                "duration": stream._duration,
                "start_position": request.start_position,
                "source_resolution": {"width": source_resolution[0], "height": source_resolution[1]} if source_resolution else None,
                "message": "Transcoding started"
            }
        else:
            return {"success": False, "error": stream.error or "Failed to start transcoding"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Transcoding error: {str(e)}"}


@router.post("/transcode/stop")
async def stop_transcode_stream():
    """Stop all active transcoding streams."""
    from ..services.transcode_stream import stop_all_transcode_streams

    stop_all_transcode_streams()
    return {"success": True, "message": "Transcode streams stopped"}


@router.get("/transcode/stream/{stream_id}/{filename:path}")
async def serve_transcode_hls_file(stream_id: str, filename: str):
    """Serve HLS playlist or segment files for transcoding stream."""
    from ..services.transcode_stream import get_active_transcode_stream

    stream = get_active_transcode_stream(stream_id)
    if not stream:
        return Response(content="Stream not found", status_code=404)

    if not stream.hls_dir:
        return Response(content="Stream not ready", status_code=404)

    file_path = stream.hls_dir / filename
    if not file_path.exists():
        return Response(content="File not found", status_code=404)

    # Determine content type
    if filename.endswith('.m3u8'):
        media_type = "application/vnd.apple.mpegurl"
    else:
        media_type = "video/mp2t"

    return FileResponse(file_path, media_type=media_type)


@router.get("/svp/stream/{stream_id}/{filename:path}")
async def serve_svp_hls_file(stream_id: str, filename: str):
    """Serve HLS playlist or segment files for the SVP stream."""
    from ..services.svp_stream import get_active_svp_stream, _active_svp_streams

    stream = get_active_svp_stream(stream_id)
    if not stream:
        print(f"[SVP] Stream {stream_id} not found. Active streams: {list(_active_svp_streams.keys())}")
        return Response(content="Stream not found", status_code=404)

    if not stream.hls_dir:
        return Response(content="Stream not ready", status_code=404)

    file_path = stream.hls_dir / filename
    if not file_path.exists():
        return Response(content="File not found", status_code=404)

    # Determine content type
    if filename.endswith('.m3u8'):
        media_type = 'application/vnd.apple.mpegurl'
    elif filename.endswith('.ts'):
        media_type = 'video/mp2t'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Access-Control-Allow-Origin': '*'
        }
    )


# =============================================================================
# Web Video SVP Endpoints (Browser Extension)
# =============================================================================

@router.post("/svp/web/play")
async def play_web_video_svp(url: str, quality: str = "best"):
    """
    Start SVP stream for a web video URL using yt-dlp.

    This endpoint:
    1. Checks if the URL is from a DRM-protected site
    2. Downloads the video via yt-dlp to a temp file
    3. Passes the temp file to the existing SVP pipeline
    4. Returns the HLS stream URL

    Args:
        url: Web video URL (YouTube, Vimeo, Twitch VOD, direct video, etc.)
        quality: Quality preference - "best", "1080p", "720p", "480p"

    Returns:
        On success: {"success": true, "stream_url": "...", "download_id": "..."}
        On pending: {"success": true, "status": "downloading", "download_id": "...", "progress": 0.5}
        On error: {"success": false, "error": "..."}
    """
    from ..services.web_video_downloader import (
        download_video,
        get_download,
        is_drm_site,
        is_live_stream,
    )
    from ..services.svp_stream import SVPStream, get_svp_status

    # Quick DRM check before starting download
    if is_drm_site(url):
        return {
            "success": False,
            "error": "This site uses DRM protection and cannot be processed",
            "drm_protected": True,
        }

    # Quick live stream check
    if is_live_stream(url):
        return {
            "success": False,
            "error": "Live streams are not supported yet (coming in v2)",
            "live_stream": True,
        }

    # Check SVP status before downloading
    config = get_svp_settings()
    status = get_svp_status()

    if not config["enabled"]:
        return {"success": False, "error": "SVP interpolation is not enabled"}

    if not status["ready"]:
        missing = []
        if not status["vapoursynth_available"]:
            missing.append("VapourSynth")
        if not status["svp_plugins_available"]:
            missing.append("SVPflow plugins")
        if not status["vspipe_available"]:
            missing.append("vspipe")
        return {"success": False, "error": f"SVP not ready. Missing: {', '.join(missing)}"}

    # Start or check download
    result = await download_video(url, quality)

    if not result.success:
        return {"success": False, "error": result.error}

    # Check download status
    download = get_download(result.download_id)
    if not download:
        return {"success": False, "error": "Download tracking error"}

    if download.status in ("pending", "downloading", "processing"):
        return {
            "success": True,
            "status": download.status,
            "download_id": download.download_id,
            "progress": download.progress,
        }

    if download.status == "error":
        return {"success": False, "error": download.error or "Download failed"}

    if download.status == "complete" and download.file_path:
        # Download complete, start SVP stream
        try:
            stream = SVPStream(
                video_path=download.file_path,
                target_fps=config["target_fps"],
                preset=config.get("preset", "balanced"),
                use_nvof=config.get("use_nvof", True),
                shader=config.get("shader", 23),
                artifact_masking=config.get("artifact_masking", 100),
                frame_interpolation=config.get("frame_interpolation", 2),
                custom_super=config.get("custom_super"),
                custom_analyse=config.get("custom_analyse"),
                custom_smooth=config.get("custom_smooth"),
            )

            success = await stream.start()

            if success:
                return {
                    "success": True,
                    "status": "streaming",
                    "download_id": download.download_id,
                    "stream_id": stream.stream_id,
                    "stream_url": f"/api/settings/svp/stream/{stream.stream_id}/stream.m3u8",
                    "duration": stream._duration,
                    "title": download.title,
                    "message": f"SVP stream started at {config['target_fps']} fps",
                }
            else:
                return {"success": False, "error": stream.error or "Failed to start SVP stream"}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"SVP stream error: {str(e)}"}

    return {"success": False, "error": "Unexpected download state"}


@router.get("/svp/web/status/{download_id}")
async def get_web_download_status(download_id: str):
    """
    Get download progress for a web video.

    Args:
        download_id: The download ID returned from /svp/web/play

    Returns:
        Download status including progress (0.0 to 1.0), status, and any errors.
    """
    from ..services.web_video_downloader import get_download

    download = get_download(download_id)
    if not download:
        return {"success": False, "error": "Download not found"}

    return {
        "success": True,
        "download_id": download.download_id,
        "status": download.status,
        "progress": download.progress,
        "title": download.title,
        "file_path": download.file_path,
        "error": download.error,
    }


@router.get("/svp/web/drm-check")
async def check_drm_site(url: str):
    """
    Check if a URL is from a DRM-protected site.

    This is a quick check that the extension can use before attempting to play.

    Args:
        url: The URL to check

    Returns:
        {"drm_protected": true/false, "live_stream": true/false/null}
    """
    from ..services.web_video_downloader import is_drm_site, is_live_stream

    return {
        "drm_protected": is_drm_site(url),
        "live_stream": is_live_stream(url),
    }


# Utility endpoints
@router.get("/util/dimensions")
async def get_file_dimensions(file_path: str):
    """
    Get dimensions for an image or video file.

    Used to fetch dimensions on-the-fly without storing in DB.
    """
    from ..services.importer import get_image_dimensions

    try:
        dims = get_image_dimensions(file_path)
        if dims:
            return {
                "success": True,
                "width": dims[0],
                "height": dims[1]
            }
        else:
            return {
                "success": False,
                "error": "Could not detect dimensions"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
