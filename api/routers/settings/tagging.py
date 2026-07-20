"""
Tagger/AI model settings endpoints - age detection, ML models.
"""
from fastapi import APIRouter
import subprocess
import sys
import os
from pathlib import Path

from ...database import get_data_dir
from .models import (
    get_setting,
    set_setting,
    AgeDetectionToggle,
    ModelDownloadRequest,
    AGE_DETECTION_ENABLED,
    AGE_DETECTION_INSTALLED,
    AGE_DETECTION_INSTALLING,
    AGE_DETECTION_INSTALL_PROGRESS,
)

router = APIRouter()


# =============================================================================
# Package/Environment Management
# =============================================================================

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


# =============================================================================
# Dependency Checking
# =============================================================================

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


# =============================================================================
# MiVOLO Patching
# =============================================================================

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


# =============================================================================
# Venv Creation
# =============================================================================

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


# =============================================================================
# Dependency Installation
# =============================================================================

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


# =============================================================================
# Age Detection Endpoints
# =============================================================================

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
    from ...services.model_downloader import get_all_models_status
    return {
        "models": get_all_models_status()
    }


@router.post("/models/download")
async def download_model(request: ModelDownloadRequest):
    """Start downloading a model"""
    import asyncio
    from ...services.model_downloader import (
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
    from ...services.model_downloader import (
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
