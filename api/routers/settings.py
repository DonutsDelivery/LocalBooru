"""
Settings router - app configuration and optional features
Uses JSON file for settings to avoid database migration issues
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
import os
import json
from pathlib import Path

from ..database import get_data_dir

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


def ensure_packages_in_path():
    """Add persistent packages directory to Python path"""
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


def check_age_detection_deps() -> dict:
    """Check if age detection dependencies are installed"""
    # Ensure persistent packages directory is in path
    ensure_packages_in_path()

    is_windows = sys.platform == "win32"

    deps = {
        "torch": False,
        "transformers": False,
        "ultralytics": False,
    }

    # Only include insightface on non-Windows (it's skipped on Windows)
    if not is_windows:
        deps["insightface"] = False

    # Catch OSError too - Windows throws this when VC++ redistributable is missing
    try:
        import torch
        deps["torch"] = True
    except (ImportError, OSError) as e:
        if isinstance(e, OSError):
            deps["torch_error"] = "Missing Visual C++ Redistributable. Install from: https://aka.ms/vs/17/release/vc_redist.x64.exe"

    try:
        import transformers
        deps["transformers"] = True
    except (ImportError, OSError):
        pass

    try:
        import ultralytics
        deps["ultralytics"] = True
    except (ImportError, OSError):
        pass

    if not is_windows:
        try:
            import insightface
            deps["insightface"] = True
        except (ImportError, OSError):
            pass

    return deps


def are_required_deps_installed() -> bool:
    """Check if required (non-optional) dependencies are installed"""
    deps = check_age_detection_deps()
    # insightface is optional - OpenCV fallback is available
    required = ["torch", "transformers", "ultralytics"]
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


def install_age_detection_deps_sync():
    """Synchronous function to install age detection dependencies"""
    try:
        set_setting(AGE_DETECTION_INSTALLING, "true")
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Starting installation...")

        # Get Python executable and persistent packages directory
        python_exe = sys.executable
        is_windows = sys.platform == "win32"
        packages_dir = get_packages_dir()

        # On Windows, install VC++ Redistributable if needed (required for PyTorch)
        if is_windows:
            try:
                # Check if VC++ is installed by trying to load a simple DLL
                import ctypes
                try:
                    ctypes.CDLL("vcruntime140.dll")
                except OSError:
                    set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installing Visual C++ Redistributable...")
                    print("[AgeDetection] Installing VC++ Redistributable...", flush=True)

                    import urllib.request
                    import tempfile

                    # Download VC++ redistributable
                    vc_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
                    vc_path = Path(tempfile.gettempdir()) / "vc_redist.x64.exe"

                    urllib.request.urlretrieve(vc_url, vc_path)

                    # Install silently
                    result = subprocess.run(
                        [str(vc_path), "/install", "/quiet", "/norestart"],
                        capture_output=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        print("[AgeDetection] VC++ Redistributable installed", flush=True)
                    else:
                        print(f"[AgeDetection] VC++ install returned {result.returncode}", flush=True)

                    # Clean up
                    try:
                        vc_path.unlink()
                    except:
                        pass
            except Exception as e:
                print(f"[AgeDetection] VC++ check/install error: {e}", flush=True)

        # Add packages dir to path so we can check for already-installed packages
        ensure_packages_in_path()

        # Install packages one by one for progress tracking
        # numpy<2 required for insightface compatibility
        packages = [
            ("numpy", "numpy<2"),
            ("torch", "torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
            ("transformers", "transformers"),
            ("ultralytics", "ultralytics"),
        ]

        # insightface is optional - OpenCV fallback works fine for face detection
        # Skip on Windows as it requires specific Python version wheels or C++ compiler
        if not is_windows:
            packages.append(("insightface", "insightface"))

        for name, package in packages:
            # Check if already installed
            try:
                __import__(name)
                print(f"[AgeDetection] {name} already installed, skipping", flush=True)
                continue
            except OSError:
                # OSError means package IS installed but has DLL issues (e.g. missing VC++)
                # Reinstalling won't help, skip it
                print(f"[AgeDetection] {name} installed but has DLL issues, skipping", flush=True)
                continue
            except ImportError:
                # Package not installed, proceed to install
                pass

            set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Installing {name}...")
            print(f"[AgeDetection] Installing {name} to {packages_dir}...", flush=True)

            try:
                # Install to persistent user directory (survives app updates)
                cmd = [python_exe, "-m", "pip", "install", "--target", str(packages_dir)] + package.split()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout per package
                )

                if result.returncode != 0:
                    print(f"[AgeDetection] Failed to install {name}: {result.stderr}", flush=True)
                    set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Failed to install {name}")
                else:
                    print(f"[AgeDetection] Installed {name}", flush=True)

            except subprocess.TimeoutExpired:
                print(f"[AgeDetection] Timeout installing {name}", flush=True)
                set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Timeout installing {name}")
            except Exception as e:
                print(f"[AgeDetection] Error installing {name}: {e}", flush=True)

        # Check final status - only required deps matter
        deps = check_age_detection_deps()
        required = ["torch", "transformers", "ultralytics"]
        required_installed = all(deps.get(r, False) for r in required)

        if required_installed:
            set_setting(AGE_DETECTION_INSTALLED, "true")
            if deps.get("insightface", False):
                set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installation complete!")
            else:
                # insightface is optional - OpenCV fallback works fine
                set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installation complete (using OpenCV fallback)")
        else:
            missing = [k for k in required if not deps.get(k, False)]
            set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Some packages failed: {', '.join(missing)}")

    except Exception as e:
        print(f"[AgeDetection] Installation error: {e}", flush=True)
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
    deps = check_age_detection_deps()

    return {
        "enabled": get_setting(AGE_DETECTION_ENABLED, "false") == "true",
        "installed": are_required_deps_installed(),
        "installing": get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
        "progress": get_setting(AGE_DETECTION_INSTALL_PROGRESS, ""),
        "dependencies": deps
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
