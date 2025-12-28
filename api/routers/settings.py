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
    deps = {
        "torch": False,
        "transformers": False,
        "ultralytics": False,
        "insightface": False
    }

    try:
        import torch
        deps["torch"] = True
    except ImportError:
        pass

    try:
        import transformers
        deps["transformers"] = True
    except ImportError:
        pass

    try:
        import ultralytics
        deps["ultralytics"] = True
    except ImportError:
        pass

    try:
        import insightface
        deps["insightface"] = True
    except ImportError:
        pass

    return deps


@router.get("")
async def get_all_settings():
    """Get all app settings"""
    deps = check_age_detection_deps()
    all_installed = all(deps.values())

    return {
        "age_detection": {
            "enabled": get_setting(AGE_DETECTION_ENABLED, "false") == "true",
            "installed": all_installed,
            "installing": get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
            "install_progress": get_setting(AGE_DETECTION_INSTALL_PROGRESS, ""),
            "dependencies": deps
        }
    }


class AgeDetectionToggle(BaseModel):
    enabled: bool


@router.post("/age-detection/toggle")
async def toggle_age_detection(data: AgeDetectionToggle):
    """Enable/disable age detection (requires dependencies to be installed)"""
    if data.enabled:
        deps = check_age_detection_deps()
        if not all(deps.values()):
            return {
                "success": False,
                "error": "Dependencies not installed. Install them first.",
                "dependencies": deps
            }

    set_setting(AGE_DETECTION_ENABLED, "true" if data.enabled else "false")
    return {"success": True, "enabled": data.enabled}


def install_age_detection_deps_sync():
    """Synchronous function to install age detection dependencies"""
    try:
        set_setting(AGE_DETECTION_INSTALLING, "true")
        set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Starting installation...")

        # Get Python executable
        python_exe = sys.executable

        # Install packages one by one for progress tracking
        packages = [
            ("torch", "torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
            ("transformers", "transformers"),
            ("ultralytics", "ultralytics"),
            ("insightface", "insightface")
        ]

        for name, package in packages:
            set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Installing {name}...")
            print(f"[AgeDetection] Installing {name}...", flush=True)

            try:
                # Use subprocess to install
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install"] + package.split(),
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

        # Check final status
        deps = check_age_detection_deps()
        if all(deps.values()):
            set_setting(AGE_DETECTION_INSTALLED, "true")
            set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installation complete!")
        else:
            missing = [k for k, v in deps.items() if not v]
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

    # Check if already installed
    deps = check_age_detection_deps()
    if all(deps.values()):
        return {"success": True, "message": "Dependencies already installed"}

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
        "installed": all(deps.values()),
        "installing": get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
        "progress": get_setting(AGE_DETECTION_INSTALL_PROGRESS, ""),
        "dependencies": deps
    }
