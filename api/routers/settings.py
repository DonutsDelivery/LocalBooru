"""
Settings router - app configuration and optional features
"""
from fastapi import APIRouter, BackgroundTasks
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
import os

from ..database import AsyncSessionLocal
from ..models import Settings

router = APIRouter()


# Settings keys
AGE_DETECTION_ENABLED = "age_detection_enabled"
AGE_DETECTION_INSTALLED = "age_detection_installed"
AGE_DETECTION_INSTALLING = "age_detection_installing"
AGE_DETECTION_INSTALL_PROGRESS = "age_detection_install_progress"


async def get_setting(key: str, default: str = None) -> Optional[str]:
    """Get a setting value from database"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Settings).where(Settings.key == key))
        setting = result.scalar_one_or_none()
        return setting.value if setting else default


async def set_setting(key: str, value: str):
    """Set a setting value in database"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Settings).where(Settings.key == key))
        setting = result.scalar_one_or_none()
        if setting:
            setting.value = value
        else:
            db.add(Settings(key=key, value=value))
        await db.commit()


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
async def get_settings():
    """Get all app settings"""
    deps = check_age_detection_deps()
    all_installed = all(deps.values())

    return {
        "age_detection": {
            "enabled": await get_setting(AGE_DETECTION_ENABLED, "false") == "true",
            "installed": all_installed,
            "installing": await get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
            "install_progress": await get_setting(AGE_DETECTION_INSTALL_PROGRESS, ""),
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

    await set_setting(AGE_DETECTION_ENABLED, "true" if data.enabled else "false")
    return {"success": True, "enabled": data.enabled}


async def install_age_detection_deps_task():
    """Background task to install age detection dependencies"""
    try:
        await set_setting(AGE_DETECTION_INSTALLING, "true")
        await set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Starting installation...")

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
            await set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Installing {name}...")
            print(f"[AgeDetection] Installing {name}...")

            try:
                # Use subprocess to install
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install"] + package.split(),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout per package
                )

                if result.returncode != 0:
                    print(f"[AgeDetection] Failed to install {name}: {result.stderr}")
                    await set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Failed to install {name}")
                else:
                    print(f"[AgeDetection] Installed {name}")

            except subprocess.TimeoutExpired:
                print(f"[AgeDetection] Timeout installing {name}")
                await set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Timeout installing {name}")
            except Exception as e:
                print(f"[AgeDetection] Error installing {name}: {e}")

        # Check final status
        deps = check_age_detection_deps()
        if all(deps.values()):
            await set_setting(AGE_DETECTION_INSTALLED, "true")
            await set_setting(AGE_DETECTION_INSTALL_PROGRESS, "Installation complete!")
        else:
            missing = [k for k, v in deps.items() if not v]
            await set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Some packages failed: {', '.join(missing)}")

    except Exception as e:
        print(f"[AgeDetection] Installation error: {e}")
        await set_setting(AGE_DETECTION_INSTALL_PROGRESS, f"Error: {str(e)}")
    finally:
        await set_setting(AGE_DETECTION_INSTALLING, "false")


@router.post("/age-detection/install")
async def install_age_detection(background_tasks: BackgroundTasks):
    """Start installing age detection dependencies (runs in background)"""
    # Check if already installing
    if await get_setting(AGE_DETECTION_INSTALLING, "false") == "true":
        return {"success": False, "error": "Installation already in progress"}

    # Check if already installed
    deps = check_age_detection_deps()
    if all(deps.values()):
        return {"success": True, "message": "Dependencies already installed"}

    # Start background installation
    background_tasks.add_task(install_age_detection_deps_task)

    return {
        "success": True,
        "message": "Installation started in background. This may take several minutes."
    }


@router.get("/age-detection/status")
async def get_age_detection_status():
    """Get current age detection installation status"""
    deps = check_age_detection_deps()

    return {
        "enabled": await get_setting(AGE_DETECTION_ENABLED, "false") == "true",
        "installed": all(deps.values()),
        "installing": await get_setting(AGE_DETECTION_INSTALLING, "false") == "true",
        "progress": await get_setting(AGE_DETECTION_INSTALL_PROGRESS, ""),
        "dependencies": deps
    }
