"""
Model Downloader Service
Downloads ML models on-demand to reduce installer size.
Models are cached in the user's data directory.
"""
import os
import asyncio
from pathlib import Path
from typing import Callable, Optional
import httpx
from ..config import get_data_dir

# Model definitions with download URLs and expected sizes
MODELS = {
    # Tagger models (WD-Tagger-V3)
    "tagger/vit-v3": {
        "files": [
            {
                "url": "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx",
                "filename": "model.onnx",
                "size_mb": 178
            },
            {
                "url": "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv",
                "filename": "selected_tags.csv",
                "size_mb": 1
            }
        ],
        "description": "ViT Tagger V3 - Fast auto-tagging model"
    },
    "tagger/eva02-large-v3": {
        "files": [
            {
                "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx",
                "filename": "model.onnx",
                "size_mb": 400
            },
            {
                "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv",
                "filename": "selected_tags.csv",
                "size_mb": 1
            }
        ],
        "description": "EVA02-Large Tagger V3 - More accurate, slower"
    },
    "tagger/swinv2-v3": {
        "files": [
            {
                "url": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx",
                "filename": "model.onnx",
                "size_mb": 180
            },
            {
                "url": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv",
                "filename": "selected_tags.csv",
                "size_mb": 1
            }
        ],
        "description": "SwinV2 Tagger V3 - Balanced speed/accuracy"
    },
    # Object detection model
    "yolov8n": {
        "files": [
            {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                "filename": "yolov8n.pt",
                "size_mb": 6
            }
        ],
        "description": "YOLOv8n - Fast object detection"
    }
}

# Download state
_download_progress: dict[str, dict] = {}
_download_locks: dict[str, asyncio.Lock] = {}


def get_models_dir() -> Path:
    """Get the models directory in user data."""
    models_dir = get_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_path(model_name: str) -> Path:
    """Get the local path for a model."""
    return get_models_dir() / model_name


def is_model_available(model_name: str) -> bool:
    """Check if all files for a model are downloaded."""
    if model_name not in MODELS:
        return False

    model_dir = get_model_path(model_name)
    for file_info in MODELS[model_name]["files"]:
        file_path = model_dir / file_info["filename"]
        if not file_path.exists():
            return False
        # Check if file is complete (at least 90% of expected size)
        expected_bytes = file_info["size_mb"] * 1024 * 1024
        if file_path.stat().st_size < expected_bytes * 0.9:
            return False
    return True


def get_download_progress(model_name: str) -> Optional[dict]:
    """Get current download progress for a model."""
    return _download_progress.get(model_name)


def get_all_models_status() -> dict:
    """Get status of all models."""
    result = {}
    for model_name, model_info in MODELS.items():
        total_size = sum(f["size_mb"] for f in model_info["files"])
        result[model_name] = {
            "available": is_model_available(model_name),
            "description": model_info["description"],
            "size_mb": total_size,
            "downloading": model_name in _download_progress
        }
        if model_name in _download_progress:
            result[model_name]["progress"] = _download_progress[model_name]
    return result


async def download_model(
    model_name: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> bool:
    """
    Download a model if not already available.

    Args:
        model_name: Name of the model to download
        progress_callback: Optional callback(filename, bytes_downloaded, total_bytes)

    Returns:
        True if model is available (downloaded or already present)
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if is_model_available(model_name):
        return True

    # Get or create lock for this model
    if model_name not in _download_locks:
        _download_locks[model_name] = asyncio.Lock()

    async with _download_locks[model_name]:
        # Check again in case another task downloaded it
        if is_model_available(model_name):
            return True

        model_dir = get_model_path(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_info = MODELS[model_name]
        total_size = sum(f["size_mb"] * 1024 * 1024 for f in model_info["files"])

        _download_progress[model_name] = {
            "status": "downloading",
            "current_file": "",
            "bytes_downloaded": 0,
            "total_bytes": total_size,
            "percent": 0
        }

        try:
            bytes_downloaded = 0

            async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
                for file_info in model_info["files"]:
                    filename = file_info["filename"]
                    url = file_info["url"]
                    file_path = model_dir / filename
                    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

                    _download_progress[model_name]["current_file"] = filename

                    # Stream download
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()
                        file_size = int(response.headers.get("content-length", 0))

                        with open(temp_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                                bytes_downloaded += len(chunk)

                                # Update progress
                                percent = int(bytes_downloaded * 100 / total_size) if total_size > 0 else 0
                                _download_progress[model_name].update({
                                    "bytes_downloaded": bytes_downloaded,
                                    "percent": percent
                                })

                                if progress_callback:
                                    progress_callback(filename, bytes_downloaded, total_size)

                    # Move temp file to final location
                    temp_path.rename(file_path)

            _download_progress[model_name]["status"] = "complete"
            return True

        except Exception as e:
            _download_progress[model_name]["status"] = "error"
            _download_progress[model_name]["error"] = str(e)

            # Clean up partial downloads
            for file_info in model_info["files"]:
                temp_path = model_dir / (file_info["filename"] + ".tmp")
                if temp_path.exists():
                    temp_path.unlink()

            raise
        finally:
            # Remove from progress after a delay to allow UI to show completion
            async def cleanup():
                await asyncio.sleep(5)
                _download_progress.pop(model_name, None)
            asyncio.create_task(cleanup())


async def ensure_model(model_name: str) -> Path:
    """
    Ensure a model is available, downloading if necessary.
    Returns the path to the model directory.
    """
    if not is_model_available(model_name):
        await download_model(model_name)
    return get_model_path(model_name)


def get_bundled_model_path(model_name: str) -> Optional[Path]:
    """
    Check if model is bundled with the app (for packaged Electron builds).
    Returns path if bundled, None otherwise.
    """
    # Check resource paths for packaged app
    possible_paths = [
        # Electron packaged (Windows)
        Path(os.environ.get("RESOURCES_PATH", "")) / model_name,
        # Development
        Path(".") / model_name.replace("/", os.sep),
        # Relative to api folder
        Path(__file__).parent.parent.parent / model_name.replace("/", os.sep),
    ]

    for path in possible_paths:
        if path.exists():
            # Verify files exist
            if model_name in MODELS:
                all_present = all(
                    (path / f["filename"]).exists()
                    for f in MODELS[model_name]["files"]
                )
                if all_present:
                    return path

    return None


def resolve_model_path(model_name: str) -> Optional[Path]:
    """
    Get the path to a model, checking bundled locations first,
    then user data directory.
    """
    # Check bundled first
    bundled = get_bundled_model_path(model_name)
    if bundled:
        return bundled

    # Check user data
    user_path = get_model_path(model_name)
    if is_model_available(model_name):
        return user_path

    return None
