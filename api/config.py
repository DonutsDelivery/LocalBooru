"""
LocalBooru configuration - simplified single-user settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os


def get_system_data_dir() -> Path:
    r"""Get the system data directory path (always returns system path, ignores portable mode).

    Used by migration to determine target/source directory regardless of current mode.
    Windows: %APPDATA%\.localbooru
    Linux/Mac: ~/.localbooru
    """
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:  # Linux/Mac
        base = Path.home()
    return base / '.localbooru'


def get_data_dir() -> Path:
    """Get LocalBooru data directory - portable or roaming"""
    # Check for portable mode (set by Electron when running from portable folder)
    portable_data = os.environ.get('LOCALBOORU_PORTABLE_DATA')
    if portable_data:
        data_dir = Path(portable_data)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    # Default: use system data directory
    data_dir = get_system_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class Settings(BaseSettings):
    # Storage
    data_dir: str = str(get_data_dir())
    thumbnails_dir: str = str(get_data_dir() / 'thumbnails')
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: set = {"png", "jpg", "jpeg", "gif", "webp", "webm", "mp4"}

    # Tagger - use legacy single-model path
    tagger_model_path: str = "./tagger/model"  # Path to model.onnx and selected_tags.csv
    tagger_base_path: str = "./tagger"  # For future multi-model support
    tagger_threshold: float = 0.35
    tagger_character_threshold: float = 0.85
    default_tagger_model: str = "vit-v3"

    # Server
    host: str = "127.0.0.1"  # Localhost only for security
    port: int = 8790
    debug: bool = False

    # Task queue
    # Increased from 1 to 2 for better throughput - SQLite WAL mode handles concurrent writes
    # Set to 1 if you experience database lock issues
    task_queue_concurrency: int = 2
    file_verify_interval: int = 1800  # Verify file locations every 30 minutes

    # GPU acceleration (auto-detected, but can be disabled)
    use_gpu: bool = True  # Set to False to force CPU-only inference
    gpu_batch_size: int = 4  # Number of images to preprocess in parallel (GPU memory dependent)

    # Tag Guardian - automatic tagging system
    # Runs periodically to catch untagged images and retry failed tasks
    tag_guardian_interval: int = 300  # Check every 5 minutes (seconds)
    tag_guardian_retry_age: int = 3600  # Retry failed tasks after 1 hour (seconds)
    tag_guardian_batch_size: int = 100  # Max images to queue per run

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "LOCALBOORU_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
