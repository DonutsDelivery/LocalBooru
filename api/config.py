"""
LocalBooru configuration - simplified single-user settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import os


def get_data_dir() -> Path:
    """Get LocalBooru data directory"""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:  # Linux/Mac
        base = Path.home()

    data_dir = base / '.localbooru'
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
    port: int = 8787
    debug: bool = False

    # Task queue
    task_queue_concurrency: int = 2  # Max concurrent tagging tasks
    file_verify_interval: int = 1800  # Verify file locations every 30 minutes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "LOCALBOORU_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
