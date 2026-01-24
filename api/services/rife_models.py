"""
RIFE model download and cache management.

Downloads RIFE (Real-Time Intermediate Flow Estimation) model weights
from Google Drive or fallback mirrors and caches them locally.
"""
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model metadata
@dataclass
class RIFEModelInfo:
    """Information about a RIFE model version."""
    version: str
    filename: str
    gdrive_id: str
    mirror_url: Optional[str]
    expected_size: int  # bytes
    sha256: Optional[str]  # for verification


# Available RIFE models
RIFE_MODELS: Dict[str, RIFEModelInfo] = {
    "4.22": RIFEModelInfo(
        version="4.22",
        filename="flownet_v4.22.pkl",
        gdrive_id="1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R",
        mirror_url="https://github.com/hzwer/Practical-RIFE/releases/download/4.22/flownet_v4.22.pkl",
        expected_size=51_000_000,  # ~51MB approximate
        sha256=None,  # Set after first verified download
    ),
    "4.22.lite": RIFEModelInfo(
        version="4.22.lite",
        filename="flownet_v4.22_lite.pkl",
        gdrive_id="1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R",  # Same ID, lite version
        mirror_url="https://github.com/hzwer/Practical-RIFE/releases/download/4.22.lite/flownet_v4.22_lite.pkl",
        expected_size=25_000_000,  # ~25MB approximate
        sha256=None,
    ),
}

# Default model version
DEFAULT_VERSION = "4.22"

# Minimum acceptable file size (to catch incomplete downloads)
MIN_MODEL_SIZE = 10_000_000  # 10MB


def get_cache_dir() -> Path:
    """Get the RIFE model cache directory."""
    from api.config import get_settings
    settings = get_settings()
    cache_dir = Path(settings.data_dir) / "models" / "rife"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_metadata_path(cache_dir: Path) -> Path:
    """Get path to the model metadata file."""
    return cache_dir / "models.json"


def load_metadata(cache_dir: Path) -> Dict[str, Any]:
    """Load model metadata from cache."""
    meta_path = get_metadata_path(cache_dir)
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load model metadata: {e}")
    return {"models": {}}


def save_metadata(cache_dir: Path, metadata: Dict[str, Any]) -> None:
    """Save model metadata to cache."""
    meta_path = get_metadata_path(cache_dir)
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except IOError as e:
        logger.warning(f"Failed to save model metadata: {e}")


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_model_cached(version: str, cache_dir: Optional[Path] = None) -> bool:
    """
    Check if a RIFE model is already downloaded and valid.

    Args:
        version: Model version string (e.g., "4.22")
        cache_dir: Optional cache directory (uses default if not provided)

    Returns:
        True if model exists and is valid, False otherwise
    """
    if version not in RIFE_MODELS:
        logger.warning(f"Unknown RIFE model version: {version}")
        return False

    if cache_dir is None:
        cache_dir = get_cache_dir()

    model_info = RIFE_MODELS[version]
    model_path = cache_dir / model_info.filename

    if not model_path.exists():
        return False

    # Check file size
    file_size = model_path.stat().st_size
    if file_size < MIN_MODEL_SIZE:
        logger.warning(f"Model file too small ({file_size} bytes), may be corrupted")
        return False

    # Check metadata for verification
    metadata = load_metadata(cache_dir)
    if version in metadata.get("models", {}):
        cached_info = metadata["models"][version]
        if cached_info.get("verified", False):
            return True

    # File exists but not verified - check hash if we have one
    if model_info.sha256:
        try:
            file_hash = compute_file_hash(model_path)
            if file_hash != model_info.sha256:
                logger.warning(f"Model hash mismatch for {version}")
                return False
        except IOError as e:
            logger.warning(f"Failed to verify model hash: {e}")
            return False

    return True


def download_rife_model(
    version: str,
    target_dir: Optional[Path] = None,
    show_progress: bool = True
) -> Path:
    """
    Download RIFE model weights.

    Args:
        version: Model version string (e.g., "4.22")
        target_dir: Directory to save model (uses default cache if not provided)
        show_progress: Whether to show download progress

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If version is unknown
        RuntimeError: If download fails
    """
    if version not in RIFE_MODELS:
        raise ValueError(f"Unknown RIFE model version: {version}. "
                        f"Available: {list(RIFE_MODELS.keys())}")

    if target_dir is None:
        target_dir = get_cache_dir()

    target_dir.mkdir(parents=True, exist_ok=True)
    model_info = RIFE_MODELS[version]
    model_path = target_dir / model_info.filename

    logger.info(f"Downloading RIFE model {version}...")

    # Try Google Drive first
    success = False
    try:
        success = _download_from_gdrive(
            model_info.gdrive_id,
            model_path,
            show_progress=show_progress
        )
    except Exception as e:
        logger.warning(f"Google Drive download failed: {e}")

    # Try mirror if Google Drive failed
    if not success and model_info.mirror_url:
        logger.info(f"Trying mirror URL...")
        try:
            success = _download_from_url(
                model_info.mirror_url,
                model_path,
                show_progress=show_progress
            )
        except Exception as e:
            logger.warning(f"Mirror download failed: {e}")

    if not success:
        raise RuntimeError(
            f"Failed to download RIFE model {version}. "
            "Check your internet connection and try again."
        )

    # Verify download
    if not model_path.exists():
        raise RuntimeError(f"Download completed but file not found at {model_path}")

    file_size = model_path.stat().st_size
    if file_size < MIN_MODEL_SIZE:
        model_path.unlink()  # Remove corrupted file
        raise RuntimeError(
            f"Downloaded file too small ({file_size} bytes). "
            "Download may have been interrupted."
        )

    # Update metadata
    metadata = load_metadata(target_dir)
    if "models" not in metadata:
        metadata["models"] = {}

    metadata["models"][version] = {
        "filename": model_info.filename,
        "size": file_size,
        "verified": True,
        "hash": compute_file_hash(model_path),
    }
    save_metadata(target_dir, metadata)

    logger.info(f"RIFE model {version} downloaded successfully ({file_size / 1_000_000:.1f}MB)")
    return model_path


def _download_from_gdrive(
    file_id: str,
    output_path: Path,
    show_progress: bool = True
) -> bool:
    """
    Download file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        output_path: Path to save downloaded file
        show_progress: Whether to show progress bar

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        import gdown
    except ImportError:
        logger.error(
            "gdown not installed. Install it with: pip install gdown"
        )
        return False

    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        result = gdown.download(
            url,
            str(output_path),
            quiet=not show_progress,
            fuzzy=True,  # Handle various Google Drive URL formats
        )
        return result is not None and output_path.exists()
    except Exception as e:
        logger.error(f"gdown download failed: {e}")
        return False


def _download_from_url(
    url: str,
    output_path: Path,
    show_progress: bool = True
) -> bool:
    """
    Download file from a direct URL.

    Args:
        url: URL to download from
        output_path: Path to save downloaded file
        show_progress: Whether to show progress

    Returns:
        True if download succeeded, False otherwise
    """
    import httpx

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / 1_000_000
                        mb_total = total_size / 1_000_000
                        print(f"\rDownloading: {mb_downloaded:.1f}/{mb_total:.1f}MB ({percent:.1f}%)", end="", flush=True)

            if show_progress:
                print()  # Newline after progress

            return True

    except httpx.HTTPError as e:
        logger.error(f"HTTP download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False
    except IOError as e:
        logger.error(f"File write failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def get_rife_model_path(version: str = DEFAULT_VERSION) -> Path:
    """
    Get path to RIFE model, downloading if needed.

    This is the main entry point for obtaining RIFE model weights.
    It checks the cache first and downloads if necessary.

    Args:
        version: Model version string (default: "4.22")

    Returns:
        Path to the model file

    Raises:
        ValueError: If version is unknown
        RuntimeError: If download fails

    Example:
        >>> model_path = get_rife_model_path("4.22")
        >>> print(model_path)
        /home/user/.localbooru/models/rife/flownet_v4.22.pkl
    """
    if version not in RIFE_MODELS:
        raise ValueError(f"Unknown RIFE model version: {version}. "
                        f"Available: {list(RIFE_MODELS.keys())}")

    cache_dir = get_cache_dir()
    model_info = RIFE_MODELS[version]
    model_path = cache_dir / model_info.filename

    # Check if already cached
    if is_model_cached(version, cache_dir):
        logger.debug(f"Using cached RIFE model: {model_path}")
        return model_path

    # Download if not cached
    return download_rife_model(version, cache_dir)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available RIFE model versions.

    Returns:
        Dict mapping version strings to model info
    """
    cache_dir = get_cache_dir()
    result = {}

    for version, info in RIFE_MODELS.items():
        cached = is_model_cached(version, cache_dir)
        model_path = cache_dir / info.filename

        result[version] = {
            "version": version,
            "filename": info.filename,
            "cached": cached,
            "size_mb": model_path.stat().st_size / 1_000_000 if cached else None,
            "expected_size_mb": info.expected_size / 1_000_000,
        }

    return result


def clear_cache(version: Optional[str] = None) -> None:
    """
    Clear cached RIFE models.

    Args:
        version: Specific version to clear, or None to clear all
    """
    cache_dir = get_cache_dir()

    if version:
        if version not in RIFE_MODELS:
            raise ValueError(f"Unknown version: {version}")

        model_info = RIFE_MODELS[version]
        model_path = cache_dir / model_info.filename
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Removed cached model: {version}")

        # Update metadata
        metadata = load_metadata(cache_dir)
        if version in metadata.get("models", {}):
            del metadata["models"][version]
            save_metadata(cache_dir, metadata)
    else:
        # Clear all models
        for v, info in RIFE_MODELS.items():
            model_path = cache_dir / info.filename
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Removed cached model: {v}")

        # Clear metadata
        meta_path = get_metadata_path(cache_dir)
        if meta_path.exists():
            meta_path.unlink()

        logger.info("Cleared all cached RIFE models")


def ensure_dependencies() -> bool:
    """
    Check if required dependencies are installed.

    Returns:
        True if all dependencies are available
    """
    missing = []

    try:
        import gdown  # noqa: F401
    except ImportError:
        missing.append("gdown")

    try:
        import httpx  # noqa: F401
    except ImportError:
        missing.append("httpx")

    if missing:
        logger.warning(
            f"Missing dependencies for RIFE model download: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )
        return False

    return True
