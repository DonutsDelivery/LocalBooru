"""
Image preprocessing for WD-Tagger-V3.
Handles image loading, resizing, and normalization for model input.
"""
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess single image for the model."""
    # Load and resize image
    img = Image.open(image_path).convert("RGB")

    # Resize to 448x448 (WD-VIT-Tagger-V3 input size)
    img = img.resize((448, 448), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    arr = np.array(img, dtype=np.float32)

    # BGR format and normalize
    arr = arr[:, :, ::-1]  # RGB to BGR
    arr = np.expand_dims(arr, axis=0)

    return arr


def preprocess_images_batch(image_paths: list[str]) -> np.ndarray:
    """
    Preprocess multiple images in a batch for more efficient GPU inference.
    Returns a single array with shape (batch_size, 448, 448, 3).
    """
    batch = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((448, 448), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            arr = arr[:, :, ::-1]  # RGB to BGR
            batch.append(arr)
        except Exception as e:
            logger.warning(f"Failed to preprocess {image_path}: {e}")
            # Add zero array as placeholder for failed images
            batch.append(np.zeros((448, 448, 3), dtype=np.float32))

    return np.stack(batch, axis=0)
