"""
WD-Tagger-V3 integration for automatic image tagging.
Supports multiple tagger models: vit-v3, eva02-large-v3, swinv2-v3.
Uses ONNX runtime for inference.

Architecture:
- Tags are stored in the main database (global definitions with post_count)
- Image-tag associations are stored in per-directory databases
- When tagging, we write tags to main DB and associations to directory DB
"""

from .tagger import (
    tag_image,
    tag_image_fallback,
    VIDEO_EXTENSIONS,
)
from .models import (
    load_model,
    get_model_path,
    ensure_model_downloaded,
    DEFAULT_MODEL,
    MODEL_DIRS,
)
from .preprocessing import (
    preprocess_image,
    preprocess_images_batch,
)
from .postprocessing import (
    get_tags_from_probs,
    run_inference,
    run_inference_batch,
    adjust_rating_by_tags,
)

__all__ = [
    # Main API
    "tag_image",
    "tag_image_fallback",
    "VIDEO_EXTENSIONS",
    # Model management
    "load_model",
    "get_model_path",
    "ensure_model_downloaded",
    "DEFAULT_MODEL",
    "MODEL_DIRS",
    # Preprocessing
    "preprocess_image",
    "preprocess_images_batch",
    # Postprocessing
    "get_tags_from_probs",
    "run_inference",
    "run_inference_batch",
    "adjust_rating_by_tags",
]
