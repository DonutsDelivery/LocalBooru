"""
Model loading and management for WD-Tagger-V3.
Handles model path resolution, downloading, and ONNX session creation.
"""
import os
import csv
import logging

from ...config import get_settings
from ...models import TaggerModel

logger = logging.getLogger(__name__)
settings = get_settings()

# Global model cache - keyed by model name
_models = {}
_tags_data_cache = {}

# Model directory names on disk
MODEL_DIRS = {
    TaggerModel.vit_v3: "vit-v3",
    TaggerModel.eva02_large_v3: "eva02-large-v3",
    TaggerModel.swinv2_v3: "swinv2-v3",
}

# Default model (fastest)
DEFAULT_MODEL = TaggerModel.vit_v3


def get_model_path(model_type: TaggerModel) -> str:
    """Get the directory path for a specific model."""
    from ..model_downloader import resolve_model_path, get_model_path as get_downloader_path

    model_dir = MODEL_DIRS.get(model_type, MODEL_DIRS[DEFAULT_MODEL])
    model_name = f"tagger/{model_dir}"

    # Try to resolve from bundled or user data
    resolved = resolve_model_path(model_name)
    if resolved:
        return str(resolved)

    # Check user data directory directly (file might exist but fail size check)
    user_path = get_downloader_path(model_name)
    if (user_path / "model.onnx").exists() and (user_path / "selected_tags.csv").exists():
        return str(user_path)

    # Fallback to legacy path (for dev environments with local models)
    base_path = getattr(settings, 'tagger_base_path', None) or os.path.dirname(settings.tagger_model_path)
    return os.path.join(base_path, model_dir)


async def ensure_model_downloaded(model_type: TaggerModel = None):
    """Ensure the tagger model is downloaded before use."""
    if model_type is None:
        model_type = DEFAULT_MODEL

    from ..model_downloader import is_model_available, download_model

    model_dir = MODEL_DIRS.get(model_type, MODEL_DIRS[DEFAULT_MODEL])
    model_name = f"tagger/{model_dir}"

    if not is_model_available(model_name):
        print(f"[Tagger] Model {model_name} not found, downloading...")
        try:
            await download_model(model_name)
            print(f"[Tagger] Model {model_name} downloaded successfully")
        except Exception as e:
            print(f"[Tagger] Failed to download model: {e}")
            raise FileNotFoundError(f"Failed to download model '{model_type.value}': {e}")


def load_model(model_type: TaggerModel = None):
    """Load a specific ONNX model and tags data."""
    global _models, _tags_data_cache

    if model_type is None:
        model_type = DEFAULT_MODEL

    # Check cache
    if model_type in _models:
        return _models[model_type], _tags_data_cache[model_type]

    model_base = get_model_path(model_type)
    model_path = os.path.join(model_base, "model.onnx")
    tags_path = os.path.join(model_base, "selected_tags.csv")

    if not os.path.exists(model_path) or not os.path.exists(tags_path):
        # Model not available - should have been downloaded
        raise FileNotFoundError(
            f"Model '{model_type.value}' not found at {model_base}. "
            f"Download may have failed."
        )

    import onnxruntime as ort

    # Load ONNX model - respect GPU settings from config
    providers = []
    if settings.use_gpu:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')  # Always have CPU fallback

    # Session options for better performance
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4  # Parallel execution within ops

    model = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )

    # Log which provider is being used
    active_provider = model.get_providers()[0] if model.get_providers() else 'Unknown'
    logger.info(f"[Tagger] Loaded {model_type.value} using {active_provider}")

    # Load tags - use row index as the model output index, not tag_id
    tags_data = {"rating": [], "general": [], "character": []}

    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for idx, row in enumerate(reader):
            if len(row) >= 3:
                tag_name, category = row[1], row[2]
                # Use row index (idx) as the model output index
                if category == "9":  # Rating tags
                    tags_data["rating"].append((idx, tag_name))
                elif category == "4":  # Character tags
                    tags_data["character"].append((idx, tag_name))
                else:  # General tags (category 0)
                    tags_data["general"].append((idx, tag_name))

    print(f"[Tagger] Loaded {model_type.value} with {len(tags_data['general'])} general tags, "
          f"{len(tags_data['character'])} character tags, {len(tags_data['rating'])} rating tags")

    # Cache the loaded model
    _models[model_type] = model
    _tags_data_cache[model_type] = tags_data

    return model, tags_data
