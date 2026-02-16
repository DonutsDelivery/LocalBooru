"""
Auto-Tagger Sidecar — Standalone FastAPI app.

Predicts tags and content ratings for images using WD-Tagger-V3 ONNX models.
No database access — returns predictions to the Rust backend which handles DB writes.

Endpoints:
  GET  /health   → health check + model status
  POST /predict  → predict tags for an image file
"""

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("auto-tagger")

# ─── Model state ──────────────────────────────────────────────────────────────

_model = None
_tags_data = None
_model_loaded = False
_use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"

# Model input size for WD-Tagger-V3
MODEL_INPUT_SIZE = 448

# Default thresholds (can be overridden via env vars)
GENERAL_THRESHOLD = float(os.environ.get("TAGGER_THRESHOLD", "0.35"))
CHARACTER_THRESHOLD = float(os.environ.get("TAGGER_CHARACTER_THRESHOLD", "0.75"))

# ─── Model directory resolution ──────────────────────────────────────────────

# The model can be at several locations:
# 1. TAGGER_MODEL_DIR env var (set by Rust sidecar launcher)
# 2. {LOCALBOORU_DATA_DIR}/models/tagger/vit-v3/
# 3. ~/.localbooru/models/tagger/vit-v3/
# Each must contain model.onnx + selected_tags.csv

def _find_model_dir() -> Optional[Path]:
    """Find the tagger model directory."""
    # Explicit env var
    env_dir = os.environ.get("TAGGER_MODEL_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "model.onnx").exists():
            return p

    # Data directory based locations
    data_dir = os.environ.get("LOCALBOORU_DATA_DIR")
    if data_dir:
        for model_name in ["vit-v3", "eva02-large-v3", "swinv2-v3"]:
            p = Path(data_dir) / "models" / "tagger" / model_name
            if (p / "model.onnx").exists():
                return p

    # Home directory fallback
    home = Path.home() / ".localbooru" / "models" / "tagger"
    for model_name in ["vit-v3", "eva02-large-v3", "swinv2-v3"]:
        p = home / model_name
        if (p / "model.onnx").exists():
            return p

    return None


def _try_download_model() -> Optional[Path]:
    """Attempt to download the default vit-v3 model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface-hub not installed, cannot auto-download model")
        return None

    data_dir = os.environ.get("LOCALBOORU_DATA_DIR", str(Path.home() / ".localbooru"))
    dest = Path(data_dir) / "models" / "tagger" / "vit-v3"
    dest.mkdir(parents=True, exist_ok=True)

    repo_id = "SmilingWolf/wd-vit-tagger-v3"
    try:
        logger.info(f"Downloading tagger model from {repo_id}...")
        for filename in ["model.onnx", "selected_tags.csv"]:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
            logger.info(f"Downloaded {filename}")
        return dest
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_model():
    """Load the ONNX model and tags data."""
    global _model, _tags_data, _model_loaded

    if _model_loaded:
        return

    model_dir = _find_model_dir()
    if model_dir is None:
        model_dir = _try_download_model()
    if model_dir is None:
        logger.error("No tagger model found. Set TAGGER_MODEL_DIR or place model in data dir.")
        return

    model_path = model_dir / "model.onnx"
    tags_path = model_dir / "selected_tags.csv"

    import onnxruntime as ort

    providers = []
    if _use_gpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    _model = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
    )

    active_provider = _model.get_providers()[0] if _model.get_providers() else "Unknown"
    logger.info(f"Loaded tagger model using {active_provider}")

    # Load tags CSV — row index maps to model output index
    _tags_data = {"rating": [], "general": [], "character": []}
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for idx, row in enumerate(reader):
            if len(row) >= 3:
                tag_name, category = row[1], row[2]
                if category == "9":
                    _tags_data["rating"].append((idx, tag_name))
                elif category == "4":
                    _tags_data["character"].append((idx, tag_name))
                else:
                    _tags_data["general"].append((idx, tag_name))

    logger.info(
        f"Loaded {len(_tags_data['general'])} general, "
        f"{len(_tags_data['character'])} character, "
        f"{len(_tags_data['rating'])} rating tags"
    )
    _model_loaded = True


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize to 448x448, normalize for WD-Tagger-V3."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB to BGR
    arr = np.expand_dims(arr, axis=0)
    return arr


# ─── Postprocessing / Rating Logic ───────────────────────────────────────────

# Tag sets for rating adjustment — allows the 5-tier system (pg/pg13/r/x/xxx)
# from the 4-tier tagger output (general/sensitive/questionable/explicit)

PG13_INDICATOR_TAGS = {
    "bare_shoulders", "off_shoulder", "off-shoulder", "shoulder_cutout",
    "short_shorts", "micro_skirt", "side_slit", "high_slit",
    "backless_outfit", "backless_dress", "strapless", "halterneck",
    "cleavage_cutout", "navel_cutout", "sports_bra", "tube_top",
    "tankini", "crop_top_overhang", "underboob_cutout", "underboob",
    "sideboob", "sleeveless", "thigh_strap", "garter_straps",
    "visible_bra", "bra_visible_through_clothes", "tight_clothes",
    "skin_tight", "wet_clothes", "see-through_silhouette", "cleavage",
    "navel", "low_neckline", "deep_neckline",
}

R_INDICATOR_TAGS = {
    "underwear", "panties", "bra", "lingerie", "bikini", "swimsuit",
    "leotard", "bodysuit", "one-piece_swimsuit", "string_bikini",
    "micro_bikini", "thong", "g-string", "negligee", "nightgown",
    "chemise", "camisole", "babydoll", "teddy_(clothing)", "garter_belt",
    "fishnets", "bunny_suit", "maid_bikini", "naked_apron", "naked_shirt",
    "naked_towel", "bath_towel", "convenient_censoring", "hair_censor",
    "light_censor", "ass_focus", "cameltoe", "panty_pull", "bra_pull",
    "lifted_by_self", "skirt_lift", "shirt_lift", "dress_lift",
    "clothes_pull", "undressing", "bare_legs", "midriff", "crop_top",
    "miniskirt", "stomach",
}

X_INDICATOR_TAGS = {
    "nipples", "areolae", "nude", "completely_nude", "naked", "topless",
    "bottomless", "pussy", "anus", "ass", "breasts_out", "no_bra",
    "no_panties", "pubic_hair", "groin", "covering_breasts",
    "covering_crotch", "nude_cover", "strategically_covered",
    "between_breasts", "paizuri_invitation", "presenting",
    "spread_pussy", "spread_legs",
}

XXX_INDICATOR_TAGS = {
    "sex", "vaginal", "anal", "oral", "penis", "erection", "cum",
    "ejaculation", "penetration", "insertion", "masturbation",
    "fingering", "handjob", "blowjob", "fellatio", "cunnilingus",
    "paizuri", "titfuck", "thighjob", "footjob", "grinding",
    "69_(position)", "doggystyle", "missionary", "cowgirl_position",
    "reverse_cowgirl", "suspended_congress", "sex_from_behind",
    "rape", "gangbang", "group_sex", "threesome", "orgy",
    "creampie", "cum_in_pussy", "cum_in_mouth", "cum_on_body",
    "cum_on_face", "cum_on_breasts", "facial", "bukkake",
    "after_sex", "used_tissue", "condom", "used_condom",
    "tentacles", "tentacle_sex", "monster_sex", "bestiality",
    "incest", "futanari", "futa", "yaoi", "yuri_sex",
    "object_insertion", "dildo", "vibrator", "sex_toy",
    "female_ejaculation", "squirting", "ahegao",
    "licking_penis", "deepthroat", "irrumatio",
}

INNOCENT_CONTEXT_TAGS = {
    "child", "young", "kid", "loli", "shota", "flat_chest",
    "school_uniform", "kindergarten_uniform", "elementary_school",
    "sports_uniform", "gym_uniform", "soccer_uniform",
    "cheerleader", "ballet", "gymnastics_leotard",
    "wedding_dress", "formal_dress", "evening_gown",
    "kimono", "yukata", "hanbok", "ao_dai", "cheongsam",
}

# Pre-compiled lookup: tag -> highest rating level
_RATING_LOOKUP = None

def _get_rating_lookup() -> dict:
    global _RATING_LOOKUP
    if _RATING_LOOKUP is None:
        _RATING_LOOKUP = {}
        for tag in PG13_INDICATOR_TAGS:
            _RATING_LOOKUP[tag] = "pg13"
        for tag in R_INDICATOR_TAGS:
            _RATING_LOOKUP[tag] = "r"
        for tag in X_INDICATOR_TAGS:
            _RATING_LOOKUP[tag] = "x"
        for tag in XXX_INDICATOR_TAGS:
            _RATING_LOOKUP[tag] = "xxx"
    return _RATING_LOOKUP


RATING_LEVELS = {"pg": 0, "pg13": 1, "r": 2, "x": 3, "xxx": 4}


def adjust_rating_by_tags(base_rating: str, tag_names: list) -> str:
    """Adjust the tagger's base rating using tag-based heuristics."""
    lookup = _get_rating_lookup()
    tag_set = {t.lower().replace(" ", "_") for t in tag_names}

    has_innocent = bool(tag_set & INNOCENT_CONTEXT_TAGS)

    max_level = 0
    for tag in tag_set:
        if tag in lookup:
            level = RATING_LEVELS[lookup[tag]]
            if level > max_level:
                max_level = level

    # cleavage + large_breasts combo → at least R
    if "cleavage" in tag_set and "large_breasts" in tag_set:
        max_level = max(max_level, RATING_LEVELS["r"])

    base_level = RATING_LEVELS.get(base_rating, 0)

    if max_level >= RATING_LEVELS["xxx"]:
        return "xxx"
    elif max_level >= RATING_LEVELS["x"]:
        return "x" if base_level < RATING_LEVELS["x"] else base_rating
    elif max_level >= RATING_LEVELS["r"]:
        return "r" if base_level < RATING_LEVELS["r"] else base_rating
    elif max_level >= RATING_LEVELS["pg13"] and not has_innocent:
        return "pg13" if base_level < RATING_LEVELS["pg13"] else base_rating

    return base_rating


def get_tags_from_probs(probs: np.ndarray) -> dict:
    """Extract tags from model output probabilities."""
    result = {
        "rating": "pg",
        "rating_scores": {},
        "general_tags": [],
        "character_tags": [],
    }

    # Rating tags
    rating_probs = {}
    for tag_id, tag_name in _tags_data["rating"]:
        rating_probs[tag_name] = float(probs[tag_id])

    result["rating_scores"] = rating_probs

    # Base rating from tagger probabilities
    rating_map = {
        "general": "pg",
        "sensitive": "r",
        "questionable": "x",
        "explicit": "xxx",
    }
    base_rating = "pg"
    if rating_probs:
        best = max(rating_probs.items(), key=lambda x: x[1])
        base_rating = rating_map.get(best[0], "pg")

    # General tags
    for tag_id, tag_name in _tags_data["general"]:
        prob = float(probs[tag_id])
        if prob >= GENERAL_THRESHOLD:
            result["general_tags"].append({
                "name": tag_name.replace(" ", "_"),
                "confidence": round(prob, 4),
                "category": "general",
            })

    # Character tags
    for tag_id, tag_name in _tags_data["character"]:
        prob = float(probs[tag_id])
        if prob >= CHARACTER_THRESHOLD:
            result["character_tags"].append({
                "name": tag_name.replace(" ", "_"),
                "confidence": round(prob, 4),
                "category": "character",
            })

    # Sort by confidence descending
    result["general_tags"].sort(key=lambda x: x["confidence"], reverse=True)
    result["character_tags"].sort(key=lambda x: x["confidence"], reverse=True)

    # Adjust rating based on detected tags
    all_names = [t["name"] for t in result["general_tags"]] + \
                [t["name"] for t in result["character_tags"]]
    result["rating"] = adjust_rating_by_tags(base_rating, all_names)

    return result


# ─── Video detection ──────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".webm", ".mp4", ".mov", ".avi", ".mkv"}


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Auto-Tagger Sidecar")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
    }


class PredictRequest(BaseModel):
    file_path: str
    image_id: Optional[int] = None


@app.post("/predict")
async def predict(req: PredictRequest):
    # Skip video files
    ext = Path(req.file_path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return {
            "tags": [],
            "rating": "pg",
            "rating_scores": {},
            "skipped": True,
            "reason": "video_file",
        }

    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    # Lazy-load model on first request
    if not _model_loaded:
        _load_model()

    if not _model_loaded or _model is None:
        raise HTTPException(
            status_code=503,
            detail="Tagger model not loaded. Check model directory.",
        )

    try:
        image_array = preprocess_image(req.file_path)

        input_name = _model.get_inputs()[0].name
        output_name = _model.get_outputs()[0].name
        outputs = _model.run([output_name], {input_name: image_array})
        probs = outputs[0][0]

        tag_results = get_tags_from_probs(probs)

        all_tags = tag_results["general_tags"] + tag_results["character_tags"]

        return {
            "tags": all_tags,
            "rating": tag_results["rating"],
            "rating_scores": tag_results["rating_scores"],
            "general_count": len(tag_results["general_tags"]),
            "character_count": len(tag_results["character_tags"]),
        }
    except Exception as e:
        logger.error(f"Prediction failed for {req.file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
