"""
WD-Tagger-V3 integration for automatic image tagging.
Supports multiple tagger models: vit-v3, eva02-large-v3, swinv2-v3.
Uses ONNX runtime for inference.

Architecture:
- Tags are stored in the main database (global definitions with post_count)
- Image-tag associations are stored in per-directory databases
- When tagging, we write tags to main DB and associations to directory DB
"""
import os
import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)
from PIL import Image
from io import BytesIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..models import (
    Tag, Image as ImageModel, TagCategory, Rating, image_tags, TaggerModel,
    DirectoryImage, directory_image_tags
)
from ..database import directory_db_manager

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


# =============================================================================
# TAG-BASED RATING ADJUSTMENTS
# =============================================================================
# These rules adjust the base rating from the tagger based on specific tag
# combinations. The tagger only outputs 4 ratings (general, sensitive,
# questionable, explicit) but we have 5 tiers (pg, pg13, r, x, xxx).
# This allows pg13 to be auto-assigned and fixes over/under-classification.

# Tags that indicate suggestive but not explicit content (elevate pg → pg13)
PG13_INDICATOR_TAGS = {
    "bare_shoulders", "off_shoulder", "off-shoulder", "shoulder_cutout",
    "short_shorts", "micro_skirt",
    "side_slit", "high_slit", "backless_outfit", "backless_dress",
    "strapless", "halterneck", "cleavage_cutout", "navel_cutout",
    "sports_bra", "tube_top", "tankini", "crop_top_overhang",
    "underboob_cutout", "underboob", "sideboob", "sleeveless", "thigh_strap",
    "garter_straps", "visible_bra", "bra_visible_through_clothes",
    "tight_clothes", "skin_tight", "wet_clothes", "see-through_silhouette",
    "cleavage", "navel", "low_neckline", "deep_neckline",  # Exposed skin
}

# Tags that should push rating to at least R (even if tagger says general)
R_INDICATOR_TAGS = {
    "underwear", "panties", "bra", "lingerie", "bikini", "swimsuit",
    "leotard", "bodysuit", "one-piece_swimsuit", "string_bikini",
    "micro_bikini", "thong", "g-string", "negligee", "nightgown",
    "chemise", "camisole", "babydoll", "teddy_(clothing)", "garter_belt",
    "fishnets", "bunny_suit", "maid_bikini",
    "naked_apron", "naked_shirt", "naked_towel", "bath_towel",
    "convenient_censoring", "hair_censor", "light_censor",
    "ass_focus", "cameltoe",
    "panty_pull", "bra_pull", "lifted_by_self", "skirt_lift",
    "shirt_lift", "dress_lift", "clothes_pull", "undressing",
    # Exposed body parts
    "bare_legs", "midriff", "crop_top", "miniskirt", "stomach",
}

# Tags that indicate X rating (nudity without explicit acts)
X_INDICATOR_TAGS = {
    "nipples", "areolae", "nude", "completely_nude", "naked",
    "topless", "bottomless", "pussy", "anus", "ass", "breasts_out",
    "no_bra", "no_panties", "pubic_hair", "groin", "covering_breasts",
    "covering_crotch", "nude_cover", "strategically_covered",
    "between_breasts", "paizuri_invitation", "presenting",
    "spread_pussy", "spread_legs",  # Often indicates explicit posing
}

# Tags that indicate XXX rating (explicit sexual content)
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
    "object_insertion", "dildo", "vibrator", "sex_toy",  # Sex toys
    "female_ejaculation", "squirting", "ahegao",  # Sexual acts
    "licking_penis", "deepthroat", "irrumatio",  # Oral variants
}

# Tags that should PREVENT rating elevation (innocent context)
INNOCENT_CONTEXT_TAGS = {
    "child", "young", "kid", "loli", "shota", "flat_chest",
    "school_uniform", "kindergarten_uniform", "elementary_school",
    "sports_uniform", "gym_uniform", "soccer_uniform",
    "cheerleader", "ballet", "gymnastics_leotard",
    "wedding_dress", "formal_dress", "evening_gown",
    "kimono", "yukata", "hanbok", "ao_dai", "cheongsam",
}

# Pre-compiled lookup dict for O(1) rating indicator checks
# Maps tag -> highest rating level it indicates (xxx > x > r > pg13)
_RATING_INDICATOR_LOOKUP = None

def _get_rating_indicator_lookup() -> dict[str, str]:
    """Get pre-compiled tag -> rating level lookup dict (cached)."""
    global _RATING_INDICATOR_LOOKUP
    if _RATING_INDICATOR_LOOKUP is None:
        _RATING_INDICATOR_LOOKUP = {}
        # Build in priority order (lowest first, highest overwrites)
        for tag in PG13_INDICATOR_TAGS:
            _RATING_INDICATOR_LOOKUP[tag] = 'pg13'
        for tag in R_INDICATOR_TAGS:
            _RATING_INDICATOR_LOOKUP[tag] = 'r'
        for tag in X_INDICATOR_TAGS:
            _RATING_INDICATOR_LOOKUP[tag] = 'x'
        for tag in XXX_INDICATOR_TAGS:
            _RATING_INDICATOR_LOOKUP[tag] = 'xxx'
    return _RATING_INDICATOR_LOOKUP


def adjust_rating_by_tags(base_rating: "Rating", tag_names: list[str]) -> "Rating":
    """
    Adjust the tagger's base rating based on detected tags.

    This allows for more nuanced rating assignment, especially for pg13
    which the tagger cannot directly assign.

    Uses pre-compiled lookup dict for O(1) indicator checks instead of
    O(n) set intersections.

    Args:
        base_rating: The rating from the tagger's probability output
        tag_names: List of detected tag names (lowercase, underscored)

    Returns:
        Adjusted Rating enum value
    """
    lookup = _get_rating_indicator_lookup()
    tag_set = set(t.lower().replace(" ", "_") for t in tag_names)

    # Check for innocent context - don't elevate ratings for these
    has_innocent_context = bool(tag_set & INNOCENT_CONTEXT_TAGS)

    # Find highest indicator level using pre-compiled lookup (O(n) single pass)
    # Rating hierarchy: xxx > x > r > pg13 > pg
    rating_levels = {'pg': 0, 'pg13': 1, 'r': 2, 'x': 3, 'xxx': 4}
    max_indicator_level = 0

    for tag in tag_set:
        if tag in lookup:
            level = rating_levels[lookup[tag]]
            if level > max_indicator_level:
                max_indicator_level = level

    # Special combination: cleavage + large_breasts -> R
    if "cleavage" in tag_set and "large_breasts" in tag_set:
        if max_indicator_level < rating_levels['r']:
            max_indicator_level = rating_levels['r']

    # Map back to rating
    base_level = rating_levels.get(base_rating.value, 0)

    # Tags can only increase rating, not decrease it
    if max_indicator_level >= rating_levels['xxx']:
        return Rating.xxx
    elif max_indicator_level >= rating_levels['x']:
        return Rating.x if base_level < rating_levels['x'] else base_rating
    elif max_indicator_level >= rating_levels['r']:
        return Rating.r if base_level < rating_levels['r'] else base_rating
    elif max_indicator_level >= rating_levels['pg13'] and not has_innocent_context:
        return Rating.pg13 if base_level < rating_levels['pg13'] else base_rating

    return base_rating


def get_model_path(model_type: TaggerModel) -> str:
    """Get the directory path for a specific model."""
    from .model_downloader import resolve_model_path, get_model_path as get_downloader_path

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

    from .model_downloader import is_model_available, download_model

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


def run_inference_batch(image_arrays: np.ndarray, model_type: TaggerModel = None) -> np.ndarray:
    """
    Run model inference on a batch of images.
    Returns array of shape (batch_size, num_tags).
    """
    model, _ = load_model(model_type)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    outputs = model.run([output_name], {input_name: image_arrays})
    return outputs[0]  # Return all batch outputs


def run_inference(image_array: np.ndarray, model_type: TaggerModel = None) -> np.ndarray:
    """Run model inference."""
    model, _ = load_model(model_type)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    outputs = model.run([output_name], {input_name: image_array})
    return outputs[0][0]  # Return first batch, first output


def get_tags_from_probs(probs: np.ndarray, model_type: TaggerModel = None) -> dict:
    """Extract tags from probability array."""
    _, tags_data = load_model(model_type)

    result = {
        "rating": None,
        "rating_scores": {},
        "general_tags": [],
        "character_tags": []
    }

    # Rating tags
    rating_probs = {}
    for tag_id, tag_name in tags_data["rating"]:
        prob = float(probs[tag_id])
        rating_probs[tag_name] = prob

    result["rating_scores"] = rating_probs

    # Determine base rating from tagger probabilities
    base_rating = Rating.pg
    if rating_probs:
        # Map tagger ratings to our 5-tier system
        # Tagger outputs: general, sensitive, questionable, explicit
        # Our system: pg, pg13, r, x, xxx
        rating_map = {
            "general": Rating.pg,       # Child-friendly content
            "sensitive": Rating.r,      # Typically catches underwear/swimsuit
            "questionable": Rating.x,   # Nudity without explicit acts
            "explicit": Rating.xxx      # Explicit sexual content
        }
        best_rating = max(rating_probs.items(), key=lambda x: x[1])
        base_rating = rating_map.get(best_rating[0], Rating.pg)

    # General tags
    for tag_id, tag_name in tags_data["general"]:
        prob = float(probs[tag_id])
        if prob >= settings.tagger_threshold:
            result["general_tags"].append({
                "name": tag_name.replace(" ", "_"),
                "confidence": prob,
                "category": TagCategory.general
            })

    # Character tags (higher threshold)
    for tag_id, tag_name in tags_data["character"]:
        prob = float(probs[tag_id])
        if prob >= settings.tagger_character_threshold:
            result["character_tags"].append({
                "name": tag_name.replace(" ", "_"),
                "confidence": prob,
                "category": TagCategory.character
            })

    # Sort by confidence
    result["general_tags"].sort(key=lambda x: x["confidence"], reverse=True)
    result["character_tags"].sort(key=lambda x: x["confidence"], reverse=True)

    # Adjust rating based on detected tags
    # This allows pg13 to be auto-assigned and fixes tagger over/under-classification
    all_tag_names = [t["name"] for t in result["general_tags"]] + \
                    [t["name"] for t in result["character_tags"]]
    result["rating"] = adjust_rating_by_tags(base_rating, all_tag_names)

    return result


async def tag_image(
    image_path: str,
    db: AsyncSession,
    image_id: int = None,
    directory_id: int = None,
    model_type: TaggerModel = None
) -> dict:
    """
    Tag an image using WD-Tagger-V3.
    Optionally updates database if image_id is provided.

    Args:
        image_path: Path to the image file
        db: Database session (main database)
        image_id: Optional image ID to update in database
        directory_id: If provided, image is in a directory database
        model_type: Which tagger model to use (vit-v3, eva02-large-v3, swinv2-v3)
    """
    if model_type is None:
        model_type = DEFAULT_MODEL

    # Ensure model is downloaded (auto-download if needed)
    await ensure_model_downloaded(model_type)

    # Preprocess and run inference
    image_array = preprocess_image(image_path)
    probs = run_inference(image_array, model_type)

    # Get tags from probabilities
    tag_results = get_tags_from_probs(probs, model_type)

    all_tags = tag_results["general_tags"] + tag_results["character_tags"]
    tag_names = []

    if image_id:
        if directory_id:
            # New architecture: image is in a directory database
            tag_names = await _tag_image_in_directory_db(
                image_id, directory_id, tag_results, all_tags, image_path, db
            )
        else:
            # Legacy: image is in main database
            tag_names = await _tag_image_in_main_db(
                image_id, tag_results, all_tags, image_path, db
            )
    else:
        tag_names = [t["name"] for t in all_tags]

    return {
        "tags": tag_names,
        "rating": tag_results["rating"].value if tag_results["rating"] else "pg",
        "rating_scores": tag_results["rating_scores"],
        "general_count": len(tag_results["general_tags"]),
        "character_count": len(tag_results["character_tags"]),
        "model_used": model_type.value
    }


async def _tag_image_in_directory_db(
    image_id: int,
    directory_id: int,
    tag_results: dict,
    all_tags: list,
    image_path: str,
    main_db: AsyncSession
) -> list[str]:
    """
    Tag an image stored in a directory database.

    - Tags are created/fetched from main DB
    - Associations are stored in directory DB
    - Post counts are updated in main DB
    """
    # Get directory database session
    dir_db = await directory_db_manager.get_session(directory_id)

    try:
        # Get image from directory database
        img_result = await dir_db.execute(
            select(DirectoryImage).where(DirectoryImage.id == image_id)
        )
        image = img_result.scalar_one_or_none()

        if not image:
            return [t["name"] for t in all_tags]

        # Update rating in directory DB
        if tag_results["rating"]:
            image.rating = tag_results["rating"]

        # Batch tag processing
        tag_names = [t["name"] for t in all_tags]
        tag_confidence_map = {t["name"]: t["confidence"] for t in all_tags}
        tag_category_map = {t["name"]: t["category"] for t in all_tags}

        # Fetch all existing tags from main DB
        existing_tags_result = await main_db.execute(
            select(Tag).where(Tag.name.in_(tag_names))
        )
        existing_tags = {t.name: t for t in existing_tags_result.scalars().all()}

        # Create missing tags in main DB
        missing_tag_names = set(tag_names) - set(existing_tags.keys())
        for tag_name in missing_tag_names:
            try:
                async with main_db.begin_nested():
                    tag = Tag(
                        name=tag_name,
                        category=tag_category_map[tag_name]
                    )
                    main_db.add(tag)
                    await main_db.flush()
                    existing_tags[tag_name] = tag
            except IntegrityError:
                tag_result = await main_db.execute(select(Tag).where(Tag.name == tag_name))
                existing_tags[tag_name] = tag_result.scalar_one()

        # Get existing associations from directory DB
        tag_ids = [t.id for t in existing_tags.values()]
        existing_assocs_result = await dir_db.execute(
            select(directory_image_tags.c.tag_id).where(
                directory_image_tags.c.image_id == image_id,
                directory_image_tags.c.tag_id.in_(tag_ids)
            )
        )
        existing_assoc_tag_ids = {row[0] for row in existing_assocs_result.all()}

        # Batch insert new associations into directory DB
        new_associations = []
        tags_to_increment = []
        for tag_name in tag_names:
            tag = existing_tags[tag_name]
            if tag.id not in existing_assoc_tag_ids:
                new_associations.append({
                    "image_id": image_id,
                    "tag_id": tag.id,
                    "confidence": tag_confidence_map[tag_name],
                    "is_manual": False
                })
                tags_to_increment.append(tag)

        if new_associations:
            await dir_db.execute(directory_image_tags.insert(), new_associations)
            # Update post counts in main DB
            for tag in tags_to_increment:
                tag.post_count += 1

        await dir_db.commit()
        await main_db.commit()

        # Run age detection
        age_result = await _run_age_detection_if_realistic_directory(
            image_path, image, directory_id, tag_names, main_db, dir_db
        )
        if age_result:
            tag_names.extend(age_result.get("age_tags", []))

        return tag_names

    finally:
        await dir_db.close()


async def _tag_image_in_main_db(
    image_id: int,
    tag_results: dict,
    all_tags: list,
    image_path: str,
    db: AsyncSession
) -> list[str]:
    """Legacy: Tag an image stored in the main database."""
    # Get image from database
    img_result = await db.execute(select(ImageModel).where(ImageModel.id == image_id))
    image = img_result.scalar_one_or_none()

    if not image:
        return [t["name"] for t in all_tags]

    # Update rating
    if tag_results["rating"]:
        image.rating = tag_results["rating"]

    # Batch tag processing for better performance
    tag_names = [t["name"] for t in all_tags]
    tag_confidence_map = {t["name"]: t["confidence"] for t in all_tags}
    tag_category_map = {t["name"]: t["category"] for t in all_tags}

    # Fetch all existing tags in one query
    existing_tags_result = await db.execute(
        select(Tag).where(Tag.name.in_(tag_names))
    )
    existing_tags = {t.name: t for t in existing_tags_result.scalars().all()}

    # Find tags that need to be created
    missing_tag_names = set(tag_names) - set(existing_tags.keys())

    # Batch create missing tags
    for tag_name in missing_tag_names:
        try:
            async with db.begin_nested():
                tag = Tag(
                    name=tag_name,
                    category=tag_category_map[tag_name]
                )
                db.add(tag)
                await db.flush()
                existing_tags[tag_name] = tag
        except IntegrityError:
            tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
            existing_tags[tag_name] = tag_result.scalar_one()

    # Get existing associations in one query
    tag_ids = [t.id for t in existing_tags.values()]
    existing_assocs_result = await db.execute(
        select(image_tags.c.tag_id).where(
            image_tags.c.image_id == image_id,
            image_tags.c.tag_id.in_(tag_ids)
        )
    )
    existing_assoc_tag_ids = {row[0] for row in existing_assocs_result.all()}

    # Batch insert new associations
    new_associations = []
    tags_to_increment = []
    for tag_name in tag_names:
        tag = existing_tags[tag_name]
        if tag.id not in existing_assoc_tag_ids:
            new_associations.append({
                "image_id": image_id,
                "tag_id": tag.id,
                "confidence": tag_confidence_map[tag_name],
                "is_manual": False
            })
            tags_to_increment.append(tag)

    if new_associations:
        await db.execute(image_tags.insert(), new_associations)
        # Update post counts
        for tag in tags_to_increment:
            tag.post_count += 1

    await db.commit()

    # Run age detection for realistic/photorealistic images
    age_result = await _run_age_detection_if_realistic(
        image_path, image, tag_names, db
    )
    if age_result:
        tag_names.extend(age_result.get("age_tags", []))

    return tag_names


async def _run_age_detection_if_realistic(
    image_path: str,
    image: 'ImageModel',
    tag_names: list[str],
    db: AsyncSession
) -> dict | None:
    """Run age detection if directory has auto_age_detect enabled (legacy main DB)."""
    import json
    from .age_detector import detect_ages, is_age_detection_enabled

    # Check if age detection is enabled globally first
    if not is_age_detection_enabled():
        return None

    # Check if the image's directory has auto_age_detect enabled
    from ..models import ImageFile, WatchDirectory
    file_query = select(ImageFile).where(ImageFile.image_id == image.id).limit(1)
    file_result = await db.execute(file_query)
    image_file = file_result.scalar_one_or_none()

    if not image_file or not image_file.watch_directory_id:
        return None

    dir_query = select(WatchDirectory).where(WatchDirectory.id == image_file.watch_directory_id)
    dir_result = await db.execute(dir_query)
    directory = dir_result.scalar_one_or_none()

    if not directory or not directory.auto_age_detect:
        return None

    try:
        result = await detect_ages(image_path)
        if result is None:
            return None

        # Store age detection results on image
        image.num_faces = result.num_faces
        image.min_detected_age = result.min_age
        image.max_detected_age = result.max_age
        image.detected_ages = json.dumps(result.ages) if result.ages else None
        image.age_detection_data = json.dumps(result.to_dict())

        # Add age-related tags
        age_tags = result.get_age_tags()
        for tag_name in age_tags:
            # Find or create tag
            tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
            tag = tag_result.scalar_one_or_none()

            if not tag:
                try:
                    async with db.begin_nested():
                        # Determine category
                        category = TagCategory.meta
                        tag = Tag(name=tag_name, category=category)
                        db.add(tag)
                        await db.flush()
                except IntegrityError:
                    tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
                    tag = tag_result.scalar_one()

            # Check if association exists
            existing = await db.execute(
                select(image_tags).where(
                    image_tags.c.image_id == image.id,
                    image_tags.c.tag_id == tag.id
                )
            )

            if not existing.first():
                await db.execute(
                    image_tags.insert().values(
                        image_id=image.id,
                        tag_id=tag.id,
                        confidence=0.95,  # High confidence for face detection
                        is_manual=False
                    )
                )
                tag.post_count += 1

        await db.commit()

        logger.info(f"Age detection for image {image.id}: {result.num_faces} faces, ages {result.ages}")
        return {"age_tags": age_tags, "result": result.to_dict()}

    except Exception as e:
        logger.error(f"Age detection failed for image {image.id}: {e}")
        return None


async def _run_age_detection_if_realistic_directory(
    image_path: str,
    image: 'DirectoryImage',
    directory_id: int,
    tag_names: list[str],
    main_db: AsyncSession,
    dir_db: AsyncSession
) -> dict | None:
    """Run age detection for an image in a directory database."""
    import json
    from .age_detector import detect_ages, is_age_detection_enabled
    from ..models import WatchDirectory

    # Check if age detection is enabled globally
    if not is_age_detection_enabled():
        return None

    # Check if the directory has auto_age_detect enabled
    dir_query = select(WatchDirectory).where(WatchDirectory.id == directory_id)
    dir_result = await main_db.execute(dir_query)
    directory = dir_result.scalar_one_or_none()

    if not directory or not directory.auto_age_detect:
        return None

    try:
        result = await detect_ages(image_path)
        if result is None:
            return None

        # Store age detection results on image in directory DB
        image.num_faces = result.num_faces
        image.min_detected_age = result.min_age
        image.max_detected_age = result.max_age
        image.detected_ages = json.dumps(result.ages) if result.ages else None
        image.age_detection_data = json.dumps(result.to_dict())

        # Add age-related tags
        age_tags = result.get_age_tags()
        for tag_name in age_tags:
            # Find or create tag in main DB
            tag_result = await main_db.execute(select(Tag).where(Tag.name == tag_name))
            tag = tag_result.scalar_one_or_none()

            if not tag:
                try:
                    async with main_db.begin_nested():
                        category = TagCategory.meta
                        tag = Tag(name=tag_name, category=category)
                        main_db.add(tag)
                        await main_db.flush()
                except IntegrityError:
                    tag_result = await main_db.execute(select(Tag).where(Tag.name == tag_name))
                    tag = tag_result.scalar_one()

            # Check if association exists in directory DB
            existing = await dir_db.execute(
                select(directory_image_tags).where(
                    directory_image_tags.c.image_id == image.id,
                    directory_image_tags.c.tag_id == tag.id
                )
            )

            if not existing.first():
                await dir_db.execute(
                    directory_image_tags.insert().values(
                        image_id=image.id,
                        tag_id=tag.id,
                        confidence=0.95,
                        is_manual=False
                    )
                )
                tag.post_count += 1

        await dir_db.commit()
        await main_db.commit()

        logger.info(f"Age detection for directory image {image.id}: {result.num_faces} faces, ages {result.ages}")
        return {"age_tags": age_tags, "result": result.to_dict()}

    except Exception as e:
        logger.error(f"Age detection failed for directory image {image.id}: {e}")
        return None


# Fallback tagger for when model isn't loaded
async def tag_image_fallback(image_path: str) -> dict:
    """Fallback when model isn't available - returns empty tags."""
    return {
        "tags": [],
        "rating": "pg",
        "rating_scores": {},
        "general_count": 0,
        "character_count": 0,
        "error": "Tagger model not loaded"
    }
