"""
WD-Tagger-V3 integration for automatic image tagging.
Supports multiple tagger models: vit-v3, eva02-large-v3, swinv2-v3.
Uses ONNX runtime for inference.
"""
import os
import csv
import numpy as np
from PIL import Image
from io import BytesIO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..models import Tag, Image as ImageModel, TagCategory, Rating, image_tags, TaggerModel

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


def adjust_rating_by_tags(base_rating: "Rating", tag_names: list[str]) -> "Rating":
    """
    Adjust the tagger's base rating based on detected tags.

    This allows for more nuanced rating assignment, especially for pg13
    which the tagger cannot directly assign.

    Args:
        base_rating: The rating from the tagger's probability output
        tag_names: List of detected tag names (lowercase, underscored)

    Returns:
        Adjusted Rating enum value
    """
    tag_set = set(t.lower().replace(" ", "_") for t in tag_names)

    # Check for innocent context - don't elevate ratings for these
    has_innocent_context = bool(tag_set & INNOCENT_CONTEXT_TAGS)

    # Check what indicator tags are present
    has_pg13_indicators = bool(tag_set & PG13_INDICATOR_TAGS)
    has_r_indicators = bool(tag_set & R_INDICATOR_TAGS)
    has_x_indicators = bool(tag_set & X_INDICATOR_TAGS)
    has_xxx_indicators = bool(tag_set & XXX_INDICATOR_TAGS)

    # Special combination: cleavage + large_breasts -> R
    if "cleavage" in tag_set and "large_breasts" in tag_set:
        has_r_indicators = True

    # Start with base rating and adjust upward based on tags
    # (tags can only increase rating, not decrease it)

    if has_xxx_indicators:
        return Rating.xxx

    if has_x_indicators:
        # X indicators push to at least X
        if base_rating in (Rating.pg, Rating.pg13, Rating.r):
            return Rating.x
        return base_rating

    if has_r_indicators:
        # R indicators push to at least R
        if base_rating in (Rating.pg, Rating.pg13):
            return Rating.r
        return base_rating

    if has_pg13_indicators and not has_innocent_context:
        # PG13 indicators only elevate PG to PG13
        if base_rating == Rating.pg:
            return Rating.pg13
        return base_rating

    return base_rating


def get_model_path(model_type: TaggerModel) -> str:
    """Get the directory path for a specific model."""
    model_dir = MODEL_DIRS.get(model_type, MODEL_DIRS[DEFAULT_MODEL])
    base_path = getattr(settings, 'tagger_base_path', None) or os.path.dirname(settings.tagger_model_path)
    return os.path.join(base_path, model_dir)


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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Download from HuggingFace.")

    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"Tags file not found at {tags_path}")

    import onnxruntime as ort

    # Load ONNX model
    model = ort.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

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
    """Preprocess image for the model."""
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


async def tag_image(image_path: str, db: AsyncSession, image_id: int = None, model_type: TaggerModel = None) -> dict:
    """
    Tag an image using WD-Tagger-V3.
    Optionally updates database if image_id is provided.

    Args:
        image_path: Path to the image file
        db: Database session
        image_id: Optional image ID to update in database
        model_type: Which tagger model to use (vit-v3, eva02-large-v3, swinv2-v3)
    """
    if model_type is None:
        model_type = DEFAULT_MODEL

    # Preprocess and run inference
    image_array = preprocess_image(image_path)
    probs = run_inference(image_array, model_type)

    # Get tags from probabilities
    tag_results = get_tags_from_probs(probs, model_type)

    all_tags = tag_results["general_tags"] + tag_results["character_tags"]
    tag_names = []

    if image_id:
        # Get image from database
        img_result = await db.execute(select(ImageModel).where(ImageModel.id == image_id))
        image = img_result.scalar_one_or_none()

        if image:
            # Update rating
            if tag_results["rating"]:
                image.rating = tag_results["rating"]

            # Add tags to database
            for tag_data in all_tags:
                tag_name = tag_data["name"]
                tag_names.append(tag_name)

                # Find or create tag (with race condition handling)
                tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
                tag = tag_result.scalar_one_or_none()

                if not tag:
                    try:
                        # Use savepoint so IntegrityError only rolls back this insert
                        async with db.begin_nested():
                            tag = Tag(
                                name=tag_name,
                                category=tag_data["category"]
                            )
                            db.add(tag)
                            await db.flush()
                    except IntegrityError:
                        # Race condition: another request created this tag
                        tag_result = await db.execute(select(Tag).where(Tag.name == tag_name))
                        tag = tag_result.scalar_one()

                # Check if association exists
                existing = await db.execute(
                    select(image_tags).where(
                        image_tags.c.image_id == image_id,
                        image_tags.c.tag_id == tag.id
                    )
                )

                if not existing.first():
                    # Add association with confidence
                    await db.execute(
                        image_tags.insert().values(
                            image_id=image_id,
                            tag_id=tag.id,
                            confidence=tag_data["confidence"],
                            is_manual=False
                        )
                    )
                    tag.post_count += 1

            await db.commit()

            # Run age detection for realistic/photorealistic images
            age_result = await _run_age_detection_if_realistic(
                image_path, image, tag_names, db
            )
            if age_result:
                tag_names.extend(age_result.get("age_tags", []))

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


async def _run_age_detection_if_realistic(
    image_path: str,
    image: 'ImageModel',
    tag_names: list[str],
    db: AsyncSession
) -> dict | None:
    """Run age detection if image has realistic/photorealistic tags."""
    import json
    from .age_detector import should_detect_age, detect_ages, REALISTIC_TAGS

    # Check if image has realistic tags
    if not should_detect_age(tag_names):
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
