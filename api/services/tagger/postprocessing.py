"""
Tag postprocessing and filtering for WD-Tagger-V3.
Handles probability extraction, rating adjustment, and tag filtering.
"""
import numpy as np

from ...config import get_settings
from ...models import TaggerModel, TagCategory, Rating
from .models import load_model

settings = get_settings()

# =============================================================================
# TAG-BASED RATING ADJUSTMENTS
# =============================================================================
# These rules adjust the base rating from the tagger based on specific tag
# combinations. The tagger only outputs 4 ratings (general, sensitive,
# questionable, explicit) but we have 5 tiers (pg, pg13, r, x, xxx).
# This allows pg13 to be auto-assigned and fixes over/under-classification.

# Tags that indicate suggestive but not explicit content (elevate pg -> pg13)
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
