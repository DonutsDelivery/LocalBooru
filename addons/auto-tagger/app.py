"""
Auto-Tagger Sidecar — Standalone FastAPI app.

Predicts tags and content ratings for images using WD-Tagger-V3 ONNX models.
No database access — returns predictions to the Rust backend which handles DB writes.

Endpoints:
  GET  /health   → health check + model status
  POST /predict  → predict tags for an image file
"""

import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
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
_model_path = None
_ort_module = None
_tags_data = None
_model_loaded = False
_requested_device = os.environ.get("TAGGER_REQUESTED_DEVICE", "auto").lower()
_available_providers = []
_registered_providers = []
_active_provider = None
_execution_state = "not_run"
_provider_node_counts = {}
_provider_duration_ms = {}
_provider_warning = None
_profile_warning = None
_last_timings_ms = None
_preload_result = {"attempted": False, "succeeded": None, "error": None}
_runtime_diagnostics = {}
_model_identity = None
_cuda_failure = None
_model_load_lock = threading.Lock()
_first_inference_lock = threading.Lock()
_runtime_diagnostic_lock = threading.Lock()
_prediction_slots = threading.BoundedSemaphore(2)

RUNTIME_DIAGNOSTIC_TIMEOUT_SECONDS = 300
RUNTIME_DIAGNOSTIC_OUTPUT_LIMIT = 64 * 1024

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
        if (p / "model.onnx").exists() and (p / "selected_tags.csv").exists():
            return p

    # Data directory based locations
    data_dir = os.environ.get("LOCALBOORU_DATA_DIR")
    if data_dir:
        for model_name in ["vit-v3", "eva02-large-v3", "swinv2-v3"]:
            p = Path(data_dir) / "models" / "tagger" / model_name
            if (p / "model.onnx").exists() and (p / "selected_tags.csv").exists():
                return p

    # Home directory fallback
    home = Path.home() / ".localbooru" / "models" / "tagger"
    for model_name in ["vit-v3", "eva02-large-v3", "swinv2-v3"]:
        p = home / model_name
        if (p / "model.onnx").exists():
            return p

    return None


def _try_download_model() -> Optional[Path]:
    """Attempt to download the selected tagger model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface-hub not installed, cannot auto-download model")
        return None

    data_dir = os.environ.get("LOCALBOORU_DATA_DIR", str(Path.home() / ".localbooru"))
    model_name = os.environ.get("TAGGER_MODEL", "vit-v3")
    model_repositories = {
        "vit-v3": "SmilingWolf/wd-vit-tagger-v3",
        "eva02-large-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
        "swinv2-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    }
    repo_id = model_repositories.get(model_name)
    if repo_id is None:
        logger.error("Unknown tagger model: %s", model_name)
        return None

    dest = Path(data_dir) / "models" / "tagger" / model_name
    dest.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading tagger model from {repo_id}...")
        for filename in ["model.onnx", "selected_tags.csv"]:
            hf_hub_download(
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


def _installed_runtime_packages():
    names = ["numpy", "onnxruntime", "onnxruntime-gpu"]
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    for distribution in importlib.metadata.distributions():
        name = (
            (distribution.metadata.get("Name") or "")
            .lower()
            .replace("_", "-")
            .replace(".", "-")
        )
        if name.startswith("nvidia-"):
            packages[name] = distribution.version
    return dict(sorted(packages.items()))


def _model_file_identity(model_path):
    path = Path(model_path)
    if not path.exists():
        return {"path": str(path), "name": path.parent.name or path.name}
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "name": path.parent.name or path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _collect_runtime_diagnostics(ort, session=None):
    provider_options = {}
    if session is not None:
        try:
            provider_options = session.get_provider_options()
        except Exception:
            provider_options = {}

    debug_output = None
    debug_error = None
    if os.environ.get("TAGGER_ORT_DEBUG", "").lower() in ("1", "true", "yes"):
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                ort.print_debug_info()
            debug_output = output.getvalue()[-16000:]
        except Exception as exc:
            debug_error = str(exc)

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "onnxruntime_version": getattr(ort, "__version__", None),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": _installed_runtime_packages(),
        "provider_options": provider_options,
        "preload": dict(_preload_result),
        "deployment": {
            "desired_revision": os.environ.get("TAGGER_DEPLOYMENT_DESIRED") or None,
            "installed_revision": os.environ.get("TAGGER_DEPLOYMENT_INSTALLED") or None,
            "runtime": os.environ.get("TAGGER_DEPLOYMENT_RUNTIME") or None,
            "warning": os.environ.get("TAGGER_DEPLOYMENT_WARNING") or None,
        },
        "ort_debug_output": debug_output,
        "ort_debug_error": debug_error,
        "cuda_failure": dict(_cuda_failure) if _cuda_failure else None,
    }


# ─── Model loading ────────────────────────────────────────────────────────────

def _new_session_options(ort):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    options.enable_profiling = True
    options.profile_file_prefix = str(
        Path(tempfile.gettempdir())
        / f"localbooru-ort-{os.getpid()}-{time.time_ns()}"
    )
    return options


def _remove_profile_prefix(prefix):
    prefix = Path(prefix)
    for profile_path in prefix.parent.glob(f"{prefix.name}*.json"):
        _remove_profile_file(profile_path)


def _create_ort_session(ort, model_path, providers):
    options = _new_session_options(ort)
    try:
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
    except Exception:
        _remove_profile_prefix(options.profile_file_prefix)
        raise


def _combine_warning(current, message):
    return f"{current} {message}" if current else message


def _session_evidence(session):
    try:
        registered = list(session.get_providers())
    except Exception as exc:
        registered = []
        providers_error = str(exc)
    else:
        providers_error = None
    try:
        options = session.get_provider_options()
    except Exception as exc:
        options = {}
        options_error = str(exc)
    else:
        options_error = None
    return {
        "registered_providers": registered,
        "provider_options": options,
        "providers_error": providers_error,
        "provider_options_error": options_error,
    }


def _record_cuda_failure(stage, error, session=None, **evidence):
    global _cuda_failure

    failure = {"stage": stage, "error": str(error)}
    if session is not None:
        failure.update(_session_evidence(session))
    failure.update(evidence)
    _cuda_failure = failure


def _create_inference_session(ort, model_path, requested_device):
    """Create a profiled provider-aware session with an explicit CPU fallback."""
    global _preload_result, _cuda_failure

    warning = None
    wants_cuda = requested_device in ("auto", "cuda")
    _cuda_failure = None
    _preload_result = {"attempted": wants_cuda, "succeeded": None, "error": None}

    if wants_cuda:
        preload_dlls = getattr(ort, "preload_dlls", None)
        if preload_dlls is not None:
            try:
                preload_dlls(directory="")
                _preload_result["succeeded"] = True
            except Exception as exc:
                _preload_result["succeeded"] = False
                _preload_result["error"] = str(exc)
                warning = f"Unable to preload packaged CUDA libraries: {exc}"
                logger.warning(warning)

    available_providers = ort.get_available_providers()
    logger.info(
        "ONNX Runtime providers for requested device %s: %s",
        requested_device,
        available_providers,
    )

    if wants_cuda and "CUDAExecutionProvider" in available_providers:
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        try:
            session = _create_ort_session(ort, model_path, providers)
        except Exception as exc:
            _record_cuda_failure(
                "session_creation",
                exc,
                available_providers=list(available_providers),
                requested_providers=providers,
                preload=dict(_preload_result),
            )
            message = (
                f"CUDA provider initialization failed ({exc}); "
                "using CPUExecutionProvider."
            )
            warning = _combine_warning(warning, message)
            logger.warning(message)
            session = _create_ort_session(
                ort, model_path, ["CPUExecutionProvider"]
            )
        else:
            evidence = _session_evidence(session)
            registered = evidence["registered_providers"]
            cuda_options = evidence["provider_options"].get("CUDAExecutionProvider")
            cuda_device = cuda_options.get("device_id") if cuda_options else None
            if (
                "CUDAExecutionProvider" not in registered
                or cuda_options is None
                or str(cuda_device) != "0"
            ):
                reason = (
                    "CUDA session verification failed: live providers and provider options "
                    "must include CUDAExecutionProvider on device_id 0"
                )
                _record_cuda_failure(
                    "session_verification", reason, session=session
                )
                message = f"{reason}; using CPUExecutionProvider."
                warning = _combine_warning(warning, message)
                logger.warning(message)
                _discard_execution_profile(session)
                session = _create_ort_session(
                    ort, model_path, ["CPUExecutionProvider"]
                )
            else:
                try:
                    session.disable_fallback()
                except Exception as exc:
                    _record_cuda_failure(
                        "session_verification",
                        f"Unable to disable ONNX Runtime provider fallback: {exc}",
                        session=session,
                    )
                    message = (
                        "Unable to disable ONNX Runtime provider fallback "
                        f"({exc}); using CPUExecutionProvider."
                    )
                    warning = _combine_warning(warning, message)
                    logger.warning(message)
                    _discard_execution_profile(session)
                    session = _create_ort_session(
                        ort, model_path, ["CPUExecutionProvider"]
                    )
        return session, available_providers, warning

    if requested_device == "cuda":
        message = "CUDA was requested but is unavailable; using CPUExecutionProvider."
        warning = _combine_warning(warning, message)
        logger.warning(message)

    session = _create_ort_session(ort, model_path, ["CPUExecutionProvider"])
    return session, available_providers, warning


def _set_loaded_session(session, available_providers, warning, model_path):
    global _model, _model_path, _model_loaded, _available_providers
    global _registered_providers, _active_provider, _execution_state
    global _provider_node_counts, _provider_duration_ms, _provider_warning
    global _profile_warning, _runtime_diagnostics, _model_identity

    _model = session
    _model_path = Path(model_path)
    _model_loaded = True
    _available_providers = list(available_providers)
    _registered_providers = list(session.get_providers())
    _active_provider = None
    _execution_state = "not_run"
    _provider_node_counts = {}
    _provider_duration_ms = {}
    _provider_warning = warning
    _profile_warning = None
    _model_identity = _model_file_identity(model_path)
    if _ort_module is not None:
        _runtime_diagnostics = _collect_runtime_diagnostics(_ort_module, session)


def _load_model():
    """Load the ONNX model and tags data once."""
    global _ort_module, _tags_data

    if _model_loaded:
        return

    with _model_load_lock:
        if _model_loaded:
            return

        model_dir = _find_model_dir()
        if model_dir is None:
            model_dir = _try_download_model()
        if model_dir is None:
            logger.error(
                "No tagger model found. Set TAGGER_MODEL_DIR or place model in data dir."
            )
            return

        model_path = model_dir / "model.onnx"
        tags_path = model_dir / "selected_tags.csv"

        import onnxruntime as ort

        session, available, warning = _create_inference_session(
            ort,
            model_path,
            _requested_device,
        )

        tags_data = {"rating": [], "general": [], "character": []}
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for idx, row in enumerate(reader):
                if len(row) >= 3:
                    tag_name, category = row[1], row[2]
                    if category == "9":
                        tags_data["rating"].append((idx, tag_name))
                    elif category == "4":
                        tags_data["character"].append((idx, tag_name))
                    else:
                        tags_data["general"].append((idx, tag_name))

        _ort_module = ort
        _tags_data = tags_data
        _set_loaded_session(session, available, warning, model_path)
        logger.info(
            "Loaded tagger model with registered providers %s; execution is unverified",
            _registered_providers,
        )
        logger.info(
            "Loaded %d general, %d character, %d rating tags",
            len(_tags_data["general"]),
            len(_tags_data["character"]),
            len(_tags_data["rating"]),
        )


def _ensure_model_loaded():
    if not _model_loaded:
        _load_model()


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize to 448x448, normalize for WD-Tagger-V3."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.ascontiguousarray(arr[:, :, ::-1][None, ...])


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


def _summarize_profile_events(events):
    node_counts = {}
    duration_us = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if not provider:
            continue
        node_counts[provider] = node_counts.get(provider, 0) + 1
        duration_us[provider] = duration_us.get(provider, 0.0) + float(
            event.get("dur", 0.0)
        )

    durations_ms = {
        provider: round(duration / 1000.0, 3)
        for provider, duration in duration_us.items()
    }
    providers = set(node_counts)
    has_cuda = "CUDAExecutionProvider" in providers
    has_cpu = "CPUExecutionProvider" in providers
    if has_cuda and has_cpu:
        state, active = "mixed", "MixedExecutionProviders"
    elif has_cuda:
        state, active = "cuda", "CUDAExecutionProvider"
    elif has_cpu:
        state, active = "cpu", "CPUExecutionProvider"
    else:
        state, active = "unknown", None
    return state, active, node_counts, durations_ms


def _remove_profile_file(profile_path):
    if not profile_path:
        return
    try:
        Path(profile_path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Unable to remove ONNX Runtime profile %s: %s", profile_path, exc)


def _finish_execution_profile(session):
    global _execution_state, _active_provider, _provider_node_counts
    global _provider_duration_ms, _provider_warning, _profile_warning

    profile_path = None
    try:
        profile_path = session.end_profiling()
        with open(profile_path, "r", encoding="utf-8") as profile_file:
            document = json.load(profile_file)
        events = document.get("traceEvents", []) if isinstance(document, dict) else document
        state, active, counts, durations = _summarize_profile_events(events)
        _execution_state = state
        _active_provider = active
        _provider_node_counts = counts
        _provider_duration_ms = durations
        if state == "unknown":
            _profile_warning = (
                "ONNX Runtime profiling did not identify which provider executed model nodes."
            )
        elif _requested_device == "cuda" and state == "cpu":
            _provider_warning = _combine_warning(
                _provider_warning,
                "CUDA was requested but observed model execution used CPU only.",
            )
        elif (
            _requested_device == "auto"
            and "CUDAExecutionProvider" in _registered_providers
            and state == "cpu"
        ):
            _provider_warning = _combine_warning(
                _provider_warning,
                "CUDA was registered but observed model execution used CPU only.",
            )
    except Exception as exc:
        _execution_state = "unknown"
        _active_provider = None
        _provider_node_counts = {}
        _provider_duration_ms = {}
        _profile_warning = f"Unable to read ONNX Runtime execution profile: {exc}"
        logger.warning(_profile_warning)
    finally:
        _remove_profile_file(profile_path)


def _discard_execution_profile(session):
    try:
        _remove_profile_file(session.end_profiling())
    except Exception:
        pass


def _replace_with_cpu_session(cuda_error, observed_execution=None):
    global _ort_module, _execution_state, _active_provider
    global _provider_node_counts, _provider_duration_ms

    old_session = _model
    _discard_execution_profile(old_session)
    ort = _ort_module
    if ort is None:
        import onnxruntime as ort
        _ort_module = ort
    message = f"CUDA execution failed ({cuda_error}); using CPUExecutionProvider."
    warning = _combine_warning(_provider_warning, message)
    logger.warning(message)
    cpu_session = _create_ort_session(
        ort, _model_path, ["CPUExecutionProvider"]
    )
    _set_loaded_session(cpu_session, _available_providers, warning, _model_path)
    if observed_execution is not None:
        _execution_state = observed_execution["state"]
        _active_provider = observed_execution["active_provider"]
        _provider_node_counts = dict(observed_execution["provider_node_counts"])
        _provider_duration_ms = dict(observed_execution["provider_duration_ms"])


def _run_model(input_name, output_name, image_array):
    if _execution_state != "not_run":
        return _model.run([output_name], {input_name: image_array})

    with _first_inference_lock:
        if _execution_state != "not_run":
            return _model.run([output_name], {input_name: image_array})

        session = _model
        try:
            outputs = session.run([output_name], {input_name: image_array})
        except Exception as exc:
            if (
                _requested_device in ("auto", "cuda")
                and _registered_providers
                and _registered_providers[0] == "CUDAExecutionProvider"
            ):
                _record_cuda_failure("first_inference", exc, session=session)
                _replace_with_cpu_session(exc)
                session = _model
                outputs = session.run([output_name], {input_name: image_array})
            else:
                raise
        _finish_execution_profile(session)
        if (
            session is _model
            and _requested_device in ("auto", "cuda")
            and "CUDAExecutionProvider" in _registered_providers
            and _execution_state == "cpu"
            and _provider_node_counts.get("CUDAExecutionProvider", 0) == 0
        ):
            observed = {
                "state": _execution_state,
                "active_provider": _active_provider,
                "provider_node_counts": dict(_provider_node_counts),
                "provider_duration_ms": dict(_provider_duration_ms),
            }
            reason = "completed inference recorded zero CUDA model nodes"
            _record_cuda_failure(
                "first_inference_verification",
                reason,
                session=session,
                **observed,
            )
            _replace_with_cpu_session(reason, observed)
        return outputs


def _predict_image(image_path):
    global _last_timings_ms

    total_started = time.perf_counter()
    phase_started = total_started
    image_array = preprocess_image(image_path)
    preprocess_ms = (time.perf_counter() - phase_started) * 1000.0

    input_name = _model.get_inputs()[0].name
    output_name = _model.get_outputs()[0].name
    phase_started = time.perf_counter()
    outputs = _run_model(input_name, output_name, image_array)
    inference_ms = (time.perf_counter() - phase_started) * 1000.0

    phase_started = time.perf_counter()
    tag_results = get_tags_from_probs(outputs[0][0])
    postprocess_ms = (time.perf_counter() - phase_started) * 1000.0
    timings = {
        "preprocess": round(preprocess_ms, 3),
        "inference": round(inference_ms, 3),
        "postprocess": round(postprocess_ms, 3),
        "total": round((time.perf_counter() - total_started) * 1000.0, 3),
    }
    _last_timings_ms = timings
    logger.info(
        "Prediction timings (ms): preprocess=%.3f inference=%.3f postprocess=%.3f total=%.3f",
        timings["preprocess"],
        timings["inference"],
        timings["postprocess"],
        timings["total"],
    )

    all_tags = tag_results["general_tags"] + tag_results["character_tags"]
    return {
        "tags": all_tags,
        "rating": tag_results["rating"],
        "rating_scores": tag_results["rating_scores"],
        "general_count": len(tag_results["general_tags"]),
        "character_count": len(tag_results["character_tags"]),
        "timings_ms": timings,
    }


# ─── Video detection ──────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".webm", ".mp4", ".mov", ".avi", ".mkv"}


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Auto-Tagger Sidecar")


@app.get("/health")
async def health():
    available_providers = list(_available_providers)
    if not available_providers:
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
        except ImportError:
            available_providers = []
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "requested_device": _requested_device,
        "requested_provider": _requested_device,
        "available_providers": available_providers,
        "registered_providers": list(_registered_providers),
        "execution_state": _execution_state,
        "execution_verified": _execution_state in ("cuda", "cpu", "mixed"),
        "active_provider": _active_provider,
        "provider_node_counts": dict(_provider_node_counts),
        "provider_duration_ms": dict(_provider_duration_ms),
        "last_timings_ms": dict(_last_timings_ms) if _last_timings_ms else None,
        "provider_warning": _provider_warning,
        "profile_warning": _profile_warning,
        "model_identity": dict(_model_identity) if _model_identity else None,
        "runtime_diagnostics": dict(_runtime_diagnostics),
        "cuda_failure": dict(_cuda_failure) if _cuda_failure else None,
    }


def _bounded_output(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) <= RUNTIME_DIAGNOSTIC_OUTPUT_LIMIT:
        return value
    return value[-RUNTIME_DIAGNOSTIC_OUTPUT_LIMIT:]


@app.post("/runtime-diagnostic")
def strict_cuda_diagnostic():
    if _model_path is None or not Path(_model_path).is_file():
        raise HTTPException(status_code=503, detail="No loaded Auto Tagger model to probe")
    if not _runtime_diagnostic_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="CUDA diagnostic is already running")

    command = [
        sys.executable,
        str(Path(__file__).with_name("runtime_probe.py")),
        str(_model_path),
        "--provider",
        "cuda",
        "--verbose",
        "--debug-info",
        "--disable-wrapper-fallback",
        "--strict-on-zero-cuda",
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=RUNTIME_DIAGNOSTIC_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return JSONResponse(
                status_code=504,
                content={
                    "status": "timed_out",
                    "timeout_seconds": RUNTIME_DIAGNOSTIC_TIMEOUT_SECONDS,
                    "probe": None,
                    "stdout": _bounded_output(exc.stdout),
                    "stderr": _bounded_output(exc.stderr),
                },
            )

        stdout = _bounded_output(completed.stdout)
        stderr = _bounded_output(completed.stderr)
        try:
            report = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "status": "invalid_output",
                    "exit_code": completed.returncode,
                    "error": str(exc),
                    "probe": None,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "probe": report,
            "stdout": stdout,
            "stderr": stderr,
        }
    finally:
        _runtime_diagnostic_lock.release()


class PredictRequest(BaseModel):
    file_path: str
    image_id: Optional[int] = None


@app.post("/predict")
def predict(req: PredictRequest):
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

    with _prediction_slots:
        _ensure_model_loaded()
        if not _model_loaded or _model is None:
            raise HTTPException(
                status_code=503,
                detail="Tagger model not loaded. Check model directory.",
            )

        try:
            return _predict_image(req.file_path)
        except Exception as exc:
            logger.error(
                "Prediction failed for %s: %s", req.file_path, exc, exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(exc))
