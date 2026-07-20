import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .constants import ADDON_VERSION, MODEL_REVISION, PINNED_MODEL_HASHES, PROTOCOL_VERSION


@dataclass(frozen=True)
class ProbeConfig:
    protocol_version: int
    upstream_revision: str
    expected_upstream_revision: str
    models: list[dict]
    model_revision: str = MODEL_REVISION
    requested_backend: str = "auto"
    fp16: bool = True
    model_probe_size: int = 64
    model_probe_frames: int = 2
    max_probe_seconds: float = 90.0


def _hash_matches(model: dict) -> bool:
    path = Path(model["path"])
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == model["sha256"]


def _model_is_pinned(model: dict) -> bool:
    role = model.get("role")
    return role in PINNED_MODEL_HASHES and model.get("sha256") in PINNED_MODEL_HASHES[role]


def _model_path(models: list[dict], role: str) -> str:
    matches = [model for model in models if model.get("role") == role]
    defaults = [model for model in matches if model.get("default") is True]
    selected = defaults if defaults else matches
    if len(selected) != 1:
        raise RuntimeError(
            f"probe requires exactly one selected {role} model, found {len(selected)}"
        )
    return selected[0]["path"]


def _weights_are_ready(models: list[dict], identity_probe: Callable[[dict], bool]) -> bool:
    try:
        if not models or not all(identity_probe(model) and _hash_matches(model) for model in models):
            return False
        for role in ("detection", "restoration"):
            _model_path(models, role)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False
    return True


def torch_backend_probe(requested: str) -> dict:
    import torch

    available = []
    errors = {}
    if requested in {"auto", "cuda"}:
        try:
            if torch.cuda.is_available():
                available.append("cuda")
        except Exception as error:
            errors["cuda"] = str(error)
    if requested in {"auto", "xpu"}:
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                available.append("xpu")
        except Exception as error:
            errors["xpu"] = str(error)

    if requested == "auto":
        active = "cuda" if "cuda" in available else "xpu" if "xpu" in available else None
    else:
        active = requested if requested in available else None
    if active is None:
        return {
            "available": available,
            "active": None,
            "tensor_operation": False,
            "device": None,
            "torch_version": torch.__version__,
            "error": errors.get(requested) if requested != "auto" else next(iter(errors.values()), None),
        }

    device = torch.device(active)
    sample = torch.ones((1, 3, 8, 8), device=device)
    result = (sample * 2).sum().item()
    if result != 384:
        raise RuntimeError("accelerator tensor verification returned an unexpected result")
    if active == "cuda":
        device_name = torch.cuda.get_device_name(device)
        runtime_version = torch.version.cuda
    else:
        device_name = torch.xpu.get_device_name(device)
        runtime_version = getattr(torch.version, "xpu", None)
    return {
        "available": available,
        "active": active,
        "tensor_operation": True,
        "device": device_name,
        "torch_version": torch.__version__,
        "runtime_version": runtime_version,
    }


def lada_model_path_probe(
    backend: str,
    models: list[dict],
    *,
    fp16: bool,
    probe_size: int,
    frame_count: int,
    max_seconds: float,
    model_loader=None,
    torch_module=None,
) -> dict:
    if not 32 <= probe_size <= 512 or not 1 <= frame_count <= 8 or not 0 < max_seconds <= 90:
        raise ValueError("model probe bounds are invalid")
    if torch_module is None:
        import torch as torch_module
    if model_loader is None:
        from lada.restorationpipeline import load_models as model_loader

    detection_path = _model_path(models, "detection")
    restoration_path = _model_path(models, "restoration")
    device = torch_module.device(backend)
    started = time.monotonic()
    detection, restoration, _ = model_loader(
        device,
        "basicvsrpp-v1.2",
        restoration_path,
        None,
        detection_path,
        fp16,
        False,
    )
    sample = torch_module.zeros((probe_size, probe_size, 3), dtype=torch_module.uint8)
    preprocessed = detection.preprocess([sample])
    detections = detection.inference_and_postprocess(preprocessed, [sample])
    restored = restoration.restore(
        [sample.clone() for _ in range(frame_count)],
        max_frames=frame_count,
    )
    accelerator = getattr(torch_module, backend, None)
    synchronize = getattr(accelerator, "synchronize", None)
    if callable(synchronize):
        synchronize()
    elapsed = time.monotonic() - started
    if elapsed > max_seconds:
        raise TimeoutError(f"model path probe exceeded {max_seconds:.1f} seconds")
    if len(detections) != 1:
        raise RuntimeError("detection model returned an unexpected batch size")
    if len(restored) != frame_count or any(frame.shape != sample.shape for frame in restored):
        raise RuntimeError("restoration model returned unexpected frame output")
    return {
        "model_path_operation": True,
        "detection_batches": len(detections),
        "restoration_frames": len(restored),
        "probe_size": probe_size,
        "elapsed_ms": round(elapsed * 1000, 3),
    }


def probe_runtime(
    config: ProbeConfig,
    *,
    backend_probe: Callable[[str], dict] = torch_backend_probe,
    model_probe: Callable[..., dict] = lada_model_path_probe,
    model_identity_probe: Callable[[dict], bool] = _model_is_pinned,
) -> dict:
    issues = []
    protocol_compatible = config.protocol_version == PROTOCOL_VERSION
    if not protocol_compatible:
        issues.append("incompatible_protocol")

    revision_compatible = config.upstream_revision == config.expected_upstream_revision
    if not revision_compatible:
        issues.append("incompatible_revision")

    model_revision_compatible = config.model_revision == MODEL_REVISION
    if not model_revision_compatible:
        issues.append("incompatible_model_revision")

    weights_ready = _weights_are_ready(config.models, model_identity_probe)
    if not weights_ready:
        issues.append("weights_invalid")

    try:
        backend = backend_probe(config.requested_backend)
    except Exception as error:
        backend = {
            "available": [],
            "active": None,
            "tensor_operation": False,
            "device": None,
            "error": str(error),
        }
    active = backend.get("active")
    accelerator_ready = active in {"cuda", "xpu"} and backend.get("tensor_operation") is True
    if not accelerator_ready:
        issues.append("accelerator_unavailable")

    model_evidence = None
    model_error = None
    prerequisites_ready = (
        protocol_compatible
        and revision_compatible
        and model_revision_compatible
        and weights_ready
        and accelerator_ready
    )
    if prerequisites_ready:
        try:
            model_evidence = model_probe(
                active,
                config.models,
                fp16=config.fp16,
                probe_size=config.model_probe_size,
                frame_count=config.model_probe_frames,
                max_seconds=config.max_probe_seconds,
            )
        except Exception as error:
            model_error = str(error)
        if not model_evidence or model_evidence.get("model_path_operation") is not True:
            issues.append("model_probe_failed")

    model_path_ready = bool(model_evidence and model_evidence.get("model_path_operation") is True)
    proven_backend = active if accelerator_ready and model_path_ready else None
    return {
        "addon_version": ADDON_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_compatible": protocol_compatible,
        "upstream_revision": config.upstream_revision,
        "revision_compatible": revision_compatible,
        "model_revision": config.model_revision,
        "model_revision_compatible": model_revision_compatible,
        "weights_ready": weights_ready,
        "requested_backend": config.requested_backend,
        "available_backends": backend.get("available", []),
        "active_backend": proven_backend,
        "device": backend.get("device") if proven_backend else None,
        "ready": not issues,
        "reason": issues[0] if issues else None,
        "issues": issues,
        "backend_error": backend.get("error"),
        "backend_evidence": backend,
        "model_evidence": model_evidence,
        "model_error": model_error,
    }
