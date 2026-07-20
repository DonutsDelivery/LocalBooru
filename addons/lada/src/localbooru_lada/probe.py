import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .constants import PROTOCOL_VERSION


@dataclass(frozen=True)
class ProbeConfig:
    protocol_version: int
    upstream_revision: str
    expected_upstream_revision: str
    models: list[dict]
    requested_backend: str = "auto"


def _hash_matches(model: dict) -> bool:
    path = Path(model["path"])
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == model["sha256"]


def torch_backend_probe(requested: str) -> dict:
    import torch

    available = []
    if torch.cuda.is_available():
        available.append("cuda")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        available.append("xpu")

    if requested == "auto":
        active = "cuda" if "cuda" in available else "xpu" if "xpu" in available else None
    else:
        active = requested if requested in available else None
    if active is None:
        return {"available": available, "active": None, "model_operation": False, "device": None}

    device = torch.device(active)
    sample = torch.ones((1, 3, 8, 8), device=device)
    result = (sample * 2).sum().item()
    if result != 384:
        raise RuntimeError("accelerator tensor verification returned an unexpected result")
    if active == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = torch.xpu.get_device_name(device)
    return {
        "available": available,
        "active": active,
        "model_operation": True,
        "device": device_name,
    }


def probe_runtime(
    config: ProbeConfig,
    *,
    backend_probe: Callable[[str], dict] = torch_backend_probe,
) -> dict:
    issues = []
    protocol_compatible = config.protocol_version == PROTOCOL_VERSION
    if not protocol_compatible:
        issues.append("incompatible_protocol")

    revision_compatible = config.upstream_revision == config.expected_upstream_revision
    if not revision_compatible:
        issues.append("incompatible_revision")

    weights_ready = bool(config.models) and all(_hash_matches(model) for model in config.models)
    if not weights_ready:
        issues.append("weights_invalid")

    try:
        backend = backend_probe(config.requested_backend)
    except Exception as error:
        backend = {
            "available": [],
            "active": None,
            "model_operation": False,
            "device": None,
            "error": str(error),
        }
    active = backend.get("active")
    accelerator_ready = active in {"cuda", "xpu"} and backend.get("model_operation") is True
    if not accelerator_ready:
        issues.append("accelerator_unavailable")
        active = None

    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_compatible": protocol_compatible,
        "upstream_revision": config.upstream_revision,
        "revision_compatible": revision_compatible,
        "weights_ready": weights_ready,
        "requested_backend": config.requested_backend,
        "available_backends": backend.get("available", []),
        "active_backend": active,
        "device": backend.get("device") if active else None,
        "ready": not issues,
        "reason": issues[0] if issues else None,
        "issues": issues,
        "backend_error": backend.get("error"),
    }
