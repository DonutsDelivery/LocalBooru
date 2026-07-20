import hashlib
import json
from pathlib import Path
from typing import Iterable

_FORBIDDEN_BASE_PARTS = (
    "/addons/lada/",
    "/site-packages/lada/",
    "/site-packages/torch/",
    "/model_weights/",
)
_FORBIDDEN_BASE_SUFFIXES = (".pt", ".pth")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_addon_metadata(root: Path) -> dict:
    addon = _read_json(root / "manifests" / "addon.json")
    model_manifest = _read_json(root / "manifests" / "models.json")
    addon["models"] = model_manifest["models"]
    addon["model_repository"] = {
        key: model_manifest[key]
        for key in ("repository", "revision", "license")
    }
    return addon


def audit_base_artifact(paths: Iterable[str]) -> None:
    for entry in paths:
        normalized = "/" + entry.replace("\\", "/").lstrip("/")
        lowered = normalized.lower()
        if lowered.endswith(_FORBIDDEN_BASE_SUFFIXES) or any(
            part in lowered for part in _FORBIDDEN_BASE_PARTS
        ):
            raise ValueError(f"LADA payload must not be present in the LocalBooru base artifact: {entry}")
        if any(token in lowered for token in ("libtorch", "libcudart", "libze_loader")):
            raise ValueError(f"LADA accelerator runtime must not be present in the base artifact: {entry}")


def _artifact(base_url: str, path: Path) -> dict:
    return {
        "url": f"{base_url}/{path.name}",
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }


def build_release_manifest(
    root: Path,
    bundles: dict[str, Path],
    *,
    source_archive: Path,
) -> dict:
    metadata = load_addon_metadata(root)
    base_url = metadata["release_base_url"].rstrip("/")
    return {
        "schema_version": 1,
        "addon_id": metadata["addon_id"],
        "version": metadata["version"],
        "protocol_version": metadata["protocol_version"],
        "license": metadata["license"],
        "source_url": metadata["source_url"],
        "upstream": metadata["upstream"],
        "model_repository": metadata["model_repository"],
        "models": metadata["models"],
        "packages": {
            name: _artifact(base_url, path)
            for name, path in sorted(bundles.items())
        },
        "corresponding_source": _artifact(base_url, source_archive),
    }
