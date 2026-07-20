import csv
import hashlib
import json
import os
import re
import shutil
from email.parser import Parser
from pathlib import Path
from typing import Iterable

_FORBIDDEN_BASE_PARTS = (
    "/addons/lada/",
    "/site-packages/lada/",
    "/site-packages/torch/",
    "/model_weights/lada/",
)
_FORBIDDEN_LADA_MODELS = {
    "lada_mosaic_detection_model_v2.pt",
    "lada_mosaic_detection_model_v4_accurate.pt",
    "lada_mosaic_detection_model_v4_fast.pt",
    "lada_mosaic_restoration_model_generic_v1.2.pth",
}
_EXPECTED_RELEASE_PACKAGES = {
    "linux_x86_64_common",
    "linux_x86_64_cuda",
    "linux_x86_64_xpu",
    "model_bundle",
}


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
        if Path(lowered).name in _FORBIDDEN_LADA_MODELS or any(
            part in lowered for part in _FORBIDDEN_BASE_PARTS
        ):
            raise ValueError(f"LADA payload must not be present in the LocalBooru base artifact: {entry}")
        if any(token in lowered for token in ("libtorch", "libcudart", "libze_loader")):
            raise ValueError(f"LADA accelerator runtime must not be present in the base artifact: {entry}")


def _same_file(left: Path, right: Path) -> bool:
    return (
        not right.is_symlink()
        and right.is_file()
        and left.stat().st_size == right.stat().st_size
        and _sha256(left) == _sha256(right)
    )


def _copy_entry(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    else:
        shutil.copy2(source, destination)


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_inventory(runtime: Path) -> dict[str, tuple[str, set[Path]]]:
    inventory = {}
    for site_packages in runtime.glob("lib/python*/site-packages"):
        for dist_info in site_packages.glob("*.dist-info"):
            metadata_path = dist_info / "METADATA"
            record_path = dist_info / "RECORD"
            if not metadata_path.is_file() or not record_path.is_file():
                continue
            metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
            name = _canonical_package_name(metadata["Name"])
            version = metadata["Version"]
            owned = set()
            with record_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.reader(handle):
                    if not row:
                        continue
                    absolute = Path(os.path.normpath(site_packages / row[0]))
                    try:
                        owned.add(absolute.relative_to(runtime))
                    except ValueError as error:
                        raise ValueError(f"distribution file escapes runtime: {row[0]}") from error
            inventory[name] = (version, owned)
    return inventory


def build_common_runtime(cuda: Path, xpu: Path, output: Path) -> None:
    cuda_distributions = _distribution_inventory(cuda)
    xpu_distributions = _distribution_inventory(xpu)
    backend_owned = set()
    for name in set(cuda_distributions) | set(xpu_distributions):
        cuda_entry = cuda_distributions.get(name)
        xpu_entry = xpu_distributions.get(name)
        if cuda_entry is None or xpu_entry is None or cuda_entry[0] != xpu_entry[0]:
            if cuda_entry is not None:
                backend_owned.update(cuda_entry[1])
            if xpu_entry is not None:
                backend_owned.update(xpu_entry[1])

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for source in sorted(cuda.rglob("*")):
        relative = source.relative_to(cuda)
        counterpart = xpu / relative
        if relative in backend_owned:
            continue
        if source.is_symlink():
            if counterpart.is_symlink() and os.readlink(source) == os.readlink(counterpart):
                _copy_entry(source, output / relative)
        elif source.is_file() and _same_file(source, counterpart):
            _copy_entry(source, output / relative)


def build_runtime_layer(base: Path, complete: Path, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for source in sorted(complete.rglob("*")):
        relative = source.relative_to(complete)
        baseline = base / relative
        destination = output / relative
        if source.is_symlink():
            if baseline.is_symlink() and os.readlink(source) == os.readlink(baseline):
                continue
            _copy_entry(source, destination)
        elif source.is_file() and not _same_file(source, baseline):
            _copy_entry(source, destination)


def _artifact(base_url: str, path: Path, installed_size: int | None = None) -> dict:
    artifact = {
        "url": f"{base_url}/{path.name}",
        "sha256": _sha256(path),
        "size": path.stat().st_size,
    }
    if installed_size is not None:
        artifact["installed_size"] = installed_size
    return artifact


def build_release_manifest(
    root: Path,
    bundles: dict[str, Path],
    *,
    source_archive: Path,
    installed_sizes: dict[str, int] | None = None,
    cuda_variant: str = "cuda",
) -> dict:
    if cuda_variant not in {"cuda", "cuda-legacy"}:
        raise ValueError(f"unsupported CUDA bundle variant: {cuda_variant}")
    package_names = set(bundles)
    if package_names != _EXPECTED_RELEASE_PACKAGES:
        missing = sorted(_EXPECTED_RELEASE_PACKAGES - package_names)
        unexpected = sorted(package_names - _EXPECTED_RELEASE_PACKAGES)
        raise ValueError(f"release package topology mismatch; missing={missing}, unexpected={unexpected}")
    installed_sizes = installed_sizes or {}
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
        "backend_compatibility": {
            "cuda": {
                "package": "linux_x86_64_cuda",
                "variant": "cu128" if cuda_variant == "cuda" else "cu126",
                "minimum_driver_major": 570 if cuda_variant == "cuda" else 560,
            },
            "xpu": {
                "package": "linux_x86_64_xpu",
                "kernel_drivers": ["i915", "xe"],
                "requires_render_node": True,
            },
        },
        "packages": {
            name: _artifact(base_url, path, installed_sizes.get(name))
            for name, path in sorted(bundles.items())
        },
        "corresponding_source": _artifact(base_url, source_archive),
    }
