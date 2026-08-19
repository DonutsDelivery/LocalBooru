import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from localbooru_lada.constants import MODEL_REVISION, PINNED_MODEL_HASHES
from localbooru_lada.release import (
    audit_base_artifact,
    build_common_runtime,
    build_release_manifest,
    build_runtime_layer,
    load_addon_metadata,
)

ROOT = Path(__file__).parents[1]


def test_addon_metadata_discloses_license_source_sizes_and_models():
    metadata = load_addon_metadata(ROOT)

    assert metadata["license"] == "AGPL-3.0-only"
    assert metadata["upstream"]["revision"] == "20cb34a20a83c72c87a991d2c949032c70085b16"
    assert metadata["source_url"].endswith("/tree/v0.1.0")
    assert metadata["packages"]["model_bundle"]["download_size"] == 174_714_035
    assert {model["role"] for model in metadata["models"]} == {"detection", "restoration"}
    assert all(model["sha256"] and model["source_url"] for model in metadata["models"])


def test_probe_identity_pins_match_packaged_model_manifest():
    manifest = json.loads((ROOT / "manifests/models.json").read_text())

    assert manifest["revision"] == MODEL_REVISION
    for role, hashes in PINNED_MODEL_HASHES.items():
        assert hashes == {
            model["sha256"] for model in manifest["models"] if model["role"] == role
        }


def test_base_artifact_audit_rejects_lada_payloads_but_allows_bridge_files():
    audit_base_artifact([
        "usr/bin/localbooru",
        "usr/lib/gstreamer-1.0/libgstlocalboorulada.so",
        "usr/share/licenses/localbooru/LADA-INTEGRATION-NOTICE.md",
        "usr/share/unrelated-addon/model.pt",
        "usr/share/unrelated-addon/checkpoint.pth",
    ])

    forbidden = [
        "usr/share/localbooru/addons/lada/lada_mosaic_detection_model_v4_fast.pt",
        "usr/share/localbooru/addons/lada/torch/lib/libtorch.so",
    ]
    for path in forbidden:
        try:
            audit_base_artifact([path])
        except ValueError as error:
            assert path in str(error)
        else:
            raise AssertionError(f"expected {path} to be rejected")


def test_release_inventory_gate_enumerates_actual_unpacked_tree(tmp_path):
    artifact = tmp_path / "linux-unpacked"
    safe = artifact / "resources" / "app" / "video-player.js"
    safe.parent.mkdir(parents=True)
    safe.write_text("ordinary playback", encoding="utf-8")

    inventory = subprocess.run(
        ["node", ROOT.parents[1] / "scripts" / "list-release-inventory.js", artifact],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    audit_base_artifact(inventory)
    assert any(entry.endswith("video-player.js") for entry in inventory)

    forbidden = artifact / "resources" / "app" / "addons" / "lada" / "runtime.py"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("forbidden", encoding="utf-8")
    inventory = subprocess.run(
        ["node", ROOT.parents[1] / "scripts" / "list-release-inventory.js", artifact],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    try:
        audit_base_artifact(inventory)
    except ValueError as error:
        assert "runtime.py" in str(error)
    else:
        raise AssertionError("actual unpacked LADA payload must fail the release gate")


def _write_distribution(runtime, name, version, files):
    site_packages = runtime / "lib" / "python3.12" / "site-packages"
    dist_info = site_packages / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    records = []
    for relative, content in files.items():
        target = site_packages / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        records.append(relative)
    (dist_info / "METADATA").write_text(
        f"Name: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    records.extend([
        f"{dist_info.name}/METADATA",
        f"{dist_info.name}/RECORD",
    ])
    (dist_info / "RECORD").write_text(
        "".join(f"{record},,\n" for record in records),
        encoding="utf-8",
    )


def _file_snapshot(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_common_runtime_excludes_files_owned_by_backend_specific_distributions(tmp_path):
    cuda = tmp_path / "cuda"
    xpu = tmp_path / "xpu"
    for runtime in (cuda, xpu):
        (runtime / "bin").mkdir(parents=True)
        (runtime / "bin" / "python").write_text("shared interpreter\n", encoding="utf-8")
        _write_distribution(runtime, "shared-dependency", "1.0", {"shared/__init__.py": "common\n"})
    _write_distribution(cuda, "torch", "2.8.0", {"torch/__init__.py": "identical shim\n"})
    _write_distribution(xpu, "torch", "2.9.1", {"torch/__init__.py": "identical shim\n"})
    _write_distribution(cuda, "nvidia-cuda-runtime", "12.8", {"nvidia/runtime.so": "cuda\n"})
    _write_distribution(xpu, "pytorch-triton-xpu", "3.5", {"triton/runtime.so": "xpu\n"})

    common = tmp_path / "common"
    cuda_layer = tmp_path / "cuda-layer"
    xpu_layer = tmp_path / "xpu-layer"
    build_common_runtime(cuda, xpu, common)
    build_runtime_layer(common, cuda, cuda_layer)
    build_runtime_layer(common, xpu, xpu_layer)

    assert (common / "bin" / "python").is_file()
    assert (common / "lib" / "python3.12" / "site-packages" / "shared" / "__init__.py").is_file()
    assert not (common / "lib" / "python3.12" / "site-packages" / "torch").exists()

    for complete, layer in ((cuda, cuda_layer), (xpu, xpu_layer)):
        reconstructed = tmp_path / f"reconstructed-{complete.name}"
        shutil.copytree(common, reconstructed)
        shutil.copytree(layer, reconstructed, dirs_exist_ok=True)
        assert _file_snapshot(reconstructed) == _file_snapshot(complete)


def test_runtime_layer_contains_only_new_and_changed_backend_files(tmp_path):
    base = tmp_path / "base"
    complete = tmp_path / "complete"
    layer = tmp_path / "layer"
    for root in (base, complete):
        (root / "site-packages").mkdir(parents=True)
        (root / "site-packages" / "common.py").write_text("same\n", encoding="utf-8")
    (base / "site-packages" / "metadata.txt").write_text("common\n", encoding="utf-8")
    (complete / "site-packages" / "metadata.txt").write_text("cuda\n", encoding="utf-8")
    (complete / "site-packages" / "torch.so").write_bytes(b"accelerator")

    build_runtime_layer(base, complete, layer)

    assert not (layer / "site-packages" / "common.py").exists()
    assert (layer / "site-packages" / "metadata.txt").read_text() == "cuda\n"
    assert (layer / "site-packages" / "torch.so").read_bytes() == b"accelerator"


def test_corresponding_source_stages_only_tracked_addon_and_upstream_files(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "upstream.py").write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
    subprocess.run(["git", "add", "upstream.py"], cwd=upstream, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=LADA Test",
            "-c",
            "user.email=lada-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=upstream,
        check=True,
    )

    contaminations = [
        ROOT / "build" / "review-contamination.bin",
        ROOT / "dist" / "review-contamination.bin",
    ]
    for contamination in contaminations:
        contamination.parent.mkdir(exist_ok=True)
        contamination.write_bytes(b"must not ship")
    stage = tmp_path / "source-stage"
    try:
        subprocess.run(
            [ROOT / "packaging" / "stage-source.sh", ROOT, upstream, stage],
            check=True,
        )
    finally:
        for contamination in contaminations:
            contamination.unlink()

    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(stage / "localbooru-lada-addon", arcname="localbooru-lada-addon")
        handle.add(stage / "lada", arcname="lada")
    with tarfile.open(archive) as handle:
        members = set(handle.getnames())

    assert "localbooru-lada-addon/LICENSE" in members
    assert "lada/upstream.py" in members
    assert not any("build" in Path(member).parts for member in members)
    assert not any("dist" in Path(member).parts for member in members)


def test_adapter_wheel_and_corresponding_source_use_same_committed_snapshot(tmp_path):
    repository = tmp_path / "repository"
    addon = repository / "addon"
    package = addon / "src" / "fixture_adapter"
    package.mkdir(parents=True)
    (addon / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = [\"setuptools>=77\"]\n"
        "build-backend = \"setuptools.build_meta\"\n\n"
        "[project]\n"
        "name = \"fixture-adapter\"\n"
        "version = \"1.0.0\"\n",
        encoding="utf-8",
    )
    implementation = package / "__init__.py"
    implementation.write_text('VALUE = "committed"\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=LADA Test",
            "-c",
            "user.email=lada-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    implementation.write_text('VALUE = "dirty-sentinel"\n', encoding="utf-8")

    upstream = tmp_path / "upstream-snapshot"
    upstream.mkdir()
    (upstream / "LICENSE").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
    subprocess.run(["git", "add", "."], cwd=upstream, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=LADA Test",
            "-c",
            "user.email=lada-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=upstream,
        check=True,
    )

    stage = tmp_path / "snapshot"
    subprocess.run(
        [ROOT / "packaging" / "stage-source.sh", addon, upstream, stage],
        check=True,
    )
    wheel_output = subprocess.run(
        [
            ROOT / "packaging" / "build-adapter-wheel.sh",
            stage / "localbooru-lada-addon",
            tmp_path / "wheels",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = Path(wheel_output.stdout.splitlines()[-1])
    with zipfile.ZipFile(wheel) as handle:
        built = handle.read("fixture_adapter/__init__.py").decode()
    staged = (stage / "localbooru-lada-addon" / "src" / "fixture_adapter" / "__init__.py").read_text()

    assert built == staged == 'VALUE = "committed"\n'
    assert "dirty-sentinel" not in built


# AC: @lada-license-provenance ac-binary-source-match
def test_release_manifest_binds_complete_bundle_topology_to_exact_source_and_hashes(tmp_path):
    bundles = {}
    for name, content in {
        "linux_x86_64_common": b"common-bundle",
        "linux_x86_64_cuda": b"cuda-layer",
        "linux_x86_64_xpu": b"xpu-layer",
        "model_bundle": b"models",
    }.items():
        path = tmp_path / f"{name}.tar.zst"
        path.write_bytes(content)
        bundles[name] = path
    source = tmp_path / "source.tar.zst"
    source.write_bytes(b"corresponding-source")

    manifest = build_release_manifest(
        ROOT,
        bundles,
        source_archive=source,
        installed_sizes={name: index for index, name in enumerate(sorted(bundles), start=1)},
    )

    assert set(manifest["packages"]) == set(bundles)
    package = manifest["packages"]["linux_x86_64_common"]
    assert package["sha256"] == "b77b4f593935b6be1ced5a6a724cf0fab0beb53f4e1e682310d28e57801eb7b9"
    assert package["size"] == len(b"common-bundle")
    assert package["installed_size"] > 0
    assert manifest["corresponding_source"]["sha256"] == "2a5399dfeffd5d8b6e57d3e6ce35b26abf63f06972b5a8f34412a45b74223587"
    assert manifest["corresponding_source"]["url"].endswith("/releases/download/v0.1.0/source.tar.zst")
    assert manifest["backend_compatibility"]["cuda"] == {
        "package": "linux_x86_64_cuda",
        "variant": "cu128",
        "minimum_driver_major": 570,
    }
    legacy = build_release_manifest(
        ROOT,
        bundles,
        source_archive=source,
        cuda_variant="cuda-legacy",
    )
    assert legacy["backend_compatibility"]["cuda"]["variant"] == "cu126"
    assert legacy["backend_compatibility"]["cuda"]["minimum_driver_major"] == 560
    json.dumps(manifest)


def test_release_manifest_rejects_partial_backend_output(tmp_path):
    bundle = tmp_path / "cuda.tar.zst"
    bundle.write_bytes(b"cuda")
    source = tmp_path / "source.tar.zst"
    source.write_bytes(b"source")

    try:
        build_release_manifest(
            ROOT,
            {"linux_x86_64_cuda": bundle},
            source_archive=source,
        )
    except ValueError as error:
        assert "topology mismatch" in str(error)
        assert "linux_x86_64_common" in str(error)
    else:
        raise AssertionError("a one-backend manifest must not overwrite the complete release manifest")
