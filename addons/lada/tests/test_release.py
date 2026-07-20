import json
import subprocess
import tarfile
from pathlib import Path

from localbooru_lada.release import audit_base_artifact, build_release_manifest, load_addon_metadata

ROOT = Path(__file__).parents[1]


def test_addon_metadata_discloses_license_source_sizes_and_models():
    metadata = load_addon_metadata(ROOT)

    assert metadata["license"] == "AGPL-3.0-only"
    assert metadata["upstream"]["revision"] == "20cb34a20a83c72c87a991d2c949032c70085b16"
    assert metadata["source_url"].endswith("/tree/v0.1.0")
    assert metadata["packages"]["model_bundle"]["download_size"] == 174_714_035
    assert {model["role"] for model in metadata["models"]} == {"detection", "restoration"}
    assert all(model["sha256"] and model["source_url"] for model in metadata["models"])


# AC: @lada-license-provenance ac-base-artifact-boundary
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


# AC: @lada-license-provenance ac-binary-source-match
def test_release_manifest_binds_bundles_to_exact_source_and_hashes(tmp_path):
    common = tmp_path / "common.tar.zst"
    common.write_bytes(b"common-bundle")
    source = tmp_path / "source.tar.zst"
    source.write_bytes(b"corresponding-source")

    manifest = build_release_manifest(
        ROOT,
        {"linux_x86_64_common": common},
        source_archive=source,
    )

    package = manifest["packages"]["linux_x86_64_common"]
    assert package["sha256"] == "b77b4f593935b6be1ced5a6a724cf0fab0beb53f4e1e682310d28e57801eb7b9"
    assert package["size"] == len(b"common-bundle")
    assert manifest["corresponding_source"]["sha256"] == "2a5399dfeffd5d8b6e57d3e6ce35b26abf63f06972b5a8f34412a45b74223587"
    assert manifest["corresponding_source"]["url"].endswith("/releases/download/v0.1.0/source.tar.zst")
    json.dumps(manifest)
