import hashlib
from pathlib import Path

from localbooru_lada.probe import ProbeConfig, probe_runtime


def _model(path: Path, contents: bytes) -> dict:
    path.write_bytes(contents)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def test_probe_reports_verified_cuda_runtime(tmp_path):
    models = [
        _model(tmp_path / "detect.pt", b"detector"),
        _model(tmp_path / "restore.pth", b"restorer"),
    ]
    config = ProbeConfig(
        protocol_version=1,
        upstream_revision="20cb34a20a83c72c87a991d2c949032c70085b16",
        expected_upstream_revision="20cb34a20a83c72c87a991d2c949032c70085b16",
        models=models,
        requested_backend="auto",
    )

    result = probe_runtime(
        config,
        backend_probe=lambda requested: {
            "available": ["cuda"],
            "active": "cuda",
            "model_operation": True,
            "device": "NVIDIA Test GPU",
        },
    )

    assert result["ready"] is True
    assert result["active_backend"] == "cuda"
    assert result["weights_ready"] is True
    assert result["reason"] is None


def test_probe_rejects_cpu_only_and_hash_mismatch(tmp_path):
    model = _model(tmp_path / "detect.pt", b"detector")
    model["sha256"] = "0" * 64
    config = ProbeConfig(
        protocol_version=1,
        upstream_revision="wrong",
        expected_upstream_revision="expected",
        models=[model],
        requested_backend="auto",
    )

    result = probe_runtime(
        config,
        backend_probe=lambda requested: {
            "available": ["cpu"],
            "active": "cpu",
            "model_operation": True,
            "device": "CPU",
        },
    )

    assert result["ready"] is False
    assert result["revision_compatible"] is False
    assert result["weights_ready"] is False
    assert result["active_backend"] is None
    assert set(result["issues"]) == {
        "incompatible_revision",
        "weights_invalid",
        "accelerator_unavailable",
    }
