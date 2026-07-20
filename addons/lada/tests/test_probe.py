import hashlib
from pathlib import Path
from types import SimpleNamespace

from localbooru_lada.probe import ProbeConfig, lada_model_path_probe, probe_runtime


def _model(path: Path, contents: bytes, role: str) -> dict:
    path.write_bytes(contents)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "role": role,
    }


def test_probe_reports_verified_cuda_runtime_only_after_model_path_success(tmp_path):
    models = [
        _model(tmp_path / "detect.pt", b"detector", "detection"),
        _model(tmp_path / "restore.pth", b"restorer", "restoration"),
    ]
    config = ProbeConfig(
        protocol_version=1,
        upstream_revision="20cb34a20a83c72c87a991d2c949032c70085b16",
        expected_upstream_revision="20cb34a20a83c72c87a991d2c949032c70085b16",
        models=models,
        requested_backend="auto",
    )
    calls = []

    def model_probe(backend, selected_models, **bounds):
        calls.append((backend, selected_models, bounds))
        return {
            "model_path_operation": True,
            "restoration_frames": bounds["frame_count"],
        }

    result = probe_runtime(
        config,
        backend_probe=lambda requested: {
            "available": ["cuda"],
            "active": "cuda",
            "tensor_operation": True,
            "device": "NVIDIA Test GPU",
            "torch_version": "2.8.0+cu128",
        },
        model_probe=model_probe,
    )

    assert result["ready"] is True
    assert result["active_backend"] == "cuda"
    assert result["weights_ready"] is True
    assert result["model_evidence"]["model_path_operation"] is True
    assert calls[0][0] == "cuda"
    assert {model["role"] for model in calls[0][1]} == {"detection", "restoration"}
    assert result["reason"] is None


def test_probe_rejects_hardware_only_success_without_model_inference(tmp_path):
    models = [
        _model(tmp_path / "detect.pt", b"detector", "detection"),
        _model(tmp_path / "restore.pth", b"restorer", "restoration"),
    ]
    config = ProbeConfig(
        protocol_version=1,
        upstream_revision="expected",
        expected_upstream_revision="expected",
        models=models,
        requested_backend="cuda",
    )

    result = probe_runtime(
        config,
        backend_probe=lambda requested: {
            "available": ["cuda"],
            "active": "cuda",
            "tensor_operation": True,
            "device": "NVIDIA Test GPU",
        },
        model_probe=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model inference failed")),
    )

    assert result["ready"] is False
    assert result["active_backend"] is None
    assert result["issues"] == ["model_probe_failed"]
    assert result["model_error"] == "model inference failed"


def test_probe_rejects_cpu_only_and_hash_mismatch_without_loading_models(tmp_path):
    model = _model(tmp_path / "detect.pt", b"detector", "detection")
    model["sha256"] = "0" * 64
    config = ProbeConfig(
        protocol_version=1,
        upstream_revision="wrong",
        expected_upstream_revision="expected",
        models=[model],
        requested_backend="auto",
    )
    model_probe_called = False

    def model_probe(*args, **kwargs):
        nonlocal model_probe_called
        model_probe_called = True

    result = probe_runtime(
        config,
        backend_probe=lambda requested: {
            "available": ["cpu"],
            "active": "cpu",
            "tensor_operation": True,
            "device": "CPU",
        },
        model_probe=model_probe,
    )

    assert result["ready"] is False
    assert result["revision_compatible"] is False
    assert result["weights_ready"] is False
    assert result["active_backend"] is None
    assert model_probe_called is False
    assert set(result["issues"]) == {
        "incompatible_revision",
        "weights_invalid",
        "accelerator_unavailable",
    }


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def clone(self):
        return FakeTensor(self.shape)


class FakeDetection:
    def preprocess(self, frames):
        assert len(frames) == 1
        return "preprocessed"

    def inference_and_postprocess(self, preprocessed, frames):
        assert preprocessed == "preprocessed"
        assert len(frames) == 1
        return ["detections"]


class FakeRestoration:
    def restore(self, frames, max_frames):
        assert len(frames) == max_frames == 2
        return [FakeTensor(frame.shape) for frame in frames]


def test_model_path_probe_loads_both_weights_and_runs_detection_and_restoration(tmp_path):
    models = [
        {"path": str(tmp_path / "detect.pt"), "role": "detection"},
        {"path": str(tmp_path / "restore.pth"), "role": "restoration"},
    ]
    loader_calls = []

    def load_models(*args):
        loader_calls.append(args)
        return FakeDetection(), FakeRestoration(), "zero"

    fake_torch = SimpleNamespace(
        uint8="uint8",
        device=lambda backend: backend,
        zeros=lambda shape, dtype: FakeTensor(shape),
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    result = lada_model_path_probe(
        "cuda",
        models,
        fp16=True,
        probe_size=64,
        frame_count=2,
        max_seconds=5,
        model_loader=load_models,
        torch_module=fake_torch,
    )

    assert result["model_path_operation"] is True
    assert result["detection_batches"] == 1
    assert result["restoration_frames"] == 2
    assert loader_calls[0][2] == str(tmp_path / "restore.pth")
    assert loader_calls[0][4] == str(tmp_path / "detect.pt")
