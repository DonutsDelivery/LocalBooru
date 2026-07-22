import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROBE_PATH = Path(__file__).parents[1] / "runtime_probe.py"
EVA02_SHA256 = "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc"


def load_probe():
    spec = importlib.util.spec_from_file_location("localbooru_runtime_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_uses_declared_model_input_type():
    probe = load_probe()

    assert probe.numpy_dtype("tensor(float)") is np.float32
    assert probe.numpy_dtype("tensor(float16)") is np.float16
    with pytest.raises(ValueError, match="Unsupported model input type"):
        probe.numpy_dtype("tensor(string)")


def test_probe_summarizes_completed_provider_events(tmp_path):
    probe = load_probe()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            [
                {
                    "cat": "Node",
                    "dur": 2000,
                    "args": {"provider": "CUDAExecutionProvider"},
                },
                {
                    "cat": "Node",
                    "dur": 500,
                    "args": {"provider": "CPUExecutionProvider"},
                },
                {"cat": "Session", "dur": 9000, "args": {}},
            ]
        ),
        encoding="utf-8",
    )

    counts, durations = probe.summarize_profile(profile)

    assert counts == {"CUDAExecutionProvider": 1, "CPUExecutionProvider": 1}
    assert durations == {"CUDAExecutionProvider": 2.0, "CPUExecutionProvider": 0.5}


# AC: @auto-tagger-runtime-acceleration-deployment ac-live-cuda-session
def test_probe_requests_explicit_cuda_device_zero():
    probe = load_probe()

    assert probe.provider_spec("cuda") == [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]


# AC: @auto-tagger-runtime-acceleration-deployment ac-live-cuda-session
def test_probe_disables_wrapper_fallback_before_running_inference(monkeypatch, tmp_path):
    probe = load_probe()
    events = []

    class Options:
        def __init__(self):
            self.profile_file_prefix = None
            self.config = {}

        def add_session_config_entry(self, key, value):
            self.config[key] = value

    class Session:
        def disable_fallback(self):
            events.append("disable_fallback")

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"device_id": "0"}}

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 1], type="tensor(float)")]

        def run(self, _outputs, _inputs):
            events.append("run")

        def end_profiling(self):
            profile = tmp_path / "profile.json"
            profile.write_text(
                json.dumps([
                    {"cat": "Node", "dur": 1000, "args": {"provider": "CUDAExecutionProvider"}}
                ]),
                encoding="utf-8",
            )
            return str(profile)

    options = Options()
    constructor_kwargs = {}

    def create_session(*args, **kwargs):
        constructor_kwargs.update(kwargs)
        return Session()

    monkeypatch.setattr(probe.ort, "SessionOptions", lambda: options)
    monkeypatch.setattr(probe.ort, "InferenceSession", create_session)
    args = SimpleNamespace(
        optimization="all",
        verbose=False,
        disable_wrapper_fallback=True,
    )

    stage = probe.execute_stage(
        tmp_path / "model.onnx",
        args,
        probe.provider_spec("cuda"),
    )

    assert events == ["disable_fallback", "run"]
    assert constructor_kwargs["enable_fallback"] == 0
    assert stage["provider_options"]["CUDAExecutionProvider"]["device_id"] == "0"
    assert stage["execution"]["provider_node_counts"] == {"CUDAExecutionProvider": 1}

    strict = probe.execute_stage(
        tmp_path / "model.onnx",
        args,
        [("CUDAExecutionProvider", {"device_id": 0})],
        disable_cpu_fallback=True,
    )
    assert options.config["session.disable_cpu_ep_fallback"] == "1"
    assert strict["cpu_ep_fallback_disabled"] is True


# AC: @auto-tagger-runtime-acceleration-deployment ac-profiler-error-preservation
def test_probe_cleanup_does_not_mask_inference_error(monkeypatch, tmp_path):
    probe = load_probe()
    events = []
    profile = tmp_path / "probe-profile_2026.json"
    profile.write_text("[]", encoding="utf-8")

    class Options:
        def __init__(self):
            self.profile_file_prefix = str(tmp_path / "probe-profile")
            self.config = {}

        def add_session_config_entry(self, key, value):
            self.config[key] = value

    class Session:
        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"device_id": "0"}}

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 1], type="tensor(float)")]

        def run(self, _outputs, _inputs):
            events.append("run")
            raise RuntimeError("CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED")

        def end_profiling(self):
            events.append("end_profiling")
            return str(profile)

        def __del__(self):
            events.append("release_session")

    def locked_unlink(self, missing_ok=False):
        if self == profile:
            events.append("unlink")
            raise PermissionError(32, "profile is still locked", str(self))
        return original_unlink(self, missing_ok=missing_ok)

    original_unlink = Path.unlink
    monkeypatch.setattr(probe.ort, "SessionOptions", Options)
    monkeypatch.setattr(probe.ort, "InferenceSession", lambda *_args, **_kwargs: Session())
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    args = SimpleNamespace(
        optimization="all",
        verbose=False,
        disable_wrapper_fallback=True,
    )

    stage = probe.execute_stage(
        tmp_path / "model.onnx",
        args,
        probe.provider_spec("cuda"),
    )

    assert stage["execution"]["error_type"] == "RuntimeError"
    assert stage["execution"]["error"] == "CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED"
    assert events == ["run", "end_profiling", "release_session", "unlink"]


# AC: @auto-tagger-runtime-acceleration-deployment ac-profiler-error-preservation
def test_probe_constructor_error_survives_cleanup(monkeypatch, tmp_path):
    probe = load_probe()

    class Options:
        def __init__(self):
            self.profile_file_prefix = str(tmp_path / "probe-profile")

        def add_session_config_entry(self, _key, _value):
            pass

    monkeypatch.setattr(probe.ort, "SessionOptions", Options)
    monkeypatch.setattr(
        probe.ort,
        "InferenceSession",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("CUDA DLL load failed")),
    )
    args = SimpleNamespace(
        optimization="all",
        verbose=False,
        disable_wrapper_fallback=True,
    )

    stage = probe.execute_stage(
        tmp_path / "model.onnx",
        args,
        probe.provider_spec("cuda"),
    )

    assert stage["execution"]["error_type"] == "OSError"
    assert stage["execution"]["error"] == "CUDA DLL load failed"



# AC: @auto-tagger-runtime-acceleration-deployment ac-strict-diagnostic
@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ({"execution": {"error": None, "provider_node_counts": {"CUDAExecutionProvider": 1}}}, True),
        ({"execution": {"error": None, "provider_node_counts": {"CPUExecutionProvider": 12}}}, False),
        ({"execution": {"error": "CUDA DLL missing", "provider_node_counts": {}}}, False),
    ],
)
def test_strict_stage_succeeds_only_with_observed_cuda(stage, expected):
    probe = load_probe()

    assert probe.strict_stage_succeeded(stage) is expected


# AC: @auto-tagger-runtime-acceleration-deployment ac-strict-diagnostic
@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"CPUExecutionProvider": 12}, True),
        ({}, True),
        ({"CUDAExecutionProvider": 1, "CPUExecutionProvider": 2}, False),
    ],
)
def test_strict_second_stage_runs_only_for_zero_cuda(counts, expected):
    probe = load_probe()

    assert probe.needs_strict_stage({"provider_node_counts": counts}) is expected


# AC: @auto-tagger-runtime-acceleration-deployment ac-native-runtime-inventory
def test_native_inventory_includes_versioned_shared_objects_and_pyd(monkeypatch, tmp_path):
    probe = load_probe()
    paths = [
        Path("nvidia/cudnn/bin/cudnn64_9.dll"),
        Path("nvidia/cudnn/lib/libcudnn.so.9"),
        Path("onnxruntime/capi/onnxruntime_pybind_state.pyd"),
        Path("nvidia/cudnn/include/cudnn.h"),
    ]
    for path in paths[:-1]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"native")

    class Distribution:
        metadata = {"Name": "nvidia-cudnn-cu12"}
        files = paths

        @staticmethod
        def locate_file(path):
            return tmp_path / path

    monkeypatch.setattr(probe.importlib.metadata, "distributions", lambda: [Distribution()])

    names = {item["name"] for item in probe.native_library_inventory()}
    assert names == {"cudnn64_9.dll", "libcudnn.so.9", "onnxruntime_pybind_state.pyd"}


# AC: @auto-tagger-runtime-acceleration-deployment ac-native-runtime-inventory
def test_inventory_only_emits_runtime_without_executing_model(monkeypatch, capsys):
    probe = load_probe()
    monkeypatch.setattr(
        probe,
        "runtime_details",
        lambda *_args: {
            "packages": {"onnxruntime-gpu": "1.23.2"},
            "native_libraries": [{"name": "cudnn64_9.dll", "exists": True}],
        },
    )
    monkeypatch.setattr(
        probe,
        "execute_stage",
        lambda *_args, **_kwargs: pytest.fail("inventory must not execute the model"),
    )
    monkeypatch.setattr(probe.ort, "preload_dlls", lambda **_kwargs: None)
    monkeypatch.setattr(sys, "argv", [str(PROBE_PATH), "--inventory-only"])

    assert probe.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inventory_only"] is True
    assert report["runtime"]["native_libraries"][0]["name"] == "cudnn64_9.dll"


# AC: @auto-tagger-runtime-acceleration-deployment ac-2
def test_runtime_inventory_includes_every_nvidia_distribution(monkeypatch):
    probe = load_probe()

    class Distribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        probe.importlib.metadata,
        "distributions",
        lambda: [
            Distribution("nvidia-cusparse-cu12", "12.5"),
            Distribution("NVIDIA-NCCL-CU12", "2.22"),
            Distribution("unrelated", "1"),
        ],
    )
    def version(name):
        versions = {"numpy": "2", "onnxruntime-gpu": "1.23.2"}
        if name not in versions:
            raise probe.importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(probe.importlib.metadata, "version", version)

    versions = probe.package_versions()

    assert versions["nvidia-cusparse-cu12"] == "12.5"
    assert versions["nvidia-nccl-cu12"] == "2.22"
    assert "unrelated" not in versions


@pytest.mark.skipif(
    not os.environ.get("LOCALBOORU_CUDA_MODEL_PATH"),
    reason="requires an explicit real-model CUDA acceptance environment",
)
def test_release_eva_model_executes_nonzero_cuda_nodes():
    # AC: @auto-tagger-runtime-acceleration-deployment ac-1
    model_path = Path(os.environ["LOCALBOORU_CUDA_MODEL_PATH"])
    completed = subprocess.run(
        [sys.executable, str(PROBE_PATH), str(model_path), "--provider", "cuda"],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = json.loads(completed.stdout)

    assert report["model"]["sha256"] == EVA02_SHA256
    assert (
        report["execution"]["provider_node_counts"].get(
            "CUDAExecutionProvider", 0
        )
        > 0
    )
