import asyncio
import inspect
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


APP_PATH = Path(__file__).parents[1] / "app.py"


def load_tagger():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"localbooru_auto_tagger_{time.time_ns()}", APP_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.intra_op_num_threads = None
        self.enable_profiling = False
        self.config = {}

    def add_session_config_entry(self, key, value):
        self.config[key] = value


class FakeSession:
    def __init__(self, providers, *, profile_path=None, run_error=None):
        self._providers = providers
        self.profile_path = profile_path
        self.run_error = run_error
        self.run_count = 0
        self.end_profile_count = 0

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        return [type("Input", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("Output", (), {"name": "output"})()]

    def run(self, _outputs, _inputs):
        self.run_count += 1
        if self.run_error is not None:
            raise self.run_error
        return [np.zeros((1, 8), dtype=np.float32)]

    def end_profiling(self):
        self.end_profile_count += 1
        return str(self.profile_path)


class FakeOrt:
    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    SessionOptions = FakeSessionOptions

    def __init__(self, available, *, fail_cuda=False, cuda_session=None, cpu_session=None):
        self.available = available
        self.fail_cuda = fail_cuda
        self.cuda_session = cuda_session
        self.cpu_session = cpu_session
        self.events = []

    def preload_dlls(self, *, directory):
        self.events.append(("preload", directory))

    def get_available_providers(self):
        self.events.append(("available",))
        return self.available

    def InferenceSession(self, _model_path, *, sess_options, providers):
        self.events.append(
            ("session", tuple(providers), dict(sess_options.config), sess_options.enable_profiling)
        )
        if providers[0] == "CUDAExecutionProvider":
            if self.fail_cuda:
                raise RuntimeError("CUDA runtime unavailable")
            return self.cuda_session or FakeSession(list(providers))
        return self.cpu_session or FakeSession(list(providers))


def write_profile(path, providers):
    events = [
        {
            "cat": "Node",
            "name": f"node-{index}",
            "dur": duration,
            "args": {"provider": provider},
        }
        for index, (provider, duration) in enumerate(providers)
    ]
    events.append({"cat": "Session", "dur": 999999, "args": {}})
    path.write_text(json.dumps(events), encoding="utf-8")


# AC: @auto-tagger-provider-diagnostics ac-1
# AC: @auto-tagger-execution-verification ac-1
def test_cuda_registration_is_not_reported_as_execution_before_prediction():
    tagger = load_tagger()
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])

    session, available, warning = tagger._create_inference_session(
        ort, "model.onnx", "auto"
    )
    tagger._set_loaded_session(session, available, warning, "model.onnx")
    status = asyncio.run(tagger.health())

    assert ort.events[0] == ("preload", "")
    assert ort.events[1] == ("available",)
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    assert status["registered_providers"][0] == "CUDAExecutionProvider"
    assert status["execution_state"] == "not_run"
    assert status["active_provider"] is None


# AC: @auto-tagger-execution-verification ac-2
def test_profile_evidence_reports_cuda_cpu_mixed_and_unknown():
    tagger = load_tagger()

    cuda = tagger._summarize_profile_events(
        [{"cat": "Node", "dur": 2500, "args": {"provider": "CUDAExecutionProvider"}}]
    )
    cpu = tagger._summarize_profile_events(
        [{"cat": "Node", "dur": 1500, "args": {"provider": "CPUExecutionProvider"}}]
    )
    mixed = tagger._summarize_profile_events(
        [
            {"cat": "Node", "dur": 2500, "args": {"provider": "CUDAExecutionProvider"}},
            {"cat": "Node", "dur": 500, "args": {"provider": "CPUExecutionProvider"}},
        ]
    )
    unknown = tagger._summarize_profile_events(
        [{"cat": "Node", "dur": 100, "args": {}}]
    )

    assert cuda == (
        "cuda",
        "CUDAExecutionProvider",
        {"CUDAExecutionProvider": 1},
        {"CUDAExecutionProvider": 2.5},
    )
    assert cpu[0:2] == ("cpu", "CPUExecutionProvider")
    assert mixed[0:2] == ("mixed", "MixedExecutionProviders")
    assert mixed[2] == {"CUDAExecutionProvider": 1, "CPUExecutionProvider": 1}
    assert mixed[3] == {"CUDAExecutionProvider": 2.5, "CPUExecutionProvider": 0.5}
    assert unknown == ("unknown", None, {}, {})


# AC: @auto-tagger-execution-verification ac-2
def test_first_inference_profiles_once_and_removes_trace(tmp_path):
    tagger = load_tagger()
    profile = tmp_path / "profile.json"
    write_profile(profile, [("CUDAExecutionProvider", 3000)])
    session = FakeSession(["CUDAExecutionProvider"], profile_path=profile)
    tagger._set_loaded_session(
        session,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        None,
        "model.onnx",
    )

    tagger._run_model("input", "output", np.zeros((1, 1), dtype=np.float32))
    tagger._run_model("input", "output", np.zeros((1, 1), dtype=np.float32))

    assert session.run_count == 2
    assert session.end_profile_count == 1
    assert tagger._execution_state == "cuda"
    assert tagger._provider_node_counts == {"CUDAExecutionProvider": 1}
    assert not profile.exists()


# AC: @auto-tagger-execution-verification ac-2
# AC: @auto-tagger-execution-verification ac-5
def test_concurrent_first_inference_finalizes_profile_once(tmp_path):
    tagger = load_tagger()
    profile = tmp_path / "profile.json"
    write_profile(profile, [("CUDAExecutionProvider", 3000)])
    started = threading.Event()
    release = threading.Event()

    class BlockingSession(FakeSession):
        def run(self, outputs, inputs):
            if self.run_count == 0:
                started.set()
                assert release.wait(timeout=1)
            return super().run(outputs, inputs)

    session = BlockingSession(["CUDAExecutionProvider"], profile_path=profile)
    tagger._set_loaded_session(
        session, ["CUDAExecutionProvider"], None, "model.onnx"
    )
    args = ("input", "output", np.zeros((1, 1), dtype=np.float32))
    threads = [threading.Thread(target=tagger._run_model, args=args) for _ in range(2)]

    threads[0].start()
    assert started.wait(timeout=1)
    threads[1].start()
    time.sleep(0.02)
    assert session.run_count == 0
    release.set()
    for thread in threads:
        thread.join(timeout=1)

    assert session.run_count == 2
    assert session.end_profile_count == 1
    assert all(not thread.is_alive() for thread in threads)


# AC: @auto-tagger-execution-verification ac-2
def test_malformed_profile_remains_unknown_with_warning(tmp_path):
    tagger = load_tagger()
    profile = tmp_path / "profile.json"
    profile.write_text("not-json", encoding="utf-8")
    session = FakeSession(["CUDAExecutionProvider"], profile_path=profile)
    tagger._set_loaded_session(
        session, ["CUDAExecutionProvider"], None, "model.onnx"
    )

    tagger._run_model("input", "output", np.zeros((1, 1), dtype=np.float32))

    assert tagger._execution_state == "unknown"
    assert tagger._active_provider is None
    assert "profile" in tagger._profile_warning.lower()
    assert not profile.exists()


# AC: @auto-tagger-provider-diagnostics ac-2
# AC: @auto-tagger-execution-verification ac-3
def test_explicit_cuda_initialization_failure_uses_fresh_cpu_options():
    tagger = load_tagger()
    ort = FakeOrt(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], fail_cuda=True
    )

    session, _available, warning = tagger._create_inference_session(
        ort, "model.onnx", "cuda"
    )

    cuda_event = next(event for event in ort.events if event[0] == "session")
    cpu_event = ort.events[-1]
    assert cuda_event[1] == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert cuda_event[2] == {}
    assert cuda_event[3] is True
    assert cpu_event[1] == ("CPUExecutionProvider",)
    assert cpu_event[2] == {}
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert "CUDA runtime unavailable" in warning


# AC: @auto-tagger-execution-verification ac-3
@pytest.mark.parametrize("requested_device", ["cuda", "auto"])
def test_cuda_first_run_failure_retries_once_on_cpu(tmp_path, requested_device):
    tagger = load_tagger()
    failed_profile = tmp_path / "failed.json"
    failed_profile.write_text("[]", encoding="utf-8")
    cpu_profile = tmp_path / "cpu.json"
    write_profile(cpu_profile, [("CPUExecutionProvider", 4000)])
    cuda_session = FakeSession(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        profile_path=failed_profile,
        run_error=RuntimeError("CUDA launch failed"),
    )
    cpu_session = FakeSession(["CPUExecutionProvider"], profile_path=cpu_profile)
    ort = FakeOrt(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        cuda_session=cuda_session,
        cpu_session=cpu_session,
    )
    tagger._ort_module = ort
    tagger._requested_device = requested_device
    tagger._set_loaded_session(
        cuda_session,
        ort.available,
        None,
        "model.onnx",
    )

    tagger._run_model("input", "output", np.zeros((1, 1), dtype=np.float32))

    assert cuda_session.run_count == 1
    assert cpu_session.run_count == 1
    assert tagger._execution_state == "cpu"
    assert tagger._registered_providers == ["CPUExecutionProvider"]
    assert "CUDA launch failed" in tagger._provider_warning


# AC: @auto-tagger-execution-verification ac-3
def test_explicit_cuda_cpu_only_profile_is_reported_as_fallback(tmp_path):
    tagger = load_tagger()
    tagger._requested_device = "cuda"
    profile = tmp_path / "cpu-only.json"
    write_profile(profile, [("CPUExecutionProvider", 4000)])
    session = FakeSession(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], profile_path=profile
    )
    tagger._set_loaded_session(
        session,
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        None,
        "model.onnx",
    )

    tagger._run_model("input", "output", np.zeros((1, 1), dtype=np.float32))

    assert tagger._execution_state == "cpu"
    assert tagger._active_provider == "CPUExecutionProvider"
    assert "CUDA was requested" in tagger._provider_warning
    assert "CPU only" in tagger._provider_warning


# AC: @auto-tagger-provider-diagnostics ac-3
def test_explicit_cpu_skips_cuda_preload_and_registration():
    tagger = load_tagger()
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])

    session, _available, warning = tagger._create_inference_session(
        ort, "model.onnx", "cpu"
    )

    assert not any(event[0] == "preload" for event in ort.events)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert warning is None


# AC: @auto-tagger-execution-verification ac-4
def test_preprocessing_returns_contiguous_float32_bgr_batch(tmp_path):
    tagger = load_tagger()
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), (255, 0, 0)).save(image_path)

    result = tagger.preprocess_image(str(image_path))

    assert result.shape == (1, 448, 448, 3)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert result[0, 0, 0].tolist() == [0.0, 0.0, 255.0]


# AC: @auto-tagger-execution-verification ac-4
def test_prediction_returns_and_caches_separate_phase_timings(monkeypatch):
    tagger = load_tagger()
    tagger._model_loaded = True
    tagger._model = FakeSession(["CPUExecutionProvider"])
    tagger._execution_state = "cpu"
    tagger._tags_data = {"rating": [], "general": [], "character": []}
    monkeypatch.setattr(
        tagger,
        "preprocess_image",
        lambda _path: np.zeros((1, 448, 448, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        tagger,
        "_run_model",
        lambda _input, _output, _array: [np.zeros((1, 8), dtype=np.float32)],
    )

    result = tagger._predict_image("image.png")
    health = asyncio.run(tagger.health())

    assert set(result["timings_ms"]) == {"preprocess", "inference", "postprocess", "total"}
    assert result["timings_ms"] == health["last_timings_ms"]
    assert all(value >= 0 for value in result["timings_ms"].values())


# AC: @auto-tagger-execution-verification ac-5
def test_prediction_handler_is_synchronous_and_model_load_is_singleton(monkeypatch):
    tagger = load_tagger()
    assert not inspect.iscoroutinefunction(tagger.predict)
    load_count = 0

    def fake_load():
        nonlocal load_count
        with tagger._model_load_lock:
            if tagger._model_loaded:
                return
            load_count += 1
            time.sleep(0.02)
            tagger._model_loaded = True
            tagger._model = FakeSession(["CPUExecutionProvider"])

    monkeypatch.setattr(tagger, "_load_model", fake_load)
    threads = [threading.Thread(target=tagger._ensure_model_loaded) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert load_count == 1
    assert asyncio.run(tagger.health())["status"] == "ok"


# AC: @auto-tagger-execution-verification ac-5
def test_health_remains_responsive_while_prediction_is_running(tmp_path, monkeypatch):
    tagger = load_tagger()
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")
    tagger._model_loaded = True
    tagger._model = FakeSession(["CPUExecutionProvider"])
    started = threading.Event()
    release = threading.Event()

    def blocked_prediction(_path):
        started.set()
        assert release.wait(timeout=1)
        return {"tags": []}

    monkeypatch.setattr(tagger, "_predict_image", blocked_prediction)
    prediction = threading.Thread(
        target=tagger.predict,
        args=(tagger.PredictRequest(file_path=str(image_path)),),
    )
    prediction.start()
    assert started.wait(timeout=1)

    before = time.perf_counter()
    status = asyncio.run(tagger.health())
    elapsed = time.perf_counter() - before
    release.set()
    prediction.join(timeout=1)

    assert status["status"] == "ok"
    assert elapsed < 0.1
    assert not prediction.is_alive()
