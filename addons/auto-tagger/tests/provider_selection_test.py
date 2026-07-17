import importlib.util
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("localbooru_auto_tagger", APP_PATH)
TAGGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TAGGER)


class FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class FakeOrt:
    def __init__(self, available, *, fail_cuda=False, active_cuda=True):
        self.available = available
        self.fail_cuda = fail_cuda
        self.active_cuda = active_cuda
        self.events = []

    def preload_dlls(self, *, directory):
        self.events.append(("preload", directory))

    def get_available_providers(self):
        self.events.append(("available",))
        return self.available

    def InferenceSession(self, _model_path, *, sess_options, providers):
        self.events.append(("session", tuple(providers)))
        if providers[0] == "CUDAExecutionProvider":
            if self.fail_cuda:
                raise RuntimeError("CUDA runtime unavailable")
            if not self.active_cuda:
                return FakeSession(["CPUExecutionProvider"])
        return FakeSession(list(providers))


# AC: @auto-tagger-provider-diagnostics ac-1
# AC: @addon-platform-dependencies ac-1
def test_cuda_preloads_packaged_libraries_before_provider_discovery():
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])

    session, available, warning = TAGGER._create_inference_session(
        ort, "model.onnx", object(), "cuda"
    )

    assert ort.events[0] == ("preload", "")
    assert ort.events[1] == ("available",)
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    assert available[0] == "CUDAExecutionProvider"
    assert warning is None


# AC: @auto-tagger-provider-diagnostics ac-2
# AC: @addon-platform-dependencies ac-2
def test_cuda_initialization_failure_retries_cpu_with_warning():
    ort = FakeOrt(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], fail_cuda=True
    )

    session, _available, warning = TAGGER._create_inference_session(
        ort, "model.onnx", object(), "cuda"
    )

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert ort.events[-1] == ("session", ("CPUExecutionProvider",))
    assert "CUDA runtime unavailable" in warning


# AC: @auto-tagger-provider-diagnostics ac-2
def test_advertised_cuda_that_activates_cpu_is_reported_as_fallback():
    ort = FakeOrt(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], active_cuda=False
    )

    session, _available, warning = TAGGER._create_inference_session(
        ort, "model.onnx", object(), "auto"
    )

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert "using CPUExecutionProvider" in warning


# AC: @auto-tagger-provider-diagnostics ac-3
def test_explicit_cpu_skips_cuda_preload_and_activation():
    ort = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])

    session, _available, warning = TAGGER._create_inference_session(
        ort, "model.onnx", object(), "cpu"
    )

    assert not any(event[0] == "preload" for event in ort.events)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert warning is None
