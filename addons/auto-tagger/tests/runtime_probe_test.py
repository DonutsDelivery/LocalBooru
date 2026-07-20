import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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
