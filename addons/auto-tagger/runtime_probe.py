#!/usr/bin/env python3
"""Profile a real Auto Tagger model in the current Python environment."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def package_versions():
    names = [
        "numpy",
        "onnxruntime",
        "onnxruntime-gpu",
        "nvidia-cublas-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-nvjitlink-cu12",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_profile(path):
    with open(path, "r", encoding="utf-8") as profile_file:
        document = json.load(profile_file)
    events = document.get("traceEvents", []) if isinstance(document, dict) else document
    counts = {}
    durations_us = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if not provider:
            continue
        counts[provider] = counts.get(provider, 0) + 1
        durations_us[provider] = durations_us.get(provider, 0.0) + float(
            event.get("dur", 0.0)
        )
    return counts, {
        provider: round(duration / 1000.0, 3)
        for provider, duration in durations_us.items()
    }


def numpy_dtype(input_type):
    types = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }
    try:
        return types[input_type]
    except KeyError as error:
        raise ValueError(f"Unsupported model input type: {input_type}") from error


def nvidia_smi_info():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "output": None, "error": "nvidia-smi not found"}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "available": True,
            "output": completed.stdout.strip(),
            "error": None,
        }
    except Exception as error:
        return {"available": True, "output": None, "error": str(error)}


def runtime_details(providers, preload, args, debug_output, debug_error):
    return {
        "python": sys.version.split()[0],
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": package_versions(),
        "available_providers": ort.get_available_providers(),
        "requested_providers": providers,
        "preload": preload,
        "optimization": args.optimization,
        "cpu_fallback_disabled": args.disable_cpu_fallback,
        "ort_debug_output": debug_output,
        "ort_debug_error": debug_error,
        "nvidia_smi": nvidia_smi_info(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--optimization", choices=("all", "basic", "disabled"), default="all"
    )
    parser.add_argument("--disable-cpu-fallback", action="store_true")
    parser.add_argument("--debug-info", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    preload = {"attempted": args.provider == "cuda", "succeeded": None, "error": None}
    if args.provider == "cuda":
        try:
            ort.preload_dlls(directory="")
            preload["succeeded"] = True
        except Exception as error:
            preload["succeeded"] = False
            preload["error"] = str(error)

    debug_output = None
    debug_error = None
    if args.debug_info:
        try:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                ort.print_debug_info()
            debug_output = captured.getvalue().strip()
        except Exception as error:
            debug_error = str(error)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(
        Path(tempfile.gettempdir())
        / f"localbooru-runtime-probe-{os.getpid()}-{time.time_ns()}"
    )
    levels = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    options.graph_optimization_level = levels[args.optimization]
    if args.disable_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if args.verbose:
        options.log_severity_level = 0
        options.log_verbosity_level = 1
        options.logid = "localbooru-runtime-probe"

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    runtime = runtime_details(
        providers, preload, args, debug_output, debug_error
    )
    model = {
        "path": str(args.model.resolve()),
        "sha256": sha256_file(args.model),
        "bytes": args.model.stat().st_size,
    }
    profile_path = None
    try:
        session = ort.InferenceSession(
            str(args.model), sess_options=options, providers=providers
        )
        model_input = session.get_inputs()[0]
        shape = [
            1 if not isinstance(value, int) or value <= 0 else value
            for value in model_input.shape
        ]
        array = np.zeros(shape, dtype=numpy_dtype(model_input.type))
        model.update(
            {
                "input_name": model_input.name,
                "input_shape": model_input.shape,
                "input_type": model_input.type,
            }
        )
        runtime.update(
            {
                "registered_providers": session.get_providers(),
                "provider_options": session.get_provider_options(),
            }
        )
        started = time.perf_counter()
        session.run(None, {model_input.name: array})
        inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
        profile_path = session.end_profiling()
        counts, durations = summarize_profile(profile_path)
        execution = {
            "inference_ms": inference_ms,
            "provider_node_counts": counts,
            "provider_duration_ms": durations,
            "error_type": None,
            "error": None,
        }
        exit_code = 0
    except Exception as error:
        execution = {
            "inference_ms": None,
            "provider_node_counts": {},
            "provider_duration_ms": {},
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_code = 1
    finally:
        if profile_path:
            Path(profile_path).unlink(missing_ok=True)
        prefix = Path(options.profile_file_prefix)
        for leftover in prefix.parent.glob(f"{prefix.name}*.json"):
            leftover.unlink(missing_ok=True)

    print(
        json.dumps(
            {"model": model, "runtime": runtime, "execution": execution},
            indent=2,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
