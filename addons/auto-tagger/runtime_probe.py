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

if "--inventory-only" in sys.argv:
    np = None
    ort = None
else:
    import numpy as np
    import onnxruntime as ort


def package_versions():
    names = ["numpy", "onnxruntime", "onnxruntime-gpu"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    for distribution in importlib.metadata.distributions():
        name = (
            (distribution.metadata.get("Name") or "")
            .lower()
            .replace("_", "-")
            .replace(".", "-")
        )
        if name.startswith("nvidia-"):
            versions[name] = distribution.version
    return dict(sorted(versions.items()))


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


def native_library_inventory():
    libraries = []
    seen = set()
    distributions = []
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith("nvidia-") or name.startswith("onnxruntime"):
            distributions.append((name, distribution))
    for package, distribution in distributions:
        for relative_path in distribution.files or []:
            filename = Path(str(relative_path)).name.lower()
            is_native_library = (
                filename.endswith((".dll", ".dylib", ".pyd", ".so"))
                or ".so." in filename
            )
            if not is_native_library:
                continue
            path = Path(distribution.locate_file(relative_path))
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            libraries.append(
                {
                    "package": package,
                    "name": path.name,
                    "path": key,
                    "exists": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                }
            )
    return sorted(libraries, key=lambda item: (item["package"], item["name"], item["path"]))


def provider_spec(provider):
    if provider == "cuda":
        return [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def needs_strict_stage(execution):
    counts = execution.get("provider_node_counts") or {}
    return counts.get("CUDAExecutionProvider", 0) == 0


def strict_stage_succeeded(stage):
    execution = stage.get("execution") or {}
    counts = execution.get("provider_node_counts") or {}
    return (
        execution.get("error") is None
        and counts.get("CUDAExecutionProvider", 0) > 0
    )


def metadata_inventory():
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": package_versions(),
        "native_libraries": native_library_inventory(),
        "nvidia_smi": nvidia_smi_info(),
    }


def runtime_details(providers, preload, args, debug_output, debug_error):
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
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
        "native_libraries": native_library_inventory(),
    }


def execute_stage(model_path, args, providers, *, disable_cpu_fallback=False):
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
    if disable_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if args.verbose:
        options.log_severity_level = 0
        options.log_verbosity_level = 1
        options.logid = "localbooru-runtime-probe"

    profile_path = None
    registered_providers = []
    provider_options = {}
    model_input_details = None
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
            enable_fallback=0 if args.disable_wrapper_fallback else 1,
        )
        if args.disable_wrapper_fallback:
            session.disable_fallback()
        registered_providers = session.get_providers()
        provider_options = session.get_provider_options()
        model_input = session.get_inputs()[0]
        shape = [
            1 if not isinstance(value, int) or value <= 0 else value
            for value in model_input.shape
        ]
        array = np.zeros(shape, dtype=numpy_dtype(model_input.type))
        model_input_details = {
            "input_name": model_input.name,
            "input_shape": model_input.shape,
            "input_type": model_input.type,
        }
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
    except Exception as error:
        execution = {
            "inference_ms": None,
            "provider_node_counts": {},
            "provider_duration_ms": {},
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        if profile_path:
            Path(profile_path).unlink(missing_ok=True)
        prefix = Path(options.profile_file_prefix)
        for leftover in prefix.parent.glob(f"{prefix.name}*.json"):
            leftover.unlink(missing_ok=True)

    return {
        "requested_providers": providers,
        "registered_providers": registered_providers,
        "provider_options": provider_options,
        "cpu_ep_fallback_disabled": disable_cpu_fallback,
        "wrapper_fallback_disabled": args.disable_wrapper_fallback,
        "model_input": model_input_details,
        "execution": execution,
    }


def main():
    global np, ort

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--optimization", choices=("all", "basic", "disabled"), default="all"
    )
    parser.add_argument("--disable-cpu-fallback", action="store_true")
    parser.add_argument("--disable-wrapper-fallback", action="store_true")
    parser.add_argument("--strict-on-zero-cuda", action="store_true")
    parser.add_argument("--debug-info", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.inventory_only and args.model is None:
        parser.error("model is required unless --inventory-only is used")

    if args.inventory_only and ort is None:
        early_report = {
            "inventory_only": True,
            "runtime": {
                **metadata_inventory(),
                "ort_import": "pending",
            },
        }
        print(json.dumps(early_report, sort_keys=True), flush=True)
        try:
            import numpy as numpy_module
            import onnxruntime as ort_module
        except Exception as error:
            early_report["runtime"]["ort_import"] = "failed"
            early_report["runtime"]["ort_import_error"] = str(error)
            print(json.dumps(early_report, indent=2, sort_keys=True), flush=True)
            return 1
        np = numpy_module
        ort = ort_module

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
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                ort.print_debug_info()
            debug_output = captured.getvalue().strip()[-65536:]
        except Exception as error:
            debug_error = str(error)

    providers = provider_spec(args.provider)
    runtime = runtime_details(
        providers, preload, args, debug_output, debug_error
    )
    if args.inventory_only:
        print(json.dumps({"inventory_only": True, "runtime": runtime}, indent=2, sort_keys=True))
        return 0

    model = {
        "path": str(args.model.resolve()),
        "name": args.model.parent.name or args.model.name,
        "sha256": sha256_file(args.model),
        "bytes": args.model.stat().st_size,
    }
    primary = execute_stage(
        args.model,
        args,
        providers,
        disable_cpu_fallback=args.disable_cpu_fallback,
    )
    if primary["model_input"]:
        model.update(primary["model_input"])
    runtime.update(
        {
            "registered_providers": primary["registered_providers"],
            "provider_options": primary["provider_options"],
            "wrapper_fallback_disabled": primary["wrapper_fallback_disabled"],
        }
    )
    execution = primary["execution"]
    strict_stage = None
    if (
        args.provider == "cuda"
        and args.strict_on_zero_cuda
        and needs_strict_stage(execution)
    ):
        strict_stage = execute_stage(
            args.model,
            args,
            [("CUDAExecutionProvider", {"device_id": 0})],
            disable_cpu_fallback=True,
        )

    report = {
        "model": model,
        "runtime": runtime,
        "execution": execution,
        "strict_stage": strict_stage,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if execution["error"] is not None:
        return 1
    if strict_stage and not strict_stage_succeeded(strict_stage):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
