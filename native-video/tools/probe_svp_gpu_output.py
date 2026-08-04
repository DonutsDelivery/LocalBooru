#!/usr/bin/env python3
"""Probe whether installed SVPflow/VapourSynth exposes exportable GPU frames.

This probe intentionally does not call VideoFrame.get_read_ptr(): it inspects the
actual SmoothFps_NVOF output object, its properties, the public VapourSynth API,
and installed plugin symbols for an external GPU-memory export contract.
"""

from __future__ import annotations

import argparse
import math
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

HANDLE_TERMS = re.compile(r"cuda|vulkan|opencl|dmabuf|dma_buf|eglimage|gpu.?handle|export.?handle", re.I)
COMPUTE_TERMS = re.compile(r"NVOF|GPURenderer|SmoothFps", re.I)
PLUGIN_CANDIDATES = (
    Path.home() / "SVP 4/plugins",
    Path("/opt/svp/plugins"),
    Path("/usr/lib/svp/plugins"),
)


def discover_plugins(configured: str | None) -> Path:
    candidates = []
    if configured:
        candidates.append(Path(configured))
    if os.environ.get("LOCALBOORU_SVP_PLUGIN_PATH"):
        candidates.append(Path(os.environ["LOCALBOORU_SVP_PLUGIN_PATH"]))
    candidates.extend(PLUGIN_CANDIDATES)
    for candidate in candidates:
        if (candidate / "libsvpflow1.so").is_file() and (candidate / "libsvpflow2.so").is_file():
            return candidate.resolve()
    raise RuntimeError("libsvpflow1.so and libsvpflow2.so were not found")


def command_output(args: list[str]) -> str:
    result = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.strip()


def public_header(vs_module: Any) -> Path | None:
    candidate = Path(vs_module.__file__).resolve().parent / "include/VapourSynth4.h"
    return candidate if candidate.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-path")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    import vapoursynth as vs

    plugin_path = discover_plugins(args.plugin_path)
    flow1 = plugin_path / "libsvpflow1.so"
    flow2 = plugin_path / "libsvpflow2.so"
    core = vs.core
    std = getattr(core, "std")
    std.LoadPlugin(str(flow1))
    std.LoadPlugin(str(flow2))

    signatures: dict[str, dict[str, str]] = {}
    for plugin in core.plugins():
        if plugin.namespace not in {"svp1", "svp2"}:
            continue
        signatures[plugin.namespace] = {
            function.name: function.signature for function in plugin.functions()
        }

    # 640x512 with a quarter-resolution vector clip is the smallest reliable
    # NVOF test geometry accepted by this installed SVPflow build.
    source_length = max(30, math.ceil(args.frames * (24000 / 1001) / 60) + 2)
    source = std.BlankClip(
        width=640,
        height=512,
        format=vs.YUV420P8,
        length=source_length,
        fpsnum=24000,
        fpsden=1001,
        color=[32, 128, 128],
    )
    vectors = source.resize.Bicubic(width=160, height=128, format=vs.YUV420P8)
    options = "{rate:{num:60,den:1,abs:true},gpuid:0,algo:23,mask:{area:100},scene:{}}"
    output = getattr(core, "svp2").SmoothFps_NVOF(
        source,
        options,
        vec_src=vectors,
        src=source,
        fps=source.fps.numerator / source.fps.denominator,
    )

    frame_attributes: set[str] = set()
    property_keys: set[str] = set()
    frame_types: set[str] = set()
    start = time.monotonic()
    for index in range(args.frames):
        frame = output.get_frame(index)
        frame_types.add(type(frame).__name__)
        frame_attributes.update(name for name in dir(frame) if not name.startswith("__"))
        property_keys.update(str(name) for name in frame.props.keys())
        frame.close()
    elapsed = time.monotonic() - start

    header = public_header(vs)
    header_handle_lines: list[str] = []
    if header:
        for line in header.read_text(errors="replace").splitlines():
            if HANDLE_TERMS.search(line):
                header_handle_lines.append(line.strip())

    plugin_symbols = command_output(["nm", "-D", "--defined-only", str(flow2)]).splitlines()
    relevant_symbols = [line for line in plugin_symbols if HANDLE_TERMS.search(line)]
    compute_symbols = [line for line in plugin_symbols if COMPUTE_TERMS.search(line)]
    dependencies = {
        flow1.name: command_output(["ldd", str(flow1)]).splitlines(),
        flow2.name: command_output(["ldd", str(flow2)]).splitlines(),
    }
    exportable_frame_attributes = sorted(name for name in frame_attributes if HANDLE_TERMS.search(name))
    exportable_frame_properties = sorted(name for name in property_keys if HANDLE_TERMS.search(name))
    export_api_proven = bool(
        exportable_frame_attributes or exportable_frame_properties or header_handle_lines
    )

    report = {
        "probe_version": 1,
        "vapoursynth": {
            "module": str(Path(vs.__file__).resolve()),
            "api_version": str(getattr(vs, "__api_version__", "unknown")),
            "core": str(core),
            "public_header": str(header) if header else None,
        },
        "plugins": {
            "path": str(plugin_path),
            "signatures": signatures,
            "dependencies": dependencies,
            "gpu_or_export_symbols": relevant_symbols,
            "nvof_or_renderer_symbols": compute_symbols,
        },
        "graph": {
            "filter": "svp2.SmoothFps_NVOF",
            "options": options,
            "input_format": source.format.name,
            "output_format": output.format.name,
            "width": output.width,
            "height": output.height,
            "input_fps": str(source.fps),
            "output_fps": str(output.fps),
            "frames_requested": args.frames,
            "seconds": elapsed,
        },
        "output_frames": {
            "types": sorted(frame_types),
            "public_attributes": sorted(frame_attributes),
            "property_keys": sorted(property_keys),
            "exportable_handle_attributes": exportable_frame_attributes,
            "exportable_handle_properties": exportable_frame_properties,
            "public_header_export_apis": header_handle_lines,
            "cpu_plane_access_api_present": "get_read_ptr" in frame_attributes,
            "cpu_plane_access_invoked": False,
        },
        "decision": {
            "exportable_gpu_output_proven": export_api_proven,
            "zero_copy_svp_supported": export_api_proven,
            "conclusion": (
                "An external GPU-frame export contract was found; adapter work requires manual validation."
                if export_api_proven
                else "No CUDA/Vulkan/OpenCL/DMA-BUF/EGLImage output handle is exposed by the installed SVPflow/VapourSynth API. Retain planar YUV420 SHM and prohibit zero-copy SVP claims."
            ),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"probe_version": 1, "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
