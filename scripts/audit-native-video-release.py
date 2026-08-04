#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate and gate native-video helper release provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def ffmpeg_configuration() -> tuple[str, list[str]]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-buildconf"],
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    options = sorted(set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_=-]*", output)))
    if not options:
        raise RuntimeError("ffmpeg -buildconf returned no configuration options")
    return "\n".join(options) + "\n", options


def linked_dependencies(binary: Path) -> list[dict[str, object]]:
    dependencies: dict[str, Path] = {}
    for raw_line in command_output("ldd", str(binary)).splitlines():
        line = raw_line.strip()
        if "not found" in line:
            raise RuntimeError(f"unresolved helper dependency: {line}")
        match = re.match(r"(?P<name>\S+)\s+=>\s+(?P<path>/\S+)\s+\(", line)
        if not match:
            match = re.match(r"(?P<path>/\S+)\s+\(", line)
        if not match:
            continue
        path = Path(match.group("path")).resolve()
        name = match.groupdict().get("name") or path.name
        dependencies[f"{name}:{path}"] = path
    return [
        {
            "name": key.split(":", 1)[0],
            "path": str(path),
            "sha256": sha256(path),
            "license_concluded": "NOASSERTION",
        }
        for key, path in sorted(dependencies.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", action="append", default=[], type=Path)
    parser.add_argument("--expected-ffmpeg-config-sha256")
    parser.add_argument("--notices", type=Path)
    parser.add_argument(
        "--licenses",
        type=Path,
        help="JSON object mapping every linked library name to an SPDX expression",
    )
    parser.add_argument("--release", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"helper is not an executable file: {binary}")
    fixtures = [fixture.resolve() for fixture in args.fixture]
    missing = [str(path) for path in fixtures if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing release fixture(s): {', '.join(missing)}")

    configuration, options = ffmpeg_configuration()
    configuration_hash = hashlib.sha256(configuration.encode()).hexdigest()
    nonfree = "--enable-nonfree" in options
    pinned = args.expected_ffmpeg_config_sha256 == configuration_hash
    notices_present = bool(
        args.notices and args.notices.is_file() and args.notices.stat().st_size > 0
    )
    licenses: dict[str, str] = {}
    if args.licenses:
        loaded = json.loads(args.licenses.read_text())
        if not isinstance(loaded, dict) or not all(
            isinstance(name, str) and isinstance(expression, str) and expression.strip()
            for name, expression in loaded.items()
        ):
            raise RuntimeError("--licenses must be a JSON object of non-empty SPDX expressions")
        licenses = loaded
    dependencies = linked_dependencies(binary)
    for dependency in dependencies:
        dependency["license_concluded"] = licenses.get(
            str(dependency["name"]), "NOASSERTION"
        )

    failures: list[str] = []
    if nonfree:
        failures.append("FFmpeg was configured with --enable-nonfree")
    if args.expected_ffmpeg_config_sha256 and not pinned:
        failures.append("FFmpeg configuration hash does not match the pinned release hash")
    if args.release and not args.expected_ffmpeg_config_sha256:
        failures.append("release audit requires --expected-ffmpeg-config-sha256")
    if args.release and not notices_present:
        failures.append("release audit requires a non-empty --notices file")
    if args.release and not args.licenses:
        failures.append("release audit requires --licenses")
    unresolved_licenses = [
        str(dependency["name"])
        for dependency in dependencies
        if dependency["license_concluded"] == "NOASSERTION"
    ]
    if args.release and unresolved_licenses:
        failures.append(
            "release dependency licenses remain unresolved: "
            + ", ".join(unresolved_licenses)
        )

    manifest = {
        "schema": "localbooru-native-video-provenance-v1",
        "distribution_approved": args.release and not failures,
        "binary": {
            "path": str(binary),
            "sha256": sha256(binary),
        },
        "ffmpeg": {
            "configuration_sha256": configuration_hash,
            "configuration": options,
            "gpl_enabled": "--enable-gpl" in options,
            "version3_enabled": "--enable-version3" in options,
            "nonfree_enabled": nonfree,
        },
        "dependencies": dependencies,
        "fixtures": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in fixtures
        ],
        "notices": str(args.notices.resolve()) if notices_present else None,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if failures:
        for failure in failures:
            print(f"release audit failed: {failure}", file=sys.stderr)
        return 2
    print(
        f"native-video provenance written: {args.output} "
        f"(dependencies={len(manifest['dependencies'])}, fixtures={len(fixtures)}, "
        f"distribution_approved={manifest['distribution_approved']})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"native-video release audit error: {error}", file=sys.stderr)
        raise SystemExit(1)
