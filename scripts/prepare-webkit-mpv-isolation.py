#!/usr/bin/env python3
"""Upgrade only the reviewed legacy DMC relay, preserving other WebKit edits."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import re
import os

OBJECTS = (
    "Source/WebCore/CMakeFiles/WebCore.dir/__/__/WebCore/DerivedSources/unified-sources/UnifiedSource-3c72abbe-58.cpp.o",
    "Source/WebKit/CMakeFiles/WebKit.dir/__/__/DerivedSources/WebKit/unified-sources/UnifiedSource-54928a2b-47.cpp.o",
)


def validate_plan(output):
    allowed = {*("Building CXX object " + path for path in OBJECTS),
               "Linking CXX shared library lib/libwebkit2gtk-4.1.so.0.21.7",
               "Creating library symlink lib/libwebkit2gtk-4.1.so.0 lib/libwebkit2gtk-4.1.so",
               "Linking CXX executable bin/WebKitWebProcess"}
    lines = [line for line in output.splitlines() if line.strip()
             and not line.startswith("ninja: Entering directory ")
             and line != "ninja: no work to do."]
    tasks = [line for line in lines if re.match(r"^\[\d+/\d+\]", line)]
    unexpected = [line for line in tasks if re.sub(r"^\[\d+/\d+\] ", "", line) not in allowed]
    if unexpected or len(tasks) != len(lines) or len(tasks) > len(allowed):
        raise ValueError(f"Unsafe incremental cache: {len(tasks)} tasks, unrelated dependencies pending; "
                         "inspect Ninja explain and restore a compatible baseline before building")
    return tasks


def check_build_plan(build):
    output = subprocess.check_output(["ninja", "-n", "-C", str(build), "bin/WebKitWebProcess"], text=True,
                                     env={**os.environ, "NINJA_STATUS": "[%f/%t] "})
    tasks = validate_plan(output)
    print(f"Bounded incremental plan: {len(tasks)} tasks")
    for task in tasks:
        print(task)

PATCH = Path(__file__).resolve().parents[1] / "patches/webkitgtk/2.52.3-existing-mpv-relay-upgrade.patch"


def prepare(source, apply=False):
    entries = [line.split()[2:] for line in PATCH.read_text().splitlines()
               if line.startswith("# preimage ")]
    states = []
    for before, after, relative in entries:
        path = source / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in (before, after):
            raise ValueError(f"Unrecognized source preimage: {relative}; inspect/rebase the upgrade patch")
        states.append(digest == after)
    if all(states):
        return "already upgraded"
    if any(states):
        raise ValueError("Partially upgraded source; inspect before continuing")
    subprocess.run(["patch", "--batch", "--forward", "--dry-run", "-p1", "-i", str(PATCH)],
                   cwd=source, check=True, stdout=subprocess.DEVNULL)
    if apply:
        subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(PATCH)],
                       cwd=source, check=True, stdout=subprocess.DEVNULL)
        for _, after, relative in entries:
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != after:
                raise ValueError(f"Unexpected upgraded source: {relative}")
    return "upgraded" if apply else "recognized legacy source; upgrade applies"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="otherwise perform a read-only preimage/dry-run check")
    parser.add_argument("--build-cache", type=Path, help="reject unrelated dirty Ninja targets before source mutation")
    args = parser.parse_args()
    if args.build_cache:
        check_build_plan(args.build_cache)
    print(prepare(args.source, args.apply))
    if args.build_cache and args.apply:
        check_build_plan(args.build_cache)
