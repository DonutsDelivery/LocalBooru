#!/usr/bin/env python3
"""Apply the small Flatpak launcher fix without discarding the WebKit cache."""
import argparse
import hashlib
from pathlib import Path
import subprocess

PATCH = Path(__file__).resolve().parents[1] / "patches/webkitgtk/2.52.3-flatpak-video-bridge.patch"


def prepare(source, apply=False, patch=PATCH):
    patch = patch.resolve()
    entry = next(line.split()[2:] for line in patch.read_text().splitlines()
                 if line.startswith("# preimage "))
    before, after, relative = entry
    digest = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    if digest == after:
        return "already upgraded"
    if digest != before:
        raise ValueError("Unrecognized native runtime source; refusing to modify the native cache")
    subprocess.run(["patch", "--batch", "--forward", "--dry-run", "-p1", "-i", str(patch)],
                   cwd=source, check=True, stdout=subprocess.DEVNULL)
    if apply:
        subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
                       cwd=source, check=True, stdout=subprocess.DEVNULL)
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != after:
            raise ValueError("Unexpected patched native source")
    return "upgraded" if apply else "recognized source; upgrade applies"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--patch", type=Path, default=PATCH)
    args = parser.parse_args()
    print(prepare(args.source, args.apply, args.patch))
