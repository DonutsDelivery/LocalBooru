#!/usr/bin/env python3
"""Keep DMC's bare executable compatible with linuxdeploy's AppRun parser."""
import argparse
from pathlib import Path


def normalize(path, check=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular staged desktop entry")
    original = path.read_bytes()
    lines = original.decode("utf-8").splitlines(keepends=True)
    count = 0
    for index, line in enumerate(lines):
        if not line.startswith("Exec="):
            continue
        count += 1
        value = line[5:]
        for token in ('"donutmediacenter"', "donutmediacenter"):
            if value.startswith(token) and (
                len(value) == len(token) or value[len(token)].isspace()
            ):
                lines[index] = "Exec=donutmediacenter" + value[len(token):]
                break
        else:
            raise ValueError("Unexpected AppImage executable; entry left unchanged")
    if not count:
        raise ValueError("Desktop entry has no executable")
    updated = "".join(lines).encode("utf-8")
    if check and updated != original:
        raise ValueError("AppImage executable still contains literal quotes")
    if not check and updated != original:
        path.write_bytes(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    normalize(args.path, check=args.check)
