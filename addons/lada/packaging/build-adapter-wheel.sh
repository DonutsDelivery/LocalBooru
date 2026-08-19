#!/usr/bin/env bash
set -euo pipefail

source_root="$(cd "${1:?staged adapter source is required}" && pwd)"
out="${2:?wheel output directory is required}"

rm -rf "$out"
mkdir -p "$out"
uv build --wheel --out-dir "$out" "$source_root"
find "$out" -maxdepth 1 -type f -name '*.whl' -print -quit
