#!/usr/bin/env bash
# Build-only dependency: Microsoft native MSIX SDK, never bundled into DMC.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gate=${HOST_HEAVY_BUILD_GATE:-$HOME/.local/bin/host-heavy-build}
cache=${LOCALBOORU_MSIX_TOOLS_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/donutmediacenter/msix-tools}
[[ -x "$gate" ]] || { echo 'The host heavy-build gate is required.' >&2; exit 75; }
args=(--cache "$cache")
if [[ -n "${LOCALBOORU_MSIX_SOURCE_ARCHIVE:-}" ]]; then args+=(--archive "$LOCALBOORU_MSIX_SOURCE_ARCHIVE"); fi
exec "$gate" run --project DonutMediaCenter --worktree "$root" --wait 21600 -- \
  python3 "$root/scripts/build-msix-tools.py" "${args[@]}"
