#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${HOST_HEAVY_BUILD_GATE:-$HOME/.local/bin/host-heavy-build}"
[[ -x "$GATE" ]] || {
  echo "LocalBooru compilation requires the host build gate: $GATE" >&2
  exit 75
}

exec "$GATE" run \
  --project localbooru-dev-compile \
  --worktree "$ROOT" \
  --wait "${HOST_HEAVY_BUILD_WAIT_SECONDS:-0}" \
  -- "$@"
