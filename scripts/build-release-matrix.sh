#!/usr/bin/env bash
# Preserve the complete stable matrix and serialize its platform wrappers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (($#)); then
  echo "Usage: scripts/build-release-matrix.sh (configure platform wrappers via environment)" >&2
  exit 2
fi
python3 "$ROOT/scripts/check-release-version.py"
export LOCALBOORU_SOURCE_REVISION="$(git -C "$ROOT" rev-parse --verify "${LOCALBOORU_SOURCE_REVISION:-HEAD}^{commit}")"
"$ROOT/scripts/build-linux-local.sh"
"$ROOT/scripts/build-windows-local.sh"
