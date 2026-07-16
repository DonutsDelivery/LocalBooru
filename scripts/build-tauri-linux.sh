#!/usr/bin/env bash
# Backwards-compatible name for the project-owned Docker release wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build-linux-local.sh" "$@"