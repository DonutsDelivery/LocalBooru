#!/usr/bin/env bash
# Build and verify release-signed Android artifacts with the permanent DMC key.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/android-release.py" "$@"
