#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_SECONDS="${LOCALBOORU_DEV_BUILD_WAIT_SECONDS:-7200}"

if [[ ! "$WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "LOCALBOORU_DEV_BUILD_WAIT_SECONDS must be a non-negative integer" >&2
    exit 2
fi

# Tauri's Dev command does not reliably preserve the host gate wait setting for
# RUSTC_WRAPPER. Set it in the wrapper process itself so a desktop launch queues
# instead of returning Cargo's fail-fast status 75.
export HOST_HEAVY_BUILD_WAIT_SECONDS="$WAIT_SECONDS"
exec "$ROOT/scripts/rustc-host-heavy-build.sh" "$@"
