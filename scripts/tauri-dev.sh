#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_BUILD_JOBS="${LOCALBOORU_DEV_BUILD_JOBS:-1}"
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-$ROOT/scripts/rustc-host-heavy-build.sh}"
exec cargo tauri dev "$@"
