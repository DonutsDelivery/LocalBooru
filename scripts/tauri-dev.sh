#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_BUILD_JOBS="${LOCALBOORU_DEV_BUILD_JOBS:-1}"
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-$ROOT/scripts/rustc-host-heavy-build.sh}"
# Trim while the shared gate is held, then release it for dev hot compilation.
"$ROOT/scripts/run-cargo.sh" metadata --no-deps --format-version=1 >/dev/null
exec cargo tauri dev "$@"
