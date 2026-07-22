#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
mkdir -p "$TEMP_DIR/bin"

cat >"$TEMP_DIR/bin/fake-host-gate" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$FAKE_GATE_ARGS"
while (($#)); do
  if [[ "$1" == -- ]]; then shift; break; fi
  shift
done
exec "$@"
EOF

cat >"$TEMP_DIR/bin/fake-rustc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$FAKE_RUSTC_ARGS"
EOF

cat >"$TEMP_DIR/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${CARGO_BUILD_JOBS:-unset}" >"$FAKE_CARGO_JOBS"
printf '%s\n' "${RUSTC_WRAPPER:-unset}" >"$FAKE_CARGO_WRAPPER"
printf '%s\n' "$@" >"$FAKE_CARGO_ARGS"
EOF
chmod +x "$TEMP_DIR/bin/"*

export FAKE_GATE_ARGS="$TEMP_DIR/gate-args"
export FAKE_RUSTC_ARGS="$TEMP_DIR/rustc-args"
export FAKE_CARGO_JOBS="$TEMP_DIR/cargo-jobs"
export FAKE_CARGO_WRAPPER="$TEMP_DIR/cargo-wrapper"
export FAKE_CARGO_ARGS="$TEMP_DIR/cargo-args"

HOST_HEAVY_BUILD_GATE="$TEMP_DIR/bin/fake-host-gate" \
  "$ROOT/scripts/rustc-host-heavy-build.sh" "$TEMP_DIR/bin/fake-rustc" --crate-name sample

grep -Fx -- '--project' "$FAKE_GATE_ARGS" >/dev/null
grep -Fx -- 'localbooru-dev-compile' "$FAKE_GATE_ARGS" >/dev/null
grep -Fx -- "$TEMP_DIR/bin/fake-rustc" "$FAKE_GATE_ARGS" >/dev/null
printf '%s\n' --crate-name sample >"$TEMP_DIR/expected-rustc"
cmp "$TEMP_DIR/expected-rustc" "$FAKE_RUSTC_ARGS"

PATH="$TEMP_DIR/bin:$PATH" LOCALBOORU_DEV_BUILD_JOBS=1 \
  "$ROOT/scripts/tauri-dev.sh" -- --sample
[[ "$(cat "$FAKE_CARGO_JOBS")" == 1 ]]
[[ "$(cat "$FAKE_CARGO_WRAPPER")" == "$ROOT/scripts/rustc-host-heavy-build.sh" ]]
printf '%s\n' tauri dev -- --sample >"$TEMP_DIR/expected-cargo"
cmp "$TEMP_DIR/expected-cargo" "$FAKE_CARGO_ARGS"

printf 'Host build integration tests passed\n'
