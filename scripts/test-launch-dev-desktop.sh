#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TEMP_DIR/bin"

cat >"$TEMP_DIR/bin/ss" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ -f "$FAKE_VITE_READY" ]]; then
    printf 'LISTEN 0 511 127.0.0.1:5210 0.0.0.0:*\n'
fi
EOF
chmod +x "$TEMP_DIR/bin/ss"

cat >"$TEMP_DIR/bin/npm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >>"$FAKE_NPM_CALLS"
touch "$FAKE_VITE_READY"
EOF
chmod +x "$TEMP_DIR/bin/npm"

cat >"$TEMP_DIR/localbooru" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >"$FAKE_BINARY_ARGS"
EOF
chmod +x "$TEMP_DIR/localbooru"

export PATH="$TEMP_DIR/bin:$PATH"
export FAKE_VITE_READY="$TEMP_DIR/vite-ready"
export FAKE_NPM_CALLS="$TEMP_DIR/npm-calls"
export FAKE_BINARY_ARGS="$TEMP_DIR/binary-args"

# A ready frontend must launch the binary immediately without starting npm.
touch "$FAKE_VITE_READY"
LOCALBOORU_DEV_BINARY="$TEMP_DIR/localbooru" "$ROOT/scripts/launch-dev-desktop.sh" "/tmp/one image.png"
[[ "$(<"$FAKE_BINARY_ARGS")" == "/tmp/one image.png" ]]
[[ ! -e "$FAKE_NPM_CALLS" ]]

# A missing frontend starts Vite, waits for readiness, then launches the binary.
rm -f "$FAKE_VITE_READY" "$FAKE_BINARY_ARGS"
LOCALBOORU_DEV_BINARY="$TEMP_DIR/localbooru" "$ROOT/scripts/launch-dev-desktop.sh" "/tmp/two image.png"
[[ "$(<"$FAKE_BINARY_ARGS")" == "/tmp/two image.png" ]]
grep -F 'run dev -- --port 5210' "$FAKE_NPM_CALLS" >/dev/null

# A desktop click must fail clearly when no previously built binary exists.
if LOCALBOORU_DEV_BINARY="$TEMP_DIR/missing" "$ROOT/scripts/launch-dev-desktop.sh" >"$TEMP_DIR/missing.out" 2>&1; then
    printf 'missing Dev binary unexpectedly launched\n' >&2
    exit 1
fi
grep -F 'Rebuild explicitly with:' "$TEMP_DIR/missing.out" >/dev/null

printf 'Desktop Dev launcher tests passed\n'
