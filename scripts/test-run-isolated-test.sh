#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d /tmp/localbooru-isolation-tests.XXXXXX)"
trap 'rm -rf "$TEST_DIR"' EXIT
REAL_MKTEMP="$(command -v mktemp)"
export REAL_MKTEMP
mkdir -p "$TEST_DIR/bin" "$TEST_DIR/caller/home" "$TEST_DIR/caller/cache"
cat >"$TEST_DIR/bin/mktemp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
root="$($REAL_MKTEMP "$@")"
printf '%s\n' "$root" >>"$SYNTHETIC_ROOTS"
printf '%s\n' "$root"
EOF
cat >"$TEST_DIR/bin/python3" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == '-' && "$2" == '18790' ]]
cat >/dev/null
EOF
cat >"$TEST_DIR/bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$SYNTHETIC_PROBES"
[[ "$*" == *'http://127.0.0.1:18790/health'* ]]
[[ "${SYNTHETIC_READY:-1}" == '1' ]]
EOF
cat >"$TEST_DIR/candidate" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 ]]
root="${HOME%/home}"
[[ "$root" == /tmp/localbooru-isolated.* ]]
[[ "$HOME" == "$root/home" && "$XDG_CONFIG_HOME" == "$root/config" ]]
[[ "$XDG_CACHE_HOME" == "$root/cache" && "$XDG_DATA_HOME" == "$root/xdg-data" ]]
[[ "$XDG_STATE_HOME" == "$root/state" && "$TMPDIR" == "$root/tmp" ]]
[[ "$LOCALBOORU_PORTABLE_DATA" == "$root/library" ]]
[[ "$LOCALBOORU_DEV_TARGET_DIR" == "$root/target" && "$CARGO_TARGET_DIR" == "$root/target" ]]
[[ "$LOCALBOORU_PORT" == '18790' && "$VITE_LOCALBOORU_PORT" == '18790' ]]
[[ "$LOCALBOORU_DISABLE_SINGLE_INSTANCE" == '1' && "$LOCALBOORU_ENABLE_NATIVE_SVP" == '0' ]]
[[ "$LOCALBOORU_DEV_LOG" == "$root/dev.log" ]]
mkdir -p "$LOCALBOORU_PORTABLE_DATA"
printf 'synthetic\n' >"$LOCALBOORU_PORTABLE_DATA/fixture.txt"
printf '%s\n' "$root" >>"$SYNTHETIC_REPORT"
printf '%s\n' "$$" >"$SYNTHETIC_PID"
if [[ "${SYNTHETIC_HOLD:-0}" == '1' ]]; then
  trap 'exit 0' TERM
  while :; do sleep 0.05; done
fi
EOF
chmod +x "$TEST_DIR/bin/"* "$TEST_DIR/candidate"
export PATH="$TEST_DIR/bin:$PATH"
export HOME="$TEST_DIR/caller/home" XDG_CACHE_HOME="$TEST_DIR/caller/cache"
export LOCALBOORU_PORTABLE_DATA="$TEST_DIR/caller/private-unused"
export LOCALBOORU_PORT=8790 LOCALBOORU_DISABLE_SINGLE_INSTANCE=0
export SYNTHETIC_ROOTS="$TEST_DIR/roots" SYNTHETIC_REPORT="$TEST_DIR/report"
export SYNTHETIC_PROBES="$TEST_DIR/probes" SYNTHETIC_PID="$TEST_DIR/pid"
export LOCALBOORU_TEST_PORT=18790 LOCALBOORU_TEST_READY_TIMEOUT_SECONDS=2
export LOCALBOORU_TEST_BINARY="$TEST_DIR/candidate"
# A prebuilt candidate performs no compiler work and requires no build gate.
export HOST_HEAVY_BUILD_GATE="$TEST_DIR/nonexistent-gate"
LAUNCHER="$ROOT/scripts/run-isolated-test.sh"
"$LAUNCHER" >"$TEST_DIR/candidate-output" 2>&1
[[ "$(wc -l <"$SYNTHETIC_REPORT")" -eq 1 ]]
[[ ! -e "$(tail -n 1 "$SYNTHETIC_REPORT")" ]]
[[ ! -e "$TEST_DIR/caller/private-unused" ]]
grep -F 'Isolated LocalBooru instance is ready' "$TEST_DIR/candidate-output" >/dev/null

# No application file arguments or relative/non-executable candidate paths.
if "$LAUNCHER" '/synthetic/file with spaces.mp4' >"$TEST_DIR/argument-output" 2>&1; then exit 1; fi
grep -F 'does not accept application file arguments' "$TEST_DIR/argument-output" >/dev/null
[[ "$(wc -l <"$SYNTHETIC_REPORT")" -eq 1 ]]
[[ ! -e "$(tail -n 1 "$SYNTHETIC_ROOTS")" ]]
for bad_binary in relative-candidate "$TEST_DIR/not-executable"; do
  touch "$TEST_DIR/not-executable"
  if LOCALBOORU_TEST_BINARY="$bad_binary" "$LAUNCHER" >"$TEST_DIR/invalid-output" 2>&1; then exit 1; fi
  grep -F 'absolute path to an executable candidate binary' "$TEST_DIR/invalid-output" >/dev/null
  [[ ! -e "$(tail -n 1 "$SYNTHETIC_ROOTS")" ]]
done

# A failed readiness check stops only the synthetic child and removes its roots.
if SYNTHETIC_READY=0 SYNTHETIC_HOLD=1 LOCALBOORU_TEST_READY_TIMEOUT_SECONDS=1 "$LAUNCHER" >"$TEST_DIR/timeout-output" 2>&1; then exit 1; fi
! kill -0 "$(cat "$SYNTHETIC_PID")" 2>/dev/null
[[ ! -e "$(tail -n 1 "$SYNTHETIC_ROOTS")" ]]
grep -F 'did not become ready within 1 seconds' "$TEST_DIR/timeout-output" >/dev/null

# Verify the unchanged default dev branch using a synthetic launcher checkout.
mkdir -p "$TEST_DIR/repo/scripts"
cp "$LAUNCHER" "$TEST_DIR/repo/scripts/run-isolated-test.sh"
cat >"$TEST_DIR/repo/run-dev.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 ]]
printf 'dev\n' >"$SYNTHETIC_DEV_MARKER"
exec "$SYNTHETIC_CANDIDATE"
EOF
cat >"$TEST_DIR/gate" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$TEST_DIR/repo/run-dev.sh" "$TEST_DIR/gate"
export SYNTHETIC_DEV_MARKER="$TEST_DIR/default-dev" SYNTHETIC_CANDIDATE="$TEST_DIR/candidate"
LOCALBOORU_TEST_BINARY='' HOST_HEAVY_BUILD_GATE="$TEST_DIR/gate" "$TEST_DIR/repo/scripts/run-isolated-test.sh" >"$TEST_DIR/default-output" 2>&1
[[ -f "$SYNTHETIC_DEV_MARKER" ]]
[[ "$(wc -l <"$SYNTHETIC_REPORT")" -eq 3 ]]
[[ ! -e "$(tail -n 1 "$SYNTHETIC_REPORT")" ]]
printf 'Isolated launcher safety tests passed\n'
