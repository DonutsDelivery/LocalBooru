#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d /tmp/localbooru-isolation-tests.XXXXXX)"
trap 'rm -rf "$TEST_DIR"' EXIT
REAL_MKTEMP="$(command -v mktemp)"
REAL_PYTHON="$(command -v python3)"
export REAL_MKTEMP REAL_PYTHON
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
if [[ "$1" == '-c' ]]; then exec "$REAL_PYTHON" "$@"; fi
[[ "$1" == '-' && "$2" == '18790' ]]
cat >/dev/null
EOF
cat >"$TEST_DIR/bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$SYNTHETIC_PROBES"
[[ "$*" == *'http://127.0.0.1:18790/health'* ]]
[[ "${SYNTHETIC_READY:-1}" == '1' ]]
case "${SYNTHETIC_RESPONSE:-own}" in
  own) printf '{"status":"healthy","isolated_instance_id":"%s"}\n' "$LOCALBOORU_ISOLATED_INSTANCE_ID" ;;
  foreign) printf '{"status":"healthy","isolated_instance_id":"another-launch"}\n' ;;
  missing) printf '{"status":"healthy"}\n' ;;
  malformed) printf 'invalid-json\n' ;;
  unhealthy) printf '{"status":"starting","isolated_instance_id":"%s"}\n' "$LOCALBOORU_ISOLATED_INSTANCE_ID" ;;
esac
EOF
cat >"$TEST_DIR/bin/dbus-run-session" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == '--' && $# == 2 ]]
shift
[[ "$1" == /* ]]
printf '%s\n' "$1" >>"$SYNTHETIC_DBUS_CALLS"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/synthetic/private-session"
export SYNTHETIC_PRIVATE_BUS=1
# A synthetic daemon shares the owned session and must end during cleanup,
# even if the application has already exited successfully.
"$SYNTHETIC_DAEMON" &
printf '%s\n' "$!" >"$SYNTHETIC_DAEMON_PID"
exec "$@"
EOF
cat >"$TEST_DIR/bin/uname" <<'EOF'
#!/usr/bin/env bash
printf 'Linux\n'
EOF
cat >"$TEST_DIR/daemon" <<'EOF'
#!/usr/bin/env bash
trap 'touch "$SYNTHETIC_DAEMON_STOPPED"; exit 0' TERM
while :; do sleep 0.05; done
EOF
cat >"$TEST_DIR/candidate" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 ]]
[[ "$SYNTHETIC_PRIVATE_BUS" == 1 && "$DBUS_SESSION_BUS_ADDRESS" == 'unix:path=/synthetic/private-session' ]]
[[ "$DISPLAY" == 'synthetic-display' && "$WAYLAND_DISPLAY" == 'synthetic-wayland' ]]
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
[[ "$LOCALBOORU_ISOLATED_INSTANCE_ID" =~ ^[0-9a-f]{64}$ ]]
[[ "$LOCALBOORU_ISOLATED_INSTANCE_ID" != 'inherited-foreign-instance' ]]
printf '%s\n' "$LOCALBOORU_ISOLATED_INSTANCE_ID" >>"$SYNTHETIC_IDENTITIES"
mkdir -p "$LOCALBOORU_PORTABLE_DATA"
printf 'synthetic\n' >"$LOCALBOORU_PORTABLE_DATA/fixture.txt"
printf '%s\n' "$root" >>"$SYNTHETIC_REPORT"
printf '%s\n' "$$" >"$SYNTHETIC_PID"
if [[ "${SYNTHETIC_HOLD:-0}" == '1' ]]; then
  trap 'exit 0' TERM
  while :; do sleep 0.05; done
fi
EOF
chmod +x "$TEST_DIR/bin/"* "$TEST_DIR/candidate" "$TEST_DIR/daemon"
export PATH="$TEST_DIR/bin:$PATH"
export HOME="$TEST_DIR/caller/home" XDG_CACHE_HOME="$TEST_DIR/caller/cache"
export LOCALBOORU_PORTABLE_DATA="$TEST_DIR/caller/private-unused"
export LOCALBOORU_PORT=8790 LOCALBOORU_DISABLE_SINGLE_INSTANCE=0
export SYNTHETIC_ROOTS="$TEST_DIR/roots" SYNTHETIC_REPORT="$TEST_DIR/report"
export SYNTHETIC_PROBES="$TEST_DIR/probes" SYNTHETIC_PID="$TEST_DIR/pid"
export SYNTHETIC_IDENTITIES="$TEST_DIR/identities"
export LOCALBOORU_ISOLATED_INSTANCE_ID='inherited-foreign-instance'
export SYNTHETIC_DBUS_CALLS="$TEST_DIR/dbus-calls" SYNTHETIC_DAEMON="$TEST_DIR/daemon"
export SYNTHETIC_DAEMON_PID="$TEST_DIR/daemon-pid" SYNTHETIC_DAEMON_STOPPED="$TEST_DIR/daemon-stopped"
export DBUS_SESSION_BUS_ADDRESS="unix:path=/synthetic/shared-unused"
export DISPLAY="synthetic-display" WAYLAND_DISPLAY="synthetic-wayland"
assert_daemon_stopped() {
  for _ in $(seq 1 40); do [[ -f "$SYNTHETIC_DAEMON_STOPPED" ]] && break; sleep 0.05; done
  [[ -f "$SYNTHETIC_DAEMON_STOPPED" ]]
  [[ -z "$(ps -o stat= -p "$(cat "$SYNTHETIC_DAEMON_PID")" 2>/dev/null | sed '/^Z/d')" ]]
  rm -f "$SYNTHETIC_DAEMON_STOPPED"
}
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
assert_daemon_stopped
[[ "$(cat "$SYNTHETIC_DBUS_CALLS")" == "$LOCALBOORU_TEST_BINARY" ]]

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
assert_daemon_stopped

# A successful response from another listener is never readiness proof. These
# cases exercise cleanup of the owned child/group without opening any socket.
for response in foreign missing malformed unhealthy; do
  if SYNTHETIC_RESPONSE="$response" SYNTHETIC_HOLD=1 "$LAUNCHER" >"$TEST_DIR/foreign-output" 2>&1; then exit 1; fi
  ! kill -0 "$(cat "$SYNTHETIC_PID")" 2>/dev/null
  [[ ! -e "$(tail -n 1 "$SYNTHETIC_ROOTS")" ]]
  ! grep -F 'Isolated LocalBooru instance is ready' "$TEST_DIR/foreign-output" >/dev/null
  grep -F 'does not belong to this launch' "$TEST_DIR/foreign-output" >/dev/null
  assert_daemon_stopped
done

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
[[ "$(wc -l <"$SYNTHETIC_REPORT")" -eq 7 ]]
[[ "$(sort -u "$SYNTHETIC_IDENTITIES" | wc -l)" -eq 7 ]]
[[ ! -e "$(tail -n 1 "$SYNTHETIC_REPORT")" ]]
assert_daemon_stopped
# Missing private-session tooling must fail before launching a child.
mkdir -p "$TEST_DIR/no-dbus"
for tool in bash dirname rm; do ln -s "$(command -v "$tool")" "$TEST_DIR/no-dbus/$tool"; done
ln -s "$TEST_DIR/bin/mktemp" "$TEST_DIR/no-dbus/mktemp"
ln -s "$TEST_DIR/bin/uname" "$TEST_DIR/no-dbus/uname"
if PATH="$TEST_DIR/no-dbus" "$LAUNCHER" >"$TEST_DIR/no-dbus-output" 2>&1; then exit 1; fi
grep -F 'requires dbus-run-session and setsid' "$TEST_DIR/no-dbus-output" >/dev/null
[[ ! -e "$(tail -n 1 "$SYNTHETIC_ROOTS")" ]]
[[ "$(wc -l <"$SYNTHETIC_DBUS_CALLS")" -eq 7 ]]
printf 'Isolated launcher safety tests passed\n'
