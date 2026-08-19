#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
INSTALL_DIR="$DATA_HOME/localbooru/local-build"
INSTALLED_APP="$INSTALL_DIR/localbooru"
INSTALLED_FRONTEND="$INSTALL_DIR/frontend/dist/index.html"
LOCK_DIR="$STATE_HOME/localbooru"
LOCK_FILE="$LOCK_DIR/install.lock"
BUILD_LOCK_FILE="$LOCK_DIR/build-cache.lock"
LOCK_TIMEOUT="${LOCALBOORU_BUILD_LOCK_TIMEOUT:-1800}"
[[ "$LOCK_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "ERROR: LOCALBOORU_BUILD_LOCK_TIMEOUT must be a nonnegative number" >&2
  exit 2
}
IF_MISSING=0

usage() {
  cat <<'EOF'
Usage: scripts/install-local-app.sh [--if-missing]

Build and atomically install the standalone LocalBooru desktop executable.

  --if-missing  Skip the build when an executable is already installed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --if-missing) IF_MISSING=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$INSTALL_DIR" "$LOCK_DIR"
exec 9>"$LOCK_FILE"
flock -w "$LOCK_TIMEOUT" 9 || {
  echo "ERROR: timed out waiting for another LocalBooru install" >&2
  exit 1
}

if [[ $IF_MISSING -eq 1 && -x "$INSTALLED_APP" && -s "$INSTALLED_FRONTEND" ]]; then
  exit 0
fi

if [[ (-e "$INSTALLED_APP" || -L "$INSTALLED_APP") && (! -f "$INSTALLED_APP" || -L "$INSTALLED_APP") ]]; then
  echo "ERROR: refusing to replace unexpected install target: $INSTALLED_APP" >&2
  exit 1
fi
if [[ -L "$INSTALL_DIR/frontend" || -L "$INSTALL_DIR/frontend/dist" ]]; then
  echo "ERROR: refusing symlinked frontend install target under $INSTALL_DIR" >&2
  exit 1
fi

exec 8>"$BUILD_LOCK_FILE"
flock -w "$LOCK_TIMEOUT" 8 || {
  echo "ERROR: timed out waiting for another LocalBooru build or cleanup" >&2
  exit 1
}

source "$HOME/.cargo/env" 2>/dev/null || true
export CARGO_BUILD_JOBS="${LOCALBOORU_BUILD_JOBS:-2}"
cd "$ROOT"

echo "[install-local-app] Building standalone release…"
cargo tauri build --ci --no-bundle

TARGET_DIR="${CARGO_TARGET_DIR:-}"
if [[ -z "$TARGET_DIR" ]]; then
  TARGET_DIR="$(cargo metadata --no-deps --format-version 1 | node -e '
    let input = ""
    process.stdin.on("data", chunk => { input += chunk })
    process.stdin.on("end", () => process.stdout.write(JSON.parse(input).target_directory))
  ')"
fi
if [[ "$TARGET_DIR" != /* ]]; then
  TARGET_DIR="$ROOT/$TARGET_DIR"
fi
BUILT_APP="$TARGET_DIR/release/localbooru"
BUILT_FRONTEND="$ROOT/frontend/dist"

if [[ ! -s "$BUILT_APP" || ! -x "$BUILT_APP" ]]; then
  echo "ERROR: expected executable at $BUILT_APP" >&2
  exit 1
fi
if [[ ! -s "$BUILT_FRONTEND/index.html" ]]; then
  echo "ERROR: expected production frontend at $BUILT_FRONTEND" >&2
  exit 1
fi

FRONTEND_ROOT="$INSTALL_DIR/frontend"
FINAL_DIST="$FRONTEND_ROOT/dist"
mkdir -p "$FRONTEND_ROOT"
STAGED_APP="$(mktemp "$INSTALL_DIR/.localbooru.XXXXXX")"
STAGED_DIST="$(mktemp -d "$FRONTEND_ROOT/.dist.XXXXXX")"
BACKUP_DIST=""
DIST_PUBLISHED=0
COMMITTED=0
cleanup() {
  rm -f "$STAGED_APP"
  rm -rf "$STAGED_DIST"
  if [[ $COMMITTED -eq 0 && $DIST_PUBLISHED -eq 1 ]]; then
    rm -rf "$FINAL_DIST"
    if [[ -n "$BACKUP_DIST" && -d "$BACKUP_DIST" ]]; then
      mv "$BACKUP_DIST" "$FINAL_DIST"
    fi
  elif [[ -n "$BACKUP_DIST" ]]; then
    rm -rf "$BACKUP_DIST"
  fi
}
trap cleanup EXIT

install -m 0755 "$BUILT_APP" "$STAGED_APP"
cp -a "$BUILT_FRONTEND/." "$STAGED_DIST/"

if [[ -e "$FINAL_DIST" ]]; then
  BACKUP_DIST="$(mktemp -d "$FRONTEND_ROOT/.dist-backup.XXXXXX")"
  rmdir "$BACKUP_DIST"
  mv "$FINAL_DIST" "$BACKUP_DIST"
fi
mv "$STAGED_DIST" "$FINAL_DIST"
DIST_PUBLISHED=1
mv -f "$STAGED_APP" "$INSTALLED_APP"
COMMITTED=1
rm -rf "$BACKUP_DIST"
BACKUP_DIST=""
trap - EXIT

printf '[install-local-app] Installed %s with browser frontend\n' "$INSTALLED_APP"
