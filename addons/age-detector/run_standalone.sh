#!/usr/bin/env bash
# Run the age-detector standalone (bound to all interfaces) so DonutBooru's VPS
# can reach it over Tailscale. The Tauri sidecar binds 127.0.0.1, which is not
# reachable remotely — use this for the auto-moderation deployment.
#
#   AGE_SCAN_SECRET=optional-shared-secret ./run_standalone.sh
#
# Uses the same Python/venv you normally run the addon with. Set
# LOCALBOORU_PACKAGES_DIR if the models live in a custom HF cache dir.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${AGE_DETECTOR_HOST:-0.0.0.0}"
PORT="${AGE_DETECTOR_PORT:-18002}"

# Prefer the dedicated venv created from requirements.txt, then a local ./venv,
# then whatever PYTHON / python is on PATH.
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HOME/.localbooru/addons/age-detector/venv/bin/python" ]; then
    PYTHON="$HOME/.localbooru/addons/age-detector/venv/bin/python"
  elif [ -x "./venv/bin/python" ]; then
    PYTHON="./venv/bin/python"
  elif [ -x "$HOME/.pyenv/versions/3.11.6/bin/python" ]; then
    # existing localbooru env on this machine (has torch+mivolo+insightface)
    PYTHON="$HOME/.pyenv/versions/3.11.6/bin/python"
  else
    PYTHON="python"
  fi
fi

echo "Starting age-detector with $PYTHON on $HOST:$PORT"
exec "$PYTHON" -m uvicorn app:app --host "$HOST" --port "$PORT"
