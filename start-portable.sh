#!/bin/bash
# Start LocalBooru in portable mode (data stored next to app, not in ~/.localbooru)
cd "$(dirname "$0")"

# Initialize pyenv if available
export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
fi

# Set portable data directory (next to this script)
export LOCALBOORU_PORTABLE_DATA="$(pwd)/data"
mkdir -p "$LOCALBOORU_PORTABLE_DATA"

echo "Starting LocalBooru in PORTABLE mode"
echo "Data directory: $LOCALBOORU_PORTABLE_DATA"

# Check for zombie processes on portable port 8791
if lsof -ti:8791 >/dev/null 2>&1; then
    echo "Found existing process on port 8791, attempting graceful shutdown..."
    lsof -ti:8791 2>/dev/null | xargs -r kill -15 2>/dev/null
    sleep 2
    if lsof -ti:8791 >/dev/null 2>&1; then
        echo "Process didn't stop gracefully, force killing..."
        lsof -ti:8791 2>/dev/null | xargs -r kill -9 2>/dev/null
        sleep 1
    fi
fi

# Start Electron app in production mode
export NODE_ENV=production
exec npm start
