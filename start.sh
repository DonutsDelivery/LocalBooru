#!/bin/bash
cd "$(dirname "$0")"

# Initialize pyenv if available
export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
fi

# Check for zombie processes on port 8790 and try graceful shutdown first
if lsof -ti:8790 >/dev/null 2>&1; then
    echo "Found existing process on port 8790, attempting graceful shutdown..."
    # Try SIGTERM first (graceful)
    lsof -ti:8790 2>/dev/null | xargs -r kill -15 2>/dev/null
    sleep 2
    # If still running, force kill
    if lsof -ti:8790 >/dev/null 2>&1; then
        echo "Process didn't stop gracefully, force killing..."
        lsof -ti:8790 2>/dev/null | xargs -r kill -9 2>/dev/null
        sleep 1
    fi
fi

# Start Electron app in production mode (uses backend server, not Vite dev server)
export NODE_ENV=production
exec npm start
