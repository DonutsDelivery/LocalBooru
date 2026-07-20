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

# DEV MODE: Start Vite dev server and Electron in dev mode
# Kill any existing Vite dev server on port 5174
if lsof -ti:5174 >/dev/null 2>&1; then
    lsof -ti:5174 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 1
fi

# Start Vite dev server in background (in subshell to not change our cwd)
(cd frontend && npm run dev) &
VITE_PID=$!

# Wait for Vite to be ready
echo "Waiting for Vite dev server..."
for i in {1..30}; do
    if curl -s http://localhost:5174 >/dev/null 2>&1; then
        echo "Vite dev server ready"
        break
    fi
    sleep 0.5
done

# Cleanup function to kill Vite when Electron exits
cleanup() {
    kill $VITE_PID 2>/dev/null
}
trap cleanup EXIT

# Start Electron in dev mode
export LOCALBOORU_DEV=true
npm run dev
cleanup
