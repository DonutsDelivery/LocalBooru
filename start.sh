#!/bin/bash
cd "$(dirname "$0")"

# Initialize pyenv if available
export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
fi

# Start Electron app in production mode (uses backend server, not Vite dev server)
export NODE_ENV=production
exec npm start
