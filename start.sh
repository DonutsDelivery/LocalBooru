#!/bin/bash
cd "$(dirname "$0")"

# Initialize pyenv if available
export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init -)"
fi

# Start Electron app (which manages backend automatically)
exec npm start
