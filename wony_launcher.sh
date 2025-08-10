#!/bin/bash
# Universal WonyBot Launcher - works globally

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    # Use virtual environment
    source "$SCRIPT_DIR/venv/bin/activate"
    python -m app.main "$@"
else
    # Try to run directly with system Python
    cd "$SCRIPT_DIR"
    python3 -m app.main "$@"
fi