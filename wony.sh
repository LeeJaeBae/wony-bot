#!/bin/bash
# WonyBot launcher script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the CLI with all arguments
python -m app.main "$@"