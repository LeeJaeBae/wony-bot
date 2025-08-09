#!/bin/bash
# Quick setup for WonyBot with minimal dependencies

echo "🚀 Quick WonyBot Setup"
echo "====================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install minimal deps
echo "📦 Installing minimal dependencies..."
source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet httpx pydantic sqlalchemy aiosqlite typer rich python-dotenv pydantic-settings

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run WonyBot:"
echo "  source venv/bin/activate"
echo "  python -m app.main chat"
echo ""
echo "Or install globally in venv:"
echo "  pip install -e ."
echo "  wony chat"