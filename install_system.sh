#!/bin/bash
# Install WonyBot to work system-wide on macOS

echo "🚀 WonyBot System-Wide Installation"
echo "===================================="
echo ""

# Create a symbolic link in /usr/local/bin
INSTALL_DIR="/usr/local/bin"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if /usr/local/bin exists
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Creating $INSTALL_DIR..."
    sudo mkdir -p "$INSTALL_DIR"
fi

# Install dependencies in virtual environment
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "📦 Setting up virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

echo "📦 Installing dependencies..."
source "$SCRIPT_DIR/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet httpx pydantic sqlalchemy aiosqlite typer rich python-dotenv pydantic-settings chromadb sentence-transformers

# Create the global launcher
echo "🔗 Creating global launcher..."
sudo ln -sf "$SCRIPT_DIR/wony_launcher.sh" "$INSTALL_DIR/wony"

echo ""
echo "✅ Installation Complete!"
echo ""
echo "You can now run WonyBot from anywhere:"
echo "  wony chat"
echo "  wony memories"
echo "  wony help"
echo ""
echo "To uninstall:"
echo "  sudo rm /usr/local/bin/wony"