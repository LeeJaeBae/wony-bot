#!/bin/bash
# Install WonyBot globally (system-wide)

echo "🌍 Installing WonyBot Globally"
echo "=============================="

# Method 1: Using pipx (Recommended)
if command -v pipx &> /dev/null; then
    echo "📦 Installing with pipx..."
    pipx install --editable .
    echo "✅ Installed! Run 'wony chat' from anywhere"
    
# Method 2: Using pip with --break-system-packages (macOS)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing with pip (macOS)..."
    echo "⚠️  This will modify system packages. Continue? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        pip3 install --user --break-system-packages -e .
        echo "✅ Installed! You may need to add ~/.local/bin to PATH"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
    
# Method 3: Standard pip install
else
    echo "📦 Installing with pip..."
    pip3 install --user -e .
    echo "✅ Installed! Run 'wony chat' from anywhere"
fi

echo ""
echo "📌 Note: You still need to install dependencies:"
echo "  pip3 install --user --break-system-packages httpx pydantic sqlalchemy aiosqlite typer rich python-dotenv pydantic-settings"
echo ""
echo "Or use pipx for isolated environment:"
echo "  brew install pipx"
echo "  pipx install --editable ."