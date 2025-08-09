#!/bin/bash
# Setup virtual environment for WonyBot

echo "🔧 Setting up Python virtual environment for WonyBot..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install httpx pydantic sqlalchemy aiosqlite typer rich python-dotenv

# Install RAG dependencies
echo "📦 Installing RAG dependencies..."
pip install chromadb sentence-transformers langchain langchain-community

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install prompt-toolkit pydantic-settings

echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run WonyBot:"
echo "  source venv/bin/activate && wony chat"