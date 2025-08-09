.PHONY: help install setup run chat history clean test

# Default target
help:
	@echo "WonyBot - GPT-OSS Personal Assistant"
	@echo ""
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make setup     - Run initial setup (download model, init DB)"
	@echo "  make run       - Start interactive chat"
	@echo "  make chat      - Alias for 'make run'"
	@echo "  make history   - Show chat history"
	@echo "  make clean     - Clean temporary files and cache"
	@echo "  make test      - Run tests (if available)"

# Install dependencies
install:
	pip install -r requirements.txt

# Initial setup
setup: install
	python -m app.main setup

# Run chat
run:
	python -m app.main chat

# Alias for run
chat: run

# Show history
history:
	python -m app.main history

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Run tests (placeholder)
test:
	@echo "Tests not implemented yet"