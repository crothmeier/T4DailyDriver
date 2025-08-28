#!/usr/bin/env bash
set -euo pipefail

# Clear informative banners
echo "========================================="
echo "Python 3.11 Virtual Environment Upgrade"
echo "========================================="

# Verify we're in the repo root
if [ ! -f "README.md" ]; then
    echo "Error: README.md not found. Please run this script from the repository root."
    exit 1
fi

echo "✓ Repository root verified"

# Detect Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is not installed or not in PATH"
    echo "Please install Python 3.11 and try again"
    exit 1
fi

echo "✓ Python 3.11 detected: $(python3.11 --version)"

# Remove old .venv safely
if [ -d ".venv" ]; then
    echo "→ Removing existing .venv directory..."
    rm -rf .venv
    echo "✓ Old virtual environment removed"
fi

# Create new virtual environment with Python 3.11
echo "→ Creating new virtual environment with Python 3.11..."
python3.11 -m venv .venv
echo "✓ Virtual environment created"

# Activate and upgrade pip
echo "→ Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip --quiet
echo "✓ pip upgraded to $(.venv/bin/pip --version)"

# Install requirements
echo "→ Installing requirements from requirements.txt..."
.venv/bin/pip install -r requirements.txt --quiet
echo "✓ Requirements installed"

# Install pre-commit if not present
if ! .venv/bin/pip show pre-commit &> /dev/null; then
    echo "→ Installing pre-commit..."
    .venv/bin/pip install pre-commit --quiet
    echo "✓ pre-commit installed"
else
    echo "✓ pre-commit already available"
fi

# Install pre-commit hooks
echo "→ Installing pre-commit hooks..."
.venv/bin/pre-commit install --install-hooks
echo "✓ Pre-commit hooks installed"

# Run pre-commit on all files
echo "→ Running pre-commit checks on all files..."
.venv/bin/pre-commit run --all-files || true
echo "✓ Pre-commit checks completed"

echo ""
echo "========================================="
echo "✅ Environment setup complete!"
echo "========================================="
echo ""
echo "Next step: pytest -q"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
