#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_PYTHON_VERSION="3.10"
VENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Find Python 3.10
find_python310() {
    local python_cmd=""

    # Check various Python command variations
    for cmd in python3.10 python310 python3; do
        if command_exists "$cmd"; then
            # Verify it's actually Python 3.10
            local version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [[ "$version" == "3.10" ]]; then
                python_cmd="$cmd"
                break
            fi
        fi
    done

    # If python3 wasn't 3.10, check if plain python is
    if [[ -z "$python_cmd" ]] && command_exists python; then
        local version=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if [[ "$version" == "3.10" ]]; then
            python_cmd="python"
        fi
    done

    echo "$python_cmd"
}

# Main setup function
main() {
    cd "$PROJECT_ROOT"

    echo "========================================="
    echo "   T4 Daily Driver Development Setup"
    echo "========================================="
    echo ""

    # Step 1: Find Python 3.10
    log_info "Detecting Python 3.10..."
    PYTHON_CMD=$(find_python310)

    if [[ -z "$PYTHON_CMD" ]]; then
        log_error "Python 3.10 not found!"
        echo ""
        echo "Please install Python 3.10 using one of these methods:"
        echo "  Ubuntu/Debian: sudo apt-get install python3.10 python3.10-venv python3.10-dev"
        echo "  macOS: brew install python@3.10"
        echo "  Or use pyenv: pyenv install 3.10.13"
        exit 1
    fi

    log_success "Found Python 3.10: $PYTHON_CMD"
    $PYTHON_CMD --version

    # Step 2: Check for venv module
    log_info "Checking Python venv module..."
    if ! $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
        log_error "Python venv module not found!"
        echo "Please install it:"
        echo "  Ubuntu/Debian: sudo apt-get install python3.10-venv"
        echo "  Other systems: Ensure Python was installed with venv support"
        exit 1
    fi

    # Step 3: Create or activate virtual environment
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Virtual environment already exists at $VENV_DIR"

        # Check if it's using the correct Python version
        if [[ -f "$VENV_DIR/bin/python" ]]; then
            VENV_PYTHON_VERSION=$("$VENV_DIR/bin/python" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [[ "$VENV_PYTHON_VERSION" != "3.10" ]]; then
                log_warning "Existing venv uses Python $VENV_PYTHON_VERSION instead of 3.10"
                read -p "Do you want to recreate the virtual environment? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Removing old virtual environment..."
                    rm -rf "$VENV_DIR"
                    log_info "Creating new virtual environment with Python 3.10..."
                    $PYTHON_CMD -m venv "$VENV_DIR"
                    log_success "Virtual environment recreated"
                else
                    log_warning "Continuing with existing virtual environment"
                fi
            else
                log_success "Virtual environment is using Python 3.10"
            fi
        fi
    else
        log_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        log_success "Virtual environment created at $VENV_DIR"
    fi

    # Step 4: Activate virtual environment
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    # Step 5: Upgrade pip, setuptools, wheel
    log_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel >/dev/null 2>&1
    log_success "Base packages upgraded"

    # Step 6: Install dependencies
    log_info "Installing project dependencies..."

    # Check for constraint file
    if [[ -f "constraints-cu121-py310.txt" ]]; then
        log_info "Using constraints file: constraints-cu121-py310.txt"
        export PIP_CONSTRAINT="constraints-cu121-py310.txt"
    elif [[ -f "constraints.txt" ]]; then
        log_info "Using constraints file: constraints.txt"
        export PIP_CONSTRAINT="constraints.txt"
    fi

    # Install main requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing requirements.txt..."
        pip install -r requirements.txt
        log_success "Main requirements installed"
    else
        log_warning "requirements.txt not found, skipping main dependencies"
    fi

    # Install dev requirements
    if [[ -f "requirements-dev.txt" ]]; then
        log_info "Installing requirements-dev.txt..."
        pip install -r requirements-dev.txt
        log_success "Development requirements installed"
    else
        log_warning "requirements-dev.txt not found, skipping dev dependencies"
    fi

    # Step 7: Install and configure pre-commit
    log_info "Setting up pre-commit hooks..."

    # Install pre-commit if not already installed
    if ! pip show pre-commit >/dev/null 2>&1; then
        log_info "Installing pre-commit..."
        pip install pre-commit
    fi

    # Check if pre-commit is already installed (hooks)
    if [[ -f ".git/hooks/pre-commit" ]] && grep -q "pre-commit" ".git/hooks/pre-commit" 2>/dev/null; then
        log_info "Pre-commit hooks already installed"
        log_info "Updating pre-commit hooks..."
        pre-commit install --overwrite
    else
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi

    # Update pre-commit hooks to latest versions
    log_info "Updating pre-commit hook versions..."
    pre-commit autoupdate || log_warning "Could not auto-update some hooks"

    log_success "Pre-commit hooks configured"

    # Step 8: Create .env.example if it doesn't exist
    if [[ ! -f ".env.example" ]]; then
        log_info "Creating .env.example file..."
        cat > .env.example << 'EOF'
# Example environment variables for T4 Daily Driver
# Copy this file to .env and update with your values

# Model configuration
MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95

# API configuration
HOST=0.0.0.0
PORT=8080

# Redis configuration (if using distributed inference)
# REDIS_HOST=redis
# REDIS_PORT=6379

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
EOF
        log_success "Created .env.example"
    fi

    # Step 9: Display summary
    echo ""
    echo "========================================="
    echo "   Setup Complete!"
    echo "========================================="
    echo ""
    echo "Virtual environment: $VENV_DIR"
    echo "Python version: $(python --version)"
    echo ""
    echo "To activate the environment in a new shell, run:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo "To run pre-commit on all files:"
    echo "  pre-commit run --all-files"
    echo ""
    echo "To run tests:"
    echo "  make test"
    echo ""

    # Check if direnv is installed
    if command_exists direnv; then
        if [[ -f ".envrc" ]]; then
            log_info "Detected direnv. Run 'direnv allow' to auto-activate the environment"
        fi
    fi

    log_success "Development environment is ready!"
}

# Run main function
main "$@"
