# vLLM Service for Tesla T4

Production-ready vLLM service optimized for Tesla T4 GPUs with Mistral-7B AWQ quantization, connection pooling, Prometheus metrics, and health checks.

## Features

- **Optimized for Tesla T4**: Configured for 16GB VRAM with AWQ quantization
- **Connection Pooling**: Efficient request handling with metrics
- **Prometheus Metrics**: Comprehensive monitoring including TTFT, token throughput, and GPU usage
- **Health Checks**: Liveness and readiness probes for production reliability
- **Kubernetes Ready**: Full K8s manifests with HPA, monitoring, and ingress
- **Docker Compose**: Local development and testing environment

## Quick Start

### Local Development (Docker Compose)

```bash
# Clone the repository
git clone <repository-url>
cd T4DailyDriver

# Build and start services
docker-compose up -d

# Check service health
curl http://localhost:8080/health

# Test generation
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Production Deployment (Kubernetes)

```bash
# Apply all manifests
kubectl apply -k k8s/

# Or apply individually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Check deployment status
kubectl -n vllm get pods
kubectl -n vllm get svc

# Port forward for testing
kubectl -n vllm port-forward svc/vllm-service 8080:80
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key configurations for T4 optimization:
- `GPU_MEMORY_UTILIZATION=0.9` - Use 90% of T4's 16GB VRAM
- `MAX_NUM_SEQS=32` - Optimal batch size for T4
- `QUANTIZATION=awq` - 4-bit quantization for Mistral-7B

### Model Selection

Default model: `TheBloke/Mistral-7B-Instruct-v0.2-AWQ`

To use a different model, update:
1. `MODEL_PATH` environment variable
2. ConfigMap in Kubernetes
3. Docker Compose environment

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check with GPU metrics
- `GET /metrics` - Prometheus metrics
- `POST /generate` - Text generation (streaming supported)

### Generation Request

```json
{
  "prompt": "Your prompt here",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50,
  "stream": false,
  "stop": ["\\n"],
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```

## Monitoring

### Prometheus Metrics

- `vllm_request_count` - Total requests by status
- `vllm_request_duration_seconds` - Request duration histogram
- `vllm_active_requests` - Currently active requests
- `vllm_token_throughput` - Tokens per second histogram
- `vllm_time_to_first_token_seconds` - TTFT histogram
- `vllm_gpu_memory_usage_gb` - GPU memory usage

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

## Performance Targets

On Tesla T4 with Mistral-7B AWQ:
- **Throughput**: 50 tokens/second
- **TTFT p95**: < 200ms
- **Max concurrent requests**: 32
- **GPU utilization**: > 70%

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `GPU_MEMORY_UTILIZATION`
   - Decrease `MAX_NUM_SEQS`
   - Use smaller model or more aggressive quantization

2. **Slow Performance**
   - Check GPU utilization with `nvidia-smi`
   - Monitor metrics at `/metrics` endpoint
   - Verify CUDA graphs are enabled

3. **Model Loading Fails**
   - Ensure sufficient disk space for model cache
   - Check network connectivity to Hugging Face
   - Verify CUDA/driver compatibility

## Development

### Prerequisites

- Python 3.10 (required)
- NVIDIA GPU with CUDA 12.1+ support
- Docker and Docker Compose
- Git

### Interpreter

Local development targets Python 3.11. Note that vLLM 0.5.3 requires Python >=3.11 due to its dependency chain (outlines→pyairports).

### Development Environment Setup

#### Quick Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd T4DailyDriver

# Run the automated setup script
make dev-setup

# Activate the virtual environment
source .venv/bin/activate
```

The `make dev-setup` command will:
- Detect and validate Python 3.10
- Create a virtual environment
- Install all dependencies with proper constraints
- Configure pre-commit hooks
- Set up the development environment

#### Manual Setup

If you prefer to set up manually or need to troubleshoot:

```bash
# 1. Ensure Python 3.10 is installed
python3.10 --version

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install base packages
pip install --upgrade pip setuptools wheel

# 4. Install dependencies with constraints
export PIP_CONSTRAINT=constraints-cu121-py310.txt
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Install pre-commit hooks
pre-commit install
pre-commit run --all-files  # Run on all files to verify
```

#### Using direnv (Optional)

For automatic environment activation:

```bash
# Install direnv (if not already installed)
# macOS: brew install direnv
# Ubuntu: sudo apt-get install direnv

# Allow direnv in this directory
direnv allow

# The environment will now activate automatically when you cd into the directory
```

### Python Version Management

This project requires Python 3.10. If you have multiple Python versions:

#### Ubuntu/Debian
```bash
# Install Python 3.10
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev

# Verify installation
python3.10 --version
```

#### macOS
```bash
# Using Homebrew
brew install python@3.10

# Verify installation
python3.10 --version
```

#### Using pyenv (Cross-platform)
```bash
# Install Python 3.10
pyenv install 3.10.13
pyenv local 3.10.13

# Verify
python --version
```

### Common Development Tasks

```bash
# Run pre-commit hooks on all files
make pre-commit

# Format code
make format

# Run linting
make lint

# Run unit tests
make test-unit

# Run all tests with coverage
make test

# Start local development server
make up-local

# View logs
make logs

# Stop services
make down
```

### Building Docker Image

```bash
# Build with CUDA support
make docker-build

# Or manually with specific options
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
  -t vllm-service:latest .
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests (requires running service)
make test-integration

# Run with coverage report
pytest --cov=app --cov-report=html tests/
# View coverage report in htmlcov/index.html

# Run security checks
make test-security
```

### Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Fast Python linting
- **hadolint**: Dockerfile linting
- **detect-secrets**: Prevent secrets in code
- Various safety checks (YAML, JSON, merge conflicts, etc.)

To bypass hooks temporarily (not recommended):
```bash
git commit --no-verify -m "your message"
```

### Troubleshooting Development Setup

#### Issue: Python 3.11 interpreter error
```bash
# Solution: Ensure Python 3.10 is used
make dev-setup  # This will detect and use Python 3.10
```

#### Issue: Virtual environment using wrong Python version
```bash
# Remove and recreate virtual environment
rm -rf .venv
make dev-setup
```

#### Issue: Pre-commit hooks failing
```bash
# Update hooks to latest versions
pre-commit autoupdate

# Clean and reinstall
pre-commit clean
pre-commit install
```

#### Issue: Dependencies conflict
```bash
# Use the constraint file
export PIP_CONSTRAINT=constraints-cu121-py310.txt
pip install -r requirements.txt
```

## License

[Your License Here]
