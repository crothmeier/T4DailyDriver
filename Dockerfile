# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Enhanced metadata for production vLLM service
LABEL description="Production-hardened vLLM Service with comprehensive features"
LABEL version="2.0.0"
LABEL gpu.architectures="Turing SM75, Ada SM89, Ampere SM86"
LABEL attention.backends="SDPA,FLASH_ATTN"
LABEL features="OpenAI-API,CircuitBreakers,Metrics,OOMPrevention"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PREFER_BINARY=1

# Production-optimized environment defaults
ENV VLLM_ATTENTION_BACKEND=AUTO \
    GPU_MEMORY_UTILIZATION=0.9 \
    TENSOR_PARALLEL_SIZE=1 \
    BLOCK_SIZE=16 \
    QUANTIZATION=awq \
    DTYPE=float16 \
    ENABLE_PREFIX_CACHING=true \
    ENABLE_FUNCTION_CALLING=true \
    ENABLE_REQUEST_TRACING=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip curl ca-certificates git \
      python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements early for better caching
COPY requirements-cuda124.txt constraints-cuda124.txt ./

# Use BuildKit cache mount for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip==24.2 setuptools wheel

# Install PyTorch CUDA 12.4 (T4 compatible) and requirements with constraints
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 && \
    python3 -m pip install -r requirements-cuda124.txt -c constraints-cuda124.txt && \
    python3 -m pip check && \
    echo "T4 build: Using SDPA attention backend, FA2 explicitly excluded"

# Copy application code
COPY . .

# Create non-root user and fix ownership
RUN useradd -m -u 1000 vllm && \
    chown -R vllm:vllm /app

USER vllm

# Expose ports
EXPOSE 8080

# Enhanced runtime environment
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-AWQ" \
    HF_HOME=/cache/hf \
    VLLM_CACHE_DIR=/cache/vllm \
    MAX_QUEUE_SIZE=100 \
    REQUEST_TIMEOUT_STREAMING=300 \
    REQUEST_TIMEOUT_BATCH=60 \
    CIRCUIT_BREAKER_FAILURES=5 \
    ENABLE_PROMETHEUS=true

# Enhanced health check with comprehensive validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health && curl -f http://localhost:8080/readyz || exit 1

# Run enhanced production vLLM server
CMD ["python3", "-m", "uvicorn", "app_enhanced:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info", "--access-log"]
