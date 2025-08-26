# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Metadata for T4 optimization
LABEL description="vLLM Service optimized for Tesla T4 GPU with SDPA backend"
LABEL gpu.architecture="Turing SM75"
LABEL attention.backend="SDPA"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PREFER_BINARY=1

# T4-optimized environment defaults
ENV VLLM_ATTENTION_BACKEND=SDPA \
    GPU_MEMORY_UTILIZATION=0.9 \
    MAX_NUM_SEQS=32 \
    BLOCK_SIZE=16 \
    QUANTIZATION=awq \
    DTYPE=float16

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

# Install PyTorch CUDA 12.4 (T4 compatible)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

# Install requirements with constraints (no flash-attn, T4-optimized)
RUN --mount=type=cache,target=/root/.cache/pip \
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

# Set T4-specific runtime environment
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-AWQ" \
    HF_HOME=/cache/hf

# Health check with T4-specific validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with T4-optimized vLLM server
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
