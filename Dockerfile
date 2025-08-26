# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Use BuildKit cache mount for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip==24.2 setuptools wheel

# --- Robust Torch (CUDA 12.1) local wheel install ---
ARG TORCH_FILE="torch-2.3.1+cu121-cp310-cp310-linux_x86_64.whl"
ARG TORCH_URL="https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

RUN set -eux; \
    curl -fSL --retry 10 --retry-delay 5 --retry-connrefused --continue-at - \
      -o "/tmp/${TORCH_FILE}" "${TORCH_URL}"; \
    python3 -m pip install "/tmp/${TORCH_FILE}"; \
    rm -f "/tmp/${TORCH_FILE}"

# Keep CUDA wheels preferred for any later installs
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"

# Copy constraints file
COPY constraints.txt .

# Install requirements with constraints using BuildKit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt -c constraints.txt && \
    python3 -m pip check

# Copy only necessary files first for better caching
COPY scripts/download_model.py scripts/

# Pre-download model with cache mount for efficiency
# This allows Docker to cache the model layer separately
ENV HF_HOME=/cache/hf
ENV MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Download model during build with BuildKit cache mount
RUN --mount=type=cache,target=/cache/hf,sharing=locked \
    if python3 scripts/download_model.py "${MODEL_PATH}" /cache/hf; then \
        echo "Model downloaded successfully" && \
        mkdir -p /model-cache && \
        { cp -r /cache/hf/* /model-cache/ 2>/dev/null || true; }; \
    else \
        echo "Model download failed, will retry at runtime" && \
        mkdir -p /model-cache; \
    fi

# Copy application code
COPY . .

# Create non-root user and fix ownership
RUN useradd -m -u 1000 vllm && \
    chown -R vllm:vllm /app && \
    if [ -d /model-cache ]; then \
        chown -R vllm:vllm /model-cache; \
    fi

USER vllm

# Expose ports
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
ENV HF_HOME=/model-cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
