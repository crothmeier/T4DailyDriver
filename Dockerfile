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

RUN python3 -m pip install --upgrade pip==24.2 setuptools wheel

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

# Create dummy pyairports package to satisfy dependency
RUN mkdir -p /tmp/pyairports && \
    echo "from setuptools import setup; setup(name='pyairports', version='2.1.1')" > /tmp/pyairports/setup.py && \
    cd /tmp/pyairports && \
    python3 -m pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/pyairports

# Copy constraints file
COPY constraints.txt .

# Install requirements with constraints
RUN python3 -m pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    python3 -m pip check

COPY . .

# Create non-root user
RUN useradd -m -u 1000 vllm && chown -R vllm:vllm /app
USER vllm

# Expose ports
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python3", "-u", "app.py"]
