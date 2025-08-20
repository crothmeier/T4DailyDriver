.PHONY: help prepare-caches up-local down smoke logs logs-all services clean-caches build stop clean deploy k8s-apply k8s-delete k8s-status k8s-logs port-forward metrics lint format test-local deploy-build test test-unit test-integration test-load test-security install install-dev dev-setup docker-build docker-run docker-stop pre-commit setup-hooks

# Variables
IMAGE_NAME := vllm-service
IMAGE_TAG := latest
NAMESPACE := vllm
REGISTRY :=
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Cache and directory management
prepare-caches: ## Create cache/logs dirs, fix ownership, ensure .gitkeep
	@echo "Preparing cache directories..."
	@mkdir -p cache/hf cache/vllm cache/config logs
	@touch cache/.gitkeep cache/hf/.gitkeep cache/vllm/.gitkeep cache/config/.gitkeep logs/.gitkeep
	@if [ "$$(uname)" = "Linux" ]; then \
		echo "Setting ownership on Linux..."; \
		sudo chown -R "$$(id -u):$$(id -g)" cache logs 2>/dev/null || true; \
	elif [ "$$(uname)" = "Darwin" ]; then \
		echo "Setting ownership on macOS..."; \
		chown -R "$$(id -u):$$(id -g)" cache logs 2>/dev/null || true; \
	fi
	@echo "Cache directories ready."

# Docker Compose operations
up-local: prepare-caches ## Start services with docker compose (depends on prepare-caches)
	docker compose up -d

down: ## Stop and remove containers
	docker compose down

smoke: ## Run smoke tests (fails if script fails)
	./scripts/smoke_compose.sh

logs: ## Tail application logs
	@docker compose logs -f vllm 2>/dev/null || docker compose logs -f vllm-t4-service

logs-all: ## Tail all service logs
	docker compose logs -f

services: ## List running services
	docker compose ps

clean-caches: ## Remove cache contents (keeps .gitkeep)
	@echo "Cleaning cache directories..."
	@find cache -mindepth 1 -name '.gitkeep' -prune -o -exec rm -rf {} + 2>/dev/null || true
	@find logs -mindepth 1 -name '.gitkeep' -prune -o -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache directories cleaned."

# Legacy/existing targets
build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

stop: ## Stop local services (alias for down)
	docker compose down

test-local: ## Test local service
	@echo "Testing health endpoint..."
	@curl -s http://localhost:8080/health | jq .
	@echo "\nTesting generation endpoint..."
	@curl -X POST http://localhost:8080/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello, how are you?", "max_tokens": 50}' | jq .

deploy: ## Deploy to Kubernetes
	./scripts/deploy.sh --namespace $(NAMESPACE) --tag $(IMAGE_TAG)

deploy-build: ## Build and deploy to Kubernetes
	./scripts/deploy.sh --build --namespace $(NAMESPACE) --tag $(IMAGE_TAG) --registry $(REGISTRY)

k8s-apply: ## Apply Kubernetes manifests
	kubectl apply -k k8s/

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -k k8s/

k8s-status: ## Check deployment status
	kubectl -n $(NAMESPACE) get pods,svc,hpa

k8s-logs: ## Show logs from Kubernetes pods
	kubectl -n $(NAMESPACE) logs -f -l app=vllm

port-forward: ## Port forward to Kubernetes service
	kubectl -n $(NAMESPACE) port-forward svc/vllm-service 8080:80

metrics: ## Show current metrics
	@curl -s http://localhost:8080/metrics | grep -E "^vllm_" | grep -v "#"

clean: ## Clean up resources
	docker-compose down -v
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) || true

lint: ## Lint Python code
	@which ruff >/dev/null 2>&1 || pip install ruff
	ruff check .
	black --check .
	isort --check-only .
	@if [ -f Dockerfile ]; then \
		if command -v hadolint >/dev/null 2>&1; then \
			hadolint Dockerfile; \
		else \
			echo "Hadolint not installed. Install from: https://github.com/hadolint/hadolint"; \
		fi \
	fi

format: ## Format Python code
	@which ruff >/dev/null 2>&1 || pip install ruff
	black .
	isort .
	ruff check . --fix

# Testing targets
test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests with coverage
	@echo "Running unit tests..."
	$(PYTEST) tests/unit/ -v --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	@if ! curl -f http://localhost:8000/health 2>/dev/null; then \
		echo "Error: Service not running on localhost:8000. Please start the service first."; \
		echo "Run: make docker-run"; \
		exit 1; \
	fi
	$(PYTEST) tests/integration/ -v -s

test-load: ## Run load tests with Artillery
	@echo "Running load tests..."
	@if ! command -v artillery >/dev/null 2>&1; then \
		echo "Installing Artillery..."; \
		npm install -g artillery artillery-plugin-metrics-by-endpoint; \
	fi
	@if ! curl -f http://localhost:8000/health 2>/dev/null; then \
		echo "Error: Service not running on localhost:8000. Please start the service first."; \
		echo "Run: make docker-run"; \
		exit 1; \
	fi
	artillery run tests/load/artillery.yml --output reports/load-test-$(shell date +%Y%m%d-%H%M%S).json
	@echo "Load test complete. Report saved to reports/"

test-security: ## Run security scans
	@echo "Running security scans..."
	@echo "Checking Python dependencies with safety..."
	safety check --json || true
	@echo ""
	@echo "Scanning with trivy..."
	@if command -v trivy >/dev/null 2>&1; then \
		trivy fs . --severity HIGH,CRITICAL; \
	else \
		echo "Trivy not installed. Install from: https://github.com/aquasecurity/trivy"; \
	fi
	@echo ""
	@echo "Checking for secrets..."
	@if command -v detect-secrets >/dev/null 2>&1; then \
		detect-secrets scan --baseline .secrets.baseline; \
	else \
		echo "detect-secrets not installed. Run: pip install detect-secrets"; \
	fi

# Installation targets
install: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1
	PIP_CONSTRAINT=constraints-cu121-py310.txt $(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-asyncio httpx ruff black isort safety pre-commit
	pre-commit install

dev-setup: ## Complete development environment setup (Python 3.10, venv, deps, pre-commit)
	@echo "Setting up development environment..."
	@bash scripts/setup-dev.sh
	@echo "Setup complete! Activate your environment with: source .venv/bin/activate"

# Docker operations
docker-build: ## Build Docker image with proper CUDA support
	docker build \
		--build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 \
		--build-arg CUDA_VERSION=12.1.0 \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(IMAGE_NAME):$(shell git rev-parse --short HEAD 2>/dev/null || echo "latest") \
		.

docker-run: ## Run Docker container with GPU
	@echo "Starting vLLM service container..."
	docker run -d \
		--name vllm-service \
		--gpus all \
		-p 8000:8000 \
		-e MODEL_NAME=mistralai/Mistral-7B-v0.1 \
		-e QUANTIZATION=awq \
		-e MAX_MODEL_LEN=4096 \
		-e GPU_MEMORY_UTILIZATION=0.9 \
		-v $(PWD)/models:/models \
		$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "Container started. Check logs with: docker logs -f vllm-service"
	@echo "Service will be available at http://localhost:8000 once model is loaded"

docker-stop: ## Stop and remove Docker container
	@echo "Stopping vLLM service container..."
	docker stop vllm-service 2>/dev/null || true
	docker rm vllm-service 2>/dev/null || true

# Pre-commit hooks
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

setup-hooks: ## Set up git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Git hooks installed successfully!"
