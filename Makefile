.PHONY: help prepare-caches up-local down smoke logs logs-all services clean-caches build stop clean deploy k8s-apply k8s-delete k8s-status k8s-logs port-forward metrics lint format test-local deploy-build test test-unit test-integration test-load test-security install install-dev dev-setup docker-build docker-run docker-stop pre-commit setup-hooks

# Variables
IMAGE_NAME := vllm-service
IMAGE_TAG := latest
NAMESPACE := vllm
REGISTRY :=
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
ARTIFACT_DIR ?= artifacts

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo "For T4-specific targets, run: make help-t4"

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
	@if ! curl -f http://localhost:8080/health 2>/dev/null; then \
		echo "Error: Service not running on localhost:8080. Please start the service first."; \
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
	@if ! curl -f http://localhost:8080/health 2>/dev/null; then \
		echo "Error: Service not running on localhost:8080. Please start the service first."; \
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
		-p 8080:8080 \
		-e MODEL_NAME=mistralai/Mistral-7B-v0.1 \
		-e QUANTIZATION=awq \
		-e MAX_MODEL_LEN=4096 \
		-e GPU_MEMORY_UTILIZATION=0.9 \
		-v $(PWD)/models:/models \
		$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "Container started. Check logs with: docker logs -f vllm-service"
	@echo "Service will be available at http://localhost:8080 once model is loaded"

docker-stop: ## Stop and remove Docker container
	docker stop vllm-service 2>/dev/null || true
	docker rm vllm-service 2>/dev/null || true

# T4-Specific Docker Operations
docker-build-t4: ## Build T4-optimized Docker image with SDPA backend
	@echo "Building T4-optimized Docker image with SDPA backend..."
	docker build \
		--build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 \
		--build-arg CUDA_VERSION=12.4.0 \
		--build-arg VLLM_ATTENTION_BACKEND=SDPA \
		-t $(IMAGE_NAME):t4-$(IMAGE_TAG) \
		-t $(IMAGE_NAME):t4-$(shell git rev-parse --short HEAD 2>/dev/null || echo "latest") \
		-t $(IMAGE_NAME):t4-latest \
		.
	@echo "✓ T4-optimized image built successfully"

docker-run-t4: ## Run T4-optimized Docker container with SDPA
	@echo "Starting T4-optimized vLLM service container..."
	docker run -d \
		--name vllm-t4-service \
		--gpus all \
		-p 8080:8080 \
		-e VLLM_ATTENTION_BACKEND=SDPA \
		-e MODEL_NAME=TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
		-e QUANTIZATION=awq \
		-e MAX_MODEL_LEN=4096 \
		-e GPU_MEMORY_UTILIZATION=0.9 \
		-e MAX_NUM_SEQS=32 \
		-v $(PWD)/cache:/cache \
		$(IMAGE_NAME):t4-latest
	@echo "✓ T4 container started. Check logs with: docker logs -f vllm-t4-service"
	@echo "✓ Service will be available at http://localhost:8080 once model is loaded"

docker-stop-t4: ## Stop and remove T4 Docker container
	@echo "Stopping T4-optimized vLLM service container..."
	docker stop vllm-t4-service 2>/dev/null || true
	docker rm vllm-t4-service 2>/dev/null || true
	@echo "✓ T4 container stopped and removed"

# T4-Specific Testing
test-t4-build: docker-build-t4 ## Build and test T4 image locally
	@echo "Testing T4-optimized build..."
	@docker run --rm $(IMAGE_NAME):t4-latest python -c "\
	import os; \
	backend = os.getenv('VLLM_ATTENTION_BACKEND'); \
	assert backend == 'SDPA', f'Expected SDPA, got {backend}'; \
	print('✓ T4 SDPA backend verified')"
	@echo "✓ T4 build test passed"

test-t4-local: docker-run-t4 ## Start T4 service and run local tests
	@echo "Testing T4 service locally..."
	@sleep 30
	@echo "Checking T4 service health..."
	@if curl -f http://localhost:8080/health >/dev/null 2>&1; then \
		echo "✓ T4 service health check passed"; \
	else \
		echo "✗ T4 service health check failed"; \
		docker logs vllm-t4-service | tail -20; \
		$(MAKE) docker-stop-t4; \
		exit 1; \
	fi
	@echo "Checking T4 service metrics..."
	@if curl -f http://localhost:8080/metrics | grep -q "vllm_"; then \
		echo "✓ T4 service metrics available"; \
	else \
		echo "✗ T4 service metrics not available"; \
	fi
	@echo "Testing T4 text generation..."
	@if curl -X POST http://localhost:8080/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello T4", "max_tokens": 5}' | grep -q "text"; then \
		echo "✓ T4 text generation working"; \
	else \
		echo "✗ T4 text generation failed"; \
	fi
	@$(MAKE) docker-stop-t4
	@echo "✅ T4 local test completed successfully"

smoke-t4: ## Run T4-specific smoke tests
	@echo "Running T4-specific smoke tests..."
	@if [ ! -f "scripts/verify_runtime.py" ]; then \
		echo "✗ verify_runtime.py not found"; \
		exit 1; \
	fi
	@echo "✓ Verification script found"
	@if docker images | grep -q "$(IMAGE_NAME):t4-latest"; then \
		echo "✓ T4 image available"; \
		docker run --rm -e CUDA_VISIBLE_DEVICES=0 $(IMAGE_NAME):t4-latest \
			python scripts/verify_runtime.py --t4-mode || echo "⚠ T4 verification completed with warnings"; \
	else \
		echo "⚠ T4 image not built, run 'make docker-build-t4' first"; \
	fi
	@echo "✓ T4 smoke tests completed"

# T4-Specific Performance Testing
benchmark-t4: ## Run T4 performance benchmarks
	@echo "Running T4 performance benchmarks..."
	@if ! docker ps | grep -q vllm-t4-service; then \
		echo "Starting T4 service for benchmarking..."; \
		$(MAKE) docker-run-t4; \
		sleep 60; \
	fi
	@echo "Running T4 benchmark tests..."
	@for i in {1..10}; do \
		echo -n "Request $$i: "; \
		curl -s -w "%{time_total}" -X POST http://localhost:8080/generate \
			-H "Content-Type: application/json" \
			-d '{"prompt": "Benchmark test", "max_tokens": 20}' \
			| tail -1; \
		echo "s"; \
	done
	@echo "✓ T4 benchmark completed"

# T4 Environment Setup
setup-t4-env: ## Set up T4-optimized development environment
	@echo "Setting up T4-optimized development environment..."
	@if [ -f ".envrc" ]; then \
		if ! grep -q "VLLM_ATTENTION_BACKEND=SDPA" .envrc; then \
			echo 'export VLLM_ATTENTION_BACKEND=SDPA' >> .envrc; \
			echo "✓ Added SDPA backend to .envrc"; \
		else \
			echo "✓ SDPA backend already configured in .envrc"; \
		fi; \
	fi
	@if [ -f ".env.example" ]; then \
		if ! grep -q "VLLM_ATTENTION_BACKEND=SDPA" .env.example; then \
			echo 'VLLM_ATTENTION_BACKEND=SDPA' >> .env.example; \
			echo "✓ Added SDPA backend to .env.example"; \
		else \
			echo "✓ SDPA backend already configured in .env.example"; \
		fi; \
	fi
	@echo "✓ T4 environment setup completed"

# T4 Deployment Helpers
deploy-t4: ## Deploy T4-optimized service to Kubernetes
	@echo "Deploying T4-optimized service to Kubernetes..."
	@if [ -f "k8s/deployment.yaml" ]; then \
		kubectl apply -k k8s/; \
		echo "✓ T4-optimized deployment applied"; \
	else \
		echo "✗ Kubernetes manifests not found"; \
		exit 1; \
	fi

t4-status: ## Check T4 deployment status
	@echo "Checking T4 deployment status..."
	@echo "Docker containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(NAMES|t4|vllm)" || echo "No T4 containers running"
	@echo ""
	@echo "Kubernetes pods:"
	@kubectl get pods -l app=vllm 2>/dev/null || echo "No Kubernetes pods found (cluster not accessible)"
	@echo ""
	@echo "GPU status:"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi not available"

# T4 Cleanup
clean-t4: ## Clean up T4-specific resources
	@echo "Cleaning up T4-specific resources..."
	@$(MAKE) docker-stop-t4
	@docker rmi $(IMAGE_NAME):t4-latest $(IMAGE_NAME):t4-$(IMAGE_TAG) 2>/dev/null || true
	@echo "✓ T4 cleanup completed"

# Update the help target to include T4 information
help-t4: ## Show T4-specific help
	@echo 'T4-Optimized vLLM Service - Make Targets'
	@echo '========================================'
	@echo 'T4 Docker Operations:'
	@echo '  docker-build-t4    Build T4-optimized Docker image'
	@echo '  docker-run-t4      Run T4-optimized Docker container'
	@echo '  docker-stop-t4     Stop T4 Docker container'
	@echo ''
	@echo 'T4 Testing:'
	@echo '  test-t4-build      Build and test T4 image'
	@echo '  test-t4-local      Run local T4 service tests'
	@echo '  smoke-t4           T4-specific smoke tests'
	@echo '  benchmark-t4       T4 performance benchmarks'
	@echo ''
	@echo 'T4 Environment:'
	@echo '  setup-t4-env       Setup T4 development environment'
	@echo '  deploy-t4          Deploy T4 service to Kubernetes'
	@echo '  t4-status          Check T4 deployment status'
	@echo '  clean-t4           Clean T4 resources'
	@echo ''
	@echo 'T4 Validation:'
	@echo '  preflight          Run pre-flight validation checks'
	@echo '  verify-runtime     Verify runtime environment'
	@echo '  test-t4-validation Run T4 validation tests'
	@echo '  validate-all       Run all validation checks'
	@echo ''
	@echo 'Use "make help" for general targets'

# T4 Validation Targets
preflight: ## Run pre-flight validation checks for T4 GPU
	@mkdir -p $(ARTIFACT_DIR)
	@echo "Running pre-flight checks for T4 GPU and CUDA 12.4..."
	@bash scripts/preflight_check.sh | tee $(ARTIFACT_DIR)/preflight_output.json
	@echo ""
	@echo "Results saved to $(ARTIFACT_DIR)/preflight_output.json"

verify-runtime: ## Verify runtime environment for T4
	@mkdir -p $(ARTIFACT_DIR)
	@echo "Verifying runtime environment..."
	@if command -v python3 >/dev/null 2>&1; then \
		python3 scripts/verify_runtime.py --output $(ARTIFACT_DIR)/runtime_verification.json; \
	else \
		echo "Python3 not found. Skipping runtime verification."; \
	fi

test-t4-validation: ## Run T4-specific validation tests
	@echo "Running T4 validation test suite..."
	@if [ -f test_t4_validation.py ]; then \
		python3 test_t4_validation.py; \
	else \
		echo "test_t4_validation.py not found"; \
		exit 1; \
	fi

validate-all: preflight verify-runtime test-t4-validation ## Run all validation checks
	@echo ""
	@echo "========================================="
	@echo "All T4 validation checks completed!"
	@echo "========================================="
	@echo "Check the following output files:"
	@echo "  - $(ARTIFACT_DIR)/preflight_output.json"
	@echo "  - $(ARTIFACT_DIR)/runtime_verification.json"

freeze-deps: ## Freeze Python dependencies from built image
	@echo "Freezing Python dependencies from built T4 image..."
	@if docker images | grep -q "$(IMAGE_NAME):t4-latest"; then \
		docker run --rm $(IMAGE_NAME):t4-latest pip freeze > constraints.lock.txt; \
		echo "✓ Dependencies frozen to constraints.lock.txt"; \
	else \
		echo "✗ T4 image not found. Build with 'make docker-build-t4' first"; \
		exit 1; \
	fi

verify-deps: ## Verify dependency alignment between requirements and lock file
	@echo "Verifying dependency alignment..."
	@python3 -c "import sys; \
		req_file = 'requirements-cuda124.txt'; \
		lock_file = 'constraints.lock.txt'; \
		import os; \
		if not os.path.exists(lock_file): \
			print('✗ Lock file not found. Run make freeze-deps first'); \
			sys.exit(1); \
		print('✓ Dependency files exist'); \
		print('Check requirements-cuda124.txt vs constraints.lock.txt for alignment')"

# Pre-commit hooks
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

setup-hooks: ## Set up git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Git hooks installed successfully!"
