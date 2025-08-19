.PHONY: help prepare-caches up-local down smoke logs logs-all services clean-caches build stop clean deploy k8s-apply k8s-delete k8s-status k8s-logs port-forward metrics lint format test-local deploy-build

# Variables
IMAGE_NAME := vllm-service
IMAGE_TAG := latest
NAMESPACE := vllm
REGISTRY :=

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
	ruff check app.py
	ruff format --check app.py

format: ## Format Python code
	@which ruff >/dev/null 2>&1 || pip install ruff
	ruff format app.py
