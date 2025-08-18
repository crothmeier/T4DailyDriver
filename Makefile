.PHONY: help build run stop clean deploy test lint

# Variables
IMAGE_NAME := vllm-service
IMAGE_TAG := latest
NAMESPACE := vllm
REGISTRY :=

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run: ## Run locally with docker-compose
	docker-compose up -d

stop: ## Stop local services
	docker-compose down

logs: ## Show logs from local services
	docker-compose logs -f vllm

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
