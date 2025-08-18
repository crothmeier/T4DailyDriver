#!/bin/bash
# Deployment script for vLLM service

set -euo pipefail

# Configuration
NAMESPACE=${NAMESPACE:-vllm}
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi

    # Check for GPU nodes
    if ! kubectl get nodes -o json | jq -r '.items[].status.allocatable' | grep -q "nvidia.com/gpu"; then
        log_warning "No GPU nodes found in the cluster. Deployment may fail."
    fi

    log_info "Prerequisites check passed."
}

# Build and push Docker image
build_and_push() {
    log_info "Building Docker image..."

    if [ -z "$REGISTRY" ]; then
        log_warning "No registry specified. Using local image."
        docker build -t vllm-service:${IMAGE_TAG} .
    else
        docker build -t ${REGISTRY}/vllm-service:${IMAGE_TAG} .
        log_info "Pushing image to registry..."
        docker push ${REGISTRY}/vllm-service:${IMAGE_TAG}
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."

    # Update image in kustomization if registry is specified
    if [ -n "$REGISTRY" ]; then
        cd k8s
        kustomize edit set image vllm-service=${REGISTRY}/vllm-service:${IMAGE_TAG}
        cd ..
    fi

    # Apply manifests
    kubectl apply -k k8s/

    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl -n ${NAMESPACE} rollout status deployment/vllm-mistral-7b --timeout=300s

    # Get pod status
    kubectl -n ${NAMESPACE} get pods -l app=vllm

    # Get service info
    log_info "Service information:"
    kubectl -n ${NAMESPACE} get svc vllm-service
}

# Health check
health_check() {
    log_info "Running health check..."

    # Port forward
    kubectl -n ${NAMESPACE} port-forward svc/vllm-service 8080:80 &
    PF_PID=$!
    sleep 5

    # Check health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_info "Health check passed!"
        curl -s http://localhost:8080/health | jq .
    else
        log_error "Health check failed!"
    fi

    # Kill port forward
    kill $PF_PID 2>/dev/null || true
}

# Main execution
main() {
    log_info "Starting vLLM service deployment..."

    check_prerequisites

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD=true
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --health-check)
                HEALTH_CHECK=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Build if requested
    if [ "${BUILD:-false}" = true ]; then
        build_and_push
    fi

    # Deploy
    deploy_k8s

    # Health check if requested
    if [ "${HEALTH_CHECK:-false}" = true ]; then
        health_check
    fi

    log_info "Deployment completed successfully!"
}

# Run main
main "$@"
