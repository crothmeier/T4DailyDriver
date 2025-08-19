#!/bin/bash
set -euo pipefail

# smoke_compose.sh - Validate Docker Compose security hardening
# Exit codes: 0=success, 1=failure

echo "=== Docker Compose Smoke Test ==="
echo "Testing security hardening and healthchecks..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MAX_WAIT=120  # Maximum wait time in seconds
HEALTH_CHECK_INTERVAL=5  # Check interval in seconds

# Function to check service health
check_service_health() {
    local service=$1
    local container=$2
    local elapsed=0

    echo -n "Checking $service health..."

    while [ $elapsed -lt $MAX_WAIT ]; do
        # Get container health status
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "not-found")

        if [ "$health_status" = "healthy" ]; then
            echo -e " ${GREEN}✓ HEALTHY${NC}"
            return 0
        elif [ "$health_status" = "not-found" ]; then
            echo -e " ${RED}✗ Container not found${NC}"
            return 1
        fi

        sleep $HEALTH_CHECK_INTERVAL
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
        echo -n "."
    done

    echo -e " ${RED}✗ UNHEALTHY (status: $health_status)${NC}"
    return 1
}

# Function to test HTTP endpoint
test_http_endpoint() {
    local service=$1
    local url=$2
    local expected_code=${3:-200}

    echo -n "Testing $service endpoint..."

    # Use curl with timeout
    response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")

    if [ "$response_code" = "$expected_code" ]; then
        echo -e " ${GREEN}✓ HTTP $response_code${NC}"
        return 0
    else
        echo -e " ${RED}✗ HTTP $response_code (expected $expected_code)${NC}"
        return 1
    fi
}

# Function to check security configurations
check_security_config() {
    local container=$1
    local service=$2

    echo "Checking $service security configuration:"

    # Check read-only root filesystem
    read_only=$(docker inspect --format='{{.HostConfig.ReadonlyRootfs}}' "$container" 2>/dev/null)
    if [ "$read_only" = "true" ]; then
        echo -e "  Read-only filesystem: ${GREEN}✓${NC}"
    else
        echo -e "  Read-only filesystem: ${YELLOW}⚠ Not enabled${NC}"
    fi

    # Check capabilities
    cap_add=$(docker inspect --format='{{join .HostConfig.CapAdd " "}}' "$container" 2>/dev/null)
    cap_drop=$(docker inspect --format='{{join .HostConfig.CapDrop " "}}' "$container" 2>/dev/null)

    if [[ "$cap_drop" == *"ALL"* ]]; then
        echo -e "  Capabilities dropped: ${GREEN}✓ ALL${NC}"
        if [ -n "$cap_add" ]; then
            echo "  Capabilities added: $cap_add"
        fi
        # Check NET_RAW is not added
        if [[ "$cap_add" != *"NET_RAW"* ]]; then
            echo -e "  NET_RAW capability: ${GREEN}✓ Not present${NC}"
        else
            echo -e "  NET_RAW capability: ${RED}✗ Should be removed${NC}"
        fi
    else
        echo -e "  Capabilities dropped: ${YELLOW}⚠ Not dropping ALL${NC}"
    fi

    echo ""
}

# Main test sequence
main() {
    local exit_code=0

    # Check if docker compose is running
    echo "Checking Docker Compose status..."
    if ! docker compose ps --format json > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker Compose is not running or not accessible${NC}"
        echo "Please run: docker compose up -d"
        exit 1
    fi

    echo ""
    echo "=== Health Check Validation ==="
    echo ""

    # Services to check with their expected container names
    declare -A services=(
        ["vLLM Service"]="vllm-t4-service"
        ["Prometheus"]="prometheus"
        ["Grafana"]="grafana"
    )

    # Check each service health
    for service in "${!services[@]}"; do
        if ! check_service_health "$service" "${services[$service]}"; then
            exit_code=1
        fi
    done

    echo ""
    echo "=== HTTP Endpoint Tests ==="
    echo ""

    # Test HTTP endpoints
    test_http_endpoint "vLLM /health" "http://localhost:8080/health" || exit_code=1
    test_http_endpoint "vLLM /metrics" "http://localhost:8080/metrics" || exit_code=1
    test_http_endpoint "Prometheus" "http://localhost:9090/-/healthy" || exit_code=1
    test_http_endpoint "Grafana API" "http://localhost:3000/api/health" || exit_code=1

    echo ""
    echo "=== Security Configuration Validation ==="
    echo ""

    # Check security configurations for main services
    check_security_config "vllm-t4-service" "vLLM"
    check_security_config "prometheus" "Prometheus"
    check_security_config "grafana" "Grafana"

    echo ""
    echo "=== Container Status Overview ==="
    echo ""
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"

    echo ""

    # Prometheus scrape test
    echo "=== Prometheus Scrape Test ==="
    echo -n "Checking if Prometheus can scrape vLLM metrics..."

    # Query Prometheus for vLLM metrics
    prom_response=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null || echo "{}")
    if echo "$prom_response" | grep -q "vllm.*up"; then
        echo -e " ${GREEN}✓ Scraping active${NC}"
    else
        echo -e " ${YELLOW}⚠ Cannot verify scraping${NC}"
    fi

    echo ""

    # Final result
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}=== All smoke tests PASSED ===${NC}"
    else
        echo -e "${RED}=== Some tests FAILED ===${NC}"
        echo "Please check the failed components above"
    fi

    exit $exit_code
}

# Run main function
main "$@"
