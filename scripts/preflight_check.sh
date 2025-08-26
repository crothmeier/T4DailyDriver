#!/bin/bash

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# JSON output structure
declare -A results
declare -A errors
EXIT_CODE=0

# Check NVIDIA Driver
check_nvidia_driver() {
    local check_name="nvidia_driver"
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &>/dev/null; then
            local driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")
            results["${check_name}_version"]="$driver_version"
            results["${check_name}_status"]="pass"

            # Extract CUDA version from nvidia-smi
            local cuda_version=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/' | head -n1 || echo "unknown")
            if [[ -n "$cuda_version" ]] && [[ "$cuda_version" != "unknown" ]]; then
                results["cuda_driver_version"]="$cuda_version"
            fi
        else
            results["${check_name}_status"]="fail"
            errors["${check_name}"]="nvidia-smi command failed"
            EXIT_CODE=1
        fi
    else
        results["${check_name}_status"]="fail"
        errors["${check_name}"]="nvidia-smi not found"
        EXIT_CODE=1
    fi
}

# Check CUDA Runtime
check_cuda_runtime() {
    local check_name="cuda_runtime"
    if command -v nvcc &> /dev/null; then
        local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' | head -n1 || echo "unknown")
        results["${check_name}_version"]="$cuda_version"
        results["${check_name}_status"]="pass"
    else
        results["${check_name}_status"]="warning"
        results["${check_name}_version"]="not_installed"
        errors["${check_name}"]="nvcc not found (may be in container)"
    fi
}

# Check GPU Presence
check_gpu_presence() {
    local check_name="gpu_t4"
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &>/dev/null; then
            local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
            if echo "$gpu_info" | grep -qi "t4"; then
                results["${check_name}_status"]="pass"
                results["${check_name}_detected"]="true"
                results["${check_name}_name"]=$(echo "$gpu_info" | head -n1)
                results["${check_name}_count"]=$(echo "$gpu_info" | grep -ci "t4" || echo "1")
            else
                results["${check_name}_status"]="fail"
                results["${check_name}_detected"]="false"
                if [[ -n "$gpu_info" ]]; then
                    errors["${check_name}"]="GPU found but not T4: $gpu_info"
                else
                    errors["${check_name}"]="No GPU detected"
                fi
                EXIT_CODE=1
            fi
        else
            results["${check_name}_status"]="fail"
            errors["${check_name}"]="Cannot query GPU"
            EXIT_CODE=1
        fi
    else
        results["${check_name}_status"]="fail"
        errors["${check_name}"]="nvidia-smi not available"
        EXIT_CODE=1
    fi
}

# Check flash-attn references
check_flash_attn_references() {
    local check_name="flash_attn_refs"
    local ref_count=0
    local ref_files=""
    local cuda121_refs=0
    local cuda124_refs=0

    # Check specific files for flash-attn references
    local files=(
        "requirements.txt"
        "requirements-cuda124.txt"
        "constraints.txt"
        "constraints-cuda124.txt"
        "Dockerfile"
        "Dockerfile.cuda124"
    )

    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            if grep -q "flash.attn\|flash_attn\|flash-attn" "$file" 2>/dev/null; then
                ((ref_count++)) || true
                ref_files="${ref_files}${file};"

                # Check for CUDA version specific references
                if grep -q "cu121\|cuda.12.1\|cuda12.1" "$file" 2>/dev/null; then
                    ((cuda121_refs++)) || true
                fi
                if grep -q "cu124\|cuda.12.4\|cuda12.4" "$file" 2>/dev/null; then
                    ((cuda124_refs++)) || true
                fi
            fi
        fi
    done

    # Also check Python files in scripts
    if [[ -d "scripts" ]]; then
        for file in scripts/*.py; do
            if [[ -f "$file" ]] && grep -q "flash.attn\|flash_attn\|flash-attn" "$file" 2>/dev/null; then
                ((ref_count++)) || true
                ref_files="${ref_files}${file};"
            fi
        done
    fi

    results["${check_name}_count"]="$ref_count"
    results["${check_name}_status"]="pass"
    if [[ $ref_count -gt 0 ]]; then
        results["${check_name}_files"]="${ref_files%;}"
        results["${check_name}_cuda121_refs"]="$cuda121_refs"
        results["${check_name}_cuda124_refs"]="$cuda124_refs"
    fi
}

# Check Docker
check_docker() {
    local check_name="docker"
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version 2>/dev/null | sed 's/Docker version \([0-9.]*\).*/\1/' | head -n1 || echo "unknown")
        results["${check_name}_version"]="$docker_version"
        results["${check_name}_status"]="pass"

        if docker info &>/dev/null; then
            results["${check_name}_daemon"]="running"
        else
            results["${check_name}_daemon"]="not_accessible"
            errors["${check_name}_daemon"]="Docker daemon not accessible"
        fi
    else
        results["${check_name}_status"]="fail"
        errors["${check_name}"]="Docker not found"
        EXIT_CODE=1
    fi
}

# Check Docker Compose
check_docker_compose() {
    local check_name="docker_compose"
    local compose_cmd=""
    local compose_version=""

    if docker compose version &>/dev/null; then
        compose_cmd="docker compose"
        compose_version=$(docker compose version 2>/dev/null | sed 's/.*version \([v0-9.]*\).*/\1/' | head -n1 || echo "unknown")
        results["${check_name}_version"]="$compose_version"
        results["${check_name}_command"]="$compose_cmd"
        results["${check_name}_status"]="pass"
    elif command -v docker-compose &> /dev/null; then
        compose_cmd="docker-compose"
        compose_version=$(docker-compose --version 2>/dev/null | sed 's/.*version \([0-9.]*\).*/\1/' | head -n1 || echo "unknown")
        results["${check_name}_version"]="$compose_version"
        results["${check_name}_command"]="$compose_cmd"
        results["${check_name}_status"]="pass"
    else
        results["${check_name}_status"]="fail"
        errors["${check_name}"]="Docker Compose not found"
        EXIT_CODE=1
    fi
}

# Check Project Files
check_project_files() {
    local check_name="project_files"
    local required_files=(
        "Dockerfile"
        "Dockerfile.cuda124"
        "docker-compose.yaml"
        "docker-compose.cuda124.yaml"
        "requirements.txt"
        "requirements-cuda124.txt"
        "constraints.txt"
        "constraints-cuda124.txt"
    )

    local missing_files=""
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files="${missing_files}${file};"
        fi
    done

    if [[ -z "$missing_files" ]]; then
        results["${check_name}_status"]="pass"
        results["${check_name}_all_present"]="true"
    else
        results["${check_name}_status"]="fail"
        results["${check_name}_missing"]="${missing_files%;}"
        errors["${check_name}"]="Missing required files"
        EXIT_CODE=1
    fi
}

# Format JSON output
format_json_output() {
    echo "{"
    echo "  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
    echo "  \"exit_code\": $EXIT_CODE,"
    echo "  \"status\": \"$([ $EXIT_CODE -eq 0 ] && echo "pass" || echo "fail")\","
    echo "  \"checks\": {"

    local first=true
    for key in "${!results[@]}"; do
        if [[ "$first" == "false" ]]; then
            echo ","
        fi
        # Escape quotes in values
        local value="${results[$key]}"
        value="${value//\"/\\\"}"
        echo -n "    \"${key}\": \"${value}\""
        first=false
    done

    echo ""
    echo "  },"
    echo "  \"errors\": {"

    first=true
    for key in "${!errors[@]}"; do
        if [[ "$first" == "false" ]]; then
            echo ","
        fi
        # Escape quotes in values
        local value="${errors[$key]}"
        value="${value//\"/\\\"}"
        echo -n "    \"${key}\": \"${value}\""
        first=false
    done

    echo ""
    echo "  },"
    echo "  \"summary\": {"
    echo "    \"total_checks\": 7,"
    echo "    \"passed\": $(echo "${results[@]}" | grep -o "pass" | wc -l || echo "0"),"
    echo "    \"failed\": $(echo "${results[@]}" | grep -o "fail" | wc -l || echo "0"),"
    echo "    \"warnings\": $(echo "${results[@]}" | grep -o "warning" | wc -l || echo "0")"
    echo "  }"
    echo "}"
}

# Main function
main() {
    echo -e "${GREEN}Running Pre-Flight Checks for CUDA 12.4 Upgrade...${NC}" >&2
    echo -e "${GREEN}================================================${NC}" >&2

    echo -e "\n${YELLOW}[1/7] Checking NVIDIA Driver...${NC}" >&2
    check_nvidia_driver

    echo -e "${YELLOW}[2/7] Checking CUDA Runtime...${NC}" >&2
    check_cuda_runtime

    echo -e "${YELLOW}[3/7] Checking GPU Presence (Tesla T4)...${NC}" >&2
    check_gpu_presence

    echo -e "${YELLOW}[4/7] Scanning flash-attn References...${NC}" >&2
    check_flash_attn_references

    echo -e "${YELLOW}[5/7] Checking Docker...${NC}" >&2
    check_docker

    echo -e "${YELLOW}[6/7] Checking Docker Compose...${NC}" >&2
    check_docker_compose

    echo -e "${YELLOW}[7/7] Checking Project Files...${NC}" >&2
    check_project_files

    echo -e "\n${GREEN}================================================${NC}" >&2
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo -e "${GREEN}✓ All pre-flight checks passed!${NC}" >&2
    else
        echo -e "${RED}✗ Some pre-flight checks failed!${NC}" >&2
        echo -e "${YELLOW}Review the JSON output for details.${NC}" >&2
    fi
    echo -e "${GREEN}================================================${NC}\n" >&2

    # Output JSON to stdout
    format_json_output

    exit $EXIT_CODE
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
