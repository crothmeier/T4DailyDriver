#!/usr/bin/env python3
"""
Validate canary deployment metrics against thresholds.
Used in CI/CD pipeline to gate promotion.
"""

import argparse
import json
import sys
from typing import Any


def load_results(results_file: str) -> dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def validate_metrics(results: dict[str, Any], ttft_threshold: float, memory_threshold: float) -> tuple[bool, list[str]]:
    """Validate metrics against thresholds."""
    issues = []

    # Check TTFT regression
    if "comparison" in results and "ttft_regression" in results["comparison"]:
        regression = results["comparison"]["ttft_regression"]["percentage"]
        if regression > ttft_threshold * 100:
            issues.append(f"TTFT regression {regression:.2f}% exceeds threshold of {ttft_threshold*100}%")

    # Check memory increase
    if "comparison" in results and "memory_delta" in results["comparison"]:
        memory_increase = results["comparison"]["memory_delta"]["percentage"]
        if memory_increase > memory_threshold * 100:
            issues.append(f"Memory increase {memory_increase:.2f}% exceeds threshold of {memory_threshold*100}%")

    # Check error rates
    if "legacy" in results and "cuda124" in results:
        legacy_errors = results["legacy"]["metrics"].get("throughput", {}).get("failed_requests", 0)
        cuda124_errors = results["cuda124"]["metrics"].get("throughput", {}).get("failed_requests", 0)

        if cuda124_errors > legacy_errors * 2 and cuda124_errors > 5:
            issues.append(f"CUDA 12.4 error count ({cuda124_errors}) is more than 2x legacy ({legacy_errors})")

    # Check throughput degradation
    if "comparison" in results and "throughput_improvement" in results["comparison"]:
        throughput_change = results["comparison"]["throughput_improvement"]["percentage"]
        if throughput_change < -10:  # More than 10% degradation
            issues.append(f"Throughput degradation {abs(throughput_change):.2f}% exceeds 10% threshold")

    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Validate canary metrics")
    parser.add_argument("--results", required=True, help="Path to benchmark results JSON")
    parser.add_argument(
        "--ttft-threshold", type=float, default=0.05, help="Maximum TTFT regression threshold (default: 0.05 = 5%)"
    )
    parser.add_argument(
        "--memory-threshold", type=float, default=0.10, help="Maximum memory increase threshold (default: 0.10 = 10%)"
    )

    args = parser.parse_args()

    # Load results
    try:
        results = load_results(args.results)
    except FileNotFoundError:
        print(f"❌ Results file not found: {args.results}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in results file: {e}")
        sys.exit(1)

    # Validate metrics
    passed, issues = validate_metrics(results, args.ttft_threshold, args.memory_threshold)

    # Print results
    print("=" * 60)
    print("CANARY VALIDATION RESULTS")
    print("=" * 60)

    if passed:
        print("✅ All metrics within acceptable thresholds")
        print(f"   TTFT threshold: {args.ttft_threshold*100}%")
        print(f"   Memory threshold: {args.memory_threshold*100}%")
    else:
        print("❌ Validation failed - Issues detected:")
        for issue in issues:
            print(f"   - {issue}")

    print("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
