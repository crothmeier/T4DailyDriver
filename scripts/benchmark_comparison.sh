#!/usr/bin/env python3
"""
Benchmark comparison script for CUDA upgrade validation.
Compares performance between legacy (CUDA 12.1) and new (CUDA 12.4) deployments.
"""

import argparse
import asyncio
import json
import time
import statistics
from typing import Dict, List, Any, Tuple
import httpx
import numpy as np
from datetime import datetime
import sys
import os

class BenchmarkRunner:
    def __init__(self, legacy_url: str, cuda124_url: str):
        self.legacy_url = legacy_url
        self.cuda124_url = cuda124_url
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "legacy": {"url": legacy_url, "metrics": {}},
            "cuda124": {"url": cuda124_url, "metrics": {}},
            "comparison": {},
            "validation": {"passed": False, "issues": []}
        }

    async def warmup(self, url: str, num_requests: int = 5):
        """Warm up the service with initial requests."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = []
            for _ in range(num_requests):
                prompt = {
                    "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                tasks.append(client.post(f"{url}/v1/chat/completions", json=prompt))

            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(2)  # Let the service stabilize

    async def measure_latency(self, url: str, prompts: List[str], max_tokens: int = 100) -> Dict[str, Any]:
        """Measure latency metrics for given prompts."""
        latencies = []
        ttft_times = []  # Time to first token
        throughputs = []
        errors = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            for prompt_text in prompts:
                try:
                    prompt = {
                        "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "stream": True
                    }

                    start_time = time.perf_counter()
                    first_token_time = None
                    tokens_received = 0

                    async with client.stream("POST", f"{url}/v1/chat/completions", json=prompt) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                    ttft_times.append(first_token_time - start_time)
                                tokens_received += 1

                                if line == "data: [DONE]":
                                    break

                    end_time = time.perf_counter()
                    total_time = end_time - start_time
                    latencies.append(total_time)

                    if tokens_received > 0:
                        throughputs.append(tokens_received / total_time)

                except Exception as e:
                    print(f"Error during request to {url}: {e}")
                    errors += 1

        if not latencies:
            return {"error": "All requests failed", "error_count": errors}

        return {
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": min(latencies),
                "max": max(latencies)
            },
            "ttft": {
                "mean": statistics.mean(ttft_times) if ttft_times else 0,
                "median": statistics.median(ttft_times) if ttft_times else 0,
                "p95": np.percentile(ttft_times, 95) if ttft_times else 0,
                "p99": np.percentile(ttft_times, 99) if ttft_times else 0
            },
            "throughput": {
                "mean_tokens_per_second": statistics.mean(throughputs) if throughputs else 0,
                "total_requests": len(latencies),
                "failed_requests": errors
            }
        }

    async def measure_memory(self, url: str) -> Dict[str, Any]:
        """Get memory metrics from the service."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{url}/metrics")
                response.raise_for_status()

                metrics_text = response.text
                memory_metrics = {}

                # Parse Prometheus metrics
                for line in metrics_text.split('\n'):
                    if 'memory_usage_bytes' in line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            memory_metrics['memory_bytes'] = float(parts[1])
                            memory_metrics['memory_gb'] = float(parts[1]) / (1024**3)
                    elif 'gpu_memory_used' in line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            memory_metrics['gpu_memory_mb'] = float(parts[1])
                            memory_metrics['gpu_memory_gb'] = float(parts[1]) / 1024

                return memory_metrics
        except Exception as e:
            return {"error": f"Failed to get memory metrics: {str(e)}"}

    async def run_concurrent_load(self, url: str, num_concurrent: int = 10, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run concurrent load test."""
        start_time = time.time()
        completed_requests = 0
        failed_requests = 0
        latencies = []

        async def single_request():
            nonlocal completed_requests, failed_requests, latencies
            try:
                prompt = {
                    "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                    "messages": [{"role": "user", "content": f"Generate a random story about {np.random.choice(['space', 'ocean', 'forest', 'city'])}"}],
                    "max_tokens": 150,
                    "temperature": 0.8
                }

                req_start = time.perf_counter()
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(f"{url}/v1/chat/completions", json=prompt)
                    response.raise_for_status()
                req_end = time.perf_counter()

                latencies.append(req_end - req_start)
                completed_requests += 1
            except Exception:
                failed_requests += 1

        tasks = []
        while time.time() - start_time < duration_seconds:
            # Maintain concurrent requests
            while len(tasks) < num_concurrent:
                tasks.append(asyncio.create_task(single_request()))

            # Wait for some to complete
            done, pending = await asyncio.wait(tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            tasks = list(pending)

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed_time = time.time() - start_time

        return {
            "duration_seconds": elapsed_time,
            "total_requests": completed_requests + failed_requests,
            "successful_requests": completed_requests,
            "failed_requests": failed_requests,
            "requests_per_second": completed_requests / elapsed_time if elapsed_time > 0 else 0,
            "latency_stats": {
                "mean": statistics.mean(latencies) if latencies else 0,
                "median": statistics.median(latencies) if latencies else 0,
                "p95": np.percentile(latencies, 95) if latencies else 0,
                "p99": np.percentile(latencies, 99) if latencies else 0
            }
        }

    def calculate_comparison(self):
        """Calculate comparison metrics between legacy and CUDA 12.4."""
        comparison = {}

        # Compare latency
        if "latency" in self.results["legacy"]["metrics"] and "latency" in self.results["cuda124"]["metrics"]:
            legacy_lat = self.results["legacy"]["metrics"]["latency"]["mean"]
            cuda124_lat = self.results["cuda124"]["metrics"]["latency"]["mean"]
            comparison["latency_improvement"] = {
                "percentage": ((legacy_lat - cuda124_lat) / legacy_lat) * 100,
                "absolute_ms": (legacy_lat - cuda124_lat) * 1000
            }

        # Compare TTFT
        if "ttft" in self.results["legacy"]["metrics"] and "ttft" in self.results["cuda124"]["metrics"]:
            legacy_ttft = self.results["legacy"]["metrics"]["ttft"]["mean"]
            cuda124_ttft = self.results["cuda124"]["metrics"]["ttft"]["mean"]
            comparison["ttft_regression"] = {
                "percentage": ((cuda124_ttft - legacy_ttft) / legacy_ttft) * 100 if legacy_ttft > 0 else 0,
                "absolute_ms": (cuda124_ttft - legacy_ttft) * 1000
            }

        # Compare memory
        if "memory" in self.results["legacy"]["metrics"] and "memory" in self.results["cuda124"]["metrics"]:
            legacy_mem = self.results["legacy"]["metrics"]["memory"].get("gpu_memory_gb", 0)
            cuda124_mem = self.results["cuda124"]["metrics"]["memory"].get("gpu_memory_gb", 0)
            comparison["memory_delta"] = {
                "percentage": ((cuda124_mem - legacy_mem) / legacy_mem) * 100 if legacy_mem > 0 else 0,
                "absolute_gb": cuda124_mem - legacy_mem
            }

        # Compare throughput
        if "load_test" in self.results["legacy"]["metrics"] and "load_test" in self.results["cuda124"]["metrics"]:
            legacy_rps = self.results["legacy"]["metrics"]["load_test"]["requests_per_second"]
            cuda124_rps = self.results["cuda124"]["metrics"]["load_test"]["requests_per_second"]
            comparison["throughput_improvement"] = {
                "percentage": ((cuda124_rps - legacy_rps) / legacy_rps) * 100 if legacy_rps > 0 else 0,
                "absolute_rps": cuda124_rps - legacy_rps
            }

        self.results["comparison"] = comparison

    def validate_thresholds(self, ttft_threshold: float = 0.05, memory_threshold: float = 0.10):
        """Validate performance against thresholds."""
        issues = []

        # Check TTFT regression
        if "ttft_regression" in self.results["comparison"]:
            regression = self.results["comparison"]["ttft_regression"]["percentage"]
            if regression > ttft_threshold * 100:
                issues.append(f"TTFT regression {regression:.2f}% exceeds threshold of {ttft_threshold*100}%")

        # Check memory increase
        if "memory_delta" in self.results["comparison"]:
            memory_increase = self.results["comparison"]["memory_delta"]["percentage"]
            if memory_increase > memory_threshold * 100:
                issues.append(f"Memory increase {memory_increase:.2f}% exceeds threshold of {memory_threshold*100}%")

        self.results["validation"]["issues"] = issues
        self.results["validation"]["passed"] = len(issues) == 0
        self.results["validation"]["ttft_threshold"] = ttft_threshold
        self.results["validation"]["memory_threshold"] = memory_threshold

    async def run_benchmark(self):
        """Run the complete benchmark suite."""
        print("=" * 60)
        print("Starting Benchmark Comparison")
        print("=" * 60)

        # Test prompts of varying complexity
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "Describe the process of photosynthesis in detail.",
            "Create a business plan outline for a tech startup.",
            "Translate 'Hello, how are you?' to Spanish, French, and German.",
            "Solve this math problem: If a train travels 120 miles in 2 hours, what is its average speed?",
            "Write a haiku about artificial intelligence.",
            "List the pros and cons of remote work.",
            "Explain the difference between machine learning and deep learning."
        ]

        # Warm up both services
        print("\n📊 Warming up services...")
        await self.warmup(self.legacy_url)
        await self.warmup(self.cuda124_url)

        # Run latency benchmarks
        print("\n⏱️  Measuring latency...")
        print(f"  Legacy: {self.legacy_url}")
        legacy_latency = await self.measure_latency(self.legacy_url, test_prompts)
        self.results["legacy"]["metrics"]["latency"] = legacy_latency.get("latency", {})
        self.results["legacy"]["metrics"]["ttft"] = legacy_latency.get("ttft", {})
        self.results["legacy"]["metrics"]["throughput"] = legacy_latency.get("throughput", {})

        print(f"  CUDA 12.4: {self.cuda124_url}")
        cuda124_latency = await self.measure_latency(self.cuda124_url, test_prompts)
        self.results["cuda124"]["metrics"]["latency"] = cuda124_latency.get("latency", {})
        self.results["cuda124"]["metrics"]["ttft"] = cuda124_latency.get("ttft", {})
        self.results["cuda124"]["metrics"]["throughput"] = cuda124_latency.get("throughput", {})

        # Get memory metrics
        print("\n💾 Measuring memory usage...")
        self.results["legacy"]["metrics"]["memory"] = await self.measure_memory(self.legacy_url)
        self.results["cuda124"]["metrics"]["memory"] = await self.measure_memory(self.cuda124_url)

        # Run concurrent load test
        print("\n🔥 Running concurrent load test (30s)...")
        print("  Legacy service...")
        self.results["legacy"]["metrics"]["load_test"] = await self.run_concurrent_load(
            self.legacy_url, num_concurrent=5, duration_seconds=30
        )

        print("  CUDA 12.4 service...")
        self.results["cuda124"]["metrics"]["load_test"] = await self.run_concurrent_load(
            self.cuda124_url, num_concurrent=5, duration_seconds=30
        )

        # Calculate comparisons
        self.calculate_comparison()

        # Validate against thresholds
        self.validate_thresholds()

        # Print summary
        self.print_summary()

        return self.results

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)

        # Latency comparison
        if "latency" in self.results["legacy"]["metrics"] and "latency" in self.results["cuda124"]["metrics"]:
            print("\n📊 Latency (seconds):")
            print(f"  Legacy:    {self.results['legacy']['metrics']['latency']['mean']:.3f}s (p95: {self.results['legacy']['metrics']['latency']['p95']:.3f}s)")
            print(f"  CUDA 12.4: {self.results['cuda124']['metrics']['latency']['mean']:.3f}s (p95: {self.results['cuda124']['metrics']['latency']['p95']:.3f}s)")
            if "latency_improvement" in self.results["comparison"]:
                imp = self.results["comparison"]["latency_improvement"]["percentage"]
                print(f"  {'✅ Improvement' if imp > 0 else '⚠️  Regression'}: {abs(imp):.1f}%")

        # TTFT comparison
        if "ttft" in self.results["legacy"]["metrics"] and "ttft" in self.results["cuda124"]["metrics"]:
            print("\n⚡ Time to First Token (seconds):")
            print(f"  Legacy:    {self.results['legacy']['metrics']['ttft']['mean']:.3f}s")
            print(f"  CUDA 12.4: {self.results['cuda124']['metrics']['ttft']['mean']:.3f}s")
            if "ttft_regression" in self.results["comparison"]:
                reg = self.results["comparison"]["ttft_regression"]["percentage"]
                print(f"  {'⚠️  Regression' if reg > 0 else '✅ Improvement'}: {abs(reg):.1f}%")

        # Memory comparison
        if "memory" in self.results["legacy"]["metrics"] and "memory" in self.results["cuda124"]["metrics"]:
            print("\n💾 GPU Memory Usage:")
            legacy_mem = self.results["legacy"]["metrics"]["memory"].get("gpu_memory_gb", 0)
            cuda124_mem = self.results["cuda124"]["metrics"]["memory"].get("gpu_memory_gb", 0)
            print(f"  Legacy:    {legacy_mem:.2f} GB")
            print(f"  CUDA 12.4: {cuda124_mem:.2f} GB")
            if "memory_delta" in self.results["comparison"]:
                delta = self.results["comparison"]["memory_delta"]["percentage"]
                print(f"  {'⚠️  Increase' if delta > 0 else '✅ Decrease'}: {abs(delta):.1f}%")

        # Throughput comparison
        if "load_test" in self.results["legacy"]["metrics"] and "load_test" in self.results["cuda124"]["metrics"]:
            print("\n🚀 Throughput (requests/second):")
            print(f"  Legacy:    {self.results['legacy']['metrics']['load_test']['requests_per_second']:.2f} req/s")
            print(f"  CUDA 12.4: {self.results['cuda124']['metrics']['load_test']['requests_per_second']:.2f} req/s")
            if "throughput_improvement" in self.results["comparison"]:
                imp = self.results["comparison"]["throughput_improvement"]["percentage"]
                print(f"  {'✅ Improvement' if imp > 0 else '⚠️  Regression'}: {abs(imp):.1f}%")

        # Validation result
        print("\n" + "=" * 60)
        if self.results["validation"]["passed"]:
            print("✅ VALIDATION PASSED - Safe to proceed with rollout")
        else:
            print("❌ VALIDATION FAILED - Issues detected:")
            for issue in self.results["validation"]["issues"]:
                print(f"   - {issue}")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='Benchmark comparison between legacy and CUDA 12.4 deployments')
    parser.add_argument('--legacy-url', default='http://localhost:8080',
                       help='URL for legacy service')
    parser.add_argument('--cuda124-url', default='http://localhost:8081',
                       help='URL for CUDA 12.4 service')
    parser.add_argument('--output', default='benchmarks/comparison-results.json',
                       help='Output file for results')
    parser.add_argument('--ttft-threshold', type=float, default=0.05,
                       help='Maximum allowed TTFT regression (default: 5%)')
    parser.add_argument('--memory-threshold', type=float, default=0.10,
                       help='Maximum allowed memory increase (default: 10%)')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Run benchmark
    runner = BenchmarkRunner(args.legacy_url, args.cuda124_url)
    results = await runner.run_benchmark()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if results["validation"]["passed"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
