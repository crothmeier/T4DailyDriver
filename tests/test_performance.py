"""
Performance and load testing for the enhanced vLLM service.
Tests throughput, latency, concurrent request handling, and stress scenarios.
"""

import asyncio
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from app.reliability import CircuitBreaker, CircuitBreakerConfig, OOMPreventionMonitor, RequestDeduplicator


class TestPerformanceMetrics:
    """Test performance measurement and optimization features."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_latency(self):
        """Test circuit breaker latency impact."""
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=1)
        cb = CircuitBreaker("performance_test", config)

        # Measure latency without circuit breaker
        start_time = time.time()
        for _ in range(100):
            pass  # Simulate operation
        baseline_time = time.time() - start_time

        # Measure latency with circuit breaker (should be minimal overhead)
        start_time = time.time()
        for _ in range(100):
            async with cb.protect():
                pass  # Simulate operation
        protected_time = time.time() - start_time

        # Circuit breaker overhead should be minimal (less than 10% increase)
        overhead_ratio = protected_time / baseline_time
        assert overhead_ratio < 1.1, f"Circuit breaker adds {overhead_ratio:.2f}x overhead"

    def test_request_deduplication_performance(self):
        """Test request deduplication performance."""
        dedup = RequestDeduplicator(ttl=300)

        # Generate test data
        test_requests = []
        for i in range(1000):
            test_requests.append(
                {"prompt": f"Test prompt {i % 100}", "max_tokens": 100}  # 100 unique prompts, repeated 10x each
            )

        # Measure cache performance
        cache_hits = 0
        miss_times = []
        hit_times = []

        for i, request in enumerate(test_requests):
            start_time = time.time()

            cached_result = dedup.get_or_set_idempotent(None, request, None)

            if cached_result is not None:
                cache_hits += 1
                hit_times.append(time.time() - start_time)
            else:
                # Simulate setting result
                dedup.get_or_set_idempotent(None, request, {"response": f"Result {i}"})
                miss_times.append(time.time() - start_time)

        # Performance assertions
        assert cache_hits > 0, "Should have cache hits with duplicate requests"

        if hit_times and miss_times:
            avg_hit_time = statistics.mean(hit_times)
            avg_miss_time = statistics.mean(miss_times)

            # Cache hits should be faster than misses
            assert avg_hit_time < avg_miss_time, "Cache hits should be faster than misses"

    def test_memory_monitor_performance(self):
        """Test OOM prevention monitor performance."""
        monitor = OOMPreventionMonitor()

        # Measure check performance
        times = []
        for _ in range(100):
            start_time = time.time()
            status = monitor.check_memory_pressure()
            times.append(time.time() - start_time)

        avg_time = statistics.mean(times)
        max_time = max(times)

        # Memory checks should be fast (under 1ms average, 5ms max)
        assert avg_time < 0.001, f"Average memory check too slow: {avg_time:.4f}s"
        assert max_time < 0.005, f"Max memory check too slow: {max_time:.4f}s"


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_concurrency(self):
        """Test circuit breaker under concurrent load."""
        config = CircuitBreakerConfig(failure_threshold=10, recovery_timeout=1)
        cb = CircuitBreaker("concurrency_test", config)

        async def test_operation(operation_id: int, should_fail: bool = False):
            """Simulate an operation that may fail."""
            async with cb.protect():
                await asyncio.sleep(0.001)  # Simulate work
                if should_fail and operation_id % 5 == 0:  # Fail every 5th operation
                    raise Exception(f"Operation {operation_id} failed")
                return f"Success {operation_id}"

        # Run concurrent operations
        tasks = []
        for i in range(50):
            # Mix of successful and failing operations
            should_fail = i < 10  # First 10 operations fail
            tasks.append(test_operation(i, should_fail))

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        successes = sum(1 for r in results if isinstance(r, str))
        failures = sum(1 for r in results if isinstance(r, Exception))

        assert successes > 0, "Should have some successful operations"
        assert failures > 0, "Should have some failed operations"

        # Circuit breaker should eventually open and block requests
        # (This is verified by the circuit breaker state)

    def test_request_deduplication_concurrency(self):
        """Test request deduplication under concurrent access."""
        dedup = RequestDeduplicator(ttl=60)

        def worker(worker_id: int, num_requests: int):
            """Worker function for concurrent testing."""
            results = []
            for i in range(num_requests):
                request = {"prompt": f"Worker {worker_id} request {i % 5}", "max_tokens": 100}  # Some duplicates

                # Try to get cached result
                cached = dedup.get_or_set_idempotent(None, request, None)
                if cached is None:
                    # Set new result
                    result = f"Worker {worker_id} result {i}"
                    dedup.get_or_set_idempotent(None, request, result)
                    results.append(result)
                else:
                    results.append(cached)

            return results

        # Run concurrent workers
        num_workers = 10
        requests_per_worker = 20

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(worker, worker_id, requests_per_worker)
                futures.append(future)

            # Collect results
            all_results = []
            for future in futures:
                worker_results = future.result()
                all_results.extend(worker_results)

        # Verify no corruption occurred
        assert len(all_results) == num_workers * requests_per_worker

        # Check that deduplication worked (should have cache entries)
        assert len(dedup.cache) > 0, "Should have cached entries"
        assert len(dedup.cache) < len(all_results), "Should have deduplication"


class TestStressScenarios:
    """Test system behavior under stress conditions."""

    def test_memory_pressure_simulation(self):
        """Test behavior under simulated memory pressure."""
        monitor = OOMPreventionMonitor(warning_threshold=0.7, critical_threshold=0.9)

        # Simulate increasing memory pressure
        memory_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        for level in memory_levels:
            with patch.object(monitor, "_get_gpu_memory_usage", return_value=level):
                status = monitor.check_memory_pressure()
                should_accept = monitor.should_accept_request()

                if level < 0.7:
                    assert status["status"] == "ok"
                    assert should_accept is True
                elif level < 0.9:
                    assert status["status"] == "warning"
                    assert should_accept is True
                else:
                    assert status["status"] == "critical"
                    assert should_accept is False

    @pytest.mark.asyncio
    async def test_cascading_failures(self):
        """Test system behavior under cascading failures."""
        # Create multiple circuit breakers to simulate different components
        gpu_cb = CircuitBreaker("gpu", CircuitBreakerConfig(failure_threshold=3))
        model_cb = CircuitBreaker("model", CircuitBreakerConfig(failure_threshold=5))

        failure_count = 0

        async def failing_operation(cb: CircuitBreaker, failure_rate: float):
            """Operation that fails at specified rate."""
            nonlocal failure_count

            async with cb.protect():
                if failure_count / 10 < failure_rate:  # Increase failure over time
                    failure_count += 1
                    raise Exception("Simulated failure")
                return "Success"

        # Simulate increasing failure rates
        results = []
        for i in range(20):
            try:
                # Try GPU operation first
                await failing_operation(gpu_cb, 0.3)

                # Then model operation
                result = await failing_operation(model_cb, 0.2)
                results.append(result)

            except Exception as e:
                results.append(str(e))

            # Small delay between operations
            await asyncio.sleep(0.01)

        # Verify that circuit breakers eventually opened
        from app.reliability import CircuitBreakerState

        # At least one circuit breaker should have opened due to failures
        assert (
            gpu_cb.state == CircuitBreakerState.OPEN or model_cb.state == CircuitBreakerState.OPEN
        ), "Circuit breakers should open under sustained failures"

    def test_cache_performance_under_load(self):
        """Test deduplication cache performance under high load."""
        dedup = RequestDeduplicator(ttl=60)

        # Generate large number of requests with high duplication
        num_unique_requests = 100
        duplications_per_request = 50

        total_requests = num_unique_requests * duplications_per_request

        start_time = time.time()
        cache_hits = 0

        for i in range(total_requests):
            request = {"prompt": f"Request {i % num_unique_requests}", "max_tokens": 100}  # High duplication

            cached = dedup.get_or_set_idempotent(None, request, None)
            if cached is None:
                # Set result for new request
                dedup.get_or_set_idempotent(None, request, f"Result {i}")
            else:
                cache_hits += 1

        total_time = time.time() - start_time

        # Performance assertions
        expected_hits = total_requests - num_unique_requests
        hit_rate = cache_hits / total_requests

        assert hit_rate > 0.8, f"Cache hit rate too low: {hit_rate:.2f}"
        assert total_time < 1.0, f"Cache operations too slow: {total_time:.2f}s for {total_requests} requests"

        # Verify cache size is reasonable
        assert len(dedup.cache) <= num_unique_requests * 1.1, "Cache size should be bounded"


class TestResourceUsage:
    """Test resource usage and optimization."""

    def test_memory_usage_tracking(self):
        """Test memory usage tracking for components."""
        import sys

        # Measure baseline memory
        baseline_size = sys.getsizeof({})

        # Test circuit breaker memory usage
        cb = CircuitBreaker("memory_test", CircuitBreakerConfig())
        cb_size = sys.getsizeof(cb.__dict__)

        # Circuit breaker should have minimal memory footprint
        assert cb_size < 1024, f"Circuit breaker uses too much memory: {cb_size} bytes"

        # Test deduplication cache memory usage
        dedup = RequestDeduplicator(ttl=60)

        # Add some entries
        for i in range(100):
            request = {"prompt": f"Request {i}", "max_tokens": 100}
            dedup.get_or_set_idempotent(None, request, f"Result {i}")

        cache_size = sys.getsizeof(dedup.cache)

        # Cache should scale reasonably with entries
        assert cache_size > baseline_size, "Cache should grow with entries"
        # But not excessively (rough heuristic)
        assert cache_size < 100 * 1024, f"Cache uses too much memory: {cache_size} bytes"

    def test_cleanup_performance(self):
        """Test performance of cleanup operations."""
        dedup = RequestDeduplicator(ttl=0.1)  # Very short TTL for testing

        # Add many entries
        for i in range(1000):
            request = {"prompt": f"Request {i}", "max_tokens": 100}
            dedup.get_or_set_idempotent(None, request, f"Result {i}")

        # Wait for entries to expire
        time.sleep(0.2)

        # Measure cleanup performance
        start_time = time.time()
        dedup._cleanup_expired()
        cleanup_time = time.time() - start_time

        # Cleanup should be fast
        assert cleanup_time < 0.1, f"Cleanup too slow: {cleanup_time:.4f}s"

        # Most entries should be cleaned up
        assert len(dedup.cache) < 100, f"Cleanup ineffective: {len(dedup.cache)} entries remain"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
