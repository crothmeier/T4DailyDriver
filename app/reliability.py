"""
Service reliability components for production vLLM deployment.
Includes circuit breakers, request deduplication, graceful shutdown, and OOM prevention.
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = 0    # Normal operation
    OPEN = 1      # Failing, blocking requests
    HALF_OPEN = 2 # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: int = 10


class CircuitBreaker:
    """Circuit breaker for GPU OOM recovery and service protection."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.failure_exceptions: deque = deque(maxlen=10)

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN."""
        return (
            self.state == CircuitBreakerState.OPEN and
            time.time() - self.last_failure_time > self.config.recovery_timeout
        )

    def _record_success(self):
        """Record a successful operation."""
        self.failure_count = max(0, self.failure_count - 1)

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0

    def _record_failure(self, exception: Exception):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.failure_exceptions.append({
            'timestamp': time.time(),
            'exception': str(exception),
            'type': type(exception).__name__
        })

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit breaker {self.name} OPENED after {self.failure_count} failures"
                )
                self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            logger.info(f"Circuit breaker {self.name} reopened after failed test")
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0

    @asynccontextmanager
    async def protect(self):
        """Context manager to protect operations with circuit breaker."""
        # Check if we should attempt reset
        if self._should_attempt_reset():
            logger.info(f"Circuit breaker {self.name} attempting recovery (HALF_OPEN)")
            self.state = CircuitBreakerState.HALF_OPEN

        # Block if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            raise HTTPException(
                status_code=503,
                detail=f"Service temporarily unavailable ({self.name} circuit breaker open)",
                headers={"Retry-After": str(max(1, self.config.recovery_timeout - int(time.time() - self.last_failure_time)))}
            )

        start_time = time.time()
        try:
            yield
            self._record_success()

        except Exception as e:
            # Check for OOM or GPU-related errors
            error_str = str(e).lower()
            if any(pattern in error_str for pattern in ['out of memory', 'cuda', 'gpu']):
                logger.error(f"GPU error detected in {self.name}: {e}")
                self._record_failure(e)

                # Import metrics here to avoid circular imports
                try:
                    from metrics import get_enhanced_metrics
                    metrics = get_enhanced_metrics()
                    metrics['enhanced']['oom_events'].labels(recovery_action="circuit_breaker_opened").inc()
                    metrics['enhanced']['circuit_breaker_state'].labels(component=self.name).set(self.state.value)
                except ImportError:
                    pass

            raise

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state for monitoring."""
        return {
            'name': self.name,
            'state': self.state.name,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'time_until_retry': max(0, self.config.recovery_timeout - int(time.time() - self.last_failure_time)) if self.state == CircuitBreakerState.OPEN else 0,
            'recent_failures': list(self.failure_exceptions)
        }


class RequestDeduplicator:
    """Request deduplication using content hashing and idempotency keys."""

    def __init__(self, ttl: int = 300):  # 5 minute TTL
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}

    def _generate_content_hash(self, content: Any) -> str:
        """Generate hash for request content."""
        content_str = str(content).encode('utf-8')
        return hashlib.sha256(content_str).hexdigest()[:16]

    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        now = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if now - access_time > self.ttl
        ]

        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def get_or_set_idempotent(
        self,
        idempotency_key: Optional[str],
        content: Any,
        result: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Get cached result for idempotent request or set new result.

        Args:
            idempotency_key: Client-provided idempotency key
            content: Request content for content-based deduplication
            result: Result to cache (if providing new result)

        Returns:
            Cached result if found, None if new request
        """
        self._cleanup_expired()

        # Use provided key or generate from content
        key = idempotency_key if idempotency_key else self._generate_content_hash(content)

        now = time.time()

        # Check if we have a cached result
        if key in self.cache:
            self.access_times[key] = now
            cached_data = self.cache[key]

            # Verify content matches for content-based deduplication
            if not idempotency_key:
                cached_hash = self._generate_content_hash(cached_data.get('content'))
                request_hash = self._generate_content_hash(content)
                if cached_hash != request_hash:
                    # Content changed, treat as new request
                    del self.cache[key]
                    del self.access_times[key]
                    return None

            logger.info(f"Returning deduplicated response for key: {key[:8]}...")
            return cached_data.get('result')

        # New request - cache result if provided
        if result is not None:
            self.cache[key] = {
                'content': content,
                'result': result,
                'timestamp': now
            }
            self.access_times[key] = now
            logger.debug(f"Cached result for key: {key[:8]}...")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        self._cleanup_expired()
        return {
            'cached_entries': len(self.cache),
            'ttl_seconds': self.ttl,
            'oldest_entry': min(self.access_times.values()) if self.access_times else None
        }


class GracefulShutdownManager:
    """Manages graceful shutdown with request draining."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.shutdown_event = asyncio.Event()
        self.active_requests: Set[str] = set()
        self.shutdown_started = False

    def register_request(self, request_id: str):
        """Register an active request."""
        if not self.shutdown_started:
            self.active_requests.add(request_id)

    def unregister_request(self, request_id: str):
        """Unregister a completed request."""
        self.active_requests.discard(request_id)

    def is_shutdown_requested(self) -> bool:
        \"\"\"Check if shutdown has been requested.\"\"\"
        return self.shutdown_event.is_set()

    async def initiate_shutdown(self):
        \"\"\"Initiate graceful shutdown.\"\"\"
        logger.info(f\"Initiating graceful shutdown with {len(self.active_requests)} active requests\")\n        self.shutdown_started = True\n        self.shutdown_event.set()\n        \n        start_time = time.time()\n        \n        # Wait for active requests to complete\n        while self.active_requests and (time.time() - start_time) < self.timeout:\n            logger.info(f\"Waiting for {len(self.active_requests)} requests to complete...\")\n            await asyncio.sleep(1)\n            \n        remaining_requests = len(self.active_requests)\n        shutdown_duration = time.time() - start_time\n        \n        if remaining_requests > 0:\n            logger.warning(f\"Shutdown timeout reached with {remaining_requests} requests still active\")\n        else:\n            logger.info(f\"Graceful shutdown completed in {shutdown_duration:.2f}s\")\n            \n        # Update metrics\n        try:\n            from metrics import get_enhanced_metrics\n            metrics = get_enhanced_metrics()\n            metrics['enhanced']['graceful_shutdown_duration'].observe(shutdown_duration)\n        except ImportError:\n            pass\n            \n        return shutdown_duration, remaining_requests\n\n\nclass OOMPreventionMonitor:\n    \"\"\"Monitors GPU memory and prevents OOM by implementing backpressure.\"\"\"\n    \n    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.95):\n        self.warning_threshold = warning_threshold\n        self.critical_threshold = critical_threshold\n        self.oom_events = 0\n        self.last_check = 0\n        self.check_interval = 5  # seconds\n        \n    def _get_gpu_memory_usage(self) -> Optional[float]:\n        \"\"\"Get current GPU memory usage ratio.\"\"\"\n        try:\n            import pynvml\n            pynvml.nvmlInit()\n            handle = pynvml.nvmlDeviceGetHandleByIndex(0)\n            info = pynvml.nvmlDeviceGetMemoryInfo(handle)\n            return info.used / info.total\n        except Exception as e:\n            logger.warning(f\"Could not get GPU memory usage: {e}\")\n            return None\n            \n    def check_memory_pressure(self) -> Dict[str, Any]:\n        \"\"\"Check current memory pressure and return status.\"\"\"\n        now = time.time()\n        \n        # Rate limit checks\n        if now - self.last_check < self.check_interval:\n            return {'status': 'ok', 'cached': True}\n            \n        self.last_check = now\n        usage_ratio = self._get_gpu_memory_usage()\n        \n        if usage_ratio is None:\n            return {'status': 'unknown', 'usage_ratio': None}\n            \n        # Update metrics\n        try:\n            from metrics import get_enhanced_metrics\n            metrics = get_enhanced_metrics()\n            metrics['enhanced']['kv_cache_usage_ratio'].set(usage_ratio)\n        except ImportError:\n            pass\n            \n        if usage_ratio >= self.critical_threshold:\n            self.oom_events += 1\n            logger.error(f\"Critical GPU memory usage: {usage_ratio:.2%}\")\n            return {\n                'status': 'critical',\n                'usage_ratio': usage_ratio,\n                'should_reject': True,\n                'message': f\"GPU memory critically high ({usage_ratio:.1%})\"}\n        elif usage_ratio >= self.warning_threshold:\n            logger.warning(f\"High GPU memory usage: {usage_ratio:.2%}\")\n            return {\n                'status': 'warning',\n                'usage_ratio': usage_ratio,\n                'should_reject': False,\n                'message': f\"GPU memory usage high ({usage_ratio:.1%})\"\n            }\n        else:\n            return {'status': 'ok', 'usage_ratio': usage_ratio}\n            \n    def should_accept_request(self) -> bool:\n        \"\"\"Determine if new requests should be accepted based on memory pressure.\"\"\"        memory_status = self.check_memory_pressure()\n        return not memory_status.get('should_reject', False)\n        \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get OOM prevention statistics.\"\"\"        return {\n            'warning_threshold': self.warning_threshold,\n            'critical_threshold': self.critical_threshold,\n            'oom_events': self.oom_events,\n            'last_check': self.last_check,\n            'current_status': self.check_memory_pressure()\n        }\n\n\n# Global instances\ncircuit_breakers: Dict[str, CircuitBreaker] = {}\nrequest_deduplicator = RequestDeduplicator()\nshutdown_manager = GracefulShutdownManager()\noom_monitor = OOMPreventionMonitor()\n\n\ndef get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:\n    \"\"\"Get or create a circuit breaker.\"\"\"    if name not in circuit_breakers:\n        circuit_breakers[name] = CircuitBreaker(name, config or CircuitBreakerConfig())\n    return circuit_breakers[name]\n\n\ndef get_reliability_status() -> Dict[str, Any]:\n    \"\"\"Get overall reliability system status.\"\"\"    return {\n        'circuit_breakers': {name: cb.get_state() for name, cb in circuit_breakers.items()},\n        'request_deduplication': request_deduplicator.get_stats(),\n        'graceful_shutdown': {\n            'shutdown_requested': shutdown_manager.is_shutdown_requested(),\n            'active_requests': len(shutdown_manager.active_requests),\n            'timeout': shutdown_manager.timeout\n        },\n        'oom_prevention': oom_monitor.get_stats()\n    }"
