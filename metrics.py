"""
Enhanced metrics collection for vLLM production service.
Provides comprehensive observability with Prometheus metrics, request tracing, and GPU monitoring.
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)

_METRICS_INIT = False

# Core metrics
REQUEST_COUNT = None
REQUEST_DURATION = None
ACTIVE_REQUESTS = None
TOKEN_THROUGHPUT = None
TTFT = None
QUEUE_SIZE = None
GPU_MEMORY_USAGE = None

# Enhanced metrics
TTFT_PERCENTILES = None
TPS_PER_REQUEST = None
QUEUE_DEPTH = None
KV_CACHE_USAGE_RATIO = None
CIRCUIT_BREAKER_STATE = None
OOM_EVENTS = None
GPU_UTILIZATION = None
GPU_TEMPERATURE = None
REQUEST_SIZE_BYTES = None
RESPONSE_SIZE_BYTES = None
CONCURRENT_CONNECTIONS = None
WARM_CACHE_HITS = None
WARM_CACHE_MISSES = None
GRACEFUL_SHUTDOWN_DURATION = None
SERVICE_INFO = None


class RequestTracer:
    """Request tracing with correlation IDs."""

    def __init__(self):
        self.active_traces: dict[str, dict[str, Any]] = {}

    def start_trace(self, correlation_id: str | None = None) -> str:
        """Start a new request trace."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        self.active_traces[correlation_id] = {"start_time": time.time(), "correlation_id": correlation_id, "events": []}
        return correlation_id

    def add_event(self, correlation_id: str, event: str, metadata: dict | None = None):
        """Add an event to a trace."""
        if correlation_id in self.active_traces:
            self.active_traces[correlation_id]["events"].append(
                {"timestamp": time.time(), "event": event, "metadata": metadata or {}}
            )

    def end_trace(self, correlation_id: str) -> dict | None:
        """End a trace and return trace data."""
        if correlation_id in self.active_traces:
            trace = self.active_traces.pop(correlation_id)
            trace["end_time"] = time.time()
            trace["duration"] = trace["end_time"] - trace["start_time"]
            return trace
        return None


# Global tracer instance
tracer = RequestTracer()


@contextmanager
def trace_request(correlation_id: str | None = None):
    """Context manager for request tracing."""
    correlation_id = tracer.start_trace(correlation_id)
    try:
        yield correlation_id
    finally:
        trace_data = tracer.end_trace(correlation_id)
        if trace_data and os.getenv("ENABLE_REQUEST_TRACING", "true").lower() == "true":
            logger.info(
                "Request trace completed",
                extra={
                    "correlation_id": correlation_id,
                    "duration": trace_data["duration"],
                    "event_count": len(trace_data["events"]),
                },
            )


def init_metrics():
    """Initialize all Prometheus metrics."""
    global _METRICS_INIT
    global REQUEST_COUNT, REQUEST_DURATION, ACTIVE_REQUESTS, TOKEN_THROUGHPUT
    global TTFT, QUEUE_SIZE, GPU_MEMORY_USAGE, TTFT_PERCENTILES, TPS_PER_REQUEST
    global QUEUE_DEPTH, KV_CACHE_USAGE_RATIO, CIRCUIT_BREAKER_STATE, OOM_EVENTS
    global GPU_UTILIZATION, GPU_TEMPERATURE, REQUEST_SIZE_BYTES, RESPONSE_SIZE_BYTES
    global CONCURRENT_CONNECTIONS, WARM_CACHE_HITS, WARM_CACHE_MISSES
    global GRACEFUL_SHUTDOWN_DURATION, SERVICE_INFO

    if _METRICS_INIT:
        return

    # Core metrics
    REQUEST_COUNT = Counter("vllm_request_total", "Total requests processed", ["status", "endpoint", "model"])
    REQUEST_DURATION = Histogram("vllm_request_duration_seconds", "Request duration in seconds", ["endpoint", "status"])
    ACTIVE_REQUESTS = Gauge("vllm_active_requests", "Currently active requests")
    TOKEN_THROUGHPUT = Histogram("vllm_token_throughput", "Tokens generated per second", ["model"])
    TTFT = Histogram(
        "vllm_ttft_seconds", "Time to first token in seconds", buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    QUEUE_SIZE = Gauge("vllm_queue_size", "Number of queued requests")
    GPU_MEMORY_USAGE = Gauge("vllm_gpu_memory_usage_gb", "GPU memory usage in GB", ["device"])

    # Enhanced metrics
    TTFT_PERCENTILES = Histogram(
        "vllm_ttft_percentiles_seconds", "TTFT percentile tracking", buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    )
    TPS_PER_REQUEST = Histogram(
        "vllm_tps_per_request", "Tokens per second per individual request", ["model", "sequence_length_bucket"]
    )
    QUEUE_DEPTH = Gauge("vllm_queue_depth", "Current queue depth with backpressure tracking")
    KV_CACHE_USAGE_RATIO = Gauge("vllm_kv_cache_usage_ratio", "KV cache utilization ratio (0-1)")
    CIRCUIT_BREAKER_STATE = Gauge(
        "vllm_circuit_breaker_state", "Circuit breaker state (0=closed, 1=open, 2=half-open)", ["component"]
    )
    OOM_EVENTS = Counter("vllm_oom_events_total", "Out of memory events detected", ["recovery_action"])
    GPU_UTILIZATION = Gauge("vllm_gpu_utilization_percent", "GPU utilization percentage", ["device", "type"])
    GPU_TEMPERATURE = Gauge("vllm_gpu_temperature_celsius", "GPU temperature in Celsius", ["device"])
    REQUEST_SIZE_BYTES = Histogram("vllm_request_size_bytes", "Request payload size in bytes", ["endpoint"])
    RESPONSE_SIZE_BYTES = Histogram(
        "vllm_response_size_bytes", "Response payload size in bytes", ["endpoint", "streaming"]
    )
    CONCURRENT_CONNECTIONS = Gauge("vllm_concurrent_connections", "Number of concurrent HTTP connections")
    WARM_CACHE_HITS = Counter("vllm_warm_cache_hits_total", "Warmup cache hits")
    WARM_CACHE_MISSES = Counter("vllm_warm_cache_misses_total", "Warmup cache misses")
    GRACEFUL_SHUTDOWN_DURATION = Histogram(
        "vllm_graceful_shutdown_duration_seconds", "Time taken for graceful shutdown"
    )

    # Service information
    SERVICE_INFO = Info("vllm_service_info", "Service version and configuration information")

    _METRICS_INIT = True
    logger.info("Enhanced metrics collection initialized")


def get_metrics():
    """Get core metrics tuple for backward compatibility."""
    if not _METRICS_INIT:
        init_metrics()

    return (
        REQUEST_COUNT,
        REQUEST_DURATION,
        ACTIVE_REQUESTS,
        TOKEN_THROUGHPUT,
        TTFT,
        QUEUE_SIZE,
        GPU_MEMORY_USAGE,
    )


def get_enhanced_metrics():
    """Get all enhanced metrics."""
    if not _METRICS_INIT:
        init_metrics()

    return {
        "core": {
            "request_count": REQUEST_COUNT,
            "request_duration": REQUEST_DURATION,
            "active_requests": ACTIVE_REQUESTS,
            "token_throughput": TOKEN_THROUGHPUT,
            "ttft": TTFT,
            "queue_size": QUEUE_SIZE,
            "gpu_memory_usage": GPU_MEMORY_USAGE,
        },
        "enhanced": {
            "ttft_percentiles": TTFT_PERCENTILES,
            "tps_per_request": TPS_PER_REQUEST,
            "queue_depth": QUEUE_DEPTH,
            "kv_cache_usage_ratio": KV_CACHE_USAGE_RATIO,
            "circuit_breaker_state": CIRCUIT_BREAKER_STATE,
            "oom_events": OOM_EVENTS,
            "gpu_utilization": GPU_UTILIZATION,
            "gpu_temperature": GPU_TEMPERATURE,
            "request_size_bytes": REQUEST_SIZE_BYTES,
            "response_size_bytes": RESPONSE_SIZE_BYTES,
            "concurrent_connections": CONCURRENT_CONNECTIONS,
            "warm_cache_hits": WARM_CACHE_HITS,
            "warm_cache_misses": WARM_CACHE_MISSES,
            "graceful_shutdown_duration": GRACEFUL_SHUTDOWN_DURATION,
        },
        "info": {"service_info": SERVICE_INFO},
    }


def update_service_info(model_path: str, gpu_type: str, version: str = "1.0.0"):
    """Update service information metrics."""
    if not _METRICS_INIT:
        init_metrics()

    SERVICE_INFO.info(
        {
            "version": version,
            "model": model_path,
            "gpu_type": gpu_type,
            "python_version": os.sys.version.split()[0],
            "attention_backend": os.getenv("VLLM_ATTENTION_BACKEND", "AUTO"),
        }
    )


# Initialize metrics on import
init_metrics()
