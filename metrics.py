from prometheus_client import Counter, Gauge, Histogram

_METRICS_INIT = False
REQUEST_COUNT = REQUEST_DURATION = ACTIVE_REQUESTS = TOKEN_THROUGHPUT = TTFT = QUEUE_SIZE = GPU_MEMORY_USAGE = None


def get_metrics():
    global _METRICS_INIT
    global REQUEST_COUNT, REQUEST_DURATION, ACTIVE_REQUESTS, TOKEN_THROUGHPUT, TTFT, QUEUE_SIZE, GPU_MEMORY_USAGE
    if not _METRICS_INIT:
        REQUEST_COUNT = Counter("vllm_request_count", "Total requests", ["status"])
        REQUEST_DURATION = Histogram("vllm_request_duration_seconds", "Request duration seconds")
        ACTIVE_REQUESTS = Gauge("vllm_active_requests", "Active requests")
        TOKEN_THROUGHPUT = Histogram("vllm_token_throughput", "Tokens per second")
        TTFT = Histogram("vllm_time_to_first_token_seconds", "TTFT seconds")
        QUEUE_SIZE = Gauge("vllm_queue_size", "Queued requests")
        GPU_MEMORY_USAGE = Gauge("vllm_gpu_memory_usage_gb", "GPU memory GB")
        _METRICS_INIT = True
    return (
        REQUEST_COUNT,
        REQUEST_DURATION,
        ACTIVE_REQUESTS,
        TOKEN_THROUGHPUT,
        TTFT,
        QUEUE_SIZE,
        GPU_MEMORY_USAGE,
    )
