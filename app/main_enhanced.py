"""
Production-hardened vLLM FastAPI service with comprehensive observability and security.
Optimized for Tesla T4 GPU with CUDA 12.4, featuring OpenAI API compatibility,
structured logging, request tracing, and enhanced Prometheus metrics.
"""

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from app.openai_compat import (
    ChatCompletionRequest,
    CompletionRequest,
    OpenAICompatibilityLayer,
)
from app.runtime_warmup import WarmupCacheManager, WarmupConfig
from auth import limiter, verify_api_key
from metrics import get_enhanced_metrics, trace_request, tracer, update_service_info

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Add custom log formatter for structured logging
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "pid": os.getpid(),
        }

        # Add extra fields if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user"):
            log_data["user"] = record.user
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "duration"):
            log_data["duration"] = record.duration

        return json.dumps(log_data)


# Apply structured formatter
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.handlers = [handler]
logger.setLevel(logging.INFO)

# Get enhanced metrics
metrics = get_enhanced_metrics()


class EnhancedVLLMConnectionPool:
    """Enhanced connection pool with circuit breaker and advanced monitoring."""

    def __init__(self, model_path: str, max_connections: int = 1):
        self.model_path = model_path
        self.max_connections = max_connections
        self.engine: AsyncLLMEngine | None = None
        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False
        self.shutdown_event = asyncio.Event()
        self.drain_timeout = 30  # seconds

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_failures = 0
        self.circuit_threshold = 5
        self.circuit_reset_time = 60  # seconds
        self.circuit_last_failure = 0

        # Request tracking
        self.request_queue = asyncio.Queue(maxsize=int(os.getenv("MAX_QUEUE_SIZE", "50")))
        self.queue_processor_task = None

        # Warmup manager
        self.warmup_manager = None

        # Connection tracking for graceful shutdown
        self.active_connections = set()

    def _check_circuit_breaker(self):
        """Check and update circuit breaker state."""
        if self.circuit_open:
            if time.time() - self.circuit_last_failure > self.circuit_reset_time:
                self.circuit_open = False
                self.circuit_failures = 0
                metrics["enhanced"]["circuit_breaker_state"].labels(component="vllm_engine").set(0)
                logger.info("Circuit breaker closed")

        if self.circuit_failures >= self.circuit_threshold:
            self.circuit_open = True
            self.circuit_last_failure = time.time()
            metrics["enhanced"]["circuit_breaker_state"].labels(component="vllm_engine").set(1)
            logger.error("Circuit breaker opened due to failures")

    def record_failure(self):
        """Record a failure for circuit breaker."""
        self.circuit_failures += 1
        self._check_circuit_breaker()

    def record_success(self):
        """Record a success for circuit breaker."""
        if self.circuit_failures > 0:
            self.circuit_failures = max(0, self.circuit_failures - 1)
        self._check_circuit_breaker()

    async def initialize(self):
        """Initialize vLLM engine with enhanced monitoring."""
        try:
            # Validate T4 GPU requirements
            self._validate_gpu_requirements()

            # Create engine args
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tokenizer=self.model_path,
                trust_remote_code=True,
                max_num_seqs=32,  # T4 optimized
                gpu_memory_utilization=0.9,
                swap_space=4,
                block_size=16,
                quantization="awq",
                dtype="float16",
                enforce_eager=False,
                max_seq_len_to_capture=4096,
                disable_custom_all_reduce=True,
            )

            # Initialize warmup cache manager
            cache_dir = os.getenv("VLLM_CACHE_DIR", "/tmp/vllm_cache")
            engine_args.download_dir = cache_dir
            self.warmup_manager = WarmupCacheManager(WarmupConfig.from_engine_args(engine_args))

            # Check cache
            warmup_key = self.warmup_manager.generate_warmup_key()
            is_cached = self.warmup_manager.is_warmed_up(warmup_key)

            if is_cached:
                metrics["enhanced"]["warm_cache_hits"].inc()
                logger.info(f"Using cached warmup for key: {warmup_key}")
            else:
                metrics["enhanced"]["warm_cache_misses"].inc()
                logger.info(f"No warmup cache found for key: {warmup_key}")

            # Initialize engine
            start_time = time.time()
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            init_time = time.time() - start_time

            logger.info(
                "vLLM engine initialized",
                extra={
                    "model": self.model_path,
                    "duration": init_time,
                    "cached": is_cached,
                },
            )

            # Update service info
            gpu_type = "Tesla T4" if self._is_t4() else "Unknown"
            update_service_info(self.model_path, gpu_type, "2.0.0")

            # Start queue processor
            self.queue_processor_task = asyncio.create_task(self._process_queue())

            self.is_ready = True

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise

    def _validate_gpu_requirements(self):
        """Validate GPU requirements for T4."""
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA devices available")

            device_props = torch.cuda.get_device_properties(0)
            device_name = torch.cuda.get_device_name(0)

            logger.info(
                "GPU detected",
                extra={
                    "name": device_name,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "memory_gb": device_props.total_memory / (1024**3),
                },
            )

            # Check for T4 and SDPA backend
            if "T4" in device_name or "Tesla T4" in device_name:
                attention_backend = os.getenv("VLLM_ATTENTION_BACKEND", "").upper()
                if attention_backend != "SDPA":
                    logger.warning(
                        f"T4 GPU detected but VLLM_ATTENTION_BACKEND={attention_backend}, "
                        "recommend using SDPA for optimal performance"
                    )

        except ImportError:
            logger.warning("PyTorch not available for GPU validation")
        except Exception as e:
            logger.error(f"GPU validation failed: {e}")

    def _is_t4(self) -> bool:
        """Check if GPU is Tesla T4."""
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                return "T4" in device_name or "Tesla T4" in device_name
        except:
            pass
        return False

    async def _process_queue(self):
        """Process queued requests with monitoring."""
        while not self.shutdown_event.is_set():
            try:
                request_data = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)

                if request_data is None:  # Shutdown signal
                    break

                # Update queue metrics
                metrics["core"]["queue_size"].set(self.request_queue.qsize())
                metrics["enhanced"]["queue_depth"].set(self.request_queue.qsize())

                # Process request
                future = request_data["future"]
                try:
                    result = await self._execute_request(request_data)
                    future.set_result(result)
                    self.record_success()
                except Exception as e:
                    future.set_exception(e)
                    self.record_failure()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")

    async def _execute_request(self, request_data):
        """Execute a queued request."""
        return request_data

    async def get_connection(self, correlation_id: str | None = None):
        """Get connection with circuit breaker and monitoring."""
        # Check circuit breaker
        self._check_circuit_breaker()
        if self.circuit_open:
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable (circuit breaker open)",
                headers={"Retry-After": str(self.circuit_reset_time)},
            )

        if self.engine is None:
            await self.initialize()

        # Track connection
        connection_id = correlation_id or str(uuid.uuid4())
        self.active_connections.add(connection_id)
        metrics["enhanced"]["concurrent_connections"].set(len(self.active_connections))

        # Check immediate availability
        if self.active_requests < self.max_connections:
            self.active_requests += 1
            metrics["core"]["active_requests"].inc()
            return self.engine, connection_id

        # Queue if possible
        if self.request_queue.full():
            metrics["enhanced"]["oom_events"].labels(recovery_action="queue_full").inc()
            raise HTTPException(status_code=503, detail="Service at capacity", headers={"Retry-After": "10"})

        # Add to queue
        future = asyncio.Future()
        request_data = {"future": future, "timestamp": time.time(), "correlation_id": correlation_id}

        await self.request_queue.put(request_data)
        await future

        self.active_requests += 1
        metrics["core"]["active_requests"].inc()
        return self.engine, connection_id

    async def release_connection(self, connection_id: str):
        """Release connection and update metrics."""
        self.active_requests -= 1
        metrics["core"]["active_requests"].dec()
        self.total_requests += 1

        # Remove from active connections
        self.active_connections.discard(connection_id)
        metrics["enhanced"]["concurrent_connections"].set(len(self.active_connections))

    async def drain_connections(self):
        """Drain active connections for graceful shutdown."""
        logger.info(f"Draining {len(self.active_connections)} active connections")

        start_time = time.time()
        while self.active_connections and (time.time() - start_time) < self.drain_timeout:
            await asyncio.sleep(0.5)

        if self.active_connections:
            logger.warning(f"Forcefully closing {len(self.active_connections)} connections after timeout")

        duration = time.time() - start_time
        metrics["enhanced"]["graceful_shutdown_duration"].observe(duration)

    async def shutdown(self):
        """Graceful shutdown with connection draining."""
        logger.info("Starting graceful shutdown")

        # Signal shutdown
        self.shutdown_event.set()

        # Stop accepting new requests
        self.is_ready = False

        # Drain connections
        await self.drain_connections()

        # Stop queue processor
        if self.queue_processor_task:
            await self.request_queue.put(None)
            await self.queue_processor_task

        # Cleanup warmup cache
        if self.warmup_manager:
            try:
                removed = self.warmup_manager.rotate_stale_flags()
                if removed > 0:
                    logger.info(f"Cleaned up {removed} stale warmup cache entries")
            except Exception as e:
                logger.warning(f"Failed to rotate warmup cache: {e}")

        # Shutdown engine
        if self.engine:
            await self.engine.shutdown()
            self.engine = None

        logger.info("Graceful shutdown completed")


# Health check models
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    timestamp: str
    model_loaded: bool
    active_requests: int
    total_requests: int
    gpu_memory_usage_gb: float | None
    gpu_utilization_percent: float | None
    gpu_temperature_celsius: float | None
    circuit_breaker_state: str
    warmup_cache: dict | None


# Initialize connection pool
model_path = os.getenv("MODEL_PATH", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
connection_pool = EnhancedVLLMConnectionPool(model_path)

# Initialize OpenAI compatibility layer
openai_compat = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with graceful shutdown."""
    global openai_compat

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown")
        asyncio.create_task(connection_pool.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Startup
    logger.info("Starting vLLM service")
    await connection_pool.initialize()

    # Initialize OpenAI compatibility
    openai_compat = OpenAICompatibilityLayer(model_path, connection_pool)

    # Perform warmup
    logger.info("Performing model warmup")
    try:
        engine, conn_id = await connection_pool.get_connection("warmup")

        warmup_params = SamplingParams(temperature=0.7, max_tokens=10, top_p=0.95)

        results_generator = engine.generate("Hello, this is a warmup test.", warmup_params, "warmup-request")

        async for _ in results_generator:
            pass

        await connection_pool.release_connection(conn_id)
        logger.info("Model warmup completed successfully")

    except Exception as e:
        logger.error(f"Model warmup failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down vLLM service")
    await connection_pool.shutdown()


# Create FastAPI app
app = FastAPI(
    title="vLLM Production Service",
    description="Production-hardened vLLM service optimized for Tesla T4 with CUDA 12.4",
    version="2.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request tracing
@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Add request tracing with correlation IDs."""
    # Get or create correlation ID
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    # Start trace
    with trace_request(correlation_id) as trace_id:
        # Add request metadata
        tracer.add_event(
            trace_id,
            "request_start",
            {
                "method": request.method,
                "path": str(request.url.path),
                "client": request.client.host if request.client else None,
            },
        )

        # Track request size
        if request.headers.get("content-length"):
            try:
                size = int(request.headers["content-length"])
                metrics["enhanced"]["request_size_bytes"].labels(endpoint=str(request.url.path)).observe(size)
            except ValueError:
                pass

        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Add trace event
        tracer.add_event(
            trace_id,
            "request_complete",
            {
                "status_code": response.status_code,
                "duration": duration,
            },
        )

        # Track response size
        if response.headers.get("content-length"):
            try:
                size = int(response.headers["content-length"])
                metrics["enhanced"]["response_size_bytes"].labels(
                    endpoint=str(request.url.path), streaming="false"
                ).observe(size)
            except ValueError:
                pass

        # Add correlation ID to response
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-Duration"] = str(duration)

        # Log request
        logger.info(
            "Request processed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "duration": duration,
            },
        )

        return response


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Liveness probe endpoint with enhanced monitoring."""
    try:
        # Get GPU metrics
        gpu_memory = None
        gpu_utilization = None
        gpu_temperature = None

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = mem_info.used / (1024**3)
            metrics["core"]["gpu_memory_usage"].labels(device="0").set(gpu_memory)

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util.gpu
            metrics["enhanced"]["gpu_utilization"].labels(device="0", type="compute").set(gpu_utilization)

            # Temperature
            gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics["enhanced"]["gpu_temperature"].labels(device="0").set(gpu_temperature)

        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")

        # Circuit breaker state
        cb_state = "open" if connection_pool.circuit_open else "closed"

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=connection_pool.engine is not None,
            active_requests=connection_pool.active_requests,
            total_requests=connection_pool.total_requests,
            gpu_memory_usage_gb=gpu_memory,
            gpu_utilization_percent=gpu_utilization,
            gpu_temperature_celsius=gpu_temperature,
            circuit_breaker_state=cb_state,
            warmup_cache=(
                connection_pool.warmup_manager.get_warmup_cache_info() if connection_pool.warmup_manager else None
            ),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    # Update system metrics
    try:
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()

        # You could add these to custom metrics if needed
        logger.debug(f"Process CPU: {cpu_percent}%, Memory: {memory_info.rss / (1024**3):.2f}GB")

    except Exception as e:
        logger.warning(f"Failed to get process metrics: {e}")

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# OpenAI-compatible endpoints
@app.post("/v1/chat/completions")
@limiter.limit("100/minute")
async def chat_completions(request: ChatCompletionRequest, req: Request, api_key: str = Depends(verify_api_key)):
    """OpenAI-compatible chat completions endpoint."""
    correlation_id = req.state.correlation_id if hasattr(req.state, "correlation_id") else str(uuid.uuid4())

    # Track request
    metrics["core"]["request_count"].labels(
        status="started", endpoint="/v1/chat/completions", model=request.model
    ).inc()

    try:
        # Log request
        logger.info(
            "Chat completion request",
            extra={
                "correlation_id": correlation_id,
                "model": request.model,
                "messages": len(request.messages),
                "stream": request.stream,
                "user": api_key,
            },
        )

        # Process request
        response = await openai_compat.chat_completion(request, correlation_id)

        # Track success
        metrics["core"]["request_count"].labels(
            status="success", endpoint="/v1/chat/completions", model=request.model
        ).inc()

        return response

    except Exception as e:
        # Track error
        metrics["core"]["request_count"].labels(
            status="error", endpoint="/v1/chat/completions", model=request.model
        ).inc()

        logger.error(f"Chat completion failed: {e}", extra={"correlation_id": correlation_id})

        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
@limiter.limit("100/minute")
async def completions(request: CompletionRequest, req: Request, api_key: str = Depends(verify_api_key)):
    """OpenAI-compatible text completions endpoint."""
    correlation_id = req.state.correlation_id if hasattr(req.state, "correlation_id") else str(uuid.uuid4())

    # Track request
    metrics["core"]["request_count"].labels(status="started", endpoint="/v1/completions", model=request.model).inc()

    try:
        # Process request
        response = await openai_compat.text_completion(request, correlation_id)

        # Track success
        metrics["core"]["request_count"].labels(status="success", endpoint="/v1/completions", model=request.model).inc()

        return response

    except Exception as e:
        # Track error
        metrics["core"]["request_count"].labels(status="error", endpoint="/v1/completions", model=request.model).inc()

        logger.error(f"Text completion failed: {e}", extra={"correlation_id": correlation_id})

        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_path,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization",
                "permission": [],
                "root": model_path,
                "parent": None,
            }
        ],
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "vLLM Production Service",
        "version": "2.0.0",
        "model": model_path,
        "status": "running",
        "gpu": "Tesla T4" if connection_pool._is_t4() else "Unknown",
        "endpoints": {
            "health": "/healthz",
            "metrics": "/metrics",
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
        },
    }


if __name__ == "__main__":
    # Configure uvicorn for production
    workers = int(os.getenv("UVICORN_WORKERS", "1"))

    uvicorn.run(
        "app.main_enhanced:app",
        host="0.0.0.0",
        port=8080,
        workers=workers,  # vLLM handles concurrency internally
        loop="uvloop",  # High-performance event loop
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        },
        access_log=True,
        use_colors=False,  # Disable colors for structured logging
    )
