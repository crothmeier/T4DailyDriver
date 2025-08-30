"""
Enhanced production-ready vLLM service with comprehensive features.
Integrates GPU optimization, OpenAI compatibility, service reliability, and advanced observability.
"""

import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import random_uuid

# Import our enhanced modules
from app.config import VLLMConfig, load_config
from app.openai_compat import ChatCompletionRequest, CompletionRequest, OpenAICompatibilityLayer
from app.reliability import (
    CircuitBreakerConfig,
    get_circuit_breaker,
    get_reliability_status,
    oom_monitor,
    request_deduplicator,
    shutdown_manager,
)
from auth import limiter, verify_api_key
from metrics import get_enhanced_metrics, trace_request, update_service_info

# Configure structured logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s:%(lineno)d"
)
logger = logging.getLogger(__name__)

# Global configuration and components
config: VLLMConfig = None
openai_layer: OpenAICompatibilityLayer = None
enhanced_metrics = None
vllm_engine: AsyncLLMEngine = None


class EnhancedVLLMConnectionPool:
    """Enhanced connection pool with all production features."""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine: AsyncLLMEngine | None = None
        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False

        # Enhanced readiness tracking
        self.model_loaded = False
        self.cuda_graphs_ready = False
        self.warmup_complete = False

        # Circuit breakers
        self.gpu_circuit_breaker = get_circuit_breaker(
            "gpu_operations",
            CircuitBreakerConfig(
                failure_threshold=config.service.circuit_breaker_failure_threshold,
                recovery_timeout=config.service.circuit_breaker_recovery_timeout,
            ),
        )

        # Request queue with backpressure
        self.request_queue = asyncio.Queue(maxsize=config.service.max_queue_size)
        self.queue_processor_task = None
        self.queued_requests = 0
        self.rejected_requests = 0

    async def initialize(self):
        """Initialize the enhanced vLLM engine."""
        logger.info("Initializing enhanced vLLM engine...")

        # Create engine args with GPU-optimized settings
        gpu_config = self.config.gpu

        engine_args = AsyncEngineArgs(
            model=self.config.model_path,
            tokenizer=self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            tensor_parallel_size=gpu_config.tensor_parallel_size,
            max_num_seqs=gpu_config.max_num_seqs,
            gpu_memory_utilization=gpu_config.gpu_memory_utilization,
            swap_space=self.config.swap_space,
            block_size=gpu_config.block_size,
            quantization=self.config.quantization,
            dtype=self.config.dtype,
            enforce_eager=not gpu_config.enable_cuda_graph,
            max_seq_len_to_capture=4096 if gpu_config.enable_cuda_graph else 0,
            disable_custom_all_reduce=True,  # Better for single GPU setups
            enable_prefix_caching=gpu_config.enable_prefix_caching,
            download_dir=self.config.download_dir,
            max_model_len=self.config.max_model_len,
        )

        # Set attention backend
        os.environ["VLLM_ATTENTION_BACKEND"] = gpu_config.attention_backend
        logger.info(f"Using attention backend: {gpu_config.attention_backend}")

        # Initialize engine with circuit breaker protection
        async with self.gpu_circuit_breaker.protect():
            start_time = time.time()
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            init_time = time.time() - start_time

            self.model_loaded = True
            logger.info(f"vLLM engine initialized in {init_time:.2f}s")

            # Update metrics
            if enhanced_metrics:
                enhanced_metrics["info"]["service_info"].info(
                    {
                        "model": self.config.model_path,
                        "gpu_type": gpu_config.gpu_type.value,
                        "initialization_time": f"{init_time:.2f}s",
                    }
                )

        # Start queue processor
        self.queue_processor_task = asyncio.create_task(self._process_queue())
        logger.info("Request queue processor started")

        # Mark components as ready
        self.cuda_graphs_ready = gpu_config.enable_cuda_graph
        self.is_ready = True
        logger.info("Enhanced vLLM engine ready for requests")

    async def _process_queue(self):
        """Process queued requests with reliability features."""
        while True:
            try:
                # Check if shutdown requested
                if shutdown_manager.is_shutdown_requested():
                    logger.info("Queue processor shutting down...")
                    break

                # Wait for a request in the queue
                request_data = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)

                if request_data is None:  # Shutdown signal
                    break

                # Check memory pressure before processing
                if not oom_monitor.should_accept_request():
                    # Reject request due to memory pressure
                    future = request_data["future"]
                    future.set_exception(
                        HTTPException(
                            status_code=503,
                            detail="GPU memory pressure detected - request rejected",
                            headers={"Retry-After": "10"},
                        )
                    )
                    continue

                # Process the request
                future = request_data["future"]
                correlation_id = request_data.get("correlation_id")

                try:
                    # Register request for graceful shutdown tracking
                    if correlation_id:
                        shutdown_manager.register_request(correlation_id)

                    result = await self._execute_request(request_data)
                    future.set_result(result)

                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queued_requests -= 1
                    if enhanced_metrics:
                        enhanced_metrics["core"]["queue_size"].set(self.queued_requests)
                        enhanced_metrics["enhanced"]["queue_depth"].set(self.queued_requests)

                    if correlation_id:
                        shutdown_manager.unregister_request(correlation_id)

            except asyncio.TimeoutError:
                continue  # Normal timeout, check shutdown status
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)  # Prevent tight error loops

    async def _execute_request(self, request_data):
        """Execute a queued request with enhanced reliability."""
        # This is a placeholder - actual execution happens in the endpoint handlers
        return request_data

    async def get_connection(self, correlation_id: str | None = None):
        """Get a connection with enhanced reliability features."""
        if self.engine is None:
            await self.initialize()

        # Check for immediate availability
        if self.active_requests < self.config.service.max_concurrent_requests:
            self.active_requests += 1
            if enhanced_metrics:
                enhanced_metrics["core"]["active_requests"].inc()
            return self.engine

        # Check queue capacity (backpressure)
        if self.request_queue.full():
            self.rejected_requests += 1
            if enhanced_metrics:
                enhanced_metrics["enhanced"]["circuit_breaker_state"].labels(component="queue").set(1)  # Open state
            raise HTTPException(
                status_code=503, detail="Service at capacity. Please retry later.", headers={"Retry-After": "10"}
            )

        # Add to queue
        self.queued_requests += 1
        if enhanced_metrics:
            enhanced_metrics["core"]["queue_size"].set(self.queued_requests)
            enhanced_metrics["enhanced"]["queue_depth"].set(self.queued_requests)

        # Create future for this request
        future = asyncio.Future()
        request_data = {"future": future, "timestamp": time.time(), "correlation_id": correlation_id}

        try:
            await self.request_queue.put(request_data)
            await future  # Wait for processing

            self.active_requests += 1
            if enhanced_metrics:
                enhanced_metrics["core"]["active_requests"].inc()
            return self.engine

        except Exception:
            self.queued_requests -= 1
            if enhanced_metrics:
                enhanced_metrics["core"]["queue_size"].set(self.queued_requests)
            raise

    async def release_connection(self):
        """Release connection and update metrics."""
        self.active_requests = max(0, self.active_requests - 1)
        self.total_requests += 1

        if enhanced_metrics:
            enhanced_metrics["core"]["active_requests"].dec()

    async def shutdown(self):
        """Enhanced graceful shutdown."""
        logger.info("Starting enhanced shutdown process...")

        # Stop accepting new requests
        if self.queue_processor_task:
            await self.request_queue.put(None)  # Shutdown signal
            await self.queue_processor_task

        # Initiate graceful shutdown of active requests
        duration, remaining = await shutdown_manager.initiate_shutdown()

        if remaining > 0:
            logger.warning(f"Force-closing {remaining} requests after timeout")

        # Shutdown vLLM engine
        if self.engine:
            await self.engine.shutdown()
            self.engine = None

        logger.info(f"Enhanced shutdown completed in {duration:.2f}s")


# Response models for enhanced endpoints
class HealthResponse(BaseModel):
    """Enhanced health response with comprehensive status."""

    status: str
    timestamp: str
    model_loaded: bool
    active_requests: int
    total_requests: int
    gpu_info: dict
    reliability_status: dict
    performance_metrics: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifecycle management."""
    global config, openai_layer, enhanced_metrics, vllm_engine

    # Load configuration
    logger.info("Loading enhanced configuration...")
    config = load_config()

    # Initialize metrics with service info
    enhanced_metrics = get_enhanced_metrics()
    update_service_info(model_path=config.model_path, gpu_type=config.gpu.gpu_type.value, version="2.0.0")

    # Initialize connection pool
    logger.info("Initializing enhanced vLLM connection pool...")
    vllm_engine = EnhancedVLLMConnectionPool(config)
    await vllm_engine.initialize()

    # Initialize OpenAI compatibility layer
    logger.info("Initializing OpenAI compatibility layer...")
    openai_layer = OpenAICompatibilityLayer(config.model_path, vllm_engine)

    # Perform warmup if needed
    logger.info("Performing enhanced model warmup...")
    try:
        async with trace_request() as correlation_id:
            engine = await vllm_engine.get_connection(correlation_id)

            # Test generation
            warmup_params = SamplingParams(temperature=0.7, max_tokens=10)
            warmup_request_id = random_uuid()
            results_generator = engine.generate("Hello, this is a warmup test.", warmup_params, warmup_request_id)

            async for _ in results_generator:
                pass

            await vllm_engine.release_connection()
            vllm_engine.warmup_complete = True
            logger.info("✓ Enhanced warmup completed successfully")

    except Exception as e:
        logger.error(f"Enhanced warmup failed: {e}")
        vllm_engine.warmup_complete = False

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(vllm_engine.shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Enhanced vLLM service startup completed")
    yield

    # Shutdown
    logger.info("Enhanced vLLM service shutting down...")
    await vllm_engine.shutdown()


# Create enhanced FastAPI app
app = FastAPI(
    title="Enhanced vLLM Service",
    description="Production-ready vLLM service with comprehensive features",
    version="2.0.0",
    lifespan=lifespan,
)

# Add enhanced middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enhanced endpoints
@app.get("/health", response_model=HealthResponse)
async def enhanced_health_check():
    """Comprehensive health check with full system status."""
    try:
        # Collect GPU information
        gpu_info = {"status": "unknown"}
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

            gpu_info = {
                "name": gpu_name,
                "memory_used_gb": memory_info.used / (1024**3),
                "memory_total_gb": memory_info.total / (1024**3),
                "memory_utilization": memory_info.used / memory_info.total,
            }

            # Update metrics
            if enhanced_metrics:
                enhanced_metrics["core"]["gpu_memory_usage"].labels(device="0").set(memory_info.used / (1024**3))
        except Exception as e:
            gpu_info = {"error": str(e)}

        # Get reliability status
        reliability_status = get_reliability_status()

        # Collect performance metrics
        performance_metrics = {
            "active_requests": vllm_engine.active_requests if vllm_engine else 0,
            "total_requests": vllm_engine.total_requests if vllm_engine else 0,
            "queued_requests": vllm_engine.queued_requests if vllm_engine else 0,
            "rejected_requests": vllm_engine.rejected_requests if vllm_engine else 0,
        }

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=vllm_engine.model_loaded if vllm_engine else False,
            active_requests=performance_metrics["active_requests"],
            total_requests=performance_metrics["total_requests"],
            gpu_info=gpu_info,
            reliability_status=reliability_status,
            performance_metrics=performance_metrics,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/readyz")
async def enhanced_readiness_check():
    """Enhanced readiness check with graduated readiness levels."""
    if not vllm_engine or not vllm_engine.is_ready:
        raise HTTPException(status_code=503, detail="Service not ready - model still loading")

    readiness_status = {
        "model_loaded": vllm_engine.model_loaded,
        "cuda_graphs_ready": vllm_engine.cuda_graphs_ready,
        "warmup_complete": vllm_engine.warmup_complete,
        "overall_ready": vllm_engine.is_ready,
    }

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "readiness": readiness_status,
            "gpu_config": {
                "type": config.gpu.gpu_type.value,
                "max_num_seqs": config.gpu.max_num_seqs,
                "attention_backend": config.gpu.attention_backend,
            },
        },
    )


@app.get("/metrics")
async def enhanced_metrics_endpoint():
    """Enhanced Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# OpenAI-compatible endpoints with enhanced features
@app.post("/v1/chat/completions")
@limiter.limit("100/minute")
async def enhanced_chat_completions(request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
    """Enhanced OpenAI-compatible chat completions endpoint."""
    async with trace_request() as correlation_id:
        start_time = time.time()

        try:
            # Check for request deduplication
            idempotency_key = None  # Could be extracted from headers
            cached_result = request_deduplicator.get_or_set_idempotent(idempotency_key, request.model_dump(), None)

            if cached_result:
                logger.info(f"Returning deduplicated chat completion: {correlation_id}")
                return cached_result

            # Update metrics
            if enhanced_metrics:
                enhanced_metrics["core"]["request_count"].labels(
                    status="started", endpoint="chat_completions", model=request.model
                ).inc()

            # Process request
            response = await openai_layer.chat_completion(request, correlation_id)

            # Cache result for deduplication
            request_deduplicator.get_or_set_idempotent(idempotency_key, request.model_dump(), response)

            # Update success metrics
            if enhanced_metrics:
                duration = time.time() - start_time
                enhanced_metrics["core"]["request_duration"].labels(
                    endpoint="chat_completions", status="success"
                ).observe(duration)
                enhanced_metrics["core"]["request_count"].labels(
                    status="success", endpoint="chat_completions", model=request.model
                ).inc()

            return response

        except Exception as e:
            # Update error metrics
            if enhanced_metrics:
                duration = time.time() - start_time
                enhanced_metrics["core"]["request_duration"].labels(
                    endpoint="chat_completions", status="error"
                ).observe(duration)
                enhanced_metrics["core"]["request_count"].labels(
                    status="error", endpoint="chat_completions", model=request.model
                ).inc()

            logger.error(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
@limiter.limit("100/minute")
async def enhanced_text_completions(request: CompletionRequest, api_key: str = Depends(verify_api_key)):
    """Enhanced OpenAI-compatible text completions endpoint."""
    async with trace_request() as correlation_id:
        start_time = time.time()

        try:
            # Update metrics
            if enhanced_metrics:
                enhanced_metrics["core"]["request_count"].labels(
                    status="started", endpoint="completions", model=request.model
                ).inc()

            # Process request
            response = await openai_layer.text_completion(request, correlation_id)

            # Update success metrics
            if enhanced_metrics:
                duration = time.time() - start_time
                enhanced_metrics["core"]["request_duration"].labels(endpoint="completions", status="success").observe(
                    duration
                )
                enhanced_metrics["core"]["request_count"].labels(
                    status="success", endpoint="completions", model=request.model
                ).inc()

            return response

        except Exception as e:
            # Update error metrics
            if enhanced_metrics:
                duration = time.time() - start_time
                enhanced_metrics["core"]["request_duration"].labels(endpoint="completions", status="error").observe(
                    duration
                )
                enhanced_metrics["core"]["request_count"].labels(
                    status="error", endpoint="completions", model=request.model
                ).inc()

            logger.error(f"Text completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": config.model_path if config else "default",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm-service",
                "permission": [],
                "root": config.model_path if config else "default",
            }
        ],
    }


# Administrative endpoints
@app.get("/admin/config")
async def get_config(api_key: str = Depends(verify_api_key)):
    """Get current service configuration (admin only)."""
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")

    return {
        "model_path": config.model_path,
        "gpu_config": {
            "type": config.gpu.gpu_type.value,
            "max_num_seqs": config.gpu.max_num_seqs,
            "memory_utilization": config.gpu.gpu_memory_utilization,
            "attention_backend": config.gpu.attention_backend,
        },
        "service_config": {
            "max_queue_size": config.service.max_queue_size,
            "max_concurrent_requests": config.service.max_concurrent_requests,
            "circuit_breaker_threshold": config.service.circuit_breaker_failure_threshold,
        },
    }


@app.get("/admin/status")
async def get_admin_status(api_key: str = Depends(verify_api_key)):
    """Get detailed administrative status."""
    return {
        "service": "Enhanced vLLM Service",
        "version": "2.0.0",
        "uptime": time.time() - (vllm_engine.total_requests if vllm_engine else 0),
        "configuration": config.model_path if config else None,
        "reliability": get_reliability_status(),
        "performance": {
            "active_requests": vllm_engine.active_requests if vllm_engine else 0,
            "total_requests": vllm_engine.total_requests if vllm_engine else 0,
            "queued_requests": vllm_engine.queued_requests if vllm_engine else 0,
        },
    }


@app.get("/")
async def root():
    """Enhanced root endpoint."""
    return {
        "service": "Enhanced vLLM Service",
        "version": "2.0.0",
        "model": config.model_path if config else "Not loaded",
        "gpu_type": config.gpu.gpu_type.value if config else "Unknown",
        "features": [
            "OpenAI API Compatibility",
            "Circuit Breakers",
            "Request Deduplication",
            "Enhanced Metrics",
            "Graceful Shutdown",
            "OOM Prevention",
        ],
        "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/models", "/health", "/readyz", "/metrics"],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app_enhanced:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_level="info",
        access_log=True,
    )
