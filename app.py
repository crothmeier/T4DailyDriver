"""
Production-ready vLLM service with connection pooling, Prometheus metrics, and health checks.
Optimized for Tesla T4 GPU with Mistral-7B AWQ quantization.
"""

import os
import time
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("vllm_request_count", "Total number of requests", ["status"])
REQUEST_DURATION = Histogram(
    "vllm_request_duration_seconds", "Request duration in seconds"
)
ACTIVE_REQUESTS = Gauge("vllm_active_requests", "Number of active requests")
TOKEN_THROUGHPUT = Histogram("vllm_token_throughput", "Tokens generated per second")
TTFT = Histogram("vllm_time_to_first_token_seconds", "Time to first token in seconds")
QUEUE_SIZE = Gauge("vllm_queue_size", "Number of requests in queue")
GPU_MEMORY_USAGE = Gauge("vllm_gpu_memory_usage_gb", "GPU memory usage in GB")


# Connection pool for vLLM engine
class VLLMConnectionPool:
    def __init__(self, model_path: str, max_connections: int = 1):
        self.model_path = model_path
        self.max_connections = max_connections
        self.engine: Optional[AsyncLLMEngine] = None
        self.active_requests = 0
        self.total_requests = 0

    async def initialize(self):
        """Initialize the vLLM engine with T4-optimized parameters."""
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tokenizer=self.model_path,
            trust_remote_code=True,
            max_num_seqs=32,  # Optimized for T4
            gpu_memory_utilization=0.9,  # T4 has 16GB memory
            swap_space=4,  # GB of CPU swap space
            block_size=16,  # Optimize for T4 architecture
            quantization="awq",  # AWQ quantization for Mistral-7B
            dtype="float16",
            enforce_eager=False,  # Enable CUDA graphs for better performance
            max_context_len_to_capture=4096,
            disable_custom_all_reduce=True,  # T4 doesn't benefit from custom all-reduce
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM engine initialized with model: {self.model_path}")

    async def get_connection(self):
        """Get a connection from the pool."""
        if self.engine is None:
            await self.initialize()
        self.active_requests += 1
        ACTIVE_REQUESTS.inc()
        return self.engine

    async def release_connection(self):
        """Release a connection back to the pool."""
        self.active_requests -= 1
        ACTIVE_REQUESTS.dec()
        self.total_requests += 1

    async def shutdown(self):
        """Shutdown the engine gracefully."""
        if self.engine:
            await self.engine.shutdown()
            self.engine = None


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    active_requests: int
    total_requests: int
    gpu_memory_usage_gb: Optional[float]


# Initialize connection pool
model_path = os.getenv("MODEL_PATH", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
connection_pool = VLLMConnectionPool(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await connection_pool.initialize()
    yield
    # Shutdown
    await connection_pool.shutdown()


# Create FastAPI app
app = FastAPI(
    title="vLLM Service",
    description="Production-ready vLLM service optimized for Tesla T4",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Get GPU memory usage if available
        gpu_memory = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = info.used / 1024**3  # Convert to GB
            GPU_MEMORY_USAGE.set(gpu_memory)
        except Exception:
            pass

        return HealthResponse(
            status="healthy" if connection_pool.engine else "initializing",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=connection_pool.engine is not None,
            active_requests=connection_pool.active_requests,
            total_requests=connection_pool.total_requests,
            gpu_memory_usage_gb=gpu_memory,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion."""
    request_id = random_uuid()
    start_time = time.time()

    try:
        REQUEST_COUNT.labels(status="started").inc()

        # Get engine from pool
        engine = await connection_pool.get_connection()

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        )

        # Generate completion
        if request.stream:
            return await generate_stream(engine, request, sampling_params, request_id)
        else:
            return await generate_batch(
                engine, request, sampling_params, request_id, start_time
            )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await connection_pool.release_connection()


async def generate_batch(engine, request, sampling_params, request_id, start_time):
    """Generate non-streaming response."""
    # Add request to engine
    results_generator = engine.generate(request.prompt, sampling_params, request_id)

    # Get the final result
    final_output = None
    ttft_recorded = False
    async for request_output in results_generator:
        if not ttft_recorded and len(request_output.outputs[0].token_ids) > 0:
            ttft = time.time() - start_time
            TTFT.observe(ttft)
            ttft_recorded = True
        final_output = request_output

    # Process response
    generation_time = time.time() - start_time
    output = final_output.outputs[0]
    tokens_generated = len(output.token_ids)
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

    # Record metrics
    REQUEST_DURATION.observe(generation_time)
    TOKEN_THROUGHPUT.observe(tokens_per_second)
    REQUEST_COUNT.labels(status="success").inc()

    return GenerateResponse(
        text=output.text,
        tokens_generated=tokens_generated,
        generation_time=generation_time,
        tokens_per_second=tokens_per_second,
        request_id=request_id,
    )


async def generate_stream(engine, request, sampling_params, request_id):
    """Generate streaming response."""

    async def stream_generator():
        start_time = time.time()
        ttft_recorded = False
        total_tokens = 0

        try:
            results_generator = engine.generate(
                request.prompt, sampling_params, request_id
            )

            async for request_output in results_generator:
                if not ttft_recorded and len(request_output.outputs[0].token_ids) > 0:
                    ttft = time.time() - start_time
                    TTFT.observe(ttft)
                    ttft_recorded = True

                output = request_output.outputs[0]
                if output.text:
                    total_tokens = len(output.token_ids)
                    yield f"data: {output.text}\n\n"

            # Send final metrics
            generation_time = time.time() - start_time
            tokens_per_second = (
                total_tokens / generation_time if generation_time > 0 else 0
            )
            TOKEN_THROUGHPUT.observe(tokens_per_second)
            REQUEST_COUNT.labels(status="success").inc()

            yield "data: [DONE]\n\n"

        except Exception as e:
            REQUEST_COUNT.labels(status="error").inc()
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"X-Request-ID": request_id},
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "vLLM Service",
        "version": "1.0.0",
        "model": model_path,
        "status": "running",
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        workers=1,  # vLLM handles concurrency internally
        log_level="info",
        access_log=True,
    )
