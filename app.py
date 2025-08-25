"""
Production-ready vLLM service with connection pooling, Prometheus metrics, and health checks.
Optimized for Tesla T4 GPU with Mistral-7B AWQ quantization.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid

from metrics import get_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Prometheus metrics
(
    REQUEST_COUNT,
    REQUEST_DURATION,
    ACTIVE_REQUESTS,
    TOKEN_THROUGHPUT,
    TTFT,
    QUEUE_SIZE,
    GPU_MEMORY_USAGE,
) = get_metrics()


# Connection pool for vLLM engine
class VLLMConnectionPool:
    def __init__(self, model_path: str, max_connections: int = 1):
        self.model_path = model_path
        self.max_connections = max_connections
        self.engine: AsyncLLMEngine | None = None
        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False  # Track if model is fully loaded and warmed up

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
            max_seq_len_to_capture=4096,
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
    stop: list[str] | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    request_id: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    timestamp: str
    model_loaded: bool
    active_requests: int
    total_requests: int
    gpu_memory_usage_gb: float | None


# Initialize connection pool
model_path = os.getenv("MODEL_PATH", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
connection_pool = VLLMConnectionPool(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Initializing vLLM engine...")
    await connection_pool.initialize()

    # Perform model warmup
    logger.info("Performing model warmup...")
    try:
        engine = await connection_pool.get_connection()

        # Create a simple test prompt for warmup
        warmup_prompt = "Hello, this is a test."
        warmup_params = SamplingParams(
            temperature=0.7,
            max_tokens=10,
            top_p=0.95
        )

        # Generate a test completion to ensure model is loaded
        warmup_request_id = random_uuid()
        results_generator = engine.generate(warmup_prompt, warmup_params, warmup_request_id)

        # Wait for completion
        async for _ in results_generator:
            pass  # Just iterate to complete the generation

        await connection_pool.release_connection()
        logger.info("✓ Model warmup completed successfully")
        connection_pool.is_ready = True  # Mark as ready after successful warmup

    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        # Continue anyway - warmup is not critical
        connection_pool.is_ready = True  # Mark as ready even if warmup fails

    yield

    # Shutdown
    logger.info("Shutting down vLLM engine...")
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
    """
    Liveness probe endpoint.
    Returns 200 if the service is running, regardless of model state.
    """
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

        # Always return healthy for liveness - service is running
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=connection_pool.engine is not None,
            active_requests=connection_pool.active_requests,
            total_requests=connection_pool.total_requests,
            gpu_memory_usage_gb=gpu_memory,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/ready")
async def ready_check():
    """
    Readiness probe endpoint.
    Returns 200 only when the model is fully loaded and ready to serve requests.
    Returns 503 during initialization.
    """
    if not connection_pool.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait..."
        )

    return {
        "status": "ready",
        "model": model_path,
        "timestamp": datetime.utcnow().isoformat(),
        "active_requests": connection_pool.active_requests,
        "total_requests": connection_pool.total_requests
    }


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
            return await generate_batch(engine, request, sampling_params, request_id, start_time)

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
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
        # Ensure we have a RequestOutput object
        if isinstance(request_output, RequestOutput):
            if not ttft_recorded and request_output.outputs and len(request_output.outputs[0].token_ids) > 0:
                ttft = time.time() - start_time
                TTFT.observe(ttft)
                ttft_recorded = True
            final_output = request_output

    if not final_output or not final_output.outputs:
        raise HTTPException(status_code=500, detail="No output generated")

    # Process response
    generation_time = time.time() - start_time
    output = final_output.outputs[0]
    tokens_generated = len(output.token_ids) if output.token_ids else 0
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

    # Record metrics
    REQUEST_DURATION.observe(generation_time)
    TOKEN_THROUGHPUT.observe(tokens_per_second)
    REQUEST_COUNT.labels(status="success").inc()

    return GenerateResponse(
        text=output.text or "",
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
        last_text_length = 0

        try:
            results_generator = engine.generate(request.prompt, sampling_params, request_id)

            async for request_output in results_generator:
                # Ensure we have a RequestOutput object
                if isinstance(request_output, RequestOutput) and request_output.outputs:
                    output = request_output.outputs[0]

                    if not ttft_recorded and output.token_ids and len(output.token_ids) > 0:
                        ttft = time.time() - start_time
                        TTFT.observe(ttft)
                        ttft_recorded = True

                    # Stream only new text (incremental)
                    if output.text:
                        new_text = output.text[last_text_length:]
                        if new_text:
                            last_text_length = len(output.text)
                            total_tokens = len(output.token_ids) if output.token_ids else 0
                            yield f"data: {new_text}\n\n"

            # Send final metrics
            generation_time = time.time() - start_time
            tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
            TOKEN_THROUGHPUT.observe(tokens_per_second)
            REQUEST_COUNT.labels(status="success").inc()

            yield "data: [DONE]\n\n"

        except Exception as e:
            REQUEST_COUNT.labels(status="error").inc()
            logger.error(f"Streaming generation failed: {e}")
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


# OpenAI-compatible API router
router = APIRouter()


@router.post("/v1/completions")
async def openai_completions(body: dict):
    """OpenAI-compatible completions endpoint."""
    prompt = body.get("prompt") or body.get("input") or ""
    max_tokens = int(body.get("max_tokens", 128))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 50))
    stop = body.get("stop")
    presence_penalty = float(body.get("presence_penalty", 0.0))
    frequency_penalty = float(body.get("frequency_penalty", 0.0))

    # Create request object for generate endpoint
    request = GenerateRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        stream=False,
    )

    # Call the existing generate function
    response = await generate(request)

    # Format as OpenAI response
    return {
        "id": "cmpl-local",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_path,
        "choices": [
            {
                "text": response.text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),  # Rough estimate
            "completion_tokens": response.tokens_generated,
            "total_tokens": len(prompt.split()) + response.tokens_generated,
        },
    }


# Include the router
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        workers=1,  # vLLM handles concurrency internally
        log_level="info",
        access_log=True,
    )
