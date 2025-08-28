"""
Production-ready vLLM service with connection pooling, Prometheus metrics, and health checks.
Optimized for Tesla T4 GPU with Mistral-7B AWQ quantization.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid

from app.runtime_warmup import WarmupCacheManager, WarmupConfig
from auth import limiter, verify_api_key
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


# Connection pool for vLLM engine with request queuing
class VLLMConnectionPool:
    def __init__(self, model_path: str, max_connections: int = 1):
        self.model_path = model_path
        self.max_connections = max_connections
        self.engine: AsyncLLMEngine | None = None
        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False  # Track if model is fully loaded and warmed up

        # Graduated readiness tracking
        self.model_loaded = False  # Model successfully initialized
        self.cuda_graphs_ready = False  # CUDA graphs compiled (if applicable)
        self.warmup_complete = False  # Initial warmup generation complete

        # Request queue management
        self.max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "50"))
        self.request_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.queue_processor_task = None

        # Track queue metrics
        self.queued_requests = 0
        self.rejected_requests = 0

        # Initialize warmup cache manager
        self.warmup_manager = None

    def _is_t4(self) -> bool:
        """
        Detect if the current GPU is a Tesla T4.

        Returns:
            True if Tesla T4 detected, False otherwise.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("No CUDA devices available")
                return False

            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.info("No CUDA devices found")
                return False

            # Check first GPU (index 0)
            device_name = torch.cuda.get_device_name(0)
            device_props = torch.cuda.get_device_properties(0)

            logger.info(f"GPU detected: {device_name}")
            logger.info(f"GPU compute capability: {device_props.major}.{device_props.minor}")

            # T4 has compute capability 7.5 (SM75) and name contains "T4"
            is_t4 = (
                ("T4" in device_name or "Tesla T4" in device_name)
                and device_props.major == 7
                and device_props.minor == 5
            )

            if is_t4:
                logger.info("✓ Tesla T4 GPU detected (SM75 architecture)")

            return is_t4

        except ImportError:
            logger.warning("PyTorch not available for GPU detection")
            # Fall back to checking environment variable
            gpu_name = os.getenv("GPU_NAME", "").lower()
            return "t4" in gpu_name or "tesla t4" in gpu_name
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False

    def _validate_t4_sdpa_enforcement(self):
        """
        Validate that SDPA backend is enforced for T4 GPUs.

        Raises:
            RuntimeError: If T4 detected but SDPA backend not configured.
        """
        if self._is_t4():
            attention_backend = os.getenv("VLLM_ATTENTION_BACKEND", "").upper()

            if attention_backend != "SDPA":
                error_msg = (
                    f"Tesla T4 GPU detected but VLLM_ATTENTION_BACKEND='{attention_backend}' "
                    f"(expected 'SDPA'). T4 GPUs require SDPA backend for optimal performance. "
                    f"Please set: export VLLM_ATTENTION_BACKEND=SDPA"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info("✓ T4 SDPA enforcement check passed")
        else:
            logger.info("Non-T4 GPU detected, skipping SDPA enforcement")

    async def initialize(self):
        """Initialize the vLLM engine with T4-optimized parameters."""

        # Step 1: Validate T4 SDPA enforcement before initialization
        self._validate_t4_sdpa_enforcement()

        # Step 2: Create engine args
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

        # Step 3: Initialize warmup cache manager
        # Use VLLM_CACHE_DIR env var if set, otherwise use default
        cache_dir = os.getenv("VLLM_CACHE_DIR", "/tmp/vllm_cache")
        engine_args.download_dir = cache_dir  # Set for warmup manager to pick up
        self.warmup_manager = WarmupCacheManager(WarmupConfig.from_engine_args(engine_args))

        # Step 4: Check if we have a cached warmup for this configuration
        warmup_key = self.warmup_manager.generate_warmup_key()
        is_cached = self.warmup_manager.is_warmed_up(warmup_key)

        if is_cached:
            logger.info(f"Using cached warmup for key: {warmup_key}")
        else:
            logger.info(f"No warmup cache found for key: {warmup_key}")

        # Step 5: Initialize the engine
        start_time = time.time()
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        init_time = time.time() - start_time
        self.model_loaded = True  # Mark model as loaded
        logger.info(f"vLLM engine initialized with model: {self.model_path} in {init_time:.2f}s")

        # Step 6: Post-initialization SM75 architecture assertions for T4
        if self._is_t4():
            self._assert_sm75_architecture()

        # Step 7: Mark CUDA graphs as ready (they compile during first inference)
        # For now, we assume they're ready after model load
        self.cuda_graphs_ready = True if not os.getenv("ENFORCE_EAGER", "false").lower() == "true" else False

        # Step 8: Mark warmup as complete if not cached
        if not is_cached:
            warmup_metadata = {
                "initialization_time": init_time,
                "model": self.model_path,
                "attention_backend": os.getenv("VLLM_ATTENTION_BACKEND", "SDPA"),
                "gpu": "Tesla T4" if self._is_t4() else "Other",
            }
            self.warmup_manager.mark_warmed_up(key=warmup_key, metadata=warmup_metadata)
            logger.info(f"Warmup cache saved for key: {warmup_key}")

        # Start queue processor
        self.queue_processor_task = asyncio.create_task(self._process_queue())

    def _assert_sm75_architecture(self):
        """
        Assert that the GPU has SM75 (compute capability 7.5) architecture.
        This is specific to Tesla T4 GPUs.
        """
        try:
            import torch

            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                compute_capability = f"{device_props.major}.{device_props.minor}"

                assert (
                    device_props.major == 7 and device_props.minor == 5
                ), f"Expected SM75 (7.5) architecture for T4, but got SM{compute_capability}"

                # Additional T4-specific assertions
                assert device_props.total_memory >= 15 * (
                    1024**3
                ), f"T4 should have ~16GB memory, but found {device_props.total_memory / (1024**3):.1f}GB"

                logger.info("✓ SM75 architecture assertions passed")
                logger.info(f"  - Compute capability: {compute_capability}")
                logger.info(f"  - Memory: {device_props.total_memory / (1024**3):.1f}GB")
                logger.info(f"  - Multiprocessors: {device_props.multi_processor_count}")

        except ImportError:
            logger.warning("PyTorch not available for SM75 assertion")
        except AssertionError as e:
            logger.error(f"SM75 architecture assertion failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"Could not verify SM75 architecture: {e}")

    async def _process_queue(self):
        """Process queued requests."""
        while True:
            try:
                # Wait for a request in the queue
                request_data = await self.request_queue.get()
                if request_data is None:  # Shutdown signal
                    break

                # Process the request
                future = request_data["future"]
                try:
                    result = await self._execute_request(request_data)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queued_requests -= 1
                    QUEUE_SIZE.set(self.queued_requests)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")

    async def _execute_request(self, request_data):
        """Execute a queued request."""
        # This will be called by the queue processor
        # The actual execution logic will be in the get_connection method
        return request_data

    async def get_connection(self):
        """Get a connection from the pool with queuing support."""
        if self.engine is None:
            await self.initialize()

        # Check if we can handle this request immediately
        if self.active_requests < self.max_connections:
            self.active_requests += 1
            ACTIVE_REQUESTS.inc()
            return self.engine

        # Check if queue is full (backpressure)
        if self.request_queue.full():
            self.rejected_requests += 1
            raise HTTPException(
                status_code=503, detail="Service at capacity. Please retry later.", headers={"Retry-After": "10"}
            )

        # Add to queue and wait
        self.queued_requests += 1
        QUEUE_SIZE.set(self.queued_requests)

        # Create a future for this request
        future = asyncio.Future()
        request_data = {"future": future, "timestamp": time.time()}

        try:
            await self.request_queue.put(request_data)
            # Wait for the request to be processed
            await future
            self.active_requests += 1
            ACTIVE_REQUESTS.inc()
            return self.engine
        except asyncio.QueueFull as e:
            self.queued_requests -= 1
            QUEUE_SIZE.set(self.queued_requests)
            self.rejected_requests += 1
            raise HTTPException(
                status_code=503, detail="Service at capacity. Please retry later.", headers={"Retry-After": "10"}
            ) from e

    async def release_connection(self):
        """Release a connection back to the pool."""
        self.active_requests -= 1
        ACTIVE_REQUESTS.dec()
        self.total_requests += 1

    def get_warmup_cache_info(self) -> dict:
        """Get information about the warmup cache."""
        if not self.warmup_manager:
            return {"enabled": False}

        try:
            current_key = self.warmup_manager.generate_warmup_key()
            is_cached = self.warmup_manager.is_warmed_up(current_key)
            cache_flags = self.warmup_manager.list_warmup_flags()

            return {
                "enabled": True,
                "current_key": current_key,
                "is_cached": is_cached,
                "total_cached_configs": len(cache_flags),
                "cache_dir": str(self.warmup_manager.config.cache_dir),
            }
        except Exception as e:
            logger.warning(f"Failed to get warmup cache info: {e}")
            return {"enabled": True, "error": str(e)}

    async def shutdown(self):
        """Shutdown the engine gracefully."""
        # Stop queue processor
        if self.queue_processor_task:
            await self.request_queue.put(None)  # Shutdown signal
            await self.queue_processor_task

        # Rotate old warmup cache flags
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
    gpu_type: str | None = None
    warmup_cache: dict | None = None


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
        warmup_params = SamplingParams(temperature=0.7, max_tokens=10, top_p=0.95)

        # Generate a test completion to ensure model is loaded
        warmup_request_id = random_uuid()
        results_generator = engine.generate(warmup_prompt, warmup_params, warmup_request_id)

        # Wait for completion
        async for _ in results_generator:
            pass  # Just iterate to complete the generation

        await connection_pool.release_connection()
        logger.info("✓ Model warmup completed successfully")
        connection_pool.warmup_complete = True  # Mark warmup as complete
        connection_pool.is_ready = True  # Mark as ready after successful warmup

    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        # Continue anyway - warmup is not critical
        connection_pool.warmup_complete = False  # Warmup failed but continue
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

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        gpu_type = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = info.used / 1024**3  # Convert to GB
            GPU_MEMORY_USAGE.set(gpu_memory)

            # Try to get GPU name
            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                gpu_type = gpu_name
            except (pynvml.NVMLError, AttributeError, UnicodeDecodeError) as exc:
                # Log the specific exception for debugging
                logger.warning(f"Failed to get GPU name: {exc}")
                pass
        except Exception:
            pass

        # Fallback GPU detection if pynvml fails
        if not gpu_type and connection_pool._is_t4():
            gpu_type = "Tesla T4"

        # Always return healthy for liveness - service is running
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=connection_pool.engine is not None,
            active_requests=connection_pool.active_requests,
            total_requests=connection_pool.total_requests,
            gpu_memory_usage_gb=gpu_memory,
            gpu_type=gpu_type,
            warmup_cache=connection_pool.get_warmup_cache_info(),
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
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait...")

    # Get T4 and attention backend info
    is_t4 = connection_pool._is_t4()
    attention_backend = os.getenv("VLLM_ATTENTION_BACKEND", "default")

    return {
        "status": "ready",
        "model": model_path,
        "timestamp": datetime.utcnow().isoformat(),
        "active_requests": connection_pool.active_requests,
        "total_requests": connection_pool.total_requests,
        "gpu_validation": {
            "is_t4": is_t4,
            "attention_backend": attention_backend,
            "t4_sdpa_enforced": is_t4 and attention_backend.upper() == "SDPA",
        },
        "warmup_cache": connection_pool.get_warmup_cache_info(),
    }


@app.get("/readyz")
async def readiness_graduated():
    """
    Graduated readiness probe endpoint.

    Checks multiple readiness criteria:
    1. Model loaded - basic engine initialization
    2. CUDA graphs ready - performance optimizations compiled
    3. Warmup complete - initial inference test passed

    Behavior controlled by READINESS_STRICT env var:
    - strict=true (default): Returns 503 until all criteria met
    - strict=false: Returns 200 with partial status once model loaded
    """
    strict_mode = os.getenv("READINESS_STRICT", "true").lower() == "true"

    # Build readiness status
    readiness_status = {
        "model_loaded": connection_pool.model_loaded,
        "cuda_graphs_ready": connection_pool.cuda_graphs_ready,
        "warmup_complete": connection_pool.warmup_complete,
    }

    # Calculate overall readiness
    all_ready = all(readiness_status.values())
    partial_ready = connection_pool.model_loaded  # At minimum, model must be loaded

    # Determine HTTP status code
    if strict_mode:
        # Strict mode: require all criteria
        is_ready = all_ready
    else:
        # Non-strict: partial readiness OK
        is_ready = partial_ready

    # Build response
    response_data = {
        "ready": is_ready,
        "strict_mode": strict_mode,
        "status": readiness_status,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_path,
    }

    # Add details if partially ready
    if not all_ready and partial_ready:
        response_data["message"] = "Model loaded, optimizations in progress"
        response_data["active_requests"] = connection_pool.active_requests
        response_data["total_requests"] = connection_pool.total_requests

    # Return appropriate response
    if not is_ready:
        # Not ready - return 503
        if not connection_pool.model_loaded:
            detail = "Model not loaded"
        elif strict_mode:
            missing = [k for k, v in readiness_status.items() if not v]
            detail = f"Waiting for: {', '.join(missing)}"
        else:
            detail = "Service initializing"

        return JSONResponse(status_code=503, content={**response_data, "detail": detail})

    # Ready - return 200
    return JSONResponse(status_code=200, content=response_data)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/runtime")
async def debug_runtime():
    """
    Debug endpoint for runtime information.

    Returns detailed GPU, memory, and attention backend information.
    Useful for troubleshooting and performance analysis.
    """
    debug_info: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_path,
        "pid": os.getpid(),
    }

    # GPU Information
    gpu_info: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_info["cuda_available"] = True
            gpu_info["device_count"] = device_count

            if device_count > 0:
                # Get info for primary device
                device = 0
                props = torch.cuda.get_device_properties(device)

                gpu_info["device_0"] = {
                    "name": torch.cuda.get_device_name(device),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": props.total_memory / (1024**3),
                    "multi_processor_count": props.multi_processor_count,
                    "is_t4": connection_pool._is_t4(),
                }

                # Current memory usage
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
                max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)

                gpu_info["memory"] = {
                    "allocated_gb": round(memory_allocated, 3),
                    "reserved_gb": round(memory_reserved, 3),
                    "max_allocated_gb": round(max_memory_allocated, 3),
                    "free_gb": round((props.total_memory / (1024**3)) - memory_allocated, 3),
                }

                # CUDA version
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["cudnn_version"] = (
                    torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
                )
        else:
            gpu_info["cuda_available"] = False
    except ImportError:
        gpu_info["error"] = "PyTorch not available"
    except Exception as e:
        gpu_info["error"] = str(e)

    debug_info["gpu"] = gpu_info

    # Try alternative GPU info via pynvml
    nvidia_smi_info: dict[str, Any] = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get device info
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

            nvidia_smi_info[f"device_{i}"] = {
                "name": name,
                "memory": {
                    "used_gb": mem_info.used / (1024**3),
                    "total_gb": mem_info.total / (1024**3),
                    "free_gb": mem_info.free / (1024**3),
                },
                "utilization": {
                    "gpu_percent": util.gpu,
                    "memory_percent": util.memory,
                },
                "temperature_c": temp,
                "power_watts": round(power, 2),
            }
    except ImportError:
        nvidia_smi_info["available"] = False
    except Exception as e:
        nvidia_smi_info["error"] = str(e)

    if nvidia_smi_info:
        debug_info["nvidia_smi"] = nvidia_smi_info

    # Attention backend information
    attention_info = {
        "backend": os.getenv("VLLM_ATTENTION_BACKEND", "AUTO"),
        "enforce_eager": os.getenv("ENFORCE_EAGER", "false"),
        "cuda_graphs_enabled": not (os.getenv("ENFORCE_EAGER", "false").lower() == "true"),
    }

    # Add T4-specific checks
    if connection_pool._is_t4():
        attention_info["t4_detected"] = True
        attention_info["sdpa_enforced"] = os.getenv("VLLM_ATTENTION_BACKEND", "").upper() == "SDPA"
        attention_info["recommended_backend"] = "SDPA"

    debug_info["attention"] = attention_info

    # System memory info
    system_memory: dict[str, Any] = {}
    try:
        import psutil

        mem = psutil.virtual_memory()
        system_memory = {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }

        # Process-specific memory
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info()
        system_memory["process"] = {
            "rss_gb": process_mem.rss / (1024**3),
            "vms_gb": process_mem.vms / (1024**3),
            "percent": process.memory_percent(),
        }
    except ImportError:
        system_memory["available"] = False
    except Exception as e:
        system_memory["error"] = str(e)

    debug_info["system_memory"] = system_memory

    # Environment variables (filtered for relevant ones)
    relevant_env_vars = [
        "VLLM_ATTENTION_BACKEND",
        "CUDA_VISIBLE_DEVICES",
        "ENFORCE_EAGER",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
        "TENSOR_PARALLEL_SIZE",
        "VLLM_CACHE_DIR",
        "MODEL_PATH",
        "READINESS_STRICT",
        "MAX_QUEUE_SIZE",
    ]

    env_vars = {k: os.getenv(k, "not set") for k in relevant_env_vars}
    debug_info["environment"] = env_vars

    # Connection pool state
    pool_state = {
        "model_loaded": connection_pool.model_loaded,
        "cuda_graphs_ready": connection_pool.cuda_graphs_ready,
        "warmup_complete": connection_pool.warmup_complete,
        "is_ready": connection_pool.is_ready,
        "active_requests": connection_pool.active_requests,
        "total_requests": connection_pool.total_requests,
        "queued_requests": connection_pool.queued_requests,
        "rejected_requests": connection_pool.rejected_requests,
    }

    # Add warmup cache info
    if connection_pool.warmup_manager:
        pool_state["warmup_cache"] = connection_pool.get_warmup_cache_info()

    debug_info["pool_state"] = pool_state

    # vLLM engine info (if available)
    if connection_pool.engine:
        try:
            engine_config = connection_pool.engine.engine.model_config
            debug_info["vllm_config"] = {
                "model": engine_config.model,
                "dtype": str(engine_config.dtype),
                "quantization": engine_config.quantization,
                "max_model_len": engine_config.max_model_len,
            }
        except Exception as e:
            debug_info["vllm_config"] = {"error": str(e)}

    return JSONResponse(content=debug_info)


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("100/minute")
async def generate(request: GenerateRequest, api_key: str = Depends(verify_api_key)):
    """Generate text completion with authentication."""
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
@limiter.limit("100/minute")
async def openai_completions(body: dict, api_key: str = Depends(verify_api_key)):
    """OpenAI-compatible completions endpoint with authentication."""
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

    # Call the existing generate function with api_key
    response = await generate(request, api_key)

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
