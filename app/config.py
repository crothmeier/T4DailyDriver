"""
Production configuration schema for vLLM service.
Handles GPU detection, optimization settings, and service configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """Supported GPU types with their characteristics."""

    T4 = "Tesla T4"
    L4 = "Tesla L4"
    A4000 = "RTX A4000"
    UNKNOWN = "Unknown"


@dataclass
class GPUConfig:
    """GPU-specific configuration."""

    gpu_type: GPUType
    memory_gb: float
    compute_capability: str
    max_num_seqs: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enable_cuda_graph: bool
    block_size: int
    attention_backend: str


@dataclass
class ServiceConfig:
    """Service reliability configuration."""

    max_queue_size: int = 100
    request_timeout_streaming: int = 300  # 5 minutes
    request_timeout_batch: int = 60  # 1 minute
    max_concurrent_requests: int = 50
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 30
    graceful_shutdown_timeout: int = 30
    health_check_interval: int = 10
    enable_request_deduplication: bool = True


@dataclass
class OpenAIConfig:
    """OpenAI API compatibility configuration."""

    enable_function_calling: bool = True
    enable_streaming_keep_alive: bool = True
    keep_alive_interval: int = 30
    model_name_aliases: dict[str, str] = field(
        default_factory=lambda: {"gpt-3.5-turbo": "default", "gpt-4": "default", "text-davinci-003": "default"}
    )
    enable_usage_stats: bool = True
    enable_tiktoken_counting: bool = True


@dataclass
class MetricsConfig:
    """Observability and metrics configuration."""

    enable_prometheus: bool = True
    enable_request_tracing: bool = True
    enable_gpu_metrics: bool = True
    enable_dcgm_integration: bool = False
    metrics_port: int = 9090
    structured_logging: bool = True
    log_request_response_sizes: bool = True


@dataclass
class VLLMConfig:
    """Complete vLLM service configuration."""

    # Core model settings
    model_path: str
    trust_remote_code: bool = True
    dtype: str = "float16"
    quantization: str | None = "awq"
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False

    # GPU settings (will be auto-configured based on detection)
    gpu: GPUConfig = None

    # Service settings
    service: ServiceConfig = field(default_factory=ServiceConfig)

    # OpenAI compatibility
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)

    # Metrics and observability
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Cache settings
    cache_dir: str = "/tmp/vllm_cache"
    swap_space: int = 4

    # Model loading
    download_dir: str | None = None
    load_format: str = "auto"
    max_model_len: int | None = None


class GPUDetector:
    """Detects GPU type and returns optimal configuration."""

    # GPU-specific configurations
    GPU_CONFIGS = {
        GPUType.T4: GPUConfig(
            gpu_type=GPUType.T4,
            memory_gb=16,
            compute_capability="7.5",
            max_num_seqs=8,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            enable_cuda_graph=True,
            block_size=16,
            attention_backend="SDPA",
        ),
        GPUType.L4: GPUConfig(
            gpu_type=GPUType.L4,
            memory_gb=24,
            compute_capability="8.9",
            max_num_seqs=16,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            enable_cuda_graph=True,
            block_size=16,
            attention_backend="FLASH_ATTN",
        ),
        GPUType.A4000: GPUConfig(
            gpu_type=GPUType.A4000,
            memory_gb=16,
            compute_capability="8.6",
            max_num_seqs=20,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            enable_cuda_graph=True,
            block_size=16,
            attention_backend="FLASH_ATTN",
        ),
    }

    @classmethod
    def detect_gpu_type(cls) -> GPUType:
        """Detect current GPU type."""
        try:
            import pynvml
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                return GPUType.UNKNOWN

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

            logger.info(f"Detected GPU: {gpu_name}")

            # Match GPU name to type
            if "T4" in gpu_name or "Tesla T4" in gpu_name:
                return GPUType.T4
            elif "L4" in gpu_name or "Tesla L4" in gpu_name:
                return GPUType.L4
            elif "A4000" in gpu_name or "RTX A4000" in gpu_name:
                return GPUType.A4000
            else:
                logger.warning(f"Unknown GPU type: {gpu_name}")
                return GPUType.UNKNOWN

        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return GPUType.UNKNOWN

    @classmethod
    def get_gpu_config(cls, gpu_type: GPUType | None = None) -> GPUConfig:
        """Get GPU configuration, detecting type if not specified."""
        if gpu_type is None:
            gpu_type = cls.detect_gpu_type()

        if gpu_type in cls.GPU_CONFIGS:
            config = cls.GPU_CONFIGS[gpu_type]
            logger.info(f"Using {gpu_type.value} configuration:")
            logger.info(f"  - Max sequences: {config.max_num_seqs}")
            logger.info(f"  - Memory utilization: {config.gpu_memory_utilization}")
            logger.info(f"  - Attention backend: {config.attention_backend}")
            return config
        else:
            # Fallback to T4 config for unknown GPUs
            logger.warning(f"Unknown GPU type {gpu_type}, falling back to T4 configuration")
            return cls.GPU_CONFIGS[GPUType.T4]


def load_config() -> VLLMConfig:
    """Load configuration from environment variables with smart defaults."""

    # Detect GPU and get optimal configuration
    gpu_config = GPUDetector.get_gpu_config()

    # Allow environment overrides for GPU settings
    if os.getenv("MAX_NUM_SEQS"):
        gpu_config.max_num_seqs = int(os.getenv("MAX_NUM_SEQS"))
    if os.getenv("GPU_MEMORY_UTILIZATION"):
        gpu_config.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION"))
    if os.getenv("VLLM_ATTENTION_BACKEND"):
        gpu_config.attention_backend = os.getenv("VLLM_ATTENTION_BACKEND")
    if os.getenv("TENSOR_PARALLEL_SIZE"):
        gpu_config.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE"))

    # Service configuration
    service_config = ServiceConfig(
        max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "100")),
        request_timeout_streaming=int(os.getenv("REQUEST_TIMEOUT_STREAMING", "300")),
        request_timeout_batch=int(os.getenv("REQUEST_TIMEOUT_BATCH", "60")),
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "50")),
        circuit_breaker_failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURES", "5")),
        circuit_breaker_recovery_timeout=int(os.getenv("CIRCUIT_BREAKER_RECOVERY", "30")),
        graceful_shutdown_timeout=int(os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30")),
        health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "10")),
        enable_request_deduplication=os.getenv("ENABLE_REQUEST_DEDUPLICATION", "true").lower() == "true",
    )

    # OpenAI configuration
    openai_config = OpenAIConfig(
        enable_function_calling=os.getenv("ENABLE_FUNCTION_CALLING", "true").lower() == "true",
        enable_streaming_keep_alive=os.getenv("ENABLE_STREAMING_KEEP_ALIVE", "true").lower() == "true",
        keep_alive_interval=int(os.getenv("STREAMING_KEEP_ALIVE_INTERVAL", "30")),
        enable_usage_stats=os.getenv("ENABLE_USAGE_STATS", "true").lower() == "true",
        enable_tiktoken_counting=os.getenv("ENABLE_TIKTOKEN_COUNTING", "true").lower() == "true",
    )

    # Metrics configuration
    metrics_config = MetricsConfig(
        enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
        enable_request_tracing=os.getenv("ENABLE_REQUEST_TRACING", "true").lower() == "true",
        enable_gpu_metrics=os.getenv("ENABLE_GPU_METRICS", "true").lower() == "true",
        enable_dcgm_integration=os.getenv("ENABLE_DCGM_INTEGRATION", "false").lower() == "true",
        metrics_port=int(os.getenv("METRICS_PORT", "9090")),
        structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true",
        log_request_response_sizes=os.getenv("LOG_REQUEST_RESPONSE_SIZES", "true").lower() == "true",
    )

    return VLLMConfig(
        model_path=os.getenv("MODEL_PATH", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"),
        trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true",
        dtype=os.getenv("DTYPE", "float16"),
        quantization=os.getenv("QUANTIZATION", "awq"),
        gpu=gpu_config,
        service=service_config,
        openai=openai_config,
        metrics=metrics_config,
        cache_dir=os.getenv("VLLM_CACHE_DIR", "/tmp/vllm_cache"),
        swap_space=int(os.getenv("SWAP_SPACE", "4")),
        download_dir=os.getenv("DOWNLOAD_DIR"),
        max_model_len=int(os.getenv("MAX_MODEL_LEN")) if os.getenv("MAX_MODEL_LEN") else None,
    )
