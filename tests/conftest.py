"""
Test configuration and fixtures for enhanced vLLM service testing.
Provides common fixtures and test utilities.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Test configuration
os.environ.update(
    {
        "MODEL_PATH": "test/model/path",
        "VLLM_CACHE_DIR": "/tmp/test_vllm_cache",
        "MAX_QUEUE_SIZE": "50",
        "GPU_MEMORY_UTILIZATION": "0.8",
        "ENABLE_PROMETHEUS": "false",  # Disable metrics for testing
        "API_KEYS": "test-key-1,test-key-2",  # pragma: allowlist secret
        "VLLM_ATTENTION_BACKEND": "SDPA",
    }
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_gpu_environment():
    """Mock GPU environment for testing."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.get_device_name", return_value="Tesla T4"),
        patch("torch.cuda.get_device_properties") as mock_props,
    ):
        # Mock GPU properties for T4
        mock_device_props = Mock()
        mock_device_props.major = 7
        mock_device_props.minor = 5
        mock_device_props.total_memory = 16 * (1024**3)  # 16GB
        mock_device_props.multi_processor_count = 40
        mock_props.return_value = mock_device_props

        yield mock_device_props


@pytest.fixture
def test_config():
    """Test configuration object."""
    from app.config import GPUConfig, GPUType, MetricsConfig, OpenAIConfig, ServiceConfig, VLLMConfig

    gpu_config = GPUConfig(
        gpu_type=GPUType.T4,
        memory_gb=16,
        compute_capability="7.5",
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        enable_cuda_graph=True,
        block_size=16,
        attention_backend="SDPA",
    )

    service_config = ServiceConfig(
        max_queue_size=50,
        request_timeout_streaming=300,
        request_timeout_batch=60,
        max_concurrent_requests=25,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_recovery_timeout=30,
    )

    openai_config = OpenAIConfig(
        enable_function_calling=True,
        enable_streaming_keep_alive=True,
        enable_usage_stats=True,
        enable_tiktoken_counting=False,  # Disable for testing
    )

    metrics_config = MetricsConfig(
        enable_prometheus=False,
        enable_request_tracing=False,
        structured_logging=False,  # Disable for testing
    )

    return VLLMConfig(
        model_path="test/model/path",
        gpu=gpu_config,
        service=service_config,
        openai=openai_config,
        metrics=metrics_config,
        cache_dir="/tmp/test_cache",
    )


# Test utilities
def assert_gpu_config_valid(gpu_config):
    """Assert that a GPU configuration is valid."""
    assert gpu_config.max_num_seqs > 0
    assert 0.0 < gpu_config.gpu_memory_utilization <= 1.0
    assert gpu_config.tensor_parallel_size > 0
    assert gpu_config.block_size > 0
    assert gpu_config.attention_backend in ["SDPA", "FLASH_ATTN", "AUTO"]


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
