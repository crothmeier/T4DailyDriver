"""
Comprehensive tests for enhanced vLLM production features.
Tests GPU detection, OpenAI compatibility, reliability features, and metrics.
"""

import asyncio
import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Import our enhanced modules
from app.config import GPUDetector, GPUType, load_config
from app.openai_compat import ChatMessage, TokenCounter
from app.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    GracefulShutdownManager,
    OOMPreventionMonitor,
    RequestDeduplicator,
)
from metrics import get_enhanced_metrics, trace_request


class TestGPUDetection:
    """Test GPU detection and configuration."""

    def test_gpu_type_enum(self):
        """Test GPU type enumeration."""
        assert GPUType.T4.value == "Tesla T4"
        assert GPUType.L4.value == "Tesla L4"
        assert GPUType.A4000.value == "RTX A4000"
        assert GPUType.UNKNOWN.value == "Unknown"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("pynvml.nvmlInit")
    @patch("pynvml.nvmlDeviceGetHandleByIndex")
    @patch("pynvml.nvmlDeviceGetName")
    def test_t4_detection(self, mock_get_name, mock_get_handle, mock_init, mock_cuda):
        """Test Tesla T4 GPU detection."""
        mock_get_name.return_value = b"Tesla T4"

        gpu_type = GPUDetector.detect_gpu_type()
        assert gpu_type == GPUType.T4

    @patch("torch.cuda.is_available", return_value=True)
    @patch("pynvml.nvmlInit")
    @patch("pynvml.nvmlDeviceGetHandleByIndex")
    @patch("pynvml.nvmlDeviceGetName")
    def test_l4_detection(self, mock_get_name, mock_get_handle, mock_init, mock_cuda):
        """Test Tesla L4 GPU detection."""
        mock_get_name.return_value = b"Tesla L4"

        gpu_type = GPUDetector.detect_gpu_type()
        assert gpu_type == GPUType.L4

    def test_gpu_config_t4(self):
        """Test T4 GPU configuration."""
        config = GPUDetector.get_gpu_config(GPUType.T4)

        assert config.gpu_type == GPUType.T4
        assert config.max_num_seqs == 8
        assert config.gpu_memory_utilization == 0.9
        assert config.attention_backend == "SDPA"
        assert config.enable_prefix_caching is True

    def test_gpu_config_l4(self):
        """Test L4 GPU configuration."""
        config = GPUDetector.get_gpu_config(GPUType.L4)

        assert config.gpu_type == GPUType.L4
        assert config.max_num_seqs == 16
        assert config.attention_backend == "FLASH_ATTN"

    def test_gpu_config_a4000(self):
        """Test A4000 GPU configuration."""
        config = GPUDetector.get_gpu_config(GPUType.A4000)

        assert config.gpu_type == GPUType.A4000
        assert config.max_num_seqs == 20
        assert config.attention_backend == "FLASH_ATTN"


class TestOpenAICompatibility:
    """Test OpenAI API compatibility layer."""

    def test_token_counter_fallback(self):
        """Test token counting without tiktoken."""
        counter = TokenCounter()

        # Test simple text
        tokens = counter.count_tokens("Hello world")
        assert tokens > 0

        # Test longer text
        long_text = "This is a longer text that should have more tokens than just hello world."
        long_tokens = counter.count_tokens(long_text)
        assert long_tokens > tokens

    def test_chat_message_token_counting(self):
        """Test chat message token counting."""
        counter = TokenCounter()

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello, how are you?"),
            ChatMessage(role="assistant", content="I'm doing well, thank you!"),
        ]

        token_count = counter.count_message_tokens(messages)
        assert token_count > 0

    def test_model_name_mapping(self):
        """Test model name mapping for OpenAI compatibility."""
        from app.openai_compat import ModelNameMapper

        mapper = ModelNameMapper("test-model-path")

        # Test known OpenAI models
        assert mapper.resolve_model("gpt-3.5-turbo") == "test-model-path"
        assert mapper.resolve_model("gpt-4") == "test-model-path"
        assert mapper.resolve_model("text-davinci-003") == "test-model-path"

        # Test unknown model
        assert mapper.resolve_model("unknown-model") == "test-model-path"

    def test_function_call_handler(self):
        """Test function calling support detection."""
        from app.openai_compat import FunctionCallHandler

        handler = FunctionCallHandler()

        # Test supported models
        assert handler.is_function_calling_supported("gpt-3.5-turbo") is True
        assert handler.is_function_calling_supported("gpt-4") is True

        # Test unsupported model
        assert handler.is_function_calling_supported("text-davinci-003") is False

    def test_function_call_parsing(self):
        """Test function call parsing from response text."""
        from app.openai_compat import FunctionCallHandler

        handler = FunctionCallHandler()

        # Test valid function call
        response_with_call = '[Function Call: {"name": "test_function", "arguments": {"param": "value"}}]'
        parsed = handler.parse_function_call_from_response(response_with_call)

        assert parsed is not None
        assert parsed["name"] == "test_function"
        assert parsed["arguments"]["param"] == "value"

        # Test no function call
        normal_response = "This is just a normal response without function calls."
        parsed = handler.parse_function_call_from_response(normal_response)
        assert parsed is None


class TestCircuitBreaker:
    """Test circuit breaker reliability feature."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=10)
        cb = CircuitBreaker("test", config)

        assert cb.name == "test"
        assert cb.config.failure_threshold == 3
        assert cb.config.recovery_timeout == 10
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test successful operation through circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        async with cb.protect():
            # Simulate successful operation
            pass

        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self):
        """Test circuit breaker opening on failures."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1)
        cb = CircuitBreaker("test", config)

        # Cause failures to open circuit breaker
        for _ in range(2):
            try:
                async with cb.protect():
                    raise Exception("GPU out of memory")
            except Exception:
                pass

        # Circuit should now be open
        from app.reliability import CircuitBreakerState

        assert cb.state == CircuitBreakerState.OPEN

        # Next request should be blocked
        with pytest.raises(Exception):
            async with cb.protect():
                pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        # Cause failure to open circuit
        try:
            async with cb.protect():
                raise Exception("Test failure")
        except Exception:
            pass

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Should now allow test request (half-open)
        async with cb.protect():
            pass  # Successful operation

        # Circuit should now be closed
        from app.reliability import CircuitBreakerState

        assert cb.state == CircuitBreakerState.CLOSED


class TestRequestDeduplication:
    """Test request deduplication feature."""

    def test_deduplication_initialization(self):
        """Test request deduplicator initialization."""
        dedup = RequestDeduplicator(ttl=60)
        assert dedup.ttl == 60
        assert len(dedup.cache) == 0

    def test_content_hash_generation(self):
        """Test content hash generation."""
        dedup = RequestDeduplicator()

        hash1 = dedup._generate_content_hash("test content")
        hash2 = dedup._generate_content_hash("test content")
        hash3 = dedup._generate_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_idempotent_caching(self):
        """Test idempotent request caching."""
        dedup = RequestDeduplicator(ttl=60)

        content = {"prompt": "test", "max_tokens": 100}
        result = {"text": "response", "tokens": 50}

        # First request - should return None (new request)
        cached = dedup.get_or_set_idempotent("key1", content, None)
        assert cached is None

        # Set result
        dedup.get_or_set_idempotent("key1", content, result)

        # Second request - should return cached result
        cached = dedup.get_or_set_idempotent("key1", content, None)
        assert cached == result

    def test_content_based_deduplication(self):
        """Test content-based deduplication without explicit key."""
        dedup = RequestDeduplicator(ttl=60)

        content = {"prompt": "test prompt"}
        result = {"response": "test response"}

        # Cache result
        dedup.get_or_set_idempotent(None, content, result)

        # Same content should return cached result
        cached = dedup.get_or_set_idempotent(None, content, None)
        assert cached == result

        # Different content should not return cached result
        different_content = {"prompt": "different prompt"}
        cached = dedup.get_or_set_idempotent(None, different_content, None)
        assert cached is None


class TestGracefulShutdown:
    """Test graceful shutdown manager."""

    def test_shutdown_manager_initialization(self):
        """Test shutdown manager initialization."""
        manager = GracefulShutdownManager(timeout=30)
        assert manager.timeout == 30
        assert len(manager.active_requests) == 0
        assert not manager.shutdown_started

    def test_request_registration(self):
        """Test request registration and unregistration."""
        manager = GracefulShutdownManager()

        manager.register_request("req1")
        manager.register_request("req2")

        assert len(manager.active_requests) == 2
        assert "req1" in manager.active_requests
        assert "req2" in manager.active_requests

        manager.unregister_request("req1")
        assert len(manager.active_requests) == 1
        assert "req1" not in manager.active_requests

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown process."""
        manager = GracefulShutdownManager(timeout=1)

        # Register some requests
        manager.register_request("req1")
        manager.register_request("req2")

        # Start shutdown
        shutdown_task = asyncio.create_task(manager.initiate_shutdown())

        # Wait a bit, then unregister requests
        await asyncio.sleep(0.1)
        manager.unregister_request("req1")
        manager.unregister_request("req2")

        # Wait for shutdown to complete
        duration, remaining = await shutdown_task

        assert remaining == 0
        assert duration < 1.0


class TestOOMPrevention:
    """Test OOM prevention monitor."""

    @patch("pynvml.nvmlInit")
    @patch("pynvml.nvmlDeviceGetHandleByIndex")
    @patch("pynvml.nvmlDeviceGetMemoryInfo")
    def test_memory_usage_detection(self, mock_memory_info, mock_handle, mock_init):
        """Test GPU memory usage detection."""
        # Mock memory info
        mock_info = Mock()
        mock_info.used = 8 * (1024**3)  # 8GB used
        mock_info.total = 16 * (1024**3)  # 16GB total
        mock_memory_info.return_value = mock_info

        monitor = OOMPreventionMonitor(warning_threshold=0.7, critical_threshold=0.9)

        status = monitor.check_memory_pressure()

        assert status["status"] == "warning"  # 50% usage triggers warning
        assert status["usage_ratio"] == 0.5
        assert status["should_reject"] is False

    def test_should_accept_request(self):
        """Test request acceptance based on memory pressure."""
        monitor = OOMPreventionMonitor(critical_threshold=0.9)

        # Mock low memory usage
        with patch.object(monitor, "_get_gpu_memory_usage", return_value=0.5):
            assert monitor.should_accept_request() is True

        # Mock high memory usage
        with patch.object(monitor, "_get_gpu_memory_usage", return_value=0.95):
            assert monitor.should_accept_request() is False


class TestMetrics:
    """Test enhanced metrics collection."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = get_enhanced_metrics()

        # Test core metrics exist
        assert "core" in metrics
        assert "request_count" in metrics["core"]
        assert "request_duration" in metrics["core"]
        assert "active_requests" in metrics["core"]

        # Test enhanced metrics exist
        assert "enhanced" in metrics
        assert "ttft_percentiles" in metrics["enhanced"]
        assert "circuit_breaker_state" in metrics["enhanced"]
        assert "oom_events" in metrics["enhanced"]

    @pytest.mark.asyncio
    async def test_request_tracing(self):
        """Test request tracing context manager."""
        async with trace_request() as correlation_id:
            assert correlation_id is not None
            assert len(correlation_id) > 0


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_config_loading_with_env_vars(self):
        """Test configuration loading with environment variables."""
        # Set test environment variables
        test_env = {
            "MODEL_PATH": "test/model/path",
            "MAX_QUEUE_SIZE": "200",
            "GPU_MEMORY_UTILIZATION": "0.8",
            "ENABLE_FUNCTION_CALLING": "false",
        }

        with patch.dict(os.environ, test_env):
            config = load_config()

            assert config.model_path == "test/model/path"
            assert config.service.max_queue_size == 200
            assert config.openai.enable_function_calling is False

    def test_config_defaults(self):
        """Test configuration defaults."""
        # Clear relevant env vars
        env_to_clear = ["MODEL_PATH", "MAX_QUEUE_SIZE", "GPU_MEMORY_UTILIZATION", "ENABLE_FUNCTION_CALLING"]

        with patch.dict(os.environ, {}, clear=True):
            config = load_config()

            # Test defaults
            assert config.model_path == "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
            assert config.service.max_queue_size == 100
            assert config.openai.enable_function_calling is True


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the enhanced service."""

    @pytest.fixture
    def test_client(self):
        """Create test client for integration tests."""
        # This would require mocking the vLLM engine
        # For now, just test that imports work
        from app_enhanced import app

        return TestClient(app)

    def test_app_creation(self, test_client):
        """Test that the enhanced app can be created."""
        # Basic smoke test - app should be created without errors
        assert test_client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
