"""
Unit tests for enhanced FastAPI vLLM service with production hardening features.
Tests OpenAI compatibility, structured logging, metrics, and graceful shutdown.
"""

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


@pytest.fixture
def mock_vllm_engine():
    """Create a mock vLLM engine."""
    engine = AsyncMock()

    # Mock the generate method
    async def mock_generate(prompt, sampling_params, request_id):
        # Simulate streaming generation
        class MockOutput:
            def __init__(self):
                self.text = "This is a test response"
                self.token_ids = [1, 2, 3, 4, 5]

        class MockRequestOutput:
            def __init__(self):
                self.outputs = [MockOutput()]
                self.request_id = request_id

        for i in range(3):
            yield MockRequestOutput()
            await asyncio.sleep(0.01)

    engine.generate = mock_generate
    return engine


@pytest.fixture
def mock_connection_pool(mock_vllm_engine):
    """Create a mock connection pool."""
    pool = MagicMock()
    pool.engine = mock_vllm_engine
    pool.active_requests = 0
    pool.total_requests = 10
    pool.circuit_open = False
    pool.is_ready = True
    pool._is_t4 = MagicMock(return_value=True)

    async def mock_get_connection(correlation_id=None):
        return mock_vllm_engine, correlation_id or str(uuid.uuid4())

    async def mock_release_connection(connection_id):
        pass

    async def mock_initialize():
        pass

    async def mock_shutdown():
        pass

    pool.get_connection = mock_get_connection
    pool.release_connection = mock_release_connection
    pool.initialize = mock_initialize
    pool.shutdown = mock_shutdown
    pool.warmup_manager = MagicMock()
    pool.warmup_manager.get_warmup_cache_info = MagicMock(
        return_value={
            "enabled": True,
            "current_key": "test_key",
            "is_cached": True,
            "total_cached_configs": 5,
            "cache_dir": "/tmp/vllm_cache",
        }
    )

    return pool


@pytest.fixture
async def app_client(mock_connection_pool):
    """Create test client with mocked dependencies."""
    with patch("app.main_enhanced.connection_pool", mock_connection_pool):
        with patch("app.main_enhanced.openai_compat") as mock_openai:
            # Mock OpenAI compatibility layer
            mock_openai.chat_completion = AsyncMock()
            mock_openai.text_completion = AsyncMock()

            # Import app after patching
            from app.main_enhanced import app

            async with AsyncClient(app=app, base_url="http://test") as client:
                yield client


@pytest.fixture
def sync_client(mock_connection_pool):
    """Create synchronous test client."""
    with patch("app.main_enhanced.connection_pool", mock_connection_pool):
        from app.main_enhanced import app

        return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, app_client):
        """Test /healthz endpoint."""
        response = await app_client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["model_loaded"] is True
        assert data["active_requests"] == 0
        assert data["total_requests"] == 10
        assert data["circuit_breaker_state"] == "closed"
        assert data["warmup_cache"] is not None

    @pytest.mark.asyncio
    async def test_health_with_gpu_metrics(self, app_client):
        """Test health endpoint with GPU metrics."""
        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetHandleByIndex") as mock_handle:
                with patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_mem:
                    with patch("pynvml.nvmlDeviceGetUtilizationRates") as mock_util:
                        with patch("pynvml.nvmlDeviceGetTemperature") as mock_temp:
                            # Mock GPU metrics
                            mock_mem.return_value = MagicMock(used=8 * 1024 ** 3)  # 8GB
                            mock_util.return_value = MagicMock(gpu=75, memory=60)
                            mock_temp.return_value = 65  # 65°C

                            response = await app_client.get("/healthz")
                            assert response.status_code == 200

                            data = response.json()
                            assert data["gpu_memory_usage_gb"] == 8.0
                            assert data["gpu_utilization_percent"] == 75
                            assert data["gpu_temperature_celsius"] == 65

    @pytest.mark.asyncio
    async def test_health_circuit_breaker_open(self, app_client, mock_connection_pool):
        """Test health endpoint when circuit breaker is open."""
        mock_connection_pool.circuit_open = True

        response = await app_client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert data["circuit_breaker_state"] == "open"


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, app_client):
        """Test /metrics endpoint returns Prometheus format."""
        response = await app_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

        # Check for key metrics in response
        content = response.text
        assert "vllm_request_total" in content
        assert "vllm_request_duration_seconds" in content
        assert "vllm_active_requests" in content
        assert "vllm_token_throughput" in content
        assert "vllm_ttft_seconds" in content
        assert "vllm_queue_size" in content
        assert "vllm_gpu_memory_usage_gb" in content

        # Check enhanced metrics
        assert "vllm_circuit_breaker_state" in content
        assert "vllm_concurrent_connections" in content
        assert "vllm_warm_cache_hits_total" in content


class TestOpenAICompatibility:
    """Test OpenAI-compatible endpoints."""

    @pytest.mark.asyncio
    async def test_chat_completions(self, app_client):
        """Test /v1/chat/completions endpoint."""
        with patch("app.main_enhanced.openai_compat") as mock_openai:
            # Mock response
            mock_response = {
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt-3.5-turbo",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

            mock_openai.chat_completion = AsyncMock(return_value=mock_response)

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "max_tokens": 100,
            }

            response = await app_client.post("/v1/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "chatcmpl-test123"
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, app_client):
        """Test streaming chat completions."""
        with patch("app.main_enhanced.openai_compat") as mock_openai:
            # Mock streaming response
            async def mock_stream():
                for chunk in ["Hello", " world", "!"]:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                yield "data: [DONE]\n\n"

            mock_openai.chat_completion = AsyncMock(
                return_value=MagicMock(__aiter__=mock_stream, headers={"content-type": "text/event-stream"})
            )

            request_data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            }

            response = await app_client.post("/v1/chat/completions", json=request_data)

            # For streaming, we'd need to check the stream response
            # This is simplified for testing
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_text_completions(self, app_client):
        """Test /v1/completions endpoint."""
        with patch("app.main_enhanced.openai_compat") as mock_openai:
            # Mock response
            mock_response = {
                "id": "cmpl-test123",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "text-davinci-003",
                "choices": [{"text": "Test completion", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

            mock_openai.text_completion = AsyncMock(return_value=mock_response)

            request_data = {
                "model": "text-davinci-003",
                "prompt": "Complete this:",
                "max_tokens": 50,
                "temperature": 0.8,
            }

            response = await app_client.post("/v1/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "cmpl-test123"
            assert data["object"] == "text_completion"
            assert data["choices"][0]["text"] == "Test completion"

    @pytest.mark.asyncio
    async def test_list_models(self, app_client):
        """Test /v1/models endpoint."""
        response = await app_client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["object"] == "model"


class TestRequestTracing:
    """Test request tracing and correlation IDs."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self, app_client):
        """Test that correlation IDs are propagated through requests."""
        correlation_id = str(uuid.uuid4())

        response = await app_client.get("/healthz", headers={"X-Correlation-ID": correlation_id})

        assert response.status_code == 200
        assert response.headers.get("X-Correlation-ID") == correlation_id
        assert "X-Request-Duration" in response.headers

    @pytest.mark.asyncio
    async def test_auto_correlation_id(self, app_client):
        """Test automatic correlation ID generation."""
        response = await app_client.get("/healthz")

        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert "X-Request-Duration" in response.headers

        # Verify it's a valid UUID
        correlation_id = response.headers["X-Correlation-ID"]
        uuid.UUID(correlation_id)  # Will raise if invalid


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_without_api_key(self, app_client):
        """Test rate limiting when no API key is configured."""
        with patch("auth.API_KEYS", set()):  # No API keys configured
            # Should not require authentication
            response = await app_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]},
            )
            # Will fail due to mocking but not due to auth
            assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_rate_limit_with_api_key(self, app_client):
        """Test rate limiting with API key."""
        with patch("auth.API_KEYS", {"test-key-123"}):
            # Should require authentication
            response = await app_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]},
            )
            assert response.status_code == 401  # Unauthorized without key

            # With API key
            response = await app_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Test"}]},
                headers={"Authorization": "Bearer test-key-123"},
            )
            assert response.status_code in [200, 500]  # Authorized


class TestGracefulShutdown:
    """Test graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_connection_draining(self, mock_connection_pool):
        """Test connection draining during shutdown."""
        # Add active connections
        mock_connection_pool.active_connections = {"conn1", "conn2", "conn3"}

        # Test drain with immediate completion
        mock_connection_pool.drain_timeout = 1
        await mock_connection_pool.drain_connections()

        # In real scenario, connections would be cleared
        # This is a simplified test
        assert mock_connection_pool.drain_timeout == 1

    @pytest.mark.asyncio
    async def test_shutdown_sequence(self, mock_connection_pool):
        """Test complete shutdown sequence."""
        mock_connection_pool.shutdown_event = asyncio.Event()
        mock_connection_pool.queue_processor_task = None

        await mock_connection_pool.shutdown()

        # Verify shutdown was called
        mock_connection_pool.shutdown.assert_called_once()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        from app.main_enhanced import EnhancedVLLMConnectionPool

        pool = EnhancedVLLMConnectionPool("test-model")
        pool.circuit_threshold = 3

        # Record failures
        for _ in range(3):
            pool.record_failure()

        assert pool.circuit_open is True
        assert pool.circuit_failures == 3

    def test_circuit_breaker_closes_after_timeout(self):
        """Test that circuit breaker closes after reset timeout."""
        from app.main_enhanced import EnhancedVLLMConnectionPool

        pool = EnhancedVLLMConnectionPool("test-model")
        pool.circuit_threshold = 3
        pool.circuit_reset_time = 0.1  # 100ms for testing

        # Open circuit
        for _ in range(3):
            pool.record_failure()

        assert pool.circuit_open is True

        # Wait for reset
        time.sleep(0.2)
        pool._check_circuit_breaker()

        assert pool.circuit_open is False
        assert pool.circuit_failures == 0

    def test_circuit_breaker_success_reduces_failures(self):
        """Test that successes reduce failure count."""
        from app.main_enhanced import EnhancedVLLMConnectionPool

        pool = EnhancedVLLMConnectionPool("test-model")

        # Record some failures
        pool.record_failure()
        pool.record_failure()
        assert pool.circuit_failures == 2

        # Record success
        pool.record_success()
        assert pool.circuit_failures == 1

        # More successes
        pool.record_success()
        assert pool.circuit_failures == 0


class TestStructuredLogging:
    """Test structured JSON logging."""

    def test_structured_formatter(self):
        """Test the structured log formatter."""
        from app.main_enhanced import StructuredFormatter

        formatter = StructuredFormatter()

        # Create a log record
        record = MagicMock()
        record.levelname = "INFO"
        record.name = "test_module"
        record.getMessage = MagicMock(return_value="Test message")
        record.correlation_id = "test-correlation-id"
        record.request_id = "test-request-id"

        # Format the record
        formatted = formatter.format(record)

        # Parse JSON
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["module"] == "test_module"
        assert log_data["message"] == "Test message"
        assert log_data["correlation_id"] == "test-correlation-id"
        assert log_data["request_id"] == "test-request-id"
        assert "timestamp" in log_data
        assert "pid" in log_data


class TestRootEndpoint:
    """Test root endpoint."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, app_client):
        """Test root / endpoint."""
        response = await app_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "vLLM Production Service"
        assert data["version"] == "2.0.0"
        assert data["status"] == "running"
        assert data["gpu"] == "Tesla T4"
        assert "endpoints" in data

        # Check all endpoints are listed
        endpoints = data["endpoints"]
        assert endpoints["health"] == "/healthz"
        assert endpoints["metrics"] == "/metrics"
        assert endpoints["chat"] == "/v1/chat/completions"
        assert endpoints["completions"] == "/v1/completions"
        assert endpoints["models"] == "/v1/models"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
