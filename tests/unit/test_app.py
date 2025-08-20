import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from unittest.mock import AsyncMock, patch

    # Patch in the namespace the app uses at runtime
    with patch("app.main.AsyncLLMEngine", AsyncMock()) as MockEngine:
        # If app calls .from_engine_args(...), return an AsyncMock engine instance
        MockEngine.from_engine_args = AsyncMock(return_value=AsyncMock())
        from app.main import app

        return TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_readiness_check(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "model" in data
        assert "version" in data


class TestGenerationEndpoints:
    def test_generate_completion(self, client):
        # Access the mock through the client fixture's patched AsyncLLMEngine
        with patch("app.main.AsyncLLMEngine") as mock_engine:
            mock_instance = AsyncMock()
            mock_engine.from_engine_args.return_value = mock_instance
            mock_instance.generate.return_value = AsyncMock(
                outputs=[
                    Mock(
                        text="Generated text response",
                        token_ids=[1, 2, 3],
                        cumulative_logprob=-1.5,
                        finish_reason="stop",
                    )
                ]
            )

        request_data = {"prompt": "Test prompt", "max_tokens": 100, "temperature": 0.7, "top_p": 0.9, "top_k": 50}

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["text"] == "Generated text response"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_generate_with_invalid_params(self, client):
        request_data = {"prompt": "Test prompt", "max_tokens": -1, "temperature": 2.5}

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 422

    def test_generate_streaming(self, client):
        with patch("app.main.AsyncLLMEngine") as mock_engine:
            mock_vllm = AsyncMock()
            mock_engine.from_engine_args.return_value = mock_vllm

        async def mock_stream():
            yield Mock(outputs=[Mock(text="Streaming ", token_ids=[1], cumulative_logprob=-0.5, finish_reason=None)])
            yield Mock(
                outputs=[
                    Mock(text="Streaming response", token_ids=[1, 2], cumulative_logprob=-1.0, finish_reason="stop")
                ]
            )

        mock_vllm.generate.return_value = mock_stream()

        request_data = {"prompt": "Test prompt", "max_tokens": 100, "stream": True}

        response = client.post("/v1/completions", json=request_data, stream=True)
        assert response.status_code == 200

        chunks = []
        for line in response.iter_lines():
            if line and line != b"data: [DONE]":
                chunk_data = json.loads(line.decode().replace("data: ", ""))
                chunks.append(chunk_data)

        assert len(chunks) >= 1
        assert all("choices" in chunk for chunk in chunks)


class TestChatEndpoints:
    def test_chat_completion(self, client):
        with patch("app.main.AsyncLLMEngine") as mock_engine:
            mock_vllm = AsyncMock()
            mock_engine.from_engine_args.return_value = mock_vllm
            mock_vllm.generate.return_value = AsyncMock(
                outputs=[Mock(text="Chat response", token_ids=[1, 2, 3], cumulative_logprob=-1.5, finish_reason="stop")]
            )

        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Chat response"

    def test_chat_with_function_calling(self, client):
        with patch("app.main.AsyncLLMEngine") as mock_engine:
            mock_vllm = AsyncMock()
            mock_engine.from_engine_args.return_value = mock_vllm
            mock_vllm.generate.return_value = AsyncMock(
                outputs=[
                    Mock(
                        text='{"function": "test_function", "arguments": {"key": "value"}}',
                        token_ids=[1, 2, 3],
                        cumulative_logprob=-1.5,
                        finish_reason="function_call",
                    )
                ]
            )

        request_data = {
            "messages": [{"role": "user", "content": "Call a function"}],
            "functions": [
                {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
                }
            ],
            "function_call": "auto",
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "function_call"


class TestModelEndpoints:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert all("id" in model for model in data["data"])
        assert all("object" in model for model in data["data"])

    def test_get_model_info(self, client):
        response = client.get("/v1/models/mistral-7b-awq")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mistral-7b-awq"
        assert data["object"] == "model"
        assert "created" in data
        assert "owned_by" in data


class TestMetricsEndpoints:
    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "vllm_request_count" in response.text
        assert "vllm_request_latency" in response.text
        assert "vllm_model_memory_usage" in response.text

    def test_stats_endpoint(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "active_requests" in data
        assert "gpu_memory_usage" in data
        assert "model_loaded" in data


class TestErrorHandling:
    def test_missing_prompt(self, client):
        request_data = {"max_tokens": 100}

        response = client.post("/v1/completions", json=request_data)
        assert response.status_code == 422
        assert "prompt" in response.text.lower()

    def test_invalid_json(self, client):
        response = client.post("/v1/completions", data="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422

    def test_rate_limiting(self, client):
        request_data = {"prompt": "Test", "max_tokens": 10}

        responses = []
        for _ in range(100):
            response = client.post("/v1/completions", json=request_data)
            responses.append(response.status_code)

        assert 429 in responses or all(r == 200 for r in responses)

    @patch("app.main.AsyncLLMEngine")
    def test_model_loading_failure(self, mock_engine, client):
        mock_engine.from_engine_args.side_effect = RuntimeError("Failed to load model")

        response = client.get("/ready")
        assert response.status_code == 503
        assert "not ready" in response.json()["status"].lower()


class TestBatchProcessing:
    def test_batch_completion(self, client):
        with patch("app.main.AsyncLLMEngine") as mock_engine:
            mock_vllm = AsyncMock()
            mock_engine.from_engine_args.return_value = mock_vllm
        mock_outputs = [
            Mock(outputs=[Mock(text=f"Response {i}", token_ids=[i], cumulative_logprob=-1.0, finish_reason="stop")])
            for i in range(3)
        ]
        mock_vllm.generate.side_effect = mock_outputs

        request_data = {"prompts": ["Prompt 1", "Prompt 2", "Prompt 3"], "max_tokens": 50, "temperature": 0.7}

        response = client.post("/v1/completions/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["completions"]) == 3
        assert all("text" in c["choices"][0] for c in data["completions"])


class TestTokenization:
    def test_tokenize_endpoint(self, client):
        request_data = {"text": "Hello, world!"}

        response = client.post("/v1/tokenize", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "token_ids" in data
        assert len(data["tokens"]) == len(data["token_ids"])

    def test_detokenize_endpoint(self, client):
        request_data = {"token_ids": [1, 2, 3, 4]}

        response = client.post("/v1/detokenize", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
