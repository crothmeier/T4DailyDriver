import asyncio
import time
from typing import Any

import httpx
import pytest


@pytest.fixture(scope="module")
def service_url():
    return "http://localhost:8080"


@pytest.fixture(scope="module")
async def async_client(service_url):
    async with httpx.AsyncClient(base_url=service_url, timeout=30.0) as client:
        yield client


class TestModelLoading:
    @pytest.mark.asyncio
    async def test_model_loads_successfully(self, async_client):
        response = await async_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model"] == "mistralai/Mistral-7B-v0.1"
        assert data["quantization"] == "awq"

    @pytest.mark.asyncio
    async def test_model_info_available(self, async_client):
        response = await async_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) > 0
        model = data["data"][0]
        assert "mistral" in model["id"].lower()
        assert model["object"] == "model"

    @pytest.mark.asyncio
    async def test_gpu_memory_allocated(self, async_client):
        response = await async_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert data["gpu_memory_usage"] > 0
        assert data["gpu_memory_usage"] < 16000


class TestGenerationQuality:
    @pytest.mark.asyncio
    async def test_basic_generation(self, async_client):
        prompt = "The capital of France is"
        response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 10,
                "temperature": 0.1
            }
        )
        assert response.status_code == 200
        data = response.json()
        text = data["choices"][0]["text"].lower()
        assert "paris" in text

    @pytest.mark.asyncio
    async def test_deterministic_generation(self, async_client):
        prompt = "1 + 1 equals"
        responses = []

        for _ in range(3):
            response = await async_client.post(
                "/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 5,
                    "temperature": 0.0,
                    "seed": 42
                }
            )
            assert response.status_code == 200
            responses.append(response.json()["choices"][0]["text"])

        assert len(set(responses)) == 1

    @pytest.mark.asyncio
    async def test_generation_with_stop_sequences(self, async_client):
        prompt = "List three colors:\n1. Red\n2. Blue\n3."
        response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.5,
                "stop": ["\n4.", "\n\n"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        text = data["choices"][0]["text"]
        assert "\n4." not in text
        assert len(text) < 50

    @pytest.mark.asyncio
    async def test_chat_format_generation(self, async_client):
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2 times 3?"}
        ]

        response = await async_client.post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": 20,
                "temperature": 0.3
            }
        )
        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"].lower()
        assert "6" in content or "six" in content

    @pytest.mark.asyncio
    async def test_long_context_handling(self, async_client):
        long_prompt = "Once upon a time, " * 500
        long_prompt += "The end of the story is"

        response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": long_prompt,
                "max_tokens": 20,
                "temperature": 0.7
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"][0]["text"]) > 0

    @pytest.mark.asyncio
    async def test_batch_generation_consistency(self, async_client):
        prompts = [
            "The sun is",
            "Water freezes at",
            "Python is a"
        ]

        response = await async_client.post(
            "/v1/completions/batch",
            json={
                "prompts": prompts,
                "max_tokens": 10,
                "temperature": 0.3
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["completions"]) == 3

        expected_keywords = ["star", "0", "programming"]
        for i, completion in enumerate(data["completions"]):
            text = completion["choices"][0]["text"].lower()
            assert any(keyword in text for keyword in [expected_keywords[i]])


class TestPerformance:
    @pytest.mark.asyncio
    async def test_generation_latency(self, async_client):
        prompt = "Hello, world!"

        start_time = time.time()
        response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.7
            }
        )
        latency = time.time() - start_time

        assert response.status_code == 200
        assert latency < 5.0

    @pytest.mark.asyncio
    async def test_streaming_performance(self, async_client):
        prompt = "Tell me a story about"

        start_time = time.time()
        first_token_time = None
        tokens_received = 0

        async with async_client.stream(
            "POST",
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line and not line.startswith("data: [DONE]"):
                    if first_token_time is None:
                        first_token_time = time.time()
                    tokens_received += 1

        time_to_first_token = first_token_time - start_time
        assert time_to_first_token < 2.0
        assert tokens_received > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        async def make_request(prompt: str) -> dict[str, Any]:
            response = await async_client.post(
                "/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 20,
                    "temperature": 0.5
                }
            )
            return response.json()

        prompts = [f"Count to {i}:" for i in range(1, 11)]

        start_time = time.time()
        results = await asyncio.gather(*[make_request(p) for p in prompts])
        total_time = time.time() - start_time

        assert len(results) == 10
        assert all("choices" in r for r in results)
        assert total_time < 30.0

    @pytest.mark.asyncio
    async def test_throughput(self, async_client):
        num_requests = 20
        tokens_generated = 0

        start_time = time.time()

        for i in range(num_requests):
            response = await async_client.post(
                "/v1/completions",
                json={
                    "prompt": f"Test prompt {i}",
                    "max_tokens": 10,
                    "temperature": 0.5
                }
            )
            assert response.status_code == 200
            data = response.json()
            tokens_generated += len(data["choices"][0]["text"].split())

        elapsed_time = time.time() - start_time
        throughput = tokens_generated / elapsed_time

        assert throughput > 10


class TestMemoryManagement:
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, async_client):
        initial_response = await async_client.get("/stats")
        initial_memory = initial_response.json()["gpu_memory_usage"]

        for _ in range(10):
            await async_client.post(
                "/v1/completions",
                json={
                    "prompt": "Memory test prompt",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )

        await asyncio.sleep(2)

        final_response = await async_client.get("/stats")
        final_memory = final_response.json()["gpu_memory_usage"]

        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000

    @pytest.mark.asyncio
    async def test_cache_efficiency(self, async_client):
        prompt = "The meaning of life is"

        first_response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.0,
                "seed": 42
            }
        )

        start_time = time.time()
        second_response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": 50,
                "temperature": 0.0,
                "seed": 42
            }
        )
        cached_latency = time.time() - start_time

        assert first_response.json()["choices"][0]["text"] == second_response.json()["choices"][0]["text"]
        assert cached_latency < 2.0


class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_handles_invalid_tokens(self, async_client):
        response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": "Test",
                "max_tokens": 10000,
                "temperature": 0.7
            }
        )
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert len(data["choices"][0]["text"]) <= 4096

    @pytest.mark.asyncio
    async def test_handles_empty_prompt(self, async_client):
        response = await async_client.post(
            "/v1/completions",
            json={
                "prompt": "",
                "max_tokens": 10,
                "temperature": 0.7
            }
        )
        assert response.status_code in [200, 400]

    @pytest.mark.asyncio
    async def test_graceful_overload_handling(self, async_client):
        tasks = []
        for i in range(50):
            task = async_client.post(
                "/v1/completions",
                json={
                    "prompt": f"Overload test {i}",
                    "max_tokens": 10,
                    "temperature": 0.5
                }
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        rate_limited = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 429)

        assert successful > 0
        assert successful + rate_limited == 50


class TestTokenization:
    @pytest.mark.asyncio
    async def test_tokenization_accuracy(self, async_client):
        test_texts = [
            "Hello, world!",
            "The quick brown fox",
            "12345",
            "🚀🌟✨"
        ]

        for text in test_texts:
            response = await async_client.post(
                "/v1/tokenize",
                json={"text": text}
            )
            assert response.status_code == 200
            data = response.json()

            assert len(data["tokens"]) > 0
            assert len(data["token_ids"]) == len(data["tokens"])

            detokenize_response = await async_client.post(
                "/v1/detokenize",
                json={"token_ids": data["token_ids"]}
            )
            assert detokenize_response.status_code == 200
            reconstructed = detokenize_response.json()["text"]

            assert text in reconstructed or reconstructed in text
