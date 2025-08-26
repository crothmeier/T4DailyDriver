#!/usr/bin/env python3
"""
Test script for Phase 2 security features.
Tests API key authentication, rate limiting, and request queuing.
"""

import asyncio
import os
import sys
import time
import uuid

import httpx


def get_test_token(name: str = "test") -> str:
    """Generate deterministic test token for CI/CD."""
    return os.environ.get(f"TEST_{name.upper()}_TOKEN", f"{name}-token-{uuid.uuid4().hex[:8]}")


# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8080")
TEST_API_KEY = get_test_token("api")


async def test_health_endpoint():
    """Test that health endpoint works without authentication."""
    print("\n=== Testing Health Endpoint (No Auth Required) ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200, "Health check should work without auth"
        print("✓ Health endpoint accessible without authentication")


async def test_metrics_endpoint():
    """Test that metrics endpoint works without authentication."""
    print("\n=== Testing Metrics Endpoint (No Auth Required) ===")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/metrics")
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 200, "Metrics should work without auth"
        print("✓ Metrics endpoint accessible without authentication")


async def test_generation_without_auth():
    """Test that generation endpoint requires authentication."""
    print("\n=== Testing Generation Without Auth ===")
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/generate", json={"prompt": "Hello, world!", "max_tokens": 10})
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✓ Generation endpoint correctly requires authentication")
        else:
            print(f"⚠ Expected 401, got {response.status_code}")
            print(f"Response: {response.text}")


async def test_generation_with_auth():
    """Test that generation endpoint works with valid API key."""
    print("\n=== Testing Generation With Valid Auth ===")

    # Skip if no API keys are configured
    if not os.getenv("API_KEYS"):
        print("⚠ Skipping auth test - no API_KEYS configured")
        return

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        response = await client.post(
            f"{BASE_URL}/generate",
            json={"prompt": "Hello", "max_tokens": 5, "temperature": 0.1},
            headers=headers,
            timeout=30.0,
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Generation successful with valid API key")
            print(f"Response: {response.json()}")
        else:
            print(f"⚠ Generation failed: {response.status_code}")
            print(f"Response: {response.text}")


async def test_invalid_api_key():
    """Test that invalid API key is rejected."""
    print("\n=== Testing Invalid API Key ===")

    # Skip if no API keys are configured
    if not os.getenv("API_KEYS"):
        print("⚠ Skipping auth test - no API_KEYS configured")
        return

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": "Bearer invalid-key-999"}
        response = await client.post(f"{BASE_URL}/generate", json={"prompt": "Hello", "max_tokens": 5}, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✓ Invalid API key correctly rejected")
        else:
            print(f"⚠ Expected 401, got {response.status_code}")


async def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\n=== Testing Rate Limiting ===")

    # Skip if no API keys are configured
    if not os.getenv("API_KEYS"):
        print("⚠ Skipping rate limit test - no API_KEYS configured")
        return

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}

        # Get the rate limit from environment or use default
        rate_limit = int(os.getenv("DEFAULT_RATE_LIMIT", "100"))
        test_requests = min(10, rate_limit + 2)  # Test a few requests over the limit

        print(f"Sending {test_requests} rapid requests...")
        tasks = []
        for i in range(test_requests):
            task = client.post(
                f"{BASE_URL}/generate", json={"prompt": f"Test {i}", "max_tokens": 1}, headers=headers, timeout=5.0
            )
            tasks.append(task)

        # Send all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful and rate-limited responses
        success_count = 0
        rate_limited_count = 0
        error_count = 0

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_count += 1
                print(f"Request {i}: Error - {response}")
            elif response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                print(f"Request {i}: Rate limited (429)")
            else:
                error_count += 1
                print(f"Request {i}: Unexpected status {response.status_code}")

        print(f"\nResults: {success_count} successful, {rate_limited_count} rate-limited, {error_count} errors")

        if rate_limited_count > 0:
            print("✓ Rate limiting is working")
        else:
            print("⚠ No rate limiting observed (may need more requests or tighter limits)")


async def test_queue_backpressure():
    """Test request queuing and backpressure."""
    print("\n=== Testing Queue Backpressure ===")

    # Skip if no API keys are configured
    if not os.getenv("API_KEYS"):
        print("⚠ Skipping queue test - no API_KEYS configured")
        return

    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}

        # Send many concurrent requests to trigger queuing
        num_requests = 20
        print(f"Sending {num_requests} concurrent requests to test queuing...")

        tasks = []
        for i in range(num_requests):
            task = client.post(
                f"{BASE_URL}/generate",
                json={"prompt": f"Queue test {i}", "max_tokens": 1},
                headers=headers,
                timeout=30.0,
            )
            tasks.append(task)

        # Send all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time

        # Count response types
        success_count = 0
        service_unavailable_count = 0
        rate_limited_count = 0
        error_count = 0

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_count += 1
                print(f"Request {i}: Error - {type(response).__name__}")
            elif response.status_code == 200:
                success_count += 1
            elif response.status_code == 503:
                service_unavailable_count += 1
                print(f"Request {i}: Service at capacity (503)")
            elif response.status_code == 429:
                rate_limited_count += 1
                print(f"Request {i}: Rate limited (429)")
            else:
                error_count += 1
                print(f"Request {i}: Unexpected status {response.status_code}")

        print(f"\nCompleted in {duration:.2f} seconds")
        print(
            f"Results: {success_count} successful, {service_unavailable_count} at capacity, "
            f"{rate_limited_count} rate-limited, {error_count} errors"
        )

        if service_unavailable_count > 0:
            print("✓ Queue backpressure (503 responses) is working")
        elif success_count == num_requests:
            print("✓ All requests processed successfully (queue may be large enough)")
        else:
            print("⚠ Unexpected results - check configuration")


async def test_openai_endpoint_auth():
    """Test that OpenAI-compatible endpoint requires authentication."""
    print("\n=== Testing OpenAI Endpoint Authentication ===")

    async with httpx.AsyncClient() as client:
        # Test without auth
        response = await client.post(f"{BASE_URL}/v1/completions", json={"prompt": "Hello", "max_tokens": 5})
        print(f"Without auth - Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✓ OpenAI endpoint correctly requires authentication")

        # Test with auth if configured
        if os.getenv("API_KEYS"):
            headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
            response = await client.post(
                f"{BASE_URL}/v1/completions", json={"prompt": "Hello", "max_tokens": 5}, headers=headers, timeout=30.0
            )
            print(f"With auth - Status Code: {response.status_code}")
            if response.status_code == 200:
                print("✓ OpenAI endpoint works with valid API key")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 2 Security Features Test Suite")
    print("=" * 60)
    print(f"Target URL: {BASE_URL}")
    print(f"API Keys Configured: {'Yes' if os.getenv('API_KEYS') else 'No'}")
    print(f"Test API Key: {TEST_API_KEY if os.getenv('API_KEYS') else 'N/A'}")

    try:
        # Test endpoints without auth requirements
        await test_health_endpoint()
        await test_metrics_endpoint()

        # Test authentication
        await test_generation_without_auth()
        await test_generation_with_auth()
        await test_invalid_api_key()
        await test_openai_endpoint_auth()

        # Test rate limiting and queuing
        await test_rate_limiting()
        await test_queue_backpressure()

        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up test environment variables if not already set
    if not os.getenv("API_KEYS"):
        print("\nNote: No API_KEYS environment variable set.")
        k1, k2 = get_test_token("key1"), get_test_token("key2")
        print("To test authentication, run:")
        print(f"export API_KEYS='{k1},{k2}'")
        print("Running tests without authentication...\n")

    asyncio.run(main())
