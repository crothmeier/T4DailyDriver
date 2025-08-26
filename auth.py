"""
Authentication and rate limiting module for vLLM service.
Provides API key authentication and per-key rate limiting.
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Initialize HTTP Bearer authentication
security = HTTPBearer(auto_error=False)

# Load API keys from environment variable
# Format: comma-separated list of keys, e.g., "key1,key2,key3"
API_KEYS = set()
api_keys_env = os.getenv("API_KEYS", "")
if api_keys_env:
    API_KEYS = {key.strip() for key in api_keys_env.split(",") if key.strip()}
    logger.info(f"Loaded {len(API_KEYS)} API keys from environment")
else:
    logger.warning("No API keys configured - authentication will be disabled")


class RateLimitManager:
    """Manages per-API-key rate limiting."""

    def __init__(self, default_limit: int = 100, window_minutes: int = 1):
        self.default_limit = default_limit
        self.window_minutes = window_minutes
        self.request_counts = defaultdict(list)
        # Allow configuring per-key limits via environment
        self.key_limits = self._load_key_limits()

    def _load_key_limits(self) -> dict:
        """Load per-key rate limits from environment.
        Format: KEY1:LIMIT1,KEY2:LIMIT2"""
        limits = {}
        limits_env = os.getenv("API_KEY_LIMITS", "")
        if limits_env:
            for pair in limits_env.split(","):
                if ":" in pair:
                    key, limit = pair.split(":", 1)
                    try:
                        limits[key.strip()] = int(limit.strip())
                    except ValueError:
                        logger.warning(f"Invalid rate limit for key {key}: {limit}")
        return limits

    def check_rate_limit(self, api_key: str) -> bool:
        """Check if the API key has exceeded its rate limit.
        Returns True if within limit, False if exceeded."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_minutes)

        # Clean old requests
        self.request_counts[api_key] = [
            timestamp for timestamp in self.request_counts[api_key] if timestamp > window_start
        ]

        # Get the limit for this key
        limit = self.key_limits.get(api_key, self.default_limit)

        # Check if under limit
        if len(self.request_counts[api_key]) >= limit:
            return False

        # Record this request
        self.request_counts[api_key].append(now)
        return True

    def get_remaining_requests(self, api_key: str) -> int:
        """Get the number of remaining requests for this key."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_minutes)

        # Clean old requests
        self.request_counts[api_key] = [
            timestamp for timestamp in self.request_counts[api_key] if timestamp > window_start
        ]

        limit = self.key_limits.get(api_key, self.default_limit)
        return max(0, limit - len(self.request_counts[api_key]))


# Initialize rate limit manager
rate_limit_manager = RateLimitManager(
    default_limit=int(os.getenv("DEFAULT_RATE_LIMIT", "100")),
    window_minutes=int(os.getenv("RATE_LIMIT_WINDOW_MINUTES", "1")),
)


def get_api_key(request: Request) -> str | None:
    """Extract API key from request headers or query parameters."""
    # Check Authorization header first
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    # Check X-API-Key header
    api_key_header = request.headers.get("X-API-Key", "")
    if api_key_header:
        return api_key_header

    # Check query parameter as fallback
    api_key_param = request.query_params.get("api_key", "")
    if api_key_param:
        return api_key_param

    return None


async def verify_api_key(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> str:
    """Verify API key and apply rate limiting.
    Returns the validated API key."""

    # Skip authentication if no keys are configured
    if not API_KEYS:
        return "default"

    # Get API key from various sources
    api_key = None
    if credentials and credentials.credentials:
        api_key = credentials.credentials
    else:
        api_key = get_api_key(request)

    if not api_key:
        raise HTTPException(status_code=401, detail="API key required", headers={"WWW-Authenticate": "Bearer"})

    # Verify the API key
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    # Check rate limit
    if not rate_limit_manager.check_rate_limit(api_key):
        remaining = rate_limit_manager.get_remaining_requests(api_key)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {rate_limit_manager.window_minutes} minute(s)",
            headers={
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Window": f"{rate_limit_manager.window_minutes}m",
                "Retry-After": str(rate_limit_manager.window_minutes * 60),
            },
        )

    # Add remaining requests to response headers
    request.state.api_key = api_key
    request.state.rate_limit_remaining = rate_limit_manager.get_remaining_requests(api_key)

    return api_key


# Create a custom key function for slowapi that uses API keys
def get_api_key_for_limiter(request: Request) -> str:
    """Get API key for rate limiting purposes."""
    api_key = get_api_key(request)
    if api_key and api_key in API_KEYS:
        return api_key
    # Fall back to IP address if no valid API key
    return get_remote_address(request)


# Initialize slowapi limiter with custom key function
limiter = Limiter(key_func=get_api_key_for_limiter, default_limits=[f"{os.getenv('DEFAULT_RATE_LIMIT', '100')}/minute"])


# Optional: Dependency for endpoints that don't require auth
async def optional_api_key(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> str | None:
    """Optional API key verification for metrics/health endpoints."""
    if not API_KEYS:
        return None

    api_key = None
    if credentials and credentials.credentials:
        api_key = credentials.credentials
    else:
        api_key = get_api_key(request)

    if api_key and api_key in API_KEYS:
        request.state.api_key = api_key
        return api_key

    return None
