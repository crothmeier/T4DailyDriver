#!/usr/bin/env python3
"""
Pre-download AWQ model with retry logic and progress reporting.
This script downloads the model during Docker build to enable proper caching.
"""

import logging
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class ModelDownloader:
    """Downloads and caches Hugging Face models with retry logic."""

    def __init__(
        self,
        model_id: str,
        cache_dir: str | None = None,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ):
        """
        Initialize the model downloader.

        Args:
            model_id: The Hugging Face model ID to download
            cache_dir: Directory to cache the model (defaults to HF_HOME)
            max_retries: Maximum number of download attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
        """
        self.model_id = model_id
        self.cache_dir = cache_dir or os.getenv("HF_HOME", "/cache/hf")
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff time with jitter."""
        backoff = min(self.initial_backoff * (2**attempt), self.max_backoff)
        # Add jitter (±20%)
        import random

        jitter = backoff * 0.2 * (2 * random.random() - 1)
        return max(0.1, backoff + jitter)

    def download_with_retry(self) -> bool:
        """
        Download the model with exponential backoff retry logic.

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading model {self.model_id} (attempt {attempt + 1}/{self.max_retries})")

                # Download the model snapshot
                snapshot_path = snapshot_download(
                    repo_id=self.model_id,
                    cache_dir=self.cache_dir,
                    resume_download=True,  # Resume partial downloads
                    local_files_only=False,
                    revision="main",
                    ignore_patterns=["*.md", "*.txt", ".git*"],  # Skip unnecessary files
                )

                logger.info(f"✓ Model downloaded successfully to: {snapshot_path}")

                # Verify essential files exist
                essential_files = [
                    "config.json",
                    "tokenizer_config.json",
                ]

                snapshot_path_obj = Path(snapshot_path)
                missing_files = []

                for file in essential_files:
                    if not (snapshot_path_obj / file).exists():
                        missing_files.append(file)

                if missing_files:
                    logger.warning(f"Missing essential files: {missing_files}")
                    logger.warning("Model may not be complete, but continuing...")

                # Check for model weights (various formats)
                weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
                has_weights = any(list(snapshot_path_obj.glob(pattern)) for pattern in weight_patterns)

                if not has_weights:
                    raise ValueError("No model weight files found!")

                # Report cache size
                total_size = sum(f.stat().st_size for f in snapshot_path_obj.rglob("*") if f.is_file())
                size_gb = total_size / (1024**3)
                logger.info(f"Total model size: {size_gb:.2f} GB")

                return True

            except RepositoryNotFoundError:
                logger.error(f"Model {self.model_id} not found on Hugging Face Hub")
                return False

            except HfHubHTTPError as e:
                if e.response.status_code == 401:
                    logger.error("Authentication required. Please set HF_TOKEN environment variable.")
                    return False
                elif e.response.status_code == 403:
                    logger.error("Access denied. Check your permissions for this model.")
                    return False
                else:
                    logger.warning(f"HTTP error {e.response.status_code}: {str(e)}")

            except ConnectionError as e:
                logger.warning(f"Connection error: {str(e)}")

            except Exception as e:
                logger.warning(f"Unexpected error: {str(e)}")

            # If we haven't returned, we need to retry
            if attempt < self.max_retries - 1:
                backoff_time = self._calculate_backoff(attempt)
                logger.info(f"Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
            else:
                logger.error(f"Failed to download model after {self.max_retries} attempts")

        return False

    def verify_cache(self) -> bool:
        """
        Verify that the model is already cached.

        Returns:
            True if model is cached and valid, False otherwise
        """
        try:
            from huggingface_hub import cached_download, hf_hub_url

            # Check if config.json is cached
            config_url = hf_hub_url(repo_id=self.model_id, filename="config.json")
            config_path = cached_download(
                config_url, cache_dir=self.cache_dir, force_download=False, resume_download=False, local_files_only=True
            )

            if config_path and Path(config_path).exists():
                logger.info(f"✓ Model {self.model_id} is already cached")
                return True

        except Exception as e:
            logger.debug(f"Cache check failed: {str(e)}")

        return False


def main():
    """Main entry point for the download script."""
    # Get model ID from environment or use default
    model_id = os.getenv("MODEL_PATH", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
    cache_dir = os.getenv("HF_HOME", "/cache/hf")

    # Parse command line arguments
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    if len(sys.argv) > 2:
        cache_dir = sys.argv[2]

    logger.info("=" * 60)
    logger.info("Model Pre-Download Script")
    logger.info("=" * 60)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Cache directory: {cache_dir}")

    # Create downloader
    downloader = ModelDownloader(
        model_id=model_id, cache_dir=cache_dir, max_retries=5, initial_backoff=2.0, max_backoff=60.0
    )

    # Check if already cached
    if downloader.verify_cache():
        logger.info("Model is already cached, skipping download")
        return 0

    # Download the model
    success = downloader.download_with_retry()

    if success:
        logger.info("=" * 60)
        logger.info("✓ Model download completed successfully!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("✗ Model download failed!")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
