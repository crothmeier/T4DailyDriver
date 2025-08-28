"""
vLLM warmup cache management with hash-based keys and automatic rotation.

This module provides utilities for managing vLLM warmup cache flags based on
runtime configuration. It generates deterministic hash keys from model and
runtime parameters, manages cache flag files, and automatically rotates stale
entries.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class WarmupConfig:
    """Configuration for warmup cache management."""

    model_path: str
    attention_backend: str
    max_seq_len_to_capture: int
    max_num_seqs: int
    quantization: str | None = None
    dtype: str = "auto"
    model_revision: str | None = None
    cache_dir: Path = Path("/cache/vllm")
    max_cache_age_days: int = 30
    max_cache_entries: int = 10

    @classmethod
    def from_env(cls) -> "WarmupConfig":
        """Create config from environment variables."""
        return cls(
            model_path=os.getenv("MODEL_PATH", ""),
            attention_backend=os.getenv("ATTENTION_BACKEND", "SDPA"),
            max_seq_len_to_capture=int(os.getenv("MAX_SEQ_LEN_TO_CAPTURE", "8192")),
            max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "256")),
            quantization=os.getenv("QUANTIZATION"),
            dtype=os.getenv("DTYPE", "auto"),
            model_revision=os.getenv("MODEL_REVISION"),
            cache_dir=Path(os.getenv("VLLM_CACHE_DIR", "/cache/vllm")),
            max_cache_age_days=int(os.getenv("MAX_CACHE_AGE_DAYS", "30")),
            max_cache_entries=int(os.getenv("MAX_CACHE_ENTRIES", "10")),
        )

    @classmethod
    def from_engine_args(cls, engine_args: Any) -> "WarmupConfig":
        """Create config from vLLM engine args object."""
        return cls(
            model_path=getattr(engine_args, "model", ""),
            attention_backend=getattr(engine_args, "attention_backend", "SDPA"),
            max_seq_len_to_capture=getattr(engine_args, "max_seq_len_to_capture", 8192),
            max_num_seqs=getattr(engine_args, "max_num_seqs", 256),
            quantization=getattr(engine_args, "quantization", None),
            dtype=getattr(engine_args, "dtype", "auto"),
            model_revision=getattr(engine_args, "revision", None),
            cache_dir=Path(getattr(engine_args, "download_dir", "/cache/vllm")),
        )


class WarmupCacheManager:
    """Manages vLLM warmup cache flags with automatic rotation."""

    def __init__(self, config: WarmupConfig | None = None):
        """
        Initialize warmup cache manager.

        Args:
            config: WarmupConfig instance. If None, creates from environment.
        """
        self.config = config or WarmupConfig.from_env()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        try:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cache directory {self.config.cache_dir}: {e}")
            raise

    def generate_warmup_key(self) -> str:
        """
        Generate deterministic hash key for warmup configuration.

        Returns:
            Hexadecimal hash string representing the configuration.
        """
        # Get CUDA version
        cuda_version = "unknown"
        if HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda or "unknown"
            except Exception as e:
                logger.warning(f"Failed to get CUDA version: {e}")

        # Build key components
        key_parts = [
            f"model:{self.config.model_path}",
            f"attention:{self.config.attention_backend}",
            f"max_seq_len:{self.config.max_seq_len_to_capture}",
            f"max_num_seqs:{self.config.max_num_seqs}",
            f"quantization:{self.config.quantization or 'none'}",
            f"dtype:{self.config.dtype}",
            f"cuda:{cuda_version}",
        ]

        # Add model revision if present
        if self.config.model_revision:
            key_parts.append(f"revision:{self.config.model_revision}")

        # Generate SHA256 hash
        key_string = "|".join(key_parts)
        hash_object = hashlib.sha256(key_string.encode())
        hash_hex = hash_object.hexdigest()[:16]  # Use first 16 chars for brevity

        logger.debug(f"Generated warmup key {hash_hex} for: {key_string}")
        return hash_hex

    def get_warmup_flag_path(self, key: str | None = None) -> Path:
        """
        Get path to warmup flag file.

        Args:
            key: Warmup key. If None, generates from current config.

        Returns:
            Path to warmup flag file.
        """
        if key is None:
            key = self.generate_warmup_key()
        return self.config.cache_dir / f".warmup_{key}"

    def is_warmed_up(self, key: str | None = None) -> bool:
        """
        Check if warmup has been completed for configuration.

        Args:
            key: Warmup key. If None, generates from current config.

        Returns:
            True if warmup flag exists, False otherwise.
        """
        flag_path = self.get_warmup_flag_path(key)
        exists = flag_path.exists()

        if exists:
            logger.info(f"Warmup flag found: {flag_path}")
        else:
            logger.info(f"No warmup flag at: {flag_path}")

        return exists

    def mark_warmed_up(self, key: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        """
        Mark configuration as warmed up by creating flag file.

        Args:
            key: Warmup key. If None, generates from current config.
            metadata: Optional metadata to store in flag file.
        """
        flag_path = self.get_warmup_flag_path(key)

        try:
            # Write metadata if provided, otherwise just touch file
            if metadata:
                import json

                metadata["timestamp"] = time.time()
                metadata["config"] = {
                    "model_path": self.config.model_path,
                    "attention_backend": self.config.attention_backend,
                    "max_seq_len_to_capture": self.config.max_seq_len_to_capture,
                    "max_num_seqs": self.config.max_num_seqs,
                    "quantization": self.config.quantization,
                    "dtype": self.config.dtype,
                    "model_revision": self.config.model_revision,
                }
                flag_path.write_text(json.dumps(metadata, indent=2))
            else:
                flag_path.touch()

            logger.info(f"Created warmup flag: {flag_path}")

            # Trigger rotation after creating new flag
            self.rotate_stale_flags()

        except Exception as e:
            logger.error(f"Failed to create warmup flag {flag_path}: {e}")
            raise

    def clear_warmup(self, key: str | None = None) -> bool:
        """
        Remove warmup flag for configuration.

        Args:
            key: Warmup key. If None, generates from current config.

        Returns:
            True if flag was removed, False if it didn't exist.
        """
        flag_path = self.get_warmup_flag_path(key)

        try:
            if flag_path.exists():
                flag_path.unlink()
                logger.info(f"Removed warmup flag: {flag_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove warmup flag {flag_path}: {e}")
            raise

    def rotate_stale_flags(self) -> int:
        """
        Remove old warmup flags based on age and count limits.

        Returns:
            Number of flags removed.
        """
        try:
            # Get all warmup flags
            flags = list(self.config.cache_dir.glob(".warmup_*"))

            if not flags:
                return 0

            # Sort by modification time (oldest first)
            flags.sort(key=lambda p: p.stat().st_mtime)

            removed_count = 0
            current_time = time.time()
            max_age_seconds = self.config.max_cache_age_days * 86400

            # Remove flags that are too old
            for flag in flags:
                age = current_time - flag.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        flag.unlink()
                        logger.info(f"Removed stale warmup flag (age={age/86400:.1f} days): {flag}")
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove stale flag {flag}: {e}")

            # Remove excess flags if over limit
            remaining_flags = list(self.config.cache_dir.glob(".warmup_*"))
            if len(remaining_flags) > self.config.max_cache_entries:
                # Sort again and remove oldest
                remaining_flags.sort(key=lambda p: p.stat().st_mtime)
                excess_count = len(remaining_flags) - self.config.max_cache_entries

                for flag in remaining_flags[:excess_count]:
                    try:
                        flag.unlink()
                        logger.info(f"Removed excess warmup flag: {flag}")
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove excess flag {flag}: {e}")

            if removed_count > 0:
                logger.info(f"Rotated {removed_count} stale/excess warmup flags")

            return removed_count

        except Exception as e:
            logger.error(f"Failed to rotate stale flags: {e}")
            return 0

    def list_warmup_flags(self) -> list[dict[str, Any]]:
        """
        List all warmup flags with their metadata.

        Returns:
            List of dictionaries containing flag information.
        """
        flags_info = []

        try:
            flags = list(self.config.cache_dir.glob(".warmup_*"))

            for flag in flags:
                info = {
                    "path": str(flag),
                    "key": flag.stem.replace(".warmup_", ""),
                    "size": flag.stat().st_size,
                    "modified": flag.stat().st_mtime,
                    "age_days": (time.time() - flag.stat().st_mtime) / 86400,
                }

                # Try to read metadata if it's JSON
                try:
                    import json

                    content = flag.read_text()
                    if content:
                        metadata = json.loads(content)
                        info["metadata"] = metadata
                except Exception:
                    pass

                flags_info.append(info)

            # Sort by modification time (newest first)
            flags_info.sort(key=lambda x: x["modified"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list warmup flags: {e}")

        return flags_info

    def clear_all_flags(self) -> int:
        """
        Remove all warmup flags.

        Returns:
            Number of flags removed.
        """
        try:
            flags = list(self.config.cache_dir.glob(".warmup_*"))
            removed_count = 0

            for flag in flags:
                try:
                    flag.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove flag {flag}: {e}")

            if removed_count > 0:
                logger.info(f"Cleared {removed_count} warmup flags")

            return removed_count

        except Exception as e:
            logger.error(f"Failed to clear all flags: {e}")
            return 0


def create_manager_from_source(source: dict[str, Any] | Any | None = None) -> WarmupCacheManager:
    """
    Create WarmupCacheManager from various sources.

    Args:
        source: Can be None (uses env), dict (env-like), or engine args object.

    Returns:
        Configured WarmupCacheManager instance.
    """
    if source is None:
        # Use environment variables
        config = WarmupConfig.from_env()
    elif isinstance(source, dict):
        # Create from dictionary (useful for testing)
        config = WarmupConfig(
            model_path=source.get("MODEL_PATH", ""),
            attention_backend=source.get("ATTENTION_BACKEND", "SDPA"),
            max_seq_len_to_capture=int(source.get("MAX_SEQ_LEN_TO_CAPTURE", 8192)),
            max_num_seqs=int(source.get("MAX_NUM_SEQS", 256)),
            quantization=source.get("QUANTIZATION"),
            dtype=source.get("DTYPE", "auto"),
            model_revision=source.get("MODEL_REVISION"),
            cache_dir=Path(source.get("VLLM_CACHE_DIR", "/cache/vllm")),
        )
    else:
        # Assume it's an engine args object
        config = WarmupConfig.from_engine_args(source)

    return WarmupCacheManager(config)


# Example usage for direct execution
if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create manager from environment
    manager = create_manager_from_source()

    # Command line interface
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "check":
            key = manager.generate_warmup_key()
            print(f"Warmup key: {key}")
            print(f"Is warmed up: {manager.is_warmed_up()}")

        elif command == "mark":
            manager.mark_warmed_up()
            print("Marked as warmed up")

        elif command == "clear":
            if manager.clear_warmup():
                print("Cleared warmup flag")
            else:
                print("No warmup flag to clear")

        elif command == "rotate":
            removed = manager.rotate_stale_flags()
            print(f"Removed {removed} stale flags")

        elif command == "list":
            flags = manager.list_warmup_flags()
            print(f"Found {len(flags)} warmup flags:")
            for flag in flags:
                print(f"  - {flag['key']}: {flag['age_days']:.1f} days old")

        elif command == "clear-all":
            removed = manager.clear_all_flags()
            print(f"Cleared {removed} flags")

        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, mark, clear, rotate, list, clear-all")
            sys.exit(1)
    else:
        # Show current status
        key = manager.generate_warmup_key()
        print("Warmup cache manager initialized")
        print(f"Cache directory: {manager.config.cache_dir}")
        print(f"Current warmup key: {key}")
        print(f"Is warmed up: {manager.is_warmed_up()}")

        flags = manager.list_warmup_flags()
        print(f"Total warmup flags: {len(flags)}")
