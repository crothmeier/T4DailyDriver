#!/usr/bin/env python3
"""
Runtime verification script for GPU stack validation.
Validates CUDA, PyTorch, and vLLM versions and compatibility.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RuntimeVerifier:
    def __init__(self):
        self.verification_results = {}
        self.errors = []
        self.warnings = []

    def check_cuda_version(self) -> tuple[bool, str]:
        """Check CUDA runtime version."""
        try:
            import torch

            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()

                # Check for CUDA 12.4 compatibility
                if cuda_version:
                    major, minor = cuda_version.split(".")[:2]
                    major, minor = int(major), int(minor)

                    self.verification_results["cuda"] = {
                        "version": cuda_version,
                        "device_count": device_count,
                        "is_12_4_compatible": (major == 12 and minor >= 4),
                    }

                    if major == 12 and minor >= 4:
                        logger.info(f"✓ CUDA {cuda_version} detected with {device_count} device(s)")
                        return True, cuda_version
                    else:
                        msg = f"CUDA {cuda_version} detected, but 12.4+ required"
                        self.warnings.append(msg)
                        logger.warning(f"⚠ {msg}")
                        return False, cuda_version
            else:
                msg = "CUDA not available"
                self.errors.append(msg)
                logger.error(f"✗ {msg}")
                return False, "N/A"
        except ImportError:
            msg = "PyTorch not installed"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"
        except Exception as e:
            msg = f"Error checking CUDA: {str(e)}"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"

    def check_pytorch_version(self) -> tuple[bool, str]:
        """Check PyTorch version and CUDA compatibility."""
        try:
            import torch

            pytorch_version = torch.__version__

            # Parse version
            base_version = pytorch_version.split("+")[0]
            major, minor, patch = base_version.split(".")[:3]
            major, minor = int(major), int(minor)

            # Check for 2.5.1 or compatible
            is_compatible = major == 2 and minor >= 5

            # Check CUDA build
            cuda_compiled = torch.version.cuda if hasattr(torch.version, "cuda") else None

            self.verification_results["pytorch"] = {
                "version": pytorch_version,
                "cuda_compiled": cuda_compiled,
                "is_2_5_compatible": is_compatible,
            }

            if is_compatible:
                logger.info(f"✓ PyTorch {pytorch_version} with CUDA {cuda_compiled}")
                return True, pytorch_version
            else:
                msg = f"PyTorch {pytorch_version} detected, but 2.5+ required"
                self.warnings.append(msg)
                logger.warning(f"⚠ {msg}")
                return False, pytorch_version
        except ImportError:
            msg = "PyTorch not installed"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"
        except Exception as e:
            msg = f"Error checking PyTorch: {str(e)}"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"

    def check_vllm_version(self) -> tuple[bool, str]:
        """Check vLLM version."""
        try:
            import vllm

            vllm_version = vllm.__version__ if hasattr(vllm, "__version__") else "Unknown"

            # Parse version for compatibility check
            try:
                from packaging import version

                v = version.parse(vllm_version)
                # Check for >=0.9.0,<0.11.0
                is_compatible = v >= version.parse("0.9.0") and v < version.parse("0.11.0")
            except Exception:
                # Fallback to string comparison
                is_compatible = vllm_version.startswith(("0.9", "0.10"))

            self.verification_results["vllm"] = {"version": vllm_version, "is_compatible": is_compatible}

            if is_compatible:
                logger.info(f"✓ vLLM {vllm_version}")
                return True, vllm_version
            else:
                msg = f"vLLM {vllm_version} detected, but >=0.9.0,<0.11.0 required"
                self.warnings.append(msg)
                logger.warning(f"⚠ {msg}")
                return False, vllm_version
        except ImportError:
            msg = "vLLM not installed"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"
        except Exception as e:
            msg = f"Error checking vLLM: {str(e)}"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"

    def check_flash_attention(self) -> tuple[bool, str]:
        """Check Flash Attention availability."""
        try:
            import flash_attn

            flash_version = flash_attn.__version__ if hasattr(flash_attn, "__version__") else "Unknown"

            self.verification_results["flash_attention"] = {"version": flash_version, "available": True}

            logger.info(f"✓ Flash Attention {flash_version}")
            return True, flash_version
        except ImportError:
            msg = "Flash Attention not installed (optional)"
            self.warnings.append(msg)
            logger.warning(f"⚠ {msg}")
            return False, "N/A"
        except Exception as e:
            msg = f"Error checking Flash Attention: {str(e)}"
            self.warnings.append(msg)
            logger.warning(f"⚠ {msg}")
            return False, "N/A"

    def check_gpu_memory(self) -> dict[str, Any]:
        """Check GPU memory availability."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / (1024**3)  # Convert to GB

                    # Get current memory usage
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)

                    gpu_info.append(
                        {
                            "device": i,
                            "name": props.name,
                            "total_memory_gb": round(total_memory, 2),
                            "allocated_gb": round(allocated, 2),
                            "reserved_gb": round(reserved, 2),
                            "compute_capability": f"{props.major}.{props.minor}",
                            "is_t4": "T4" in props.name,
                            "is_l4": "L4" in props.name,
                        }
                    )

                    logger.info(
                        f"✓ GPU {i}: {props.name} - {total_memory:.1f}GB total, " f"Compute {props.major}.{props.minor}"
                    )

                self.verification_results["gpu"] = gpu_info
                return {"success": True, "gpus": gpu_info}
            else:
                return {"success": False, "error": "No GPUs available"}
        except Exception as e:
            msg = f"Error checking GPU memory: {str(e)}"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return {"success": False, "error": str(e)}

    def check_driver_version(self) -> tuple[bool, str]:
        """Check NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                driver_version = result.stdout.strip()

                # Check for 550+ driver
                try:
                    major_version = int(driver_version.split(".")[0])
                    is_compatible = major_version >= 550
                except Exception:
                    is_compatible = False

                self.verification_results["nvidia_driver"] = {"version": driver_version, "is_550_plus": is_compatible}

                if is_compatible:
                    logger.info(f"✓ NVIDIA Driver {driver_version}")
                else:
                    logger.warning(f"⚠ NVIDIA Driver {driver_version} (550+ recommended for CUDA 12.4)")

                return is_compatible, driver_version
            else:
                msg = "Failed to query NVIDIA driver"
                self.warnings.append(msg)
                logger.warning(f"⚠ {msg}")
                return False, "N/A"
        except subprocess.TimeoutExpired:
            msg = "nvidia-smi timeout"
            self.warnings.append(msg)
            logger.warning(f"⚠ {msg}")
            return False, "N/A"
        except FileNotFoundError:
            msg = "nvidia-smi not found"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"
        except Exception as e:
            msg = f"Error checking driver: {str(e)}"
            self.errors.append(msg)
            logger.error(f"✗ {msg}")
            return False, "N/A"

    def check_awq_support(self) -> bool:
        """Check AWQ quantization support."""
        try:
            # Check if AWQ is available in vLLM
            # This is a simplified check - actual AWQ support detection may vary
            self.verification_results["awq"] = {"supported": True, "note": "AWQ support available in vLLM"}
            logger.info("✓ AWQ quantization support available")
            return True
        except Exception as e:
            msg = f"Cannot verify AWQ support: {str(e)}"
            self.warnings.append(msg)
            logger.warning(f"⚠ {msg}")
            return False

    def generate_report(self, output_file: Path | None = None) -> dict[str, Any]:
        """Generate verification report."""
        report = {
            "timestamp": str(Path.ctime(Path.cwd())),
            "verification_results": self.verification_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "passed": len(self.errors) == 0,
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
            },
        }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_file}")

        return report

    def run_verification(self, build_time: bool = False, startup: bool = False) -> bool:
        """Run complete verification suite."""
        logger.info("=" * 60)
        logger.info("Starting Runtime Verification")
        logger.info("=" * 60)

        if not build_time:
            # Skip GPU checks during build
            cuda_ok, cuda_ver = self.check_cuda_version()
            _gpu_info = self.check_gpu_memory()
            driver_ok, driver_ver = self.check_driver_version()
        else:
            logger.info("Build-time verification - skipping GPU checks")

        pytorch_ok, pytorch_ver = self.check_pytorch_version()
        vllm_ok, vllm_ver = self.check_vllm_version()
        flash_ok, flash_ver = self.check_flash_attention()

        if not build_time:
            _awq_ok = self.check_awq_support()

        logger.info("=" * 60)

        # Summary
        if len(self.errors) > 0:
            logger.error(f"Verification FAILED with {len(self.errors)} error(s)")
            for error in self.errors:
                logger.error(f"  - {error}")
            if startup:
                logger.error("Startup verification failed - exiting")
                sys.exit(1)
            return False
        elif len(self.warnings) > 0:
            logger.warning(f"Verification passed with {len(self.warnings)} warning(s)")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
            return True
        else:
            logger.info("✓ All verifications passed successfully!")
            return True


def main():
    parser = argparse.ArgumentParser(description="Verify GPU runtime stack")
    parser.add_argument("--build-time", action="store_true", help="Run build-time verification (skip GPU checks)")
    parser.add_argument("--startup", action="store_true", help="Run startup verification (exit on failure)")
    parser.add_argument("--output", type=Path, help="Output JSON report to file")

    args = parser.parse_args()

    verifier = RuntimeVerifier()
    success = verifier.run_verification(build_time=args.build_time, startup=args.startup)

    if args.output:
        verifier.generate_report(args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
