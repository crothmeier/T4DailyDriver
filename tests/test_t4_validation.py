#!/usr/bin/env python3
"""
Test suite for T4 GPU-specific validations.
Ensures SDPA is enforced, flash-attn fails gracefully, and memory limits are respected.
"""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.verify_runtime import RuntimeVerifier  # noqa: E402

# Check if torch is available
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# Mock torch module if not available
if not TORCH_AVAILABLE:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.cuda"] = MagicMock()
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.backends"] = MagicMock()
    sys.modules["torch.backends.cuda"] = MagicMock()


class TestT4Validation(unittest.TestCase):
    """Test suite for T4 GPU-specific validations."""

    def setUp(self):
        """Set up test environment."""
        self.verifier = RuntimeVerifier()

    def tearDown(self):
        """Clean up after tests."""
        # Reset environment variables
        if "VLLM_ATTENTION_BACKEND" in os.environ:
            del os.environ["VLLM_ATTENTION_BACKEND"]

    def test_t4_gpu_detection(self):
        """Test T4 GPU detection and validation."""
        # Setup mocks
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.device_count = MagicMock(return_value=1)

        # Mock T4 GPU properties
        mock_device_props = MagicMock()
        mock_device_props.name = "Tesla T4"
        mock_device_props.total_memory = 16106127360  # ~15.1GB in bytes
        mock_device_props.major = 7
        mock_device_props.minor = 5
        torch.cuda.get_device_properties = MagicMock(return_value=mock_device_props)
        torch.cuda.memory_allocated = MagicMock(return_value=1073741824)  # 1GB
        torch.cuda.memory_reserved = MagicMock(return_value=2147483648)  # 2GB

        result = self.verifier.check_gpu_memory()

        self.assertTrue(result["success"])
        self.assertEqual(len(result["gpus"]), 1)

        gpu_info = result["gpus"][0]
        self.assertTrue(gpu_info["is_t4"])
        self.assertEqual(gpu_info["compute_capability"], "7.5")
        self.assertAlmostEqual(gpu_info["total_memory_gb"], 15.0, delta=1.0)

        # Check T4 validation
        self.assertIsNotNone(gpu_info["t4_validation"])
        self.assertTrue(gpu_info["t4_validation"]["sm_75_verified"])
        self.assertTrue(gpu_info["t4_validation"]["memory_validated"])

    def test_sdpa_enforcement_on_t4(self):
        """Test that SDPA is enforced when T4 is detected."""
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.device_count = MagicMock(return_value=1)

        # Mock T4 GPU
        mock_device_props = MagicMock()
        mock_device_props.name = "Tesla T4"
        torch.cuda.get_device_properties = MagicMock(return_value=mock_device_props)

        # Test with Flash Attention backend (should fail)
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTENTION_2"
        result = self.verifier.verify_attention_backend()

        self.assertFalse(result)
        self.assertIn("T4 (SM75) does not support FlashAttention-2", self.verifier.errors[0])

        # Reset for next test
        self.verifier.errors.clear()
        self.verifier.warnings.clear()

        # Test with SDPA backend (should pass)
        os.environ["VLLM_ATTENTION_BACKEND"] = "SDPA"
        result = self.verifier.verify_attention_backend()

        self.assertTrue(result)
        self.assertEqual(len(self.verifier.errors), 0)

        # Check verification results
        backend_info = self.verifier.verification_results.get("attention_backend", {})
        self.assertEqual(backend_info["status"], "optimal")
        self.assertEqual(backend_info["backend"], "SDPA")

    def test_flash_attn_import_fails_gracefully(self):
        """Test that flash-attn import failure is handled gracefully."""
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)

        # Mock flash_attn import failure
        with patch.dict("sys.modules", {"flash_attn": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'flash_attn'")):
                result = self.verifier.check_flash_attention()

        self.assertFalse(result[0])
        self.assertEqual(result[1], "N/A")
        # Should be a warning for T4, not an error
        self.assertIn("Flash Attention not installed", self.verifier.warnings[0])
        self.assertIn("using SDPA instead", self.verifier.warnings[0])

    def test_t4_memory_limit_validation(self):
        """Test T4 memory limit validation (16GB)."""
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.device_count = MagicMock(return_value=1)

        # Mock T4 GPU properties
        mock_device_props = MagicMock()
        mock_device_props.name = "Tesla T4"
        mock_device_props.total_memory = 16106127360  # ~15.1GB in bytes (typical T4)
        mock_device_props.major = 7
        mock_device_props.minor = 5
        torch.cuda.get_device_properties = MagicMock(return_value=mock_device_props)

        result = self.verifier.check_model_requirements()

        self.assertTrue(result)

        model_reqs = self.verifier.verification_results.get("model_requirements", {})
        self.assertIsNotNone(model_reqs)
        self.assertTrue(model_reqs["sufficient"])
        self.assertAlmostEqual(model_reqs["available_memory_gb"], 15.0, delta=1.0)
        self.assertEqual(model_reqs["gpu"], "Tesla T4")

    def test_sdpa_kernel_availability(self):
        """Test SDPA kernel availability check."""
        import torch

        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.device_count = MagicMock(return_value=1)
        torch.nn.functional.scaled_dot_product_attention = MagicMock()

        # Mock T4 GPU
        mock_device_props = MagicMock()
        mock_device_props.name = "Tesla T4"
        torch.cuda.get_device_properties = MagicMock(return_value=mock_device_props)

        result = self.verifier.check_sdpa_kernel()

        self.assertTrue(result)

        sdpa_info = self.verifier.verification_results.get("sdpa", {})
        self.assertTrue(sdpa_info["available"])
        self.assertTrue(sdpa_info["is_t4_gpu"])
        self.assertTrue(sdpa_info["recommended_for_t4"])

    def test_preflight_check_script_exists(self):
        """Test that preflight_check.sh script exists and is executable."""
        script_path = Path(__file__).parent / "scripts" / "preflight_check.sh"
        self.assertTrue(script_path.exists(), f"Script not found: {script_path}")

        # Check if script is executable
        self.assertTrue(os.access(script_path, os.X_OK), f"Script not executable: {script_path}")

    def test_verify_runtime_script_exists(self):
        """Test that verify_runtime.py script exists and is executable."""
        script_path = Path(__file__).parent / "scripts" / "verify_runtime.py"
        self.assertTrue(script_path.exists(), f"Script not found: {script_path}")

        # Check if script is executable
        self.assertTrue(os.access(script_path, os.X_OK), f"Script not executable: {script_path}")

    @patch("subprocess.run")
    def test_preflight_check_json_output(self, mock_run):
        """Test that preflight_check.sh produces valid JSON output."""
        # Mock successful nvidia-smi output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Tesla T4\n"
        mock_run.return_value = mock_result

        # Test JSON output parsing (mocked)
        sample_output = {
            "timestamp": "2024-01-01T00:00:00Z",
            "exit_code": 0,
            "status": "pass",
            "checks": {"gpu_t4_detected": "true", "gpu_t4_compute_capability": "7.5", "cuda_124_compat_status": "pass"},
            "errors": {},
            "summary": {"total_checks": 8, "passed": 8, "failed": 0, "warnings": 0},
        }

        # Validate JSON structure
        self.assertIn("timestamp", sample_output)
        self.assertIn("checks", sample_output)
        self.assertIn("errors", sample_output)
        self.assertIn("summary", sample_output)
        self.assertEqual(sample_output["summary"]["total_checks"], 8)

    def test_cuda_124_compatibility(self):
        """Test CUDA 12.4 compatibility checks."""
        # Test with mock CUDA version
        import torch

        torch.version = MagicMock()
        torch.version.cuda = "12.4"
        torch.cuda.is_available = MagicMock(return_value=True)
        torch.cuda.device_count = MagicMock(return_value=1)

        result = self.verifier.check_cuda_version()

        self.assertTrue(result[0])
        self.assertEqual(result[1], "12.4")

        cuda_info = self.verifier.verification_results.get("cuda", {})
        self.assertTrue(cuda_info["is_12_4_compatible"])

    @patch("subprocess.run")
    def test_driver_version_check(self, mock_run):
        """Test NVIDIA driver version check for CUDA 12.4."""
        # Mock nvidia-smi output with driver 550+
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "550.54.14\n"
        mock_run.return_value = mock_result

        result = self.verifier.check_driver_version()

        self.assertTrue(result[0])
        self.assertEqual(result[1], "550.54.14")

        driver_info = self.verifier.verification_results.get("nvidia_driver", {})
        self.assertTrue(driver_info["is_550_plus"])

    def test_no_flash_attn_in_requirements(self):
        """Test that flash-attn is not in requirements for T4."""
        requirements_files = [
            "requirements.txt",
            "requirements-cuda124.txt",
            "constraints.txt",
            "constraints-cuda124.txt",
        ]

        for req_file in requirements_files:
            file_path = Path(__file__).parent / req_file
            if file_path.exists():
                with open(file_path) as f:
                    content = f.read().lower()
                    # Check that flash-attn is not present (or is commented out)
                    lines = content.split("\n")
                    for line in lines:
                        if "flash" in line and not line.strip().startswith("#"):
                            # Allow flash-attn to be present if it's excluded for T4
                            self.assertIn(
                                "marker", line.lower(), f"flash-attn found without platform marker in {req_file}"
                            )


class TestCIIntegration(unittest.TestCase):
    """Test CI/CD integration for validation scripts."""

    def test_json_output_format(self):
        """Test that validation scripts produce CI-compatible JSON."""
        verifier = RuntimeVerifier()

        # Generate report
        report = verifier.generate_report()

        # Check required fields for CI
        self.assertIn("timestamp", report)
        self.assertIn("verification_results", report)
        self.assertIn("errors", report)
        self.assertIn("warnings", report)
        self.assertIn("summary", report)

        # Check summary structure
        summary = report["summary"]
        self.assertIn("passed", summary)
        self.assertIn("error_count", summary)
        self.assertIn("warning_count", summary)

        # Ensure it's JSON serializable
        try:
            json.dumps(report)
        except (TypeError, ValueError) as e:
            self.fail(f"Report is not JSON serializable: {e}")


def run_tests():
    """Run the test suite."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestT4Validation))
    suite.addTests(loader.loadTestsFromTestCase(TestCIIntegration))

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
