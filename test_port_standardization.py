#!/usr/bin/env python3
"""
Guard test to ensure port 8000 is not used in the codebase.
This test enforces the port standardization to 8080.
"""

import subprocess
import sys


def test_no_port_8000_references():
    """Ensure no references to port 8000 exist in the codebase."""
    # Run git grep to find any references to port 8000
    # Exclude changelog, migration files, and this test file itself
    try:
        result = subprocess.run(
            [
                "git",
                "grep",
                "-nE",
                r"(^|[^0-9])8000([^0-9]|$)",
                "--",
                ".",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            # Filter out allowed files
            lines = result.stdout.strip().split("\n")
            violations = []
            allowed_patterns = ["CHANGELOG", "MIGRATION", "test_port_standardization.py", "scripts/validate_ports.sh"]

            for line in lines:
                if line and not any(pattern in line for pattern in allowed_patterns):
                    violations.append(line)

            if violations:
                print("❌ FAIL: Found references to port 8000 in the following files:")
                for violation in violations:
                    print(f"  {violation}")
                print("\nAll ports should be standardized to 8080.")
                sys.exit(1)

        print("✅ PASS: No references to port 8000 found (port standardization verified)")

    except subprocess.SubprocessError as e:
        print(f"Error running git grep: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_no_port_8000_references()
