"""Project test runner. Requires pytest because the suite uses pytest fixtures/APIs."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_with_pytest() -> int:
    try:
        import pytest
        print(f"[CryptoGraph Test Runner] Found pytest at {pytest.__file__}")
        print(f"[CryptoGraph Test Runner] Executing test suite with Python {sys.version.split()[0]}...\n")
        return pytest.main(["-v", str(REPO_ROOT / "tests")])
    except ImportError:
        return -1


def main():
    exit_code = run_with_pytest()
    if exit_code == -1:
        print(
            "[CryptoGraph Test Runner] pytest is required but is not installed. "
            "Install the project development dependencies with `pip install -e .[dev]`.",
            file=sys.stderr,
        )
        exit_code = 2
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
