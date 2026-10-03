"""Universal test runner for CryptoGraph-Secure.

Executes all unit and integration tests using pytest if available,
or falls back seamlessly to Python's standard library unittest loader
so tests can be verified in any standard Python environment without
requiring external test runner packages.
"""

import sys
import os
from pathlib import Path
import importlib.util
import unittest
import inspect
import tempfile
import shutil

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


def run_with_unittest() -> int:
    print("[CryptoGraph Test Runner] pytest not found in current environment.")
    print(f"[CryptoGraph Test Runner] Executing via standard library unittest on Python {sys.version.split()[0]}...\n")

    suite = unittest.TestSuite()
    tests_dir = REPO_ROOT / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))

    total_discovered = 0
    for test_file in test_files:
        module_name = f"tests.{test_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, test_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[ERROR] Failed to load {test_file.name}: {e}")
            continue

        # Discover TestCase classes
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if inspect.isclass(attr) and issubclass(attr, unittest.TestCase):
                class_suite = unittest.defaultTestLoader.loadTestsFromTestCase(attr)
                suite.addTest(class_suite)
                total_discovered += class_suite.countTestCases()

        # Discover standalone test functions (pytest style)
        for attr_name in dir(module):
            if attr_name.startswith("test_") and callable(getattr(module, attr_name)):
                func = getattr(module, attr_name)
                if not inspect.isclass(func):
                    # Wrap function in a TestCase
                    class StandaloneFunctionTestCase(unittest.TestCase):
                        def __init__(self, fn, fn_name, mod_name):
                            super().__init__()
                            self._fn = fn
                            self._fn_name = fn_name
                            self._mod_name = mod_name

                        def runTest(self):
                            sig = inspect.signature(self._fn)
                            kwargs = {}
                            temp_dirs = []
                            for param in sig.parameters.values():
                                if param.name == "tmp_path":
                                    td = Path(tempfile.mkdtemp(prefix="cryptograph_test_"))
                                    temp_dirs.append(td)
                                    kwargs["tmp_path"] = td

                            try:
                                self._fn(**kwargs)
                            finally:
                                for td in temp_dirs:
                                    shutil.rmtree(td, ignore_errors=True)

                        def id(self):
                            return f"{self._mod_name}::{self._fn_name}"

                        def __str__(self):
                            return f"{self._mod_name}::{self._fn_name}"

                    tc = StandaloneFunctionTestCase(func, attr_name, test_file.name)
                    suite.addTest(tc)
                    total_discovered += 1

    print(f"[CryptoGraph Test Runner] Discovered {total_discovered} tests across {len(test_files)} test files.\n")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    exit_code = run_with_pytest()
    if exit_code == -1:
        exit_code = run_with_unittest()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
