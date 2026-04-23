"""Convenience runner: discover and run every test in this folder.

Usage (from the project root):
    python -m tests.run_all
or  python tests/run_all.py

Exits with a non-zero status if any test fails, so it is safe to use in CI.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(HERE), pattern="test_*.py", top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
