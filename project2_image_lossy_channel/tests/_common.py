"""Shared helpers for the test suite.

Ensures the project root is on `sys.path` so `from src import ...` works
regardless of how the tests are launched (pytest, unittest, or
`python tests/run_all.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
