"""Automated tests for the lossless source-coding project.

Run with::

    python -m unittest discover -s tests -v

or equivalently::

    python -m unittest tests.test_units tests.test_codecs
"""

import sys
from pathlib import Path

# Make ``from src import ...`` work when tests are run from any cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
