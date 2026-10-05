"""Shared pytest configuration for the sphinx-asr test suite.

The Python code under ``scripts/`` is not an installable package, so the
directories the scripts themselves rely on are put on ``sys.path`` here:

* ``scripts/`` so ``lib.config``, ``corpus`` and the top-level CLI modules
  (``train``, ``lm``, ...) import the same way they do under ``sphinx.sh``.
* ``scripts/lib/`` so helper modules without relative imports (``asr_util``)
  can be imported directly.
* ``tools/`` so developer tooling such as ``check_test_docstrings`` is
  importable from its tests.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for _sub in ("scripts", "scripts/lib", "tools"):
    _path = str(REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
