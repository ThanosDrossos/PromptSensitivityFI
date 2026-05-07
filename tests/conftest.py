"""Shared pytest fixtures.

The tests deliberately avoid loading the singleton config so each test gets a
fresh state.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `pytest` to import the package without installation when running from
# the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
