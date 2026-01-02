"""Test configuration.

The migration repository is a multi-package tree (``matrixlayout``, ``itikz``,
``jupyter_tikz``, and now ``la_figures``). Depending on how pytest determines
the root directory, imports may not resolve without a small sys.path tweak.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    # Ensure both the la_figures project root and the monorepo root are importable.
    here = Path(__file__).resolve()
    la_figures_root = here.parents[1]
    repo_root = here.parents[2]
    for p in (la_figures_root, repo_root):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
