from __future__ import annotations

import os
import sys

import pytest
from pathlib import Path


def _ensure_monorepo_imports() -> None:
    """Ensure monorepo sibling packages are importable for cross-package tests.

    During the migration, developers commonly run pytest from a checkout without
    installing the packages in editable mode. Several tests intentionally
    exercise cross-package integration (e.g., :mod:`la_figures` ->
    :mod:`matrixlayout`), so we make the sibling package roots importable.

    This is test-harness-only wiring; production dependencies remain governed
    by each package's packaging metadata.
    """

    # This file lives at la_figures/tests/...; the monorepo root is two levels up.
    repo_root = Path(__file__).resolve().parents[2]

    # Add per-package roots. These are the directories that contain the
    # corresponding importable packages.
    pkg_roots = [
        repo_root / "la_figures",
        repo_root / "matrixlayout",
        repo_root / "jupyter_tikz",
        repo_root / "itikz",  # legacy characterization tests
    ]

    # Ensure predictable precedence: insert in reverse order so the first item
    # ends up first in sys.path.
    for p in reversed(pkg_roots):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


_ensure_monorepo_imports()

# --- render test controls -------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--skip-render-tests",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.render",
    )


def pytest_collection_modifyitems(config, items):
    skip = config.getoption("--skip-render-tests") or os.environ.get("ITIKZ_SKIP_RENDER_TESTS") == "1"
    if not skip:
        return

    marker = pytest.mark.skip(reason="render tests skipped (set ITIKZ_SKIP_RENDER_TESTS=0 / omit --skip-render-tests)")
    for item in items:
        if "render" in item.keywords:
            item.add_marker(marker)

