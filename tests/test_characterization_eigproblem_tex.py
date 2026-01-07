"""Characterization tests: legacy itikz nicematrix vs migrated matrixlayout.

These tests compare the TeX emitted by legacy ``itikz.nicematrix`` with the TeX
emitted by the migrated ``matrixlayout`` implementation, when fed description
objects produced by :mod:`la_figures`.

If the legacy package is not importable (e.g. in a trimmed install), the tests
are skipped.
"""

from __future__ import annotations

import sys


def _ensure_repo_on_path() -> None:
    """Make the monorepo root importable for cross-package tests."""

    # This file lives at la_figures/tests/...; the repo root is two levels up.
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]

    # The monorepo uses per-package roots (e.g. ./itikz/itikz is the legacy
    # package). Add the itikz root so `import itikz.nicematrix` works when
    # running from a checkout without installing the legacy package.
    itikz_root = root / "itikz"

    # Ensure itikz_root appears before the monorepo root to avoid Python
    # resolving `itikz` as a namespace package rooted at ./itikz.
    for p in (root, itikz_root):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)

    for p in (root, itikz_root):
        sys.path.insert(0, str(p))


def _normalize_tex(s: str) -> str:
    """Normalization that is stable across minor whitespace differences."""

    # Keep line breaks (helps debugging), but strip trailing spaces.
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def test_eigenproblem_tex_matches_legacy_for_simple_matrix():
    _ensure_repo_on_path()

    try:
        from itikz.nicematrix import eig_tbl  # type: ignore
    except Exception:
        # Legacy not present in this environment.
        import pytest

        pytest.skip("legacy itikz.nicematrix not importable")

    import sympy as sym

    from la_figures import eig_tbl_spec
    from matrixlayout import eigproblem_tex

    A = sym.Matrix([[2, 0], [0, 3]])

    legacy = eig_tbl(A).nm_latex_doc(case="S", formater=sym.latex)
    new = eigproblem_tex(eig_tbl_spec(A), case="S", formater=sym.latex)

    assert _normalize_tex(new) == _normalize_tex(legacy)


def test_svd_tex_matches_legacy_for_rank_deficient_matrix():
    _ensure_repo_on_path()

    try:
        from itikz.nicematrix import svd_tbl  # type: ignore
    except Exception:
        import pytest

        pytest.skip("legacy itikz.nicematrix not importable")

    import sympy as sym

    from la_figures import svd_tbl_spec
    from matrixlayout import eigproblem_tex

    A = sym.Matrix([[3, 0], [0, 0]])

    legacy = svd_tbl(A).nm_latex_doc(case="SVD", formater=sym.latex, sz=A.shape)
    new = eigproblem_tex(svd_tbl_spec(A), case="SVD", formater=sym.latex)

    assert _normalize_tex(new) == _normalize_tex(legacy)
