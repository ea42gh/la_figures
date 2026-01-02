"""Eigen-decomposition description objects for :mod:`matrixlayout`.

The functions in this module compute eigenvalues/eigenvectors and package them
into a *description object* (a plain dictionary) suitable for
``matrixlayout.eigproblem_tex``.

This module intentionally contains *algorithmic* logic (it may call SymPy
decompositions). It does not render.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import sympy as sym

from ._sympy_utils import to_sympy_matrix
from .gram_schmidt import q_gram_schmidt


def _maybe_round(x: Any, digits: Optional[int]) -> Any:
    """Round a scalar-like value if ``digits`` is provided.

    The legacy implementation applied Python's ``round`` directly. We keep the
    same best-effort approach, but leave non-roundable symbolic values as-is.
    """

    if digits is None:
        return x
    try:
        return round(x) if digits == 0 else round(x, digits)
    except Exception:
        return x


def _maybe_round_vectors(vecs: List[sym.Matrix], digits: Optional[int]) -> List[sym.Matrix]:
    """Round vector entries elementwise if ``digits`` is provided."""

    if digits is None:
        return vecs

    out: List[sym.Matrix] = []
    for v in vecs:
        vv = sym.Matrix(v)
        out.append(vv.applyfunc(lambda t: _maybe_round(t, digits)))
    return out


def eig_tbl_spec(
    A: Any,
    *,
    normal: bool = False,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the eigen-table description object consumed by matrixlayout.

    Parameters
    ----------
    A:
        Square matrix (array-like). Converted to a SymPy Matrix.
    normal:
        If true, compute an orthonormal basis within each eigenspace (``qvecs``).
    Ascale:
        If provided, eigenvalues are divided by ``Ascale`` (matches legacy).
    eig_digits, vec_digits:
        Optional rounding controls kept for compatibility with the legacy API.

    Returns
    -------
    dict
        Keys:
          - ``lambda``: distinct eigenvalues (ordered as in legacy)
          - ``ma``: corresponding algebraic multiplicities
          - ``evecs``: eigenvector groups (list per eigenvalue)
          - ``qvecs``: (optional) orthonormal eigenvector groups
    """

    A = to_sympy_matrix(A)
    if A is None:
        raise ValueError("A must not be None")

    if A.rows != A.cols:
        raise ValueError(f"eig_tbl_spec requires a square matrix; got shape {A.shape}")

    eig: Dict[str, Any] = {
        "lambda": [],
        "ma": [],
        "evecs": [],
    }
    if normal:
        eig["qvecs"] = []

    res = A.eigenvects()
    for (e, m, vecs) in res:
        e_scaled = e if Ascale is None else (e / Ascale)
        e_scaled = _maybe_round(e_scaled, eig_digits)
        vecs2 = [sym.Matrix(v) for v in vecs]
        vecs2 = _maybe_round_vectors(vecs2, vec_digits)

        # Legacy inserts at 0 to reverse the order returned by SymPy.
        eig["lambda"].insert(0, e_scaled)
        eig["ma"].insert(0, int(m))
        eig["evecs"].insert(0, vecs2)

        if normal:
            eig["qvecs"].insert(0, q_gram_schmidt(vecs2))

    return eig
