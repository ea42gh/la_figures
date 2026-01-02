"""SVD description objects for :mod:`matrixlayout`.

The legacy ``itikz.nicematrix`` implementation built an SVD table by computing
the eigen-decomposition of ``A^T A`` and converting eigenvalues to singular
values.

This module ports that algorithm into :mod:`la_figures` and emits a dictionary
compatible with :func:`matrixlayout.eigproblem_tex` (case="SVD").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import sympy as sym

from ._sympy_utils import to_sympy_matrix
from .gram_schmidt import q_gram_schmidt
from .eig import _maybe_round, _maybe_round_vectors


def _svd_eig_pairs(AtA: sym.Matrix) -> List[tuple[Any, int, List[sym.Matrix]]]:
    """Return eigenpairs of ``AtA`` sorted by eigenvalue descending."""

    pairs: List[tuple[Any, int, List[sym.Matrix]]] = []
    for (e, m, vecs) in AtA.eigenvects():
        pairs.append((e, int(m), [sym.Matrix(v) for v in vecs]))
    return sorted(pairs, key=lambda t: t[0], reverse=True)


def svd_tbl_spec(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the SVD-table description object consumed by matrixlayout.

    Parameters
    ----------
    A:
        Matrix (array-like). Converted to a SymPy Matrix.
    Ascale:
        Optional scaling. Matches the legacy SVD helper:
        - eigenvalues are divided by ``Ascale**2``
        - left singular vectors use ``1/(sigma*Ascale)``
    eig_digits, sigma_digits, vec_digits:
        Optional rounding controls kept for compatibility with the legacy API.

    Returns
    -------
    dict
        Keys:
          - ``sigma``: distinct singular values (descending)
          - ``lambda``: eigenvalues of ``A^T A`` (descending)
          - ``ma``: multiplicities
          - ``evecs``: eigenvector groups of ``A^T A``
          - ``qvecs``: orthonormalized right singular vector groups (V columns)
          - ``uvecs``: corresponding left singular vector groups (U columns)
          - ``sz``: original matrix size ``(m, n)`` (useful for layout)
    """

    A = to_sympy_matrix(A)
    if A is None:
        raise ValueError("A must not be None")

    eig: Dict[str, Any] = {
        "sigma": [],
        "lambda": [],
        "ma": [],
        "evecs": [],
        "qvecs": [],
        "uvecs": [],
        "sz": tuple(A.shape),
    }

    AtA = A.T * A
    for (e, m, vecs) in _svd_eig_pairs(AtA):
        e_scaled = e if Ascale is None else (e / (Ascale * Ascale))
        e_scaled = _maybe_round(e_scaled, eig_digits)
        sigma = sym.sqrt(e_scaled)
        sigma = _maybe_round(sigma, sigma_digits)

        vecs2 = _maybe_round_vectors(vecs, vec_digits)
        vvecs = q_gram_schmidt(vecs2)

        eig["lambda"].append(e_scaled)
        eig["sigma"].append(sigma)
        eig["ma"].append(int(m))
        eig["evecs"].append(vecs2)
        eig["qvecs"].append(vvecs)

        # Left singular vectors for non-zero singular values.
        if not getattr(sigma, "is_zero", False):
            s_inv = (1 / sigma) if Ascale is None else (1 / (sigma * Ascale))
            eig["uvecs"].append([s_inv * A * v for v in vvecs])

    # If A is rank-deficient, complement U with an orthonormal basis for null(A^T).
    ns = A.T.nullspace()
    if ns:
        eig["uvecs"].append(q_gram_schmidt([sym.Matrix(v) for v in ns]))

    return eig
