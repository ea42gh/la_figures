"""SVD description objects for :mod:`matrixlayout`.

This module builds an SVD table by computing the eigen-decomposition of
``A^T A`` and converting eigenvalues to singular values. It emits a dictionary
compatible with :func:`matrixlayout.eigproblem_tex` (case="SVD").
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


RightSingularSpaces = Iterable[Tuple[Any, int, Sequence[sym.Matrix]]]


def svd_tbl_spec_from_right_singular_vectors(
    A: Any,
    right_singular_spaces: RightSingularSpaces,
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    """Build an SVD-table spec from precomputed right singular subspaces.

    Parameters
    ----------
    A:
        The original matrix.
    right_singular_spaces:
        An iterable of ``(sigma2, multiplicity, vectors)`` triples.
        In exact arithmetic, these are the eigenspaces of the Gramian
        ``G = A.T * A``. The vectors are the *right singular vectors* (columns
        of ``V``) grouped by distinct ``sigma^2``.
    Ascale:
        Scaling behavior matches :func:`svd_tbl_spec`.
    sigma2_digits, sigma_digits, vec_digits:
        Optional rounding controls.
        ``sigma2_digits`` rounds the distinct ``sigma^2`` values.

    Returns
    -------
    dict
        Compatible with :func:`matrixlayout.eigproblem_tex` (case="SVD").
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

    triples: List[Tuple[Any, int, List[sym.Matrix]]] = []
    for (sigma2, m, vecs) in right_singular_spaces:
        triples.append((sigma2, int(m), [sym.Matrix(v) for v in vecs]))

    # Sort by sigma^2 descending to match the standard table ordering.
    triples.sort(key=lambda t: t[0], reverse=True)

    for (sigma2, m, vecs) in triples:
        sigma2_scaled = sigma2 if Ascale is None else (sigma2 / (Ascale * Ascale))
        sigma2_scaled = _maybe_round(sigma2_scaled, sigma2_digits)
        sigma = sym.sqrt(sigma2_scaled)
        sigma = _maybe_round(sigma, sigma_digits)

        vecs2 = _maybe_round_vectors(vecs, vec_digits)
        vvecs = q_gram_schmidt(vecs2)

        eig["lambda"].append(sigma2_scaled)
        eig["sigma"].append(sigma)
        eig["ma"].append(int(m))
        eig["evecs"].append(vecs2)
        eig["qvecs"].append(vvecs)

        if not getattr(sigma, "is_zero", False):
            s_inv = (1 / sigma) if Ascale is None else (1 / (sigma * Ascale))
            eig["uvecs"].append([s_inv * A * v for v in vvecs])

    # Complement U with an orthonormal basis for null(A^T) when rank-deficient.
    ns = A.T.nullspace()
    if ns:
        eig["uvecs"].append(q_gram_schmidt([sym.Matrix(v) for v in ns]))

    return eig


def svd_tbl_spec(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    sigma2_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the SVD-table description object consumed by matrixlayout.

    Parameters
    ----------
    A:
        Matrix (array-like). Converted to a SymPy Matrix.
    Ascale:
        Optional scaling:
        - eigenvalues are divided by ``Ascale**2``
        - left singular vectors use ``1/(sigma*Ascale)``
    eig_digits, sigma2_digits, sigma_digits, vec_digits:
        Optional rounding controls.
        ``sigma2_digits`` is an alias for ``eig_digits``.

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

    if eig_digits is None and sigma2_digits is not None:
        eig_digits = sigma2_digits

    # Use the same implementation path as the precomputed-right-singular-vectors API.
    G = A.T * A
    return svd_tbl_spec_from_right_singular_vectors(
        A,
        G.eigenvects(),
        Ascale=Ascale,
        sigma2_digits=eig_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )
