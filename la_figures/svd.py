r"""SVD description objects for :mod:`matrixlayout`.

The SVD table layout in :mod:`matrixlayout` expects an ``eig``-style dictionary
containing (for ``case='SVD'``):

- ``sigma``: distinct singular values (descending)
- ``lambda``: corresponding values of :math:`\sigma^2` (eigenvalues of the
  Gramian :math:`A^T A`, aligned with ``sigma``)
- ``ma``: multiplicities / dimensions of the associated singular subspaces
- ``qvecs``: orthonormal right singular vector groups (columns of ``V``)
- ``uvecs``: corresponding left singular vector groups (columns of ``U``)
- ``sz``: original matrix shape ``(m, n)``

Historically, the legacy implementation obtained the right singular vectors by
eigendecomposing the Gramian ``A.T*A`` and taking square-roots of eigenvalues.
This module keeps that algorithmic path, but uses standard SVD terminology
throughout.

This module owns algorithmic logic only; it does not render.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sym

from ._sympy_utils import to_sympy_matrix
from .eig import _maybe_round, _maybe_round_vectors
from .gram_schmidt import q_gram_schmidt


# Each item describes one singular subspace via (sigma^2, multiplicity, vectors).
# This matches SymPy's ``Matrix.eigenvects()`` structure when applied to A.T*A,
# but the vectors should be interpreted as *right singular vectors*.
RightSingularSpaceItem = Tuple[Any, Any, Sequence[Any]]


def _sorted_right_singular_spaces(
    spaces: Iterable[RightSingularSpaceItem],
) -> List[tuple[Any, int, List[sym.Matrix]]]:
    """Normalize and sort singular subspaces by ``sigma^2`` descending."""
    out: List[tuple[Any, int, List[sym.Matrix]]] = []
    for (sigma2, m, vecs) in spaces:
        out.append((sigma2, int(m), [sym.Matrix(v) for v in vecs]))
    return sorted(out, key=lambda t: t[0], reverse=True)


def svd_tbl_spec_from_right_singular_vectors(
    A: Any,
    right_singular_spaces: Iterable[RightSingularSpaceItem],
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Build an SVD-table spec from precomputed right singular subspaces.

    This is the primary "populate the eig structure" entrypoint for SVD tables.
    It is useful when an upstream algorithm has already computed the right
    singular subspaces (typically via the Gramian :math:`A^T A`) and wants to
    reuse them for multiple downstream visualizations.

    Parameters
    ----------
    A:
        The original matrix.
    right_singular_spaces:
        Iterable of ``(sigma2, multiplicity, vectors)`` where ``sigma2`` is
        :math:`\sigma^2` for that singular subspace and ``vectors`` span the
        corresponding **right singular** subspace (columns of :math:`V`).

        This is compatible with ``sympy.Matrix(A.T*A).eigenvects()`` (where the
        eigenvectors are exactly right singular vectors), but this function uses
        SVD terminology throughout.
    Ascale:
        Optional scaling factor carried over from the legacy API.
        - The provided ``sigma2`` values are divided by ``Ascale**2`` before
          taking square-roots.
        - Left singular vectors use ``1/(sigma*Ascale)`` when ``Ascale`` is set.
    sigma2_digits / eig_digits:
        Rounding control for :math:`\sigma^2`. ``eig_digits`` is kept as a
        backwards-compatible alias (do not pass both).
    sigma_digits:
        Rounding control for :math:`\sigma`.
    vec_digits:
        Rounding control for vector entries.

    Returns
    -------
    dict
        A dictionary compatible with ``matrixlayout.eigproblem_tex(case='SVD')``.
    """

    if sigma2_digits is not None and eig_digits is not None:
        raise ValueError("Provide at most one of sigma2_digits or eig_digits")
    if sigma2_digits is None:
        sigma2_digits = eig_digits

    A = to_sympy_matrix(A)
    if A is None:
        raise ValueError("A must not be None")

    eig: Dict[str, Any] = {
        "sigma": [],
        "lambda": [],  # stores sigma^2 to match matrixlayout's expected key name
        "ma": [],
        "evecs": [],  # legacy key; in SVD case this stores (possibly non-orthonormal) right singular vectors
        "qvecs": [],  # orthonormal right singular vectors (V)
        "uvecs": [],  # left singular vectors (U), grouped by distinct sigma
        "sz": tuple(A.shape),
    }

    # Sort by sigma^2 descending (matches legacy behavior for A.T*A eigenpairs).
    pairs = _sorted_right_singular_spaces(right_singular_spaces)

    for (sigma2, m, vecs) in pairs:
        sigma2_scaled = sigma2 if Ascale is None else (sigma2 / (Ascale * Ascale))
        sigma2_scaled = _maybe_round(sigma2_scaled, sigma2_digits)

        sigma = sym.sqrt(sigma2_scaled)
        sigma = _maybe_round(sigma, sigma_digits)

        vecs2 = _maybe_round_vectors(vecs, vec_digits)
        V_cols = q_gram_schmidt(vecs2)

        eig["lambda"].append(sigma2_scaled)
        eig["sigma"].append(sigma)
        eig["ma"].append(int(m))
        eig["evecs"].append(vecs2)
        eig["qvecs"].append(V_cols)

        # Left singular vectors for non-zero singular values: u = (1/sigma) A v
        if not getattr(sigma, "is_zero", False):
            s_inv = (1 / sigma) if Ascale is None else (1 / (sigma * Ascale))
            eig["uvecs"].append([s_inv * A * v for v in V_cols])

    # If A is rank-deficient, complement U with an orthonormal basis for null(A^T).
    ns = A.T.nullspace()
    if ns:
        eig["uvecs"].append(q_gram_schmidt([sym.Matrix(v) for v in ns]))

    return eig


def svd_tbl_spec(
    A: Any,
    *,
    Ascale: Optional[Any] = None,
    sigma2_digits: Optional[int] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    r"""Compute the SVD-table description object consumed by matrixlayout.

    This computes right singular vectors via an eigendecomposition of the
    Gramian :math:`A^T A` and converts :math:`\sigma^2` to :math:`\sigma`.

    Parameters
    ----------
    A:
        Matrix (array-like). Converted to a SymPy Matrix.
    Ascale:
        Optional scaling. Matches the legacy SVD helper:
        - :math:`\sigma^2` values are divided by ``Ascale**2`` before taking square-roots
        - left singular vectors use ``1/(sigma*Ascale)``
    sigma2_digits / eig_digits:
        Optional rounding control for :math:`\sigma^2`. ``eig_digits`` is kept
        as a backwards-compatible alias (do not pass both).
    sigma_digits, vec_digits:
        Optional rounding controls kept for compatibility with the legacy API.

    Returns
    -------
    dict
        Keys:
          - ``sigma``: distinct singular values (descending)
          - ``lambda``: corresponding :math:`\sigma^2` values (descending)
          - ``ma``: multiplicities
          - ``evecs``: right singular vector groups (not necessarily orthonormal)
          - ``qvecs``: orthonormal right singular vector groups (V columns)
          - ``uvecs``: corresponding left singular vector groups (U columns)
          - ``sz``: original matrix size ``(m, n)`` (useful for layout)
    """

    if sigma2_digits is not None and eig_digits is not None:
        raise ValueError("Provide at most one of sigma2_digits or eig_digits")

    A = to_sympy_matrix(A)
    if A is None:
        raise ValueError("A must not be None")

    G = A.T * A  # Gramian
    return svd_tbl_spec_from_right_singular_vectors(
        A,
        G.eigenvects(),
        Ascale=Ascale,
        sigma2_digits=sigma2_digits,
        eig_digits=eig_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )


# Backwards-compatible name retained for earlier patch iterations.
# Prefer the SVD-terminology name going forward.
def svd_tbl_spec_from_AtA_eigenvects(
    A: Any,
    AtA_eigenvects: Iterable[RightSingularSpaceItem],
    *,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    sigma_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
) -> Dict[str, Any]:
    """Deprecated alias for :func:`svd_tbl_spec_from_right_singular_vectors`."""

    return svd_tbl_spec_from_right_singular_vectors(
        A,
        AtA_eigenvects,
        Ascale=Ascale,
        eig_digits=eig_digits,
        sigma_digits=sigma_digits,
        vec_digits=vec_digits,
    )
