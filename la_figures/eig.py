"""Eigen-decomposition description objects for :mod:`matrixlayout`.

The functions in this module compute eigenvalues/eigenvectors and package them
into a *description object* (a plain dictionary) suitable for
``matrixlayout.eigproblem_tex``.

This module intentionally contains *algorithmic* logic (it may call SymPy
decompositions). It does not render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


EigenVectsItem = Tuple[Any, Any, Sequence[Any]]


@dataclass
class EigenDecomposition:
    """Structured eigen-decomposition result.

    This is an algorithmic object intended to be convertible into the
    ``eig`` description dictionary consumed by :func:`matrixlayout.eigproblem_tex`.
    """

    lambdas: List[Any]
    multiplicities: List[int]
    evecs: List[List[sym.Matrix]]
    qvecs: Optional[List[List[sym.Matrix]]] = None

    def to_spec(self) -> Dict[str, Any]:
        eig: Dict[str, Any] = {
            "lambda": list(self.lambdas),
            "ma": list(self.multiplicities),
            "evecs": [list(g) for g in self.evecs],
        }
        if self.qvecs is not None:
            eig["qvecs"] = [list(g) for g in self.qvecs]
        return eig


def eig_spec_from_eigenvects(
    eigenvects: Iterable[EigenVectsItem],
    *,
    normal: bool = False,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    order: str = "legacy",
) -> Dict[str, Any]:
    """Convert a SymPy-style ``eigenvects()`` result into an eigproblem spec.

    Parameters
    ----------
    eigenvects:
        An iterable of ``(eigenvalue, multiplicity, eigenvectors)``.
        This is compatible with the output of ``sympy.Matrix.eigenvects()``.
    normal:
        If true, compute orthonormal bases within each eigenspace (``qvecs``).
    Ascale:
        If provided, eigenvalues are divided by ``Ascale``.
    eig_digits, vec_digits:
        Optional rounding controls for eigenvalues and eigenvector entries.
    order:
        Ordering policy for distinct eigenvalues.

        - ``"legacy"`` (default): matches the historical ``itikz`` behavior by
          reversing the order returned by SymPy.
        - ``"sympy"``: preserve SymPy's order.

    Returns
    -------
    dict
        An ``eig`` dictionary with keys ``lambda``, ``ma``, ``evecs``, and
        optionally ``qvecs``.
    """

    eig: Dict[str, Any] = {
        "lambda": [],
        "ma": [],
        "evecs": [],
    }
    if normal:
        eig["qvecs"] = []

    def _emit(e_scaled: Any, m_int: int, vecs2: List[sym.Matrix]) -> None:
        if order == "sympy":
            eig["lambda"].append(e_scaled)
            eig["ma"].append(m_int)
            eig["evecs"].append(vecs2)
            if normal:
                eig["qvecs"].append(q_gram_schmidt(vecs2))
        else:
            # Legacy inserts at 0 to reverse the order returned by SymPy.
            eig["lambda"].insert(0, e_scaled)
            eig["ma"].insert(0, m_int)
            eig["evecs"].insert(0, vecs2)
            if normal:
                eig["qvecs"].insert(0, q_gram_schmidt(vecs2))

    for (e, m, vecs) in eigenvects:
        e_scaled = e if Ascale is None else (e / Ascale)
        e_scaled = _maybe_round(e_scaled, eig_digits)

        vecs2 = [sym.Matrix(v) for v in vecs]
        vecs2 = _maybe_round_vectors(vecs2, vec_digits)

        _emit(e_scaled, int(m), vecs2)

    if order not in ("legacy", "sympy"):
        raise ValueError(f"Unsupported order={order!r}; expected 'legacy' or 'sympy'")

    return eig


def eigendecomposition(
    A: Any,
    *,
    normal: bool = False,
    Ascale: Optional[Any] = None,
    eig_digits: Optional[int] = None,
    vec_digits: Optional[int] = None,
    order: str = "legacy",
) -> EigenDecomposition:
    """Compute a structured eigen-decomposition of ``A``.

    This is a convenience routine intended for algorithmic consumers that want
    a typed object (rather than a plain dictionary) and/or want to reuse a
    decomposition output to populate multiple layouts.
    """

    A = to_sympy_matrix(A)
    if A is None:
        raise ValueError("A must not be None")
    if A.rows != A.cols:
        raise ValueError(f"eigendecomposition requires a square matrix; got shape {A.shape}")

    spec = eig_spec_from_eigenvects(
        A.eigenvects(),
        normal=normal,
        Ascale=Ascale,
        eig_digits=eig_digits,
        vec_digits=vec_digits,
        order=order,
    )
    return EigenDecomposition(
        lambdas=list(spec["lambda"]),
        multiplicities=[int(x) for x in spec["ma"]],
        evecs=[list(g) for g in spec["evecs"]],
        qvecs=[list(g) for g in spec["qvecs"]] if "qvecs" in spec else None,
    )


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

    return eig_spec_from_eigenvects(
        A.eigenvects(),
        normal=normal,
        Ascale=Ascale,
        eig_digits=eig_digits,
        vec_digits=vec_digits,
        order="legacy",
    )
