"""SymPy conversion helpers.

These utilities exist primarily to keep the public APIs tolerant to the data
representations that appear in notebooks and cross-language bridges (e.g.
Julia/PythonCall passing rationals as integer tuples).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, overload

import sympy as sym


def _tuples_to_rationals_2d(A: Sequence[Sequence[Tuple[int, int]]]) -> sym.Matrix:
    """Convert a 2D array of ``(num, denom)`` pairs to a SymPy Matrix."""

    return sym.Matrix([[sym.Rational(num, denom) for (num, denom) in row] for row in A])


def _tuples_to_rationals_1d(v: Sequence[Tuple[int, int]]) -> sym.Matrix:
    """Convert a 1D list of ``(num, denom)`` pairs to a SymPy column vector."""

    return sym.Matrix([sym.Rational(num, denom) for (num, denom) in v])


def to_sympy_matrix(A: Any) -> Optional[sym.Matrix]:
    """Best-effort conversion of ``A`` to a SymPy Matrix.

    Parameters
    ----------
    A:
        None, a SymPy Matrix, a (nested) Python sequence, or a NumPy array.
        For convenience when crossing language boundaries, ``(num, denom)``
        tuples are interpreted as rationals.
    """

    if A is None:
        return None
    if isinstance(A, sym.MatrixBase):
        return sym.Matrix(A)

    # NumPy arrays (and similar) expose ``shape`` and are iterable; try Matrix(A) directly.
    try:
        return sym.Matrix(A)
    except Exception:
        pass

    # Handle Julia-style rationals represented as tuples.
    if isinstance(A, list) and A:
        if isinstance(A[0], list) and A[0] and isinstance(A[0][0], tuple):
            return _tuples_to_rationals_2d(A)  # type: ignore[arg-type]
        if isinstance(A[0], tuple):
            return _tuples_to_rationals_1d(A)  # type: ignore[arg-type]

    raise TypeError(f"Cannot convert object of type {type(A)} to sympy.Matrix")


def to_sympy_col(rhs: Any) -> Optional[sym.Matrix]:
    """Convert a right-hand side to a SymPy column vector (or None)."""

    if rhs is None:
        return None
    m = to_sympy_matrix(rhs)
    if m is None:
        return None
    # Accept either shape (m,1) or (m,)
    if m.cols == 1:
        return m
    if m.rows == 1:
        return m.T
    # If it's a flat list, Matrix(list) returns (n,1); otherwise error.
    if m.cols != 1:
        raise ValueError(f"rhs must be a vector; got shape {m.shape}")
    return m
