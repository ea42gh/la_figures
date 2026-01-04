"""SymPy conversion helpers.

These utilities exist primarily to keep the public APIs tolerant to the data
representations that appear in notebooks and cross-language bridges (e.g.
Julia/PyCall or Julia/PythonCall passing rationals as integer tuples).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numbers
import sympy as sym


def _looks_like_juliacall_wrapper(x: Any) -> bool:
    """Heuristic check for PythonCall/JuliaCall wrapper objects.

    When calling Python from Julia via PythonCall.jl, Julia objects that do not
    have a direct scalar conversion are wrapped as ``juliacall.*Value`` Python
    objects (e.g. arrays as ``juliacall.ArrayValue``). We avoid importing
    juliacall here; instead, we rely on the module name.
    """

    mod = getattr(type(x), "__module__", "")
    return mod.startswith("juliacall")


def _juliacall_array_to_nested_list(A: Any) -> Optional[list]:
    """Best-effort conversion for ``juliacall.ArrayValue`` inputs.

    SymPy does not understand the JuliaCall array wrappers directly.
    The wrappers often satisfy the NumPy array interface, but not all call
    sites have NumPy available or want to depend on it. This helper attempts a
    minimal, dependency-free conversion.
    """

    if not _looks_like_juliacall_wrapper(A):
        return None

    # Prefer a dedicated conversion hook if present.
    to_numpy = getattr(A, "to_numpy", None)
    if callable(to_numpy):
        try:
            arr = to_numpy()
            # NumPy arrays provide ``tolist``.
            if hasattr(arr, "tolist"):
                return arr.tolist()
        except Exception:
            pass

    shp = getattr(A, "shape", None)
    if shp is None:
        return None

    try:
        dims = tuple(int(d) for d in shp)
    except Exception:
        return None

    # Julia arrays are typically 1-based when accessed via the wrapper.
    if len(dims) == 2:
        m, n = dims
        try:
            return [[A[i + 1, j + 1] for j in range(n)] for i in range(m)]
        except Exception:
            return None
    if len(dims) == 1:
        n = dims[0]
        try:
            return [A[i + 1] for i in range(n)]
        except Exception:
            return None
    return None


def _tuples_to_rationals_2d(A: Sequence[Sequence[Tuple[int, int]]]) -> sym.Matrix:
    """Convert a 2D array of ``(num, denom)`` pairs to a SymPy Matrix."""
    return sym.Matrix([[sym.Rational(num, denom) for (num, denom) in row] for row in A])


def _tuples_to_rationals_1d(v: Sequence[Tuple[int, int]]) -> sym.Matrix:
    """Convert a 1D list of ``(num, denom)`` pairs to a SymPy column vector."""
    return sym.Matrix([sym.Rational(num, denom) for (num, denom) in v])


def _is_rational_tuple(x: Any) -> bool:
    """True iff ``x`` looks like a rational encoded as ``(num, denom)``."""
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    a, b = x
    return isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral)


def _is_int(x: Any) -> bool:
    return isinstance(x, numbers.Integral)


def to_sympy_matrix(A: Any) -> Optional[sym.Matrix]:
    """Best-effort conversion of ``A`` to a SymPy Matrix.

    Parameters
    ----------
    A:
        None, a SymPy Matrix, a (nested) Python sequence, or a NumPy-like array.
        For convenience when crossing language boundaries, ``(num, denom)``
        tuples are interpreted as rationals.

    Notes
    -----
    The tuple-to-rational detection intentionally runs *before* calling
    ``sym.Matrix(A)`` because SymPy can accept tuple objects as valid matrix
    elements, which would silently bypass the intended rational conversion.

    Some bridges also materialize a matrix of ``(num, denom)`` tuples as a
    rank-3 numeric array of shape ``(m, n, 2)``; that representation is also
    recognized and converted.
    """

    if A is None:
        return None
    if isinstance(A, sym.MatrixBase):
        return sym.Matrix(A)

    # ---- PythonCall/JuliaCall: Julia arrays are wrapped as juliacall.ArrayValue --------
    # SymPy does not natively understand these wrappers, so convert them to nested lists
    # (or to a NumPy array if available through the wrapper).
    jl_list = _juliacall_array_to_nested_list(A)
    if jl_list is not None:
        return sym.Matrix(jl_list)

    # ---- Julia bridge convenience: rationals passed as integer tuples -----------------
    # 2D Python sequences: [[(n,d), ...], ...]
    if isinstance(A, (list, tuple)) and A:
        first = A[0]
        if isinstance(first, (list, tuple)) and first and _is_rational_tuple(first[0]):
            return _tuples_to_rationals_2d(A)  # type: ignore[arg-type]
        if _is_rational_tuple(first):
            return _tuples_to_rationals_1d(A)  # type: ignore[arg-type]

    # NumPy-like "matrix of pairs": shape (m,n,2)
    shp = getattr(A, "shape", None)
    if shp is not None:
        try:
            if len(shp) == 3 and int(shp[2]) == 2:
                m, n, _ = map(int, shp)
                return sym.Matrix(
                    [[sym.Rational(int(A[i, j, 0]), int(A[i, j, 1])) for j in range(n)] for i in range(m)]
                )
        except Exception:
            pass

    # NumPy-like arrays: A[0,0] exists and may itself be a tuple.
    try:
        a00 = A[0, 0]
        if _is_rational_tuple(a00) and hasattr(A, "tolist"):
            return _tuples_to_rationals_2d(A.tolist())  # type: ignore[arg-type]
    except Exception:
        pass

    # 1D array-likes (vectors) with tuple rationals
    try:
        a0 = A[0]
        if _is_rational_tuple(a0):
            if hasattr(A, "tolist"):
                return _tuples_to_rationals_1d(A.tolist())  # type: ignore[arg-type]
            return _tuples_to_rationals_1d(list(A))  # type: ignore[arg-type]
    except Exception:
        pass

    # ---- Default path: let SymPy attempt to build a Matrix -----------------------------
    try:
        return sym.Matrix(A)
    except Exception as e:
        raise TypeError(f"Cannot convert object of type {type(A)} to sympy.Matrix") from e


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
    if m.cols != 1:
        raise ValueError(f"rhs must be a vector; got shape {m.shape}")
    return m
